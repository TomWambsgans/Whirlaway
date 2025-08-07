use std::{any::TypeId, borrow::Borrow};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{BasedVectorSpace, PackedValue};
use p3_field::{ExtensionField, Field, TwoAdicField};
use rayon::prelude::*;
use tracing::instrument;
use utils::{
    batch_fold_multilinear_in_large_field, batch_fold_multilinear_in_small_field,
    batch_fold_multilinear_in_small_field_no_skip, univariate_selectors,
};
use whir_p3::{
    fiat_shamir::prover::ProverState,
    poly::{dense::WhirDensePolynomial, evals::EvaluationsList},
};

use crate::{SumcheckComputation, SumcheckComputationPacked, SumcheckGrinding};

pub const MIN_VARS_FOR_GPU: usize = 0; // When there are a small number of variables, it's not worth using GPU

#[allow(clippy::too_many_arguments)]
pub fn prove<F, NF, EF, M, SC, Challenger>(
    skips: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: &[M],
    computation: &SC,
    constraints_degree: usize,
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut ProverState<F, EF, Challenger>,
    mut sum: EF,
    n_rounds: Option<usize>,
    grinding: SumcheckGrinding,
    mut missing_mul_factor: Option<EF>,
) -> (Vec<EF>, Vec<EvaluationsList<EF>>, EF)
where
    F: TwoAdicField,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F> + TwoAdicField,
    M: Borrow<EvaluationsList<NF>>,
    SC: SumcheckComputation<F, NF, EF>
        + SumcheckComputation<F, EF, EF>
        + SumcheckComputationPacked<F, EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let multilinears = multilinears.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    let mut n_vars = multilinears[0].num_variables();
    assert!(multilinears.iter().all(|m| m.num_variables() == n_vars));

    let mut challenges = Vec::new();
    let n_rounds = n_rounds.unwrap_or(n_vars - skips + 1);
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.len(), n_vars - skips + 1);
    }

    let mut folded_multilinears = sc_round(
        skips,
        &multilinears,
        &mut n_vars,
        computation,
        eq_factor,
        batching_scalars,
        is_zerofier,
        fs_prover,
        constraints_degree,
        &mut sum,
        grinding,
        &mut challenges,
        0,
        &mut missing_mul_factor,
    );

    for i in 1..n_rounds {
        folded_multilinears = sc_round(
            1,
            &folded_multilinears.iter().collect::<Vec<_>>(),
            &mut n_vars,
            computation,
            eq_factor,
            batching_scalars,
            false,
            fs_prover,
            constraints_degree,
            &mut sum,
            grinding,
            &mut challenges,
            i,
            &mut missing_mul_factor,
        );
    }

    (challenges, folded_multilinears, sum)
}

#[instrument(name = "sumcheck_round", skip_all, fields(round))]
#[allow(clippy::too_many_arguments)]
pub fn sc_round<F, NF, EF, SC, Challenger>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &[&EvaluationsList<NF>],
    n_vars: &mut usize,
    computation: &SC,
    eq_factor: Option<&[EF]>,
    batching_scalars: &[EF],
    is_zerocheck: bool,
    fs_prover: &mut ProverState<F, EF, Challenger>,
    comp_degree: usize,
    sum: &mut EF,
    grinding: SumcheckGrinding,
    challenges: &mut Vec<EF>,
    round: usize,
    missing_mul_factor: &mut Option<EF>,
) -> Vec<EvaluationsList<EF>>
where
    F: TwoAdicField,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F> + TwoAdicField,
    SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let eq_mle = eq_factor.map(|eq_factor| EvaluationsList::eval_eq(&eq_factor[1 + round..]));

    let selectors = univariate_selectors::<F>(skips);

    let mut p_evals = Vec::<(F, EF)>::new();
    let start = if is_zerocheck {
        p_evals.extend((0..1 << skips).map(|i| (F::from_usize(i), EF::ZERO)));
        1 << skips
    } else {
        0
    };
    for z in start..=comp_degree * ((1 << skips) - 1) {
        let sum_z = if z == (1 << skips) - 1 {
            if let Some(eq_factor) = eq_factor {
                (*sum
                    - (0..(1 << skips) - 1)
                        .map(|i| p_evals[i].1 * selectors[i].evaluate(eq_factor[round]))
                        .sum::<EF>())
                    / selectors[(1 << skips) - 1].evaluate(eq_factor[round])
            } else {
                *sum - p_evals.iter().map(|(_, s)| *s).sum::<EF>()
            }
        } else {
            let folding_scalars = selectors
                .iter()
                .map(|s| s.evaluate(F::from_usize(z)))
                .collect::<Vec<_>>();
            // If skips == 1 (ie classic sumcheck round, we could avoid 1 multiplication below: TODO not urgent)
            let folded = batch_fold_multilinear_in_small_field(multilinears, &folding_scalars);
            let mut sum_z =
                compute_over_hypercube(&folded, computation, batching_scalars, eq_mle.as_ref());
            if let Some(missing_mul_factor) = missing_mul_factor {
                sum_z *= *missing_mul_factor;
            }

            sum_z
        };

        p_evals.push((F::from_usize(z), sum_z));
    }

    let mut p = WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some(eq_factor) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        p *= &WhirDensePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| (F::from_usize(i), selectors[i].evaluate(eq_factor[round])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }

    fs_prover.add_extension_scalars(&p.coeffs);

    let challenge = fs_prover.sample();
    challenges.push(challenge);
    *sum = p.evaluate(challenge);
    *n_vars -= skips;

    let pow_bits = grinding
        .pow_bits::<EF>((comp_degree + usize::from(eq_factor.is_some())) * ((1 << skips) - 1));
    fs_prover.pow_grinding(pow_bits);

    let folding_scalars = selectors
        .iter()
        .map(|s| s.evaluate(challenge))
        .collect::<Vec<_>>();
    if let Some(eq_factor) = eq_factor {
        *missing_mul_factor = Some(
            selectors
                .iter()
                .map(|s| s.evaluate(eq_factor[round]) * s.evaluate(challenge))
                .sum::<EF>()
                * missing_mul_factor.unwrap_or(EF::ONE),
        );
    }
    // If skips == 1 (ie classic sumcheck round, we could avoid 1 multiplication below: TODO not urgent)
    batch_fold_multilinear_in_large_field(multilinears, &folding_scalars)
}

#[instrument(name = "sumcheck_round", skip_all, fields(round))]
#[allow(clippy::too_many_arguments)]
pub fn sc_round_no_skip<F, NF, EF, SC, Challenger>(
    // skips: usize,
    multilinears: &[&EvaluationsList<NF>],
    n_vars: &mut usize,
    computation: &SC,
    eq_factor: Option<&[EF]>,
    batching_scalars: &[EF],
    is_zerofier: bool,
    fs_prover: &mut ProverState<F, EF, Challenger>,
    comp_degree: usize,
    sum: &mut EF,
    grinding: SumcheckGrinding,
    challenges: &mut Vec<EF>,
    round: usize,
    missing_mul_factor: &mut Option<EF>,
) -> Vec<EvaluationsList<EF>>
where
    F: TwoAdicField,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F> + TwoAdicField,
    SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let eq_mle = eq_factor.map(|eq_factor| EvaluationsList::eval_eq(&eq_factor[1 + round..]));

    // let selectors = univariate_selectors::<F>(skips);
    // S_0(x) = 1 - x
    // S_1(x) = x

    let mut p_evals = Vec::<(F, EF)>::new();
    let start = if is_zerofier {
        p_evals.push((F::ZERO, EF::ZERO));
        p_evals.push((F::ONE, EF::ZERO));
        2
    } else {
        0
    };

    for z in start..=comp_degree {
        let sum_z = if z == 1 {
            if let Some(eq_factor) = eq_factor {
                *sum - p_evals[0].1 * (EF::ONE - eq_factor[round]) / eq_factor[round]
            } else {
                *sum - p_evals[0].1
            }
        } else {
            let folded = if z == 0 {
                multilinears
                    .par_iter()
                    .map(|poly| EvaluationsList::new(poly.evals()[..((1 << *n_vars) / 2)].to_vec()))
                    .collect()
            } else {
                let folding_scalars = vec![F::ONE - F::from_usize(z), F::from_usize(z)];
                batch_fold_multilinear_in_small_field_no_skip(multilinears, &folding_scalars)
            };
            let mut sum_z =
                compute_over_hypercube(&folded, computation, batching_scalars, eq_mle.as_ref());

            if let Some(missing_mul_factor) = missing_mul_factor {
                sum_z *= *missing_mul_factor;
            }

            sum_z
        };

        p_evals.push((F::from_usize(z), sum_z));
    }

    let mut p = WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some(eq_factor) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        // p *= &WhirDensePolynomial::lagrange_interpolation(
        //     vec![
        //         (F::ZERO, EF::ONE - eq_factor[round]),
        //         (F::ONE, eq_factor[round]),
        //     ], // &(0..2)
        //        //     .into_par_iter()
        //        //     .map(|i| (F::from_usize(i), selectors[i].evaluate(eq_factor[round])))
        //        //     .collect::<Vec<_>>(),
        // )
        //.unwrap();
        // El polinomio que quiero interpolar es uno lineal por lo que lo busco directamente
        // El P(x) = y0 + (y1 - y0) * x entonces en nuestro caso y0 = 1 - eq_factor[round]
        // y y1 = eq_factor[round], al sustituir nos queda algo como
        // P(x) = (1 - eq_factor[round]) + (eq_factor[round] - (1 - eq_factor[round])) * x
        // P(x) = (1 - eq_factor[round]) + (eq_factor[round] - 1 + eq_factor[round]) * x
        // P(x) = (1 - eq_factor[round]) + (2 * eq_factor[round] - 1) * x

        let a = EF::ONE - eq_factor[round];
        let b = EF::from_usize(2) * eq_factor[round] - EF::ONE;
        let selector_poly = WhirDensePolynomial::from_coefficients_vec(vec![a, b]);
        p *= &selector_poly;
    }

    fs_prover.add_extension_scalars(&p.coeffs);

    let challenge = fs_prover.sample();
    challenges.push(challenge);
    *sum = p.evaluate(challenge);
    *n_vars -= 1;

    let pow_bits = grinding.pow_bits::<EF>(comp_degree + usize::from(eq_factor.is_some()));
    fs_prover.pow_grinding(pow_bits);

    // let folding_scalars = selectors
    //     .iter()
    //     .map(|s| s.evaluate(challenge))
    //     .collect::<Vec<_>>();
    let folding_scalars = vec![EF::ONE - challenge, challenge];
    // Aca se puuede hacer una manipulacion similar a la de arriba cuando interpolamos
    //  sum = selector[0](eq_factor[round]) * selector[0](challenge) +
    //       selector[1](eq_factor[round]) * selector[1](challenge)
    // let sum = (1 - eq_factor[round]) * (1 - challenge) + eq_factor[round] * challenge

    if let Some(eq_factor) = eq_factor {
        let sum =
            (EF::ONE - eq_factor[round]) * (EF::ONE - challenge) + eq_factor[round] * challenge;
        *missing_mul_factor = Some(sum * missing_mul_factor.unwrap_or(EF::ONE));
        // selectors
        //     .iter()
        //     .map(|s| s.evaluate(eq_factor[round]) * s.evaluate(challenge))
        //     .sum::<EF>()
        //*missing_mul_factor.unwrap_or(EF::ONE),
        // );
    }
    // If skips == 1 (ie classic sumcheck round, we could avoid 1 multiplication below: TODO not urgent)
    batch_fold_multilinear_in_large_field(multilinears, &folding_scalars)
}

fn compute_over_hypercube<F, NF, EF, SC>(
    pols: &[EvaluationsList<NF>],
    computation: &SC,
    batching_scalars: &[EF],
    eq_mle: Option<&EvaluationsList<EF>>,
) -> EF
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
{
    assert!(
        pols.iter()
            .all(|p| p.num_variables() == pols[0].num_variables())
    );
    let n_vars = pols[0].num_variables();
    if TypeId::of::<NF>() == TypeId::of::<F>() {
        let pols: &[EvaluationsList<F>] = unsafe { std::mem::transmute(pols) };
        let packed_pols = pols
            .iter()
            .map(|p| F::Packing::pack_slice(p.evals()))
            .collect::<Vec<_>>();

        let decomposed_batching_scalars: Vec<_> = (0..<EF as BasedVectorSpace<F>>::DIMENSION)
            .map(|i| {
                batching_scalars
                    .iter()
                    .map(|x| x.as_basis_coefficients_slice()[i])
                    .collect()
            })
            .collect();

        (0..(1 << n_vars) / F::Packing::WIDTH)
            .into_par_iter()
            .enumerate()
            .map(|(x, i)| {
                let point = packed_pols.iter().map(|pol| pol[x]).collect::<Vec<_>>();
                let res =
                    computation.eval_packed(&point, batching_scalars, &decomposed_batching_scalars);
                if let Some(eq_mle) = eq_mle {
                    res.enumerate()
                        .map(|(idx_in_packing, res)| {
                            res * eq_mle.evals()[i * F::Packing::WIDTH + idx_in_packing]
                        })
                        .sum()
                } else {
                    res.sum()
                }
            })
            .sum()
    } else {
        // TODO packing everywhere
        assert_eq!(TypeId::of::<NF>(), TypeId::of::<EF>());
        (0..1 << n_vars)
            .into_par_iter()
            .map(|x| {
                let point = pols.iter().map(|pol| pol.evals()[x]).collect::<Vec<_>>();
                let eq_mle_eval = eq_mle.map(|p| p.evals()[x]);
                eval_sumcheck_computation(computation, batching_scalars, &point, eq_mle_eval)
            })
            .sum()
    }
}

pub fn eval_sumcheck_computation<F, NF, EF, SC>(
    computation: &SC,
    batching_scalars: &[EF],
    point: &[NF],
    eq_mle_eval: Option<EF>,
) -> EF
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF>,
    SC: SumcheckComputation<F, NF, EF>,
{
    let res = computation.eval(point, batching_scalars);
    eq_mle_eval.map_or(res, |factor| res * factor)
}
