use std::any::TypeId;

use p3_field::{BasedVectorSpace, PackedValue};
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use tracing::info_span;
use utils::{
    FSChallenger, FSProver, PF, batch_fold_multilinear_in_large_field,
    batch_fold_multilinear_in_small_field, univariate_selectors,
};
use whir_p3::poly::multilinear::MultilinearPoint;
use whir_p3::poly::{dense::WhirDensePolynomial, evals::EvaluationsList};
use whir_p3::utils::uninitialized_vec;

use crate::{SumcheckComputation, SumcheckComputationPacked};

#[allow(clippy::too_many_arguments)]
pub fn prove<F, NF, EF, SC>(
    skips: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: Vec<&[NF]>,
    computation: &SC,
    constraints_degree: usize,
    batching_scalars: &[EF],
    mut eq_factor: Option<(&[EF], Option<Vec<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    is_zerofier: bool,
    fs_prover: &mut FSProver<EF, impl FSChallenger<EF>>,
    mut sum: EF,
    n_rounds: Option<usize>,
    mut missing_mul_factor: Option<EF>,
    log: bool,
) -> (MultilinearPoint<EF>, Vec<Vec<EF>>, EF)
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputation<F, NF, EF>
        + SumcheckComputation<F, EF, EF>
        + SumcheckComputationPacked<F, EF>,
{
    let mut n_vars = multilinears[0].len().ilog2() as usize;
    assert!(multilinears.iter().all(|m| m.len() == 1 << n_vars));

    let mut challenges = Vec::new();
    let n_rounds = n_rounds.unwrap_or(n_vars - skips + 1);
    if let Some((eq_point, _)) = &mut eq_factor {
        assert_eq!(eq_point.len(), n_vars - skips + 1);
    }
    let mut eq_factor = eq_factor.map(|(eq_point, eq_mle)| {
        (
            eq_point.to_vec(),
            eq_mle.unwrap_or_else(|| EvaluationsList::eval_eq(&eq_point[1..]).into_evals()),
        )
    });

    let mut folded_multilinears = sc_round(
        skips,
        &multilinears,
        &mut n_vars,
        computation,
        &mut eq_factor,
        batching_scalars,
        is_zerofier,
        fs_prover,
        constraints_degree,
        &mut sum,
        &mut challenges,
        &mut missing_mul_factor,
        log,
    );

    for _ in 1..n_rounds {
        folded_multilinears = sc_round(
            1,
            &folded_multilinears
                .iter()
                .map(|m| m.as_slice())
                .collect::<Vec<_>>(),
            &mut n_vars,
            computation,
            &mut eq_factor,
            batching_scalars,
            false,
            fs_prover,
            constraints_degree,
            &mut sum,
            &mut challenges,
            &mut missing_mul_factor,
            log,
        );
    }

    (MultilinearPoint(challenges), folded_multilinears, sum)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sc_round<F, NF, EF, SC>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &[&[NF]],
    n_vars: &mut usize,
    computation: &SC,
    eq_factor: &mut Option<(Vec<EF>, Vec<EF>)>, // (a, b, c ...), eq_poly(b, c, ...)
    batching_scalars: &[EF],
    is_zerofier: bool,
    fs_prover: &mut FSProver<EF, impl FSChallenger<EF>>,
    comp_degree: usize,
    sum: &mut EF,
    challenges: &mut Vec<EF>,
    missing_mul_factor: &mut Option<EF>,
    log: bool,
) -> Vec<Vec<EF>>
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
{
    let _info_span = log.then(|| info_span!("sumcheck round"));

    let selectors = univariate_selectors::<F>(skips);

    let mut p_evals = Vec::<(F, EF)>::new();
    let start = if is_zerofier {
        p_evals.extend((0..1 << skips).map(|i| (F::from_usize(i), EF::ZERO)));
        1 << skips
    } else {
        0
    };

    let zs = (start..=comp_degree * ((1 << skips) - 1))
        .filter(|&i| i != (1 << skips) - 1)
        .collect::<Vec<_>>();

    let folding_scalars = zs
        .iter()
        .map(|&z| {
            selectors
                .iter()
                .map(|s| s.evaluate(F::from_usize(z)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let fold_size = 1 << (*n_vars - skips);

    let all_sums = unsafe { uninitialized_vec::<EF>(zs.len() * fold_size) }; // sums for zs[0], sums for zs[1], ...
    (0..fold_size).into_par_iter().for_each(|i| {
        let eq_mle_eval = eq_factor.as_ref().map(|(_, eq_mle)| eq_mle[i]);
        let rows = multilinears
            .iter()
            .map(|m| {
                (0..selectors.len())
                    .map(|j| m[i + j * fold_size])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (z_index, folding_scalars_z) in folding_scalars.iter().enumerate() {
            let point = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_scalars_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<NF>()
                })
                .collect::<Vec<_>>();
            unsafe {
                let sum_ptr = all_sums.as_ptr() as *mut EF;
                *sum_ptr.add(z_index * fold_size + i) =
                    eval_sumcheck_computation(computation, batching_scalars, &point, eq_mle_eval);
            }
        }
    });
    for (z_index, z) in zs.iter().enumerate() {
        let mut sum_z = all_sums[z_index * fold_size..(z_index + 1) * fold_size]
            .par_iter()
            .copied()
            .sum::<EF>();
        if let Some(missing_mul_factor) = missing_mul_factor {
            sum_z *= *missing_mul_factor;
        }
        p_evals.push((F::from_usize(*z), sum_z));
    }
    if !is_zerofier {
        let missing_sum_z = if let Some((eq_factor, _)) = eq_factor {
            (*sum
                - (0..(1 << skips) - 1)
                    .map(|i| p_evals[i].1 * selectors[i].evaluate(eq_factor[0]))
                    .sum::<EF>())
                / selectors[(1 << skips) - 1].evaluate(eq_factor[0])
        } else {
            *sum - p_evals[..(1 << skips) - 1]
                .iter()
                .map(|(_, s)| *s)
                .sum::<EF>()
        };
        p_evals.push((F::from_usize((1 << skips) - 1), missing_sum_z));
    }

    let mut p = WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some((eq_factor, _)) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        p *= &WhirDensePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| (F::from_usize(i), selectors[i].evaluate(eq_factor[0])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }

    fs_prover.add_extension_scalars(&p.coeffs);

    let challenge = fs_prover.sample();
    challenges.push(challenge);
    *sum = p.evaluate(challenge);
    *n_vars -= skips;

    let folding_scalars = selectors
        .iter()
        .map(|s| s.evaluate(challenge))
        .collect::<Vec<_>>();
    if let Some((eq_factor, eq_mle)) = eq_factor {
        *missing_mul_factor = Some(
            selectors
                .iter()
                .map(|s| s.evaluate(eq_factor[0]) * s.evaluate(challenge))
                .sum::<EF>()
                * missing_mul_factor.unwrap_or(EF::ONE)
                / (EF::ONE - eq_factor.get(1).copied().unwrap_or_default()),
        );
        eq_factor.remove(0);
        eq_mle.resize(eq_mle.len() / 2, Default::default());
    }
    // If skips == 1 (ie classic sumcheck round, we could avoid 1 multiplication below: TODO not urgent)
    batch_fold_multilinear_in_large_field(multilinears, &folding_scalars)
}

fn compute_over_hypercube<F, NF, EF, SC>(
    pols: &[Vec<NF>],
    computation: &SC,
    batching_scalars: &[EF],
    eq_mle: Option<&[EF]>,
) -> EF
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
{
    let n_vars = pols[0].len().ilog2() as usize;
    assert!(pols.iter().all(|p| p.len() == 1 << n_vars));
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
                let res = computation.eval_packed_base(
                    &point,
                    batching_scalars,
                    &decomposed_batching_scalars,
                );
                if let Some(eq_mle) = eq_mle {
                    res.enumerate()
                        .map(|(idx_in_packing, res)| {
                            res * eq_mle[i * F::Packing::WIDTH + idx_in_packing]
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
                let point = pols.iter().map(|pol| pol[x]).collect::<Vec<_>>();
                let eq_mle_eval = eq_mle.map(|p| p[x]);
                eval_sumcheck_computation(computation, batching_scalars, &point, eq_mle_eval)
            })
            .sum()
    }
}

fn eval_sumcheck_computation<F, NF, EF, SC>(
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
