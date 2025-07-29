use std::borrow::Borrow;

use p3_field::PackedFieldExtension;
use p3_field::PackedValue;
use p3_field::PrimeCharacteristicRing;
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use tracing::info_span;
use utils::batch_fold_multilinear_in_large_field_packed;
use utils::{EFPacking, FSChallenger, FSProver, PF, PFPacking, univariate_selectors};
use whir_p3::poly::multilinear::MultilinearPoint;
use whir_p3::poly::{dense::WhirDensePolynomial, evals::EvaluationsList};
use whir_p3::utils::uninitialized_vec;

use crate::sc_round;
use crate::{SumcheckComputation, SumcheckComputationPacked};

#[allow(clippy::too_many_arguments)]
pub fn prove_packed<EF, M, SC>(
    skips: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: &[M],
    computation: &SC,
    constraints_degree: usize,
    eq_factor: &mut Option<(&[EF], Vec<EFPacking<EF>>)>,
    is_zerofier: bool,
    fs_prover: &mut FSProver<EF, impl FSChallenger<EF>>,
    mut sum: EF,
    n_rounds: Option<usize>,
    mut missing_mul_factor: Option<EF>,
    log: bool,
) -> (MultilinearPoint<EF>, Vec<EvaluationsList<EF>>, EF)
where
    EF: Field + ExtensionField<PF<PF<EF>>> + ExtensionField<PF<EF>>,
    M: Borrow<[EFPacking<EF>]>,
    SC: SumcheckComputation<PF<PF<EF>>, EF, EF>
        + SumcheckComputation<PF<EF>, EF, EF>
        + SumcheckComputationPacked<PF<EF>, EF>,
{
    let multilinears = multilinears.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    let mut n_vars = (multilinears[0].len() * PFPacking::<EF>::WIDTH).ilog2() as usize;
    assert!(
        multilinears
            .iter()
            .all(|m| m.len() * PFPacking::<EF>::WIDTH == 1 << n_vars)
    );

    let mut challenges = Vec::new();
    let n_rounds = n_rounds.unwrap_or(n_vars - skips + 1);
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.0.len(), n_vars - skips + 1);
        assert_eq!(
            eq_factor.1.len() * PFPacking::<EF>::WIDTH,
            1 << (n_vars - skips)
        );
    }

    let mut folded_multilinears = sc_round_packed(
        skips,
        &multilinears,
        &mut n_vars,
        computation,
        eq_factor,
        is_zerofier,
        fs_prover,
        constraints_degree,
        &mut sum,
        &mut challenges,
        0,
        &mut missing_mul_factor,
        log,
    );

    assert!(n_rounds > 3 + PFPacking::<EF>::WIDTH.ilog2() as usize);

    let last_round_packed = n_rounds - PFPacking::<EF>::WIDTH.ilog2() as usize - 2;

    for i in 1..last_round_packed {
        folded_multilinears = sc_round_packed(
            1,
            &folded_multilinears
                .iter()
                .map(|m| m.as_slice())
                .collect::<Vec<_>>(),
            &mut n_vars,
            computation,
            eq_factor,
            false,
            fs_prover,
            constraints_degree,
            &mut sum,
            &mut challenges,
            i,
            &mut missing_mul_factor,
            log,
        );
    }

    let mut folded_multilinears_unpacked = folded_multilinears
        .into_iter()
        .map(|m| EvaluationsList::new(EFPacking::<EF>::to_ext_iter(m).collect::<Vec<_>>()))
        .collect::<Vec<_>>();

    let mut eq_factor = eq_factor.clone().map(|(eq_factor, eq_mle)| {
        (
            eq_factor[last_round_packed..].to_vec(),
            EFPacking::<EF>::to_ext_iter(eq_mle).collect::<Vec<_>>(),
        )
    });

    for _ in last_round_packed..n_rounds {
        folded_multilinears_unpacked = sc_round::<PF<EF>, EF, EF, _>(
            skips,
            &folded_multilinears_unpacked
                .iter()
                .map(|m| m.evals())
                .collect::<Vec<_>>(),
            &mut n_vars,
            computation,
            &mut eq_factor,
            &[EF::ONE],
            is_zerofier,
            fs_prover,
            constraints_degree,
            &mut sum,
            &mut challenges,
            &mut missing_mul_factor,
            log,
        );
    }

    (
        MultilinearPoint(challenges),
        folded_multilinears_unpacked,
        sum,
    )
}

#[allow(clippy::too_many_arguments)]
fn sc_round_packed<EF, SC>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &[&[EFPacking<EF>]],
    n_vars: &mut usize,
    computation: &SC,
    eq_factor: &mut Option<(&[EF], Vec<EFPacking<EF>>)>,
    is_zerofier: bool,
    fs_prover: &mut FSProver<EF, impl FSChallenger<EF>>,
    comp_degree: usize,
    sum: &mut EF,
    challenges: &mut Vec<EF>,
    round: usize,
    missing_mul_factor: &mut Option<EF>,
    log: bool,
) -> Vec<Vec<EFPacking<EF>>>
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputation<PF<PF<EF>>, EF, EF> + SumcheckComputationPacked<PF<EF>, EF>,
{
    let _info_span = log.then(|| info_span!("sumcheck round", round,));

    let selectors = univariate_selectors::<PF<EF>>(skips);

    let mut p_evals = Vec::<(PF<EF>, EF)>::new();
    let start = if is_zerofier {
        p_evals.extend((0..1 << skips).map(|i| (PF::<EF>::from_usize(i), EF::ZERO)));
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
                .map(|s| s.evaluate(PF::<EF>::from_usize(z)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let folding_scalars_packed = folding_scalars
        .iter()
        .map(|fs| {
            fs.iter()
                .map(|s| PFPacking::<EF>::from_slice(&vec![*s; PFPacking::<EF>::WIDTH]).clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let fold_size = 1 << (*n_vars - skips);
    let packed_fold_size = fold_size / PFPacking::<EF>::WIDTH;

    let all_sums = unsafe { uninitialized_vec::<EFPacking<EF>>(zs.len() * packed_fold_size) }; // sums for zs[0], sums for zs[1], ...
    (0..packed_fold_size).into_par_iter().for_each(|i| {
        let eq_mle_eval = eq_factor.as_ref().map(|(_, eq_mle)| eq_mle[i]);
        let rows = multilinears
            .iter()
            .map(|m| {
                (0..selectors.len())
                    .map(|j| m[i + j * packed_fold_size])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (z_index, folding_scalars_z) in folding_scalars_packed.iter().enumerate() {
            let point = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_scalars_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<EFPacking<EF>>()
                })
                .collect::<Vec<_>>();
            unsafe {
                let sum_ptr = all_sums.as_ptr() as *mut EFPacking<EF>;
                *sum_ptr.add(z_index * packed_fold_size + i) =
                    eval_sumcheck_computation(computation, &point, eq_mle_eval);
            }
        }
    });
    for (z_index, z) in zs.iter().enumerate() {
        let mut sum_z = all_sums[z_index * packed_fold_size..(z_index + 1) * packed_fold_size]
            .par_iter()
            .copied()
            .sum::<EFPacking<EF>>();
        if let Some(missing_mul_factor) = missing_mul_factor {
            sum_z *= *missing_mul_factor;
        }
        p_evals.push((
            PF::<EF>::from_usize(*z),
            EFPacking::<EF>::to_ext_iter([sum_z]).sum::<EF>(),
        ));
    }
    let missing_sum_z = if let Some((eq_factor, _)) = eq_factor {
        (*sum
            - (0..(1 << skips) - 1)
                .map(|i| p_evals[i].1 * selectors[i].evaluate(eq_factor[round]))
                .sum::<EF>())
            / selectors[(1 << skips) - 1].evaluate(eq_factor[round])
    } else {
        *sum - p_evals[..(1 << skips) - 1]
            .iter()
            .map(|(_, s)| *s)
            .sum::<EF>()
    };
    p_evals.push((PF::<EF>::from_usize((1 << skips) - 1), missing_sum_z));

    let mut p = WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some((eq_factor, _)) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        p *= &WhirDensePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| {
                    (
                        PF::<EF>::from_usize(i),
                        selectors[i].evaluate(eq_factor[round]),
                    )
                })
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
    if let Some((eq_factor, _)) = eq_factor {
        *missing_mul_factor = Some(
            selectors
                .iter()
                .map(|s| s.evaluate(eq_factor[round]) * s.evaluate(challenge))
                .sum::<EF>()
                * missing_mul_factor.unwrap_or(EF::ONE)
                / (EF::ONE - eq_factor[round + 1]),
        );
    }
    if let Some((_, eq_mle)) = eq_factor {
        eq_mle.resize(eq_mle.len() / 2, Default::default());
    }
    batch_fold_multilinear_in_large_field_packed(multilinears, &folding_scalars)
}

fn eval_sumcheck_computation<EF, SC>(
    computation: &SC,
    point: &[EFPacking<EF>],
    eq_mle_eval: Option<EFPacking<EF>>,
) -> EFPacking<EF>
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputationPacked<PF<EF>, EF>,
{
    let res = computation.eval_packed_extension(point);
    eq_mle_eval.map_or(res, |factor| res * factor)
}
