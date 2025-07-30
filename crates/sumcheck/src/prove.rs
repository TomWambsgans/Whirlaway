use p3_field::PackedFieldExtension;
use p3_field::PackedValue;
use p3_field::PrimeCharacteristicRing;
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use tracing::info_span;
use utils::batch_fold_multilinear_in_large_field_packed;
use utils::pack_extension;
use utils::packing_log_width;
use utils::packing_width;
use utils::unpack_extension;
use utils::{
    EFPacking, FSChallenger, FSProver, PF, PFPacking, batch_fold_multilinear_in_large_field,
    univariate_selectors,
};
use whir_p3::poly::multilinear::MultilinearPoint;
use whir_p3::poly::{dense::WhirDensePolynomial, evals::EvaluationsList};
use whir_p3::utils::uninitialized_vec;

use crate::{SumcheckComputation, SumcheckComputationPacked};

#[allow(clippy::too_many_arguments)]
pub fn prove_generic<NF, EF, SC>(
    skips: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: Vec<&[NF]>,
    computation: &SC,
    constraints_degree: usize,
    batching_scalars: &[EF],
    mut eq_factor: Option<(&[EF], Option<Vec<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    mut sum: EF,
    mut missing_mul_factor: Option<EF>,
    log: bool,
) -> (MultilinearPoint<EF>, Vec<EF>, EF)
where
    NF: ExtensionField<PF<EF>>,
    EF: ExtensionField<NF> + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputation<NF, EF> + SumcheckComputation<EF, EF> + SumcheckComputationPacked<EF>,
{
    let mut n_vars = multilinears[0].len().ilog2() as usize;
    assert!(multilinears.iter().all(|m| m.len() == 1 << n_vars));

    let mut challenges = Vec::new();
    let n_rounds = n_vars - skips + 1;
    if let Some((eq_point, _)) = &mut eq_factor {
        assert_eq!(eq_point.len(), n_vars - skips + 1);
    }
    let mut eq_factor = eq_factor.map(|(eq_point, eq_mle)| {
        (
            eq_point.to_vec(),
            eq_mle.unwrap_or_else(|| EvaluationsList::eval_eq(&eq_point[1..]).into_evals()),
        )
    });

    let mut folded_multilinears = sc_round_generic(
        skips,
        &multilinears,
        &mut n_vars,
        computation,
        &mut eq_factor,
        batching_scalars,
        is_zerofier,
        prover_state,
        constraints_degree,
        &mut sum,
        &mut challenges,
        &mut missing_mul_factor,
        log,
    );

    for _ in 1..n_rounds {
        folded_multilinears = sc_round_generic(
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
            prover_state,
            constraints_degree,
            &mut sum,
            &mut challenges,
            &mut missing_mul_factor,
            log,
        );
    }

    let final_folds = folded_multilinears
        .into_iter()
        .map(|m| {
            debug_assert_eq!(m.len(), 1);
            m[0]
        })
        .collect::<Vec<_>>();

    (MultilinearPoint(challenges), final_folds, sum)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_extension_packed<EF, SC>(
    skips: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: Vec<&[EFPacking<EF>]>,
    computation: &SC,
    constraints_degree: usize,
    batching_scalars: &[EF],
    eq_factor: Option<(&[EF], Option<Vec<EFPacking<EF>>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    mut sum: EF,
    mut missing_mul_factor: Option<EF>,
    log: bool,
) -> (MultilinearPoint<EF>, Vec<EF>, EF)
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputation<EF, EF> + SumcheckComputationPacked<EF>,
{
    let mut n_vars = packing_log_width::<EF>() + multilinears[0].len().ilog2() as usize;
    assert!(
        multilinears
            .iter()
            .all(|m| m.len() * packing_width::<EF>() == 1 << n_vars)
    );

    let mut challenges = Vec::new();
    let n_rounds = n_vars - skips + 1;
    let mut eq_factor: Option<(Vec<EF>, Vec<EFPacking<EF>>)> =
        eq_factor.map(|(eq_point, eq_mle)| {
            (
                eq_point.to_vec(),
                eq_mle.unwrap_or_else(|| {
                    pack_extension(EvaluationsList::eval_eq(&eq_point[1..]).evals())
                }),
            )
        });
    if let Some((eq_point, eq_mle)) = &eq_factor {
        assert_eq!(eq_point.len(), n_vars - skips + 1);
        assert_eq!(
            eq_mle.len() * packing_width::<EF>(),
            1 << (eq_point.len() - 1)
        );
    }

    assert!(
        n_vars > packing_log_width::<EF>() + skips,
        "too little variables for packing sumcheck"
    );

    let mut folded_multilinears = sc_round_packed_extension(
        skips,
        &multilinears,
        &mut n_vars,
        computation,
        &mut eq_factor,
        batching_scalars,
        is_zerofier,
        prover_state,
        constraints_degree,
        &mut sum,
        &mut challenges,
        &mut missing_mul_factor,
        log,
    );

    let last_round_packed = n_rounds - packing_log_width::<EF>() - 1;
    for _ in 1..last_round_packed {
        folded_multilinears = sc_round_packed_extension(
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
            prover_state,
            constraints_degree,
            &mut sum,
            &mut challenges,
            &mut missing_mul_factor,
            log,
        );
    }

    let unpacked_eq_factor = eq_factor
        .as_ref()
        .map(|(eq_point, eq_mle)| (eq_point.as_slice(), Some(unpack_extension(eq_mle))));
    let folded_multilinears = folded_multilinears
        .into_iter()
        .map(|m| unpack_extension(&m))
        .collect::<Vec<Vec<EF>>>();
    let folded_multilinears = folded_multilinears
        .iter()
        .map(|m| m.as_slice())
        .collect::<Vec<_>>();

    let (final_challenges, folds, final_sum) = prove_generic(
        1,
        folded_multilinears,
        computation,
        constraints_degree,
        batching_scalars,
        unpacked_eq_factor,
        false,
        prover_state,
        sum,
        missing_mul_factor,
        log,
    );

    challenges.extend(final_challenges.0);

    (MultilinearPoint(challenges), folds, final_sum)
}

#[allow(clippy::too_many_arguments)]
pub fn prove_base_packed<EF, SC>(
    skips: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: Vec<&[PFPacking<EF>]>,
    computation: &SC,
    constraints_degree: usize,
    batching_scalars: &[EF],
    eq_factor: Option<(&[EF], Option<Vec<EFPacking<EF>>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    mut sum: EF,
    mut missing_mul_factor: Option<EF>,
    log: bool,
) -> (MultilinearPoint<EF>, Vec<EF>, EF)
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputation<EF, EF> + SumcheckComputationPacked<EF>,
{
    let mut n_vars = packing_log_width::<EF>() + multilinears[0].len().ilog2() as usize;
    assert!(
        multilinears
            .iter()
            .all(|m| m.len() * packing_width::<EF>() == 1 << n_vars)
    );

    let mut challenges = Vec::new();
    let mut eq_factor: Option<(Vec<EF>, Vec<EFPacking<EF>>)> =
        eq_factor.map(|(eq_point, eq_mle)| {
            (
                eq_point.to_vec(),
                eq_mle.unwrap_or_else(|| {
                    pack_extension(EvaluationsList::eval_eq(&eq_point[1..]).evals())
                }),
            )
        });
    if let Some((eq_point, eq_mle)) = &eq_factor {
        assert_eq!(eq_point.len(), n_vars - skips + 1);
        assert_eq!(
            eq_mle.len() * packing_width::<EF>(),
            1 << (eq_point.len() - 1)
        );
    }

    assert!(
        n_vars > packing_log_width::<EF>() + skips,
        "too little variables for packing sumcheck"
    );

    let folded_multilinears = sc_round_packed_base(
        skips,
        &multilinears,
        &mut n_vars,
        computation,
        &mut eq_factor,
        batching_scalars,
        is_zerofier,
        prover_state,
        constraints_degree,
        &mut sum,
        &mut challenges,
        &mut missing_mul_factor,
        log,
    );

    let eq_factor = eq_factor
        .as_mut()
        .map(|(eq_point, eq_mle)| (eq_point.as_slice(), Some(std::mem::take(eq_mle))));
    let folded_multilinears = folded_multilinears
        .iter()
        .map(|m| m.as_slice())
        .collect::<Vec<_>>();

    let (final_challenges, folds, final_sum) = prove_extension_packed(
        1,
        folded_multilinears,
        computation,
        constraints_degree,
        batching_scalars,
        eq_factor,
        false,
        prover_state,
        sum,
        missing_mul_factor,
        log,
    );

    challenges.extend(final_challenges.0);
    (MultilinearPoint(challenges), folds, final_sum)
}

#[allow(clippy::too_many_arguments)]
fn sc_round_generic<NF, EF, SC>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &[&[NF]],
    n_vars: &mut usize,
    computation: &SC,
    eq_factor: &mut Option<(Vec<EF>, Vec<EF>)>, // (a, b, c ...), eq_poly(b, c, ...)
    batching_scalars: &[EF],
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    comp_degree: usize,
    sum: &mut EF,
    challenges: &mut Vec<EF>,
    missing_mul_factor: &mut Option<EF>,
    log: bool,
) -> Vec<Vec<EF>>
where
    NF: ExtensionField<PF<EF>>,
    EF: ExtensionField<NF> + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputation<NF, EF>,
{
    let _info_span = log.then(|| info_span!("sumcheck round").entered());

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
                let mut res = computation.eval(&point, batching_scalars);
                if let Some(eq_mle_eval) = eq_mle_eval {
                    res *= eq_mle_eval;
                }
                *sum_ptr.add(z_index * fold_size + i) = res;
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
        p_evals.push((PF::<EF>::from_usize(*z), sum_z));
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
        p_evals.push((PF::<EF>::from_usize((1 << skips) - 1), missing_sum_z));
    }

    let mut p = WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some((eq_factor, _)) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        p *= &WhirDensePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| (PF::<EF>::from_usize(i), selectors[i].evaluate(eq_factor[0])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }

    prover_state.add_extension_scalars(&p.coeffs);

    let challenge = prover_state.sample();
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

#[allow(clippy::too_many_arguments)]
fn sc_round_packed_base<EF, SC>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &[&[PFPacking<EF>]],
    n_vars: &mut usize,
    computation: &SC,
    eq_factor: &mut Option<(Vec<EF>, Vec<EFPacking<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    batching_scalars: &[EF],
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    comp_degree: usize,
    sum: &mut EF,
    challenges: &mut Vec<EF>,
    missing_mul_factor: &mut Option<EF>,
    log: bool,
) -> Vec<Vec<EFPacking<EF>>>
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputationPacked<EF>,
{
    let _info_span = log.then(|| info_span!("sumcheck round").entered());

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
        .collect::<Vec<Vec<PF<EF>>>>();

    let fold_size = 1 << (*n_vars - skips);
    let packed_fold_size = fold_size / packing_width::<EF>();

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
        for (z_index, folding_scalars_z) in folding_scalars.iter().enumerate() {
            let point = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_scalars_z.iter())
                        .map(|(x, s)| *x * *s)
                        .sum::<PFPacking<EF>>()
                })
                .collect::<Vec<_>>();

            let mut res = computation.eval_packed_base(&point, batching_scalars);
            if let Some(eq_mle_eval) = eq_mle_eval {
                res *= eq_mle_eval;
            }

            unsafe {
                let sum_ptr = all_sums.as_ptr() as *mut EFPacking<EF>;
                *sum_ptr.add(z_index * packed_fold_size + i) = res;
            }
        }
    });

    for (z_index, z) in zs.iter().enumerate() {
        let sum_z_packed = all_sums[z_index * packed_fold_size..(z_index + 1) * packed_fold_size]
            .par_iter()
            .copied()
            .sum::<EFPacking<EF>>();
        let mut sum_z = EFPacking::<EF>::to_ext_iter([sum_z_packed]).sum::<EF>();
        if let Some(missing_mul_factor) = missing_mul_factor {
            sum_z *= *missing_mul_factor;
        }
        p_evals.push((PF::<EF>::from_usize(*z), sum_z));
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
        p_evals.push((PF::<EF>::from_usize((1 << skips) - 1), missing_sum_z));
    }

    let mut p = WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some((eq_factor, _)) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        p *= &WhirDensePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| (PF::<EF>::from_usize(i), selectors[i].evaluate(eq_factor[0])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }

    prover_state.add_extension_scalars(&p.coeffs);

    let challenge = prover_state.sample();
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

    // TODO this is ugly and not optimized
    batch_fold_multilinear_in_large_field(
        &multilinears
            .iter()
            .copied()
            .map(PFPacking::<EF>::unpack_slice)
            .collect::<Vec<_>>(),
        &folding_scalars,
    )
    .iter()
    .map(|m| pack_extension(m))
    .collect()
}

#[allow(clippy::too_many_arguments)]
fn sc_round_packed_extension<EF, SC>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &[&[EFPacking<EF>]],
    n_vars: &mut usize,
    computation: &SC,
    eq_factor: &mut Option<(Vec<EF>, Vec<EFPacking<EF>>)>, // (a, b, c ...), eq_poly(b, c, ...)
    batching_scalars: &[EF],
    is_zerofier: bool,
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    comp_degree: usize,
    sum: &mut EF,
    challenges: &mut Vec<EF>,
    missing_mul_factor: &mut Option<EF>,
    log: bool,
) -> Vec<Vec<EFPacking<EF>>>
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    SC: SumcheckComputationPacked<EF>,
{
    let _info_span = log.then(|| info_span!("sumcheck round").entered());

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
        .collect::<Vec<Vec<PF<EF>>>>();

    let fold_size = 1 << (*n_vars - skips);
    let packed_fold_size = fold_size / packing_width::<EF>();

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
        for (z_index, folding_scalars_z) in folding_scalars.iter().enumerate() {
            let point = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(folding_scalars_z.iter())
                        .map(|(x, s)| *x * PFPacking::<EF>::from(*s))
                        .sum::<EFPacking<EF>>()
                })
                .collect::<Vec<_>>();

            let mut res = computation.eval_packed_extension(&point, batching_scalars);
            if let Some(eq_mle_eval) = eq_mle_eval {
                res *= eq_mle_eval;
            }

            unsafe {
                let sum_ptr = all_sums.as_ptr() as *mut EFPacking<EF>;
                *sum_ptr.add(z_index * packed_fold_size + i) = res;
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
        p_evals.push((PF::<EF>::from_usize((1 << skips) - 1), missing_sum_z));
    }

    let mut p = WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some((eq_factor, _)) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        p *= &WhirDensePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| (PF::<EF>::from_usize(i), selectors[i].evaluate(eq_factor[0])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }

    prover_state.add_extension_scalars(&p.coeffs);

    let challenge = prover_state.sample();
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

    batch_fold_multilinear_in_large_field_packed(multilinears, &folding_scalars)
}
