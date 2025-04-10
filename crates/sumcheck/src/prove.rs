use std::borrow::Borrow;

use algebra::pols::{MultilinearHost, UnivariatePolynomial};
use arithmetic_circuit::CircuitComputation;
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use utils::{HypercubePoint, PartialHypercubePoint, eq_extension};

use crate::{eval_batched_exprs_on_partial_hypercube, sum_batched_exprs_over_hypercube};

pub fn prove<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    ML: Borrow<MultilinearHost<NF>>,
>(
    multilinears: &[ML],
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
) -> (Vec<EF>, Vec<MultilinearHost<EF>>) {
    prove_with_initial_rounds(
        multilinears,
        exprs,
        batching_scalars,
        eq_factor,
        Vec::new(),
        is_zerofier,
        fs_prover,
        sum,
        n_rounds,
        pow_bits,
    )
}

pub fn prove_with_initial_rounds<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    ML: Borrow<MultilinearHost<NF>>,
>(
    multilinears: &[ML],
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    mut challenges: Vec<EF>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
) -> (Vec<EF>, Vec<MultilinearHost<EF>>) {
    let multilinears: Vec<&MultilinearHost<NF>> =
        multilinears.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    let mut n_vars = multilinears[0].n_vars;
    assert!(multilinears.iter().all(|m| m.n_vars == n_vars));
    assert_eq!(exprs.len(), batching_scalars.len());
    assert!(batching_scalars[0].is_one());

    let starting_round = challenges.len();
    let n_rounds = n_rounds.unwrap_or(n_vars);
    let max_degree_per_vars = exprs
        .iter()
        .map(|expr| expr.composition_degree)
        .max_by_key(|d| *d)
        .unwrap();
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.len(), n_vars + starting_round);
    }
    let mut sum = sum.unwrap_or_else(|| {
        sum_batched_exprs_over_hypercube(&multilinears, n_vars, exprs, batching_scalars)
    });
    let mut folded_multilinears;

    folded_multilinears = sc_round(
        &multilinears,
        &mut n_vars,
        exprs,
        batching_scalars,
        eq_factor,
        is_zerofier,
        fs_prover,
        max_degree_per_vars,
        &mut sum,
        pow_bits,
        &mut challenges,
        starting_round,
    );
    for i in starting_round + 1..n_rounds {
        folded_multilinears = sc_round(
            &folded_multilinears,
            &mut n_vars,
            exprs,
            batching_scalars,
            eq_factor,
            is_zerofier,
            fs_prover,
            max_degree_per_vars,
            &mut sum,
            pow_bits,
            &mut challenges,
            i,
        );
    }
    (challenges, folded_multilinears)
}

fn sc_round<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    ML: Borrow<MultilinearHost<NF>> + Sync,
>(
    multilinears: &[ML],
    n_vars: &mut usize,
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    degree: usize,
    sum: &mut EF,
    pow_bits: usize,
    challenges: &mut Vec<EF>,
    round: usize,
) -> Vec<MultilinearHost<EF>> {
    let _span = if *n_vars >= 6 {
        Some(tracing::span!(tracing::Level::INFO, "Sumcheck round").entered())
    } else {
        None
    };
    let mut p_evals = Vec::<(EF, EF)>::new();
    let eq_mle = if let Some(eq_factor) = &eq_factor {
        MultilinearHost::eq_mle(&eq_factor[1 + round..])
    } else {
        MultilinearHost::zero(0)
    };

    let start = if is_zerofier && round == 0 {
        p_evals.push((EF::ZERO, EF::ZERO));
        p_evals.push((EF::ONE, EF::ZERO));
        2
    } else {
        0
    };
    for z in start..=degree as u32 {
        let sum_z = if z == 1 {
            if let Some(eq_factor) = eq_factor {
                let f = eq_extension(&eq_factor[..round], &challenges);
                (*sum - p_evals[0].1 * f * (EF::ONE - eq_factor[round])) / (f * eq_factor[round])
            } else {
                *sum - p_evals[0].1
            }
        } else {
            if eq_factor.is_some() {
                (0..1 << (*n_vars - 1))
                    .into_par_iter()
                    .map(|x| {
                        eval_batched_exprs_on_partial_hypercube(
                            &multilinears,
                            exprs,
                            batching_scalars,
                            &PartialHypercubePoint::new(z, *n_vars - 1, x),
                        ) * eq_mle.eval_hypercube(&HypercubePoint::new(eq_mle.n_vars, x))
                    })
                    .sum::<EF>()
            } else {
                (0..1 << (*n_vars - 1))
                    .into_par_iter()
                    .map(|x| {
                        eval_batched_exprs_on_partial_hypercube(
                            &multilinears,
                            exprs,
                            batching_scalars,
                            &PartialHypercubePoint::new(z, *n_vars - 1, x),
                        )
                    })
                    .sum::<EF>()
            }
        };
        p_evals.push((EF::from_u32(z), sum_z));
    }

    let mut p = UnivariatePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some(eq_factor) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        let f = eq_extension(&eq_factor[..round], &challenges);
        p *= UnivariatePolynomial::new(vec![
            f * (EF::ONE - eq_factor[round]),
            f * ((eq_factor[round] * EF::TWO) - EF::ONE),
        ]);
    }

    fs_prover.add_scalars(&p.coeffs);
    let challenge = fs_prover.challenge_scalars(1)[0];
    challenges.push(challenge);
    *sum = p.eval(&challenge);
    *n_vars -= 1;

    // Do PoW if needed
    if pow_bits > 0 {
        fs_prover.challenge_pow(pow_bits);
    }

    multilinears
        .into_iter()
        .map(|pol| pol.borrow().fix_variable(challenge))
        .collect()
}
