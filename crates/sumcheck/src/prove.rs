use algebra::pols::{
    Multilinear, MultilinearsSlice, MultilinearsVec, UnivariatePolynomial, univariate_selectors,
};
use arithmetic_circuit::{CircuitComputation, max_composition_degree};
use cuda_engine::{SumcheckComputation, cuda_sync};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

pub const MIN_VARS_FOR_GPU: usize = 6; // When there are a small number of variables, it's not worth using GPU

pub fn prove<
    'a,
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    ML: Into<MultilinearsSlice<'a, NF>>,
>(
    skips: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: ML,
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    mut sum: EF,
    n_rounds: Option<usize>,
    pow_bits: usize,
    mut missing_mul_factor: Option<EF>,
) -> (Vec<EF>, Vec<Multilinear<EF>>, EF) {
    let multilinears: MultilinearsSlice<'_, _> = multilinears.into();
    let on_device = multilinears.is_device();
    let mut n_vars = multilinears.n_vars();

    let mut challenges = Vec::new();
    let n_rounds = n_rounds.unwrap_or(n_vars - skips + 1);
    let comp_degree = max_composition_degree(&exprs);
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.len(), n_vars - skips + 1);
    }

    let sumcheck_computation = SumcheckComputation {
        exprs: &exprs,
        n_multilinears: multilinears.len() + eq_factor.is_some() as usize,
        eq_mle_multiplier: eq_factor.is_some(),
    };

    let mut folded_multilinears;
    folded_multilinears = sc_round(
        skips,
        &multilinears,
        &mut n_vars,
        &sumcheck_computation,
        batching_scalars,
        eq_factor,
        is_zerofier,
        fs_prover,
        comp_degree,
        &mut sum,
        pow_bits,
        &mut challenges,
        0,
        &mut missing_mul_factor,
    );

    let mut need_to_transfer_back_to_device = false;
    for i in 1..n_rounds {
        if on_device && !need_to_transfer_back_to_device && n_vars < MIN_VARS_FOR_GPU {
            // transfer GPU -> CPU
            let _span = tracing::span!(tracing::Level::INFO, "Sumcheck transfer to CPU").entered();
            folded_multilinears = folded_multilinears.as_ref().transfer_to_host();
            need_to_transfer_back_to_device = true;
        }

        folded_multilinears = sc_round(
            1,
            &folded_multilinears.as_ref(),
            &mut n_vars,
            &sumcheck_computation,
            batching_scalars,
            eq_factor,
            false,
            fs_prover,
            comp_degree,
            &mut sum,
            pow_bits,
            &mut challenges,
            i,
            &mut missing_mul_factor,
        );
    }
    if need_to_transfer_back_to_device {
        let _span = tracing::span!(tracing::Level::INFO, "Sumcheck transfer back to GPU").entered();
        folded_multilinears = folded_multilinears.transfer_to_device();
    }
    (challenges, folded_multilinears.decompose(), sum)
}

pub fn sc_round<'a, F: Field, NF: ExtensionField<F>, EF: ExtensionField<NF> + ExtensionField<F>>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &MultilinearsSlice<'a, NF>,
    n_vars: &mut usize,
    sumcheck_computation: &SumcheckComputation<F>,
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    degree: usize,
    sum: &mut EF,
    pow_bits: usize,
    challenges: &mut Vec<EF>,
    round: usize,
    missing_mul_factor: &mut Option<EF>,
) -> MultilinearsVec<EF> {
    let _span = if round < 30 {
        Some(tracing::span!(tracing::Level::INFO, "Sumcheck round").entered())
    } else {
        None
    };
    let eq_mle = eq_factor
        .map(|eq_factor| Multilinear::eq_mle(&eq_factor[1 + round..], multilinears.is_device()));

    let selectors = univariate_selectors::<F>(skips);

    let mut p_evals = Vec::<(F, EF)>::new();
    let start = if is_zerofier {
        p_evals.extend((0..1 << skips).map(|i| (F::from_usize(i), EF::ZERO)));
        1 << skips
    } else {
        0
    };
    for z in start..=degree * ((1 << skips) - 1) {
        let sum_z = if z == (1 << skips) - 1 {
            if let Some(eq_factor) = eq_factor {
                (*sum
                    - (0..(1 << skips) - 1)
                        .map(|i| p_evals[i].1 * selectors[i].eval(&eq_factor[round]))
                        .sum::<EF>())
                    / selectors[(1 << skips) - 1].eval(&eq_factor[round])
            } else {
                *sum - p_evals.iter().map(|(_, s)| *s).sum::<EF>()
            }
        } else {
            let folding_scalars = selectors
                .iter()
                .map(|s| s.eval(&F::from_usize(z)))
                .collect::<Vec<_>>();
            // If skips == 1 (ie classic sumcheck round, we could avoid 1 multiplication below: TODO not urgent)
            let folded = multilinears.fold_rectangular_in_small_field(&folding_scalars);
            let mut sum_z = folded.as_ref().sum_over_hypercube_of_computation(
                &sumcheck_computation,
                batching_scalars,
                eq_mle.as_ref(),
            );
            if let Some(missing_mul_factor) = missing_mul_factor {
                sum_z *= *missing_mul_factor;
            }

            sum_z
        };

        p_evals.push((F::from_usize(z), sum_z));
    }
    cuda_sync();

    let mut p = UnivariatePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some(eq_factor) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        p *= UnivariatePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| (F::from_usize(i), selectors[i].eval(&eq_factor[round])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }

    fs_prover.add_scalars(&p.coeffs);
    let challenge = fs_prover.challenge_scalars::<EF>(1)[0];
    challenges.push(challenge);
    *sum = p.eval(&challenge);
    *n_vars -= 1;

    // Do PoW if needed
    if pow_bits > 0 {
        fs_prover.challenge_pow(pow_bits);
    }

    let folding_scalars = selectors
        .iter()
        .map(|s| s.eval(&challenge))
        .collect::<Vec<_>>();
    if let Some(eq_factor) = eq_factor {
        *missing_mul_factor = Some(
            selectors
                .iter()
                .map(|s| s.eval(&eq_factor[round]) * s.eval(&challenge))
                .sum::<EF>()
                * missing_mul_factor.unwrap_or(EF::ONE),
        );
    }
    // If skips == 1 (ie classic sumcheck round, we could avoid 1 multiplication below: TODO not urgent)
    let res = multilinears.fold_rectangular_in_large_field(&folding_scalars);

    res
}
