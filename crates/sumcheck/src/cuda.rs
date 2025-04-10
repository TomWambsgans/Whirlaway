use std::borrow::Borrow;

use algebra::{
    pols::{CircuitComputation, MultilinearPolynomial, UnivariatePolynomial},
    utils::eq_extension,
};
use p3_field::{ExtensionField, Field};

use cuda_bindings::{
    CudaSlice, MultilinearPolynomialCuda, SumcheckComputation, cuda_sum_over_hypercube, cuda_sync,
    fold_ext_by_ext, fold_ext_by_prime, memcpy_dtoh, memcpy_htod,
};
use fiat_shamir::FsProver;

const MIN_VARS_FOR_GPU: usize = 9;

pub fn prove_with_cuda<
    F: Field,
    EF: ExtensionField<F>,
    ML: Borrow<MultilinearPolynomialCuda<EF>>,
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
) -> (Vec<EF>, Vec<MultilinearPolynomial<EF>>) {
    let multilinears: Vec<&MultilinearPolynomialCuda<EF>> =
        multilinears.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    let mut n_vars = multilinears[0].n_vars;
    assert!(multilinears.iter().all(|m| m.n_vars == n_vars));
    assert_eq!(exprs.len(), batching_scalars.len());
    assert!(batching_scalars[0].is_one());

    let n_rounds = n_rounds.unwrap_or(n_vars);
    if n_rounds < MIN_VARS_FOR_GPU {
        let multilinears = multilinears
            .into_iter()
            .map(|m| MultilinearPolynomial::new(memcpy_dtoh(&m.evals)))
            .collect::<Vec<_>>();
        cuda_sync();
        return super::prove(
            &multilinears,
            exprs,
            batching_scalars,
            eq_factor,
            is_zerofier,
            fs_prover,
            sum,
            Some(n_rounds),
            pow_bits,
        );
    }

    let sumcheck_computation = SumcheckComputation {
        inner: exprs.to_vec(),
        n_multilinears: multilinears.len() + eq_factor.is_some() as usize,
        eq_mle_multiplier: eq_factor.is_some(),
    };

    let batching_scalars_dev = memcpy_htod(&batching_scalars);

    let max_degree_per_vars = exprs
        .iter()
        .map(|expr| expr.composition_degree)
        .max_by_key(|d| *d)
        .unwrap();
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.len(), n_vars);
    }
    let mut challenges = Vec::new();
    let mut sum = sum.unwrap_or_else(|| {
        cuda_sum_over_hypercube(&sumcheck_computation, &multilinears, &batching_scalars_dev)
    });

    let mut folded_multilinears_dev = sc_round(
        &multilinears,
        &mut n_vars,
        &batching_scalars_dev,
        eq_factor,
        is_zerofier,
        fs_prover,
        max_degree_per_vars,
        &mut sum,
        pow_bits,
        &mut challenges,
        0,
        &sumcheck_computation,
    );
    for i in 1..=n_rounds - MIN_VARS_FOR_GPU {
        folded_multilinears_dev = sc_round(
            &folded_multilinears_dev,
            &mut n_vars,
            &batching_scalars_dev,
            eq_factor,
            is_zerofier,
            fs_prover,
            max_degree_per_vars,
            &mut sum,
            pow_bits,
            &mut challenges,
            i,
            &sumcheck_computation,
        );
    }
    let mut folded_multilinears = folded_multilinears_dev
        .into_iter()
        .map(|multilinear_dev| MultilinearPolynomial::new(memcpy_dtoh(&multilinear_dev.evals)))
        .collect::<Vec<_>>();
    cuda_sync();
    (challenges, folded_multilinears) = super::prove_with_initial_rounds(
        &folded_multilinears,
        exprs,
        batching_scalars,
        eq_factor,
        challenges,
        is_zerofier,
        fs_prover,
        Some(sum),
        Some(n_rounds),
        pow_bits,
    );
    (challenges, folded_multilinears)
}

fn sc_round<F: Field, EF: ExtensionField<F>, ML: Borrow<MultilinearPolynomialCuda<EF>>>(
    multilinears: &[ML],
    n_vars: &mut usize,
    batching_scalars_dev: &CudaSlice<EF>,
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    degree: usize,
    sum: &mut EF,
    pow_bits: usize,
    challenges: &mut Vec<EF>,
    round: usize,
    sumcheck_computation: &SumcheckComputation<F>,
) -> Vec<MultilinearPolynomialCuda<EF>> {
    let _span = tracing::span!(tracing::Level::INFO, "Cuda sumcheck round").entered();
    let mut p_evals = Vec::<(EF, EF)>::new();
    let eq_mle =
        eq_factor.map(|eq_factor| MultilinearPolynomialCuda::eq_mle(&eq_factor[1 + round..]));

    let start = if is_zerofier && round == 0 {
        p_evals.push((EF::ZERO, EF::ZERO));
        p_evals.push((EF::ONE, EF::ZERO));
        2
    } else {
        0
    };
    for z in start..=degree as u32 {
        cuda_sync(); // I don't really understand why it is neccesary but it is
        let sum_z = if z == 1 {
            if let Some(eq_factor) = eq_factor {
                let f = eq_extension(&eq_factor[..round], &challenges);
                (*sum - p_evals[0].1 * f * (EF::ONE - eq_factor[round])) / (f * eq_factor[round])
            } else {
                *sum - p_evals[0].1
            }
        } else {
            let mut folded = fold_ext_by_prime(&multilinears, F::from_u32(z as u32));
            if let Some(eq_mle) = &eq_mle {
                folded.push(eq_mle.clone());
            }
            cuda_sum_over_hypercube(sumcheck_computation, &folded, batching_scalars_dev)
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

    fold_ext_by_ext(&multilinears, challenge)
}
