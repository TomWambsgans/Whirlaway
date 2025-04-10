use algebra::{
    pols::{MultilinearPolynomial, TransparentPolynomial},
    utils::powers,
};
use cuda_bindings::{MultilinearPolynomialCuda, SumcheckComputation, memcpy_htod};
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::extension::BinomialExtensionField;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::*;

type F = p3_koala_bear::KoalaBear;
type EF = BinomialExtensionField<F, 8>;

#[test]
fn test_sumcheck() {
    let n_vars = 18;
    let n_exprs = 10;
    let n_multilinears = 20;
    let rng = &mut StdRng::seed_from_u64(0);
    let multilinears = (0..n_multilinears)
        .map(|_| MultilinearPolynomial::<F>::random(rng, n_vars))
        .collect::<Vec<_>>();
    let exprs = (0..n_exprs)
        .map(|_| TransparentPolynomial::random(rng, n_multilinears, 1).fix_computation(true))
        .collect::<Vec<_>>();
    let batching_scalar: EF = rng.random();
    let batching_scalars = powers(batching_scalar, n_exprs);
    let mut fs_prover = FsProver::new();
    let sum = sum_batched_exprs_over_hypercube(&multilinears, n_vars, &exprs, &batching_scalars);

    let time = std::time::Instant::now();
    prove(
        &multilinears,
        &exprs,
        &batching_scalars,
        None,
        false,
        &mut fs_prover,
        Some(sum),
        None,
        0,
    );
    println!("CPU sumcheck: {} ms", time.elapsed().as_millis());

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    let max_degree_per_vars = exprs
        .iter()
        .map(|expr| expr.composition_degree)
        .max()
        .unwrap();
    let (claimed_sum, postponed_verification) =
        verify::<EF>(&mut fs_verifier, &vec![max_degree_per_vars; n_vars], 0).unwrap();
    assert_eq!(sum, claimed_sum);
    assert_eq!(
        eval_batched_exprs_mle(
            &multilinears,
            &exprs,
            &batching_scalars,
            &postponed_verification.point
        ),
        postponed_verification.value
    );
}

#[test]
fn test_cuda_sumcheck() {
    let n_vars = 18;
    let n_exprs = 10;
    let n_multilinears = 20;
    let rng = &mut StdRng::seed_from_u64(0);
    let multilinears = (0..n_multilinears)
        .map(|_| MultilinearPolynomial::<F>::random(rng, n_vars))
        .collect::<Vec<_>>();
    let exprs = (0..n_exprs)
        .map(|_| TransparentPolynomial::random(rng, n_multilinears, 1).fix_computation(true))
        .collect::<Vec<_>>();
    let batching_scalar: EF = rng.random();
    let batching_scalars = powers(batching_scalar, n_exprs);

    cuda_bindings::init(
        &[SumcheckComputation {
            n_multilinears,
            inner: exprs.clone(),
            eq_mle_multiplier: false,
        }],
        0,
    );

    let mut fs_prover = FsProver::new();
    let sum = sum_batched_exprs_over_hypercube(&multilinears, n_vars, &exprs, &batching_scalars);

    let _multilinears_dev = multilinears.iter().map(|m| m.embed()).collect::<Vec<_>>();
    let multilinears_dev = _multilinears_dev
        .iter()
        .map(|m| MultilinearPolynomialCuda::new(memcpy_htod(&m.evals)))
        .collect::<Vec<_>>();

    let time = std::time::Instant::now();
    prove_with_cuda(
        &multilinears_dev,
        &exprs,
        &batching_scalars,
        None,
        false,
        &mut fs_prover,
        Some(sum),
        None,
        0,
    );
    println!("GPU sumcheck: {} ms", time.elapsed().as_millis());

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    let max_degree_per_vars = exprs
        .iter()
        .map(|expr| expr.composition_degree)
        .max()
        .unwrap();
    let (claimed_sum, postponed_verification) =
        verify::<EF>(&mut fs_verifier, &vec![max_degree_per_vars; n_vars], 0).unwrap();
    assert_eq!(sum, claimed_sum);
    assert_eq!(
        eval_batched_exprs_mle(
            &multilinears,
            &exprs,
            &batching_scalars,
            &postponed_verification.point
        ),
        postponed_verification.value
    );
}
