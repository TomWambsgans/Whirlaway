use super::*;
use algebra::pols::{
    Multilinear, MultilinearDevice, MultilinearHost, MultilinearsVec, eval_sumcheck_computation,
    univariate_selectors,
};
use arithmetic_circuit::{CircuitComputation, TransparentPolynomial};
use cuda_engine::{
    CudaFunctionInfo, SumcheckComputation, cuda_init, cuda_load_function,
    cuda_preprocess_sumcheck_computation, memcpy_htod,
};
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, PrimeField32, extension::BinomialExtensionField,
};
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use utils::{HypercubePoint, eq_extension, powers};

type F = KoalaBear;
type EF = BinomialExtensionField<KoalaBear, 8>;

// TODO make it work with multilinears in the prime field

fn setup<F: PrimeField32, EF: ExtensionField<F>>(sumcheck_computation: &SumcheckComputation<F>) {
    cuda_init();
    cuda_preprocess_sumcheck_computation(sumcheck_computation, 1, 1, 8);
    cuda_preprocess_sumcheck_computation(sumcheck_computation, 1, 8, 8);
    cuda_load_function(CudaFunctionInfo::one_field::<EF>(
        "multilinear.cu",
        "eq_mle",
    ));
    cuda_load_function(CudaFunctionInfo::one_field::<EF>(
        "multilinear.cu",
        "piecewise_sum",
    ));
    cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
        "multilinear.cu",
        "dot_product",
    ));
    cuda_load_function(CudaFunctionInfo::two_fields::<EF, F>(
        "multilinear.cu",
        "fold_rectangular",
    ));
    cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
        "multilinear.cu",
        "fold_rectangular",
    ));
    cuda_load_function(CudaFunctionInfo::two_fields::<EF, EF>(
        "multilinear.cu",
        "fold_rectangular",
    ));
    cuda_load_function(CudaFunctionInfo::two_fields::<F, F>(
        "multilinear.cu",
        "fold_rectangular",
    ));
}

#[test]
fn test_sumcheck() {
    let n_vars = 11;
    let n_exprs = 10;
    let n_multilinears = 20;
    let rng = &mut StdRng::seed_from_u64(0);
    let exprs = (0..n_exprs)
        .map(|_| TransparentPolynomial::<F>::random(rng, n_multilinears, 1).fix_computation(true))
        .collect::<Vec<_>>();
    let eq_factor = (0..n_vars).map(|_| EF::random(rng)).collect::<Vec<_>>();

    let sumcheck_computation = SumcheckComputation {
        exprs: &exprs,
        n_multilinears: n_multilinears + 1,
        eq_mle_multiplier: true,
    };

    setup::<F, EF>(&sumcheck_computation);

    for gpu in [true, false] {
        let multilinears_host = (0..n_multilinears)
            .map(|_| MultilinearHost::<EF>::random(rng, n_vars))
            .collect::<Vec<_>>();
        let multilinears = if gpu {
            MultilinearsVec::Device(
                multilinears_host
                    .iter()
                    .map(|m| MultilinearDevice::new(memcpy_htod(&m.evals)))
                    .collect::<Vec<_>>(),
            )
        } else {
            MultilinearsVec::Host(multilinears_host.clone())
        };

        let batching_scalar: EF = rng.random();
        let batching_scalars = powers(batching_scalar, n_exprs);
        let eq_mle = Multilinear::eq_mle(&eq_factor, gpu);
        let sum = multilinears.as_ref().compute_over_hypercube(
            &sumcheck_computation,
            &batching_scalars,
            Some(&eq_mle),
        );

        let mut fs_prover = FsProver::new();

        let time = std::time::Instant::now();
        prove(
            1,
            multilinears.as_ref(),
            &exprs,
            &batching_scalars,
            Some(&eq_factor),
            false,
            &mut fs_prover,
            sum,
            None,
            0,
            None,
        );
        println!(
            "{} sumcheck: {} ms",
            if gpu { "GPU" } else { "CPU" },
            time.elapsed().as_millis()
        );

        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let max_degree_per_vars = exprs
            .iter()
            .map(|expr| expr.composition_degree)
            .max()
            .unwrap();
        let (claimed_sum, postponed_verification) =
            verify::<EF>(&mut fs_verifier, n_vars, 1 + max_degree_per_vars, 0).unwrap();
        assert_eq!(sum, claimed_sum);

        assert_eq!(
            eval_batched_exprs_of_multilinears(
                &multilinears_host,
                &exprs,
                &batching_scalars,
                &postponed_verification.point
            ) * eq_extension(&postponed_verification.point, &eq_factor),
            postponed_verification.value
        );
    }
}

#[test]
fn test_univariate_skip() {
    let skips = 3;
    let n_vars = 4;
    let n_exprs = 5;
    let n_multilinears = 7;
    let rng = &mut StdRng::seed_from_u64(0);
    let exprs = (0..n_exprs)
        .map(|_| {
            TransparentPolynomial::<KoalaBear>::random(rng, n_multilinears, 1).fix_computation(true)
        })
        .collect::<Vec<_>>();
    let eq_factor = (0..n_vars - skips + 1)
        .map(|_| EF::random(rng))
        .collect::<Vec<_>>();
    let selectors = univariate_selectors::<F>(skips);

    let sumcheck_computation = SumcheckComputation {
        exprs: &exprs,
        n_multilinears: n_multilinears + 1,
        eq_mle_multiplier: true,
    };
    setup::<F, EF>(&sumcheck_computation);

    for gpu in [false, true] {
        let multilinears_host = (0..n_multilinears)
            .map(|_| MultilinearHost::<F>::random(rng, n_vars))
            .collect::<Vec<_>>();
        let multilinears = if gpu {
            MultilinearsVec::Device(
                multilinears_host
                    .iter()
                    .map(|m| MultilinearDevice::new(memcpy_htod(&m.evals)))
                    .collect::<Vec<_>>(),
            )
        } else {
            MultilinearsVec::Host(multilinears_host.clone())
        };
        let batching_scalar: EF = rng.random();
        let batching_scalars = powers(batching_scalar, n_exprs);

        let eval_eq_factor = |point: &[EF]| {
            assert_eq!(point.len(), n_vars - skips + 1);
            selectors
                .iter()
                .map(|sel| sel.eval(&point[0]) * sel.eval(&eq_factor[0]))
                .sum::<EF>()
                * eq_extension(&point[1..], &eq_factor[1..])
        };

        let sum = HypercubePoint::par_iter(n_vars)
            .map(|x| {
                let point = multilinears_host
                    .iter()
                    .map(|pol| pol.eval_hypercube(&x))
                    .collect::<Vec<_>>();
                assert!(x.val >> (n_vars - skips) < (1 << skips));
                let mut eq_point = vec![EF::from_usize(x.val >> (n_vars - skips))];
                eq_point.extend_from_slice(&x.to_vec()[skips..]);
                let eq_mle_eval = eval_eq_factor(&eq_point);
                eval_sumcheck_computation(
                    &sumcheck_computation,
                    &batching_scalars,
                    &point,
                    Some(eq_mle_eval),
                )
            })
            .sum::<EF>();

        let mut fs_prover = FsProver::new();

        prove(
            skips,
            multilinears.as_ref(),
            &exprs,
            &batching_scalars,
            Some(&eq_factor),
            false,
            &mut fs_prover,
            sum,
            None,
            0,
            None,
        );

        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let degree = 1 + exprs
            .iter()
            .map(|expr| expr.composition_degree)
            .max()
            .unwrap();
        let (claimed_sum, postponed_verification) =
            verify_with_univariate_skip::<EF>(&mut fs_verifier, degree, n_vars, skips, 0).unwrap();
        assert_eq!(sum, claimed_sum);

        let selector_evals = selectors
            .iter()
            .map(|s| s.eval(&postponed_verification.point[0]))
            .collect::<Vec<_>>();
        let folded_multilinears_host = multilinears_host
            .iter()
            .map(|m| m.fold_rectangular_in_large_field(&selector_evals))
            .collect::<Vec<_>>();

        assert_eq!(
            eval_batched_exprs_of_multilinears(
                &folded_multilinears_host,
                &exprs,
                &batching_scalars,
                &postponed_verification.point[1..]
            ) * eval_eq_factor(&postponed_verification.point),
            postponed_verification.value
        );
    }
}

pub fn eval_batched_exprs<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    point: &[NF],
) -> EF {
    if exprs.len() == 1 {
        EF::from(exprs[0].eval(point))
    } else {
        exprs
            .iter()
            .zip(batching_scalars)
            .skip(1)
            .map(|(expr, scalar)| *scalar * expr.eval(point))
            .sum::<EF>()
            + exprs[0].eval(point)
    }
}

pub fn eval_batched_exprs_of_multilinears<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    multilinears: &[MultilinearHost<NF>],
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    point: &[EF],
) -> EF {
    let inner_evals = multilinears
        .iter()
        .map(|pol| pol.evaluate(point))
        .collect::<Vec<_>>();
    eval_batched_exprs(exprs, batching_scalars, &inner_evals)
}
