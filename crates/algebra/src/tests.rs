use crate::ntt::*;
use crate::pols::*;
use arithmetic_circuit::ArithmeticCircuit;
use cuda_bindings::*;
use cuda_engine::*;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_field::TwoAdicField;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

/*
To run all the tests here:

for test in test_cuda_hypercube_sum test_cuda_whir_fold test_cuda_eq_mle test_cuda_eval_multilinear_in_lagrange_basis test_cuda_eval_multilinear_in_monomial_basis test_cuda_keccak test_cuda_expanded_ntt test_cuda_expanded_ntt test_cuda_lagrange_to_monomial_basis test_cuda_monomial_to_lagrange_basis test_cuda_ntt test_cuda_restructure_evaluations; do
  cargo test --release --package algebra --lib -- "tests::$test" --exact --nocapture --ignored
done

(TODO make `cargo test` work (the issue is that tests dont have the same cuda_init config))
*/
#[test]
#[ignore]
fn test_cuda_hypercube_sum() {
    type F = KoalaBear;
    const EXT_DEGREE: usize = 8;
    type EF = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let n_multilinears = 50;
    let n_vars = 14;
    let depth = 25;
    let n_batching_scalars = 3;

    let rng = &mut StdRng::seed_from_u64(0);
    let compositions = (0..n_batching_scalars)
        .map(|_| ArithmeticCircuit::random(rng, n_multilinears, depth).fix_computation(true))
        .collect::<Vec<_>>();

    let sumcheck_computation = SumcheckComputation {
        n_multilinears,
        inner: compositions.clone(),
        eq_mle_multiplier: false,
    };
    let time = std::time::Instant::now();
    cuda_engine::init(&[sumcheck_computation.clone()], 0);
    println!("CUDA initialized in {} ms", time.elapsed().as_millis());
    let cuda = cuda_info();

    let rng = &mut StdRng::seed_from_u64(0);

    let multilinears = (0..n_multilinears)
        .map(|_| MultilinearHost::<EF>::random(rng, n_vars))
        .collect::<Vec<_>>();

    let batching_scalar: EF = rng.random();
    let batching_scalars = (0..n_batching_scalars)
        .map(|e| batching_scalar.exp_u64(e))
        .collect::<Vec<_>>();

    let time = std::time::Instant::now();
    let expected_sum = (0..1 << n_vars)
        .into_par_iter()
        .map(|i| {
            compositions.iter().zip(&batching_scalars).map(|(comp, b)| {
                comp.eval(
                    &(0..n_multilinears)
                        .map(|j| multilinears[j].evals[i])
                        .collect::<Vec<_>>(),
                )
            } * *b).sum::<EF>()
        })
        .sum::<EF>();
    println!("CPU hypercube sum took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let mut batching_scalars_dev =
        unsafe { cuda.stream.alloc::<EF>(batching_scalars.len()).unwrap() };
    cuda.stream
        .memcpy_htod(&batching_scalars, &mut batching_scalars_dev)
        .unwrap();

    let multilinears_dev = multilinears
        .iter()
        .map(|multilinear| {
            let mut multiliner_dev = unsafe { cuda.stream.alloc::<EF>(1 << n_vars).unwrap() };
            cuda.stream
                .memcpy_htod(&multilinear.evals, &mut multiliner_dev)
                .unwrap();
            MultilinearDevice::new(multiliner_dev)
        })
        .collect::<Vec<_>>();
    let copy_duration = time.elapsed();

    let time = std::time::Instant::now();
    let cuda_sum = cuda_sum_over_hypercube::<F, EF, _>(
        &sumcheck_computation,
        &multilinears_dev,
        &batching_scalars_dev,
    );

    println!(
        "CUDA hypercube sum took {} ms (copy duration: {} ms)",
        time.elapsed().as_millis(),
        copy_duration.as_millis()
    );

    assert_eq!(cuda_sum, expected_sum);
}

#[test]
#[ignore]
pub fn test_cuda_expanded_ntt() {
    cuda_engine::init::<KoalaBear>(&[], 0);

    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;

    let rng = &mut StdRng::seed_from_u64(0);
    let log_len = 19;
    let len = 1 << log_len;
    let log_expension_factor: usize = 3;
    let expansion_factor = 1 << log_expension_factor;
    let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();

    println!(
        "number of field elements: {}, expension factor: {}",
        len, expansion_factor
    );

    let time = std::time::Instant::now();
    let coeffs_dev = memcpy_htod(&coeffs);
    cuda_sync();
    println!("CUDA memcpy_htod took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let cuda_result = cuda_expanded_ntt(&coeffs_dev, expansion_factor);
    cuda_sync();
    println!("CUDA NTT took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let cuda_result = memcpy_dtoh(&cuda_result);
    cuda_sync();
    println!("CUDA memcpy_dtoh took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let expected_result = expand_from_coeff::<F, EF>(&coeffs, expansion_factor);
    println!("CPU NTT took {} ms", time.elapsed().as_millis());

    assert!(cuda_result == expected_result);
}

#[test]
#[ignore]
pub fn test_cuda_ntt() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;

    cuda_engine::init::<F>(&[], 0);

    let rng = &mut StdRng::seed_from_u64(0);
    let log_len = 15;
    let len = 1 << log_len;
    for log_chunck_size in [3, 11] {
        let mut coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();
        let cuda_result = cuda_ntt(&coeffs, log_chunck_size);
        ntt_batch::<F, EF>(&mut coeffs, 1 << log_chunck_size);
        assert!(cuda_result == coeffs);
    }
}

#[test]
#[ignore]
pub fn test_cuda_monomial_to_lagrange_basis() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;

    cuda_engine::init::<F>(&[], 0);

    let rng = &mut StdRng::seed_from_u64(0);
    let n_vars = 20;
    let coeffs = (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
    let coeffs_dev = memcpy_htod(&coeffs);
    cuda_sync();
    let time = std::time::Instant::now();
    let cuda_result_dev = cuda_monomial_to_lagrange_basis_rev(&coeffs_dev);
    cuda_sync();
    println!(
        "CUDA lagrange_to_monomial_basis transform took {} ms",
        time.elapsed().as_millis()
    );
    let cuda_result = memcpy_dtoh(&cuda_result_dev);
    cuda_sync();
    let time = std::time::Instant::now();
    let expected_result = CoefficientListHost::new(coeffs)
        .reverse_vars()
        .to_lagrange_basis()
        .evals;
    println!(
        "CPU lagrange_to_monomial_basis transform took {} ms",
        time.elapsed().as_millis()
    );
    assert!(cuda_result == expected_result);
}

#[test]
#[ignore]
pub fn test_cuda_lagrange_to_monomial_basis() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;

    cuda_engine::init::<F>(&[], 0);

    let rng = &mut StdRng::seed_from_u64(0);
    let n_vars = 20;
    let evals = (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
    let evals_dev = memcpy_htod(&evals);
    cuda_sync();
    let time = std::time::Instant::now();
    let cuda_result_dev = cuda_lagrange_to_monomial_basis(&evals_dev);
    cuda_sync();
    println!(
        "CUDA lagrange_to_monomial_basis transform took {} ms",
        time.elapsed().as_millis()
    );
    let cuda_result = memcpy_dtoh(&cuda_result_dev);
    cuda_sync();
    let time = std::time::Instant::now();
    let expected_result = MultilinearHost::new(evals).to_monomial_basis().coeffs;
    println!(
        "CPU lagrange_to_monomial_basis transform took {} ms",
        time.elapsed().as_millis()
    );
    assert_eq!(cuda_result, expected_result);
}
#[test]
#[ignore]
pub fn test_cuda_restructure_evaluations() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;
    let whir_folding_factor = 4;

    cuda_engine::init::<F>(&[], whir_folding_factor);

    let rng = &mut StdRng::seed_from_u64(0);
    let log_len = 24;
    let len = 1 << log_len;
    let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();
    let coeffs_dev = memcpy_htod(&coeffs);
    cuda_sync();

    let time = std::time::Instant::now();
    let cuda_result = cuda_restructure_evaluations(&coeffs_dev, whir_folding_factor);
    cuda_sync();
    println!(
        "CUDA restructuraction took {} ms",
        time.elapsed().as_millis()
    );
    let time = std::time::Instant::now();
    let cuda_result = memcpy_dtoh(&cuda_result);
    cuda_sync();
    println!("CUDA memcpy_dtoh took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let domain_gen_inv = F::two_adic_generator(log_len).inverse();
    let expected_result =
        restructure_evaluations::<F, EF>(coeffs, domain_gen_inv, whir_folding_factor);
    println!(
        "CPU restructuraction took {} ms",
        time.elapsed().as_millis()
    );

    assert!(cuda_result == expected_result);
}

#[test]
#[ignore]
pub fn test_cuda_eval_multilinear_in_monomial_basis() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;
    cuda_engine::init::<F>(&[], 0);

    let rng = &mut StdRng::seed_from_u64(0);
    let n_vars = 20;
    let len = 1 << n_vars;
    let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
    let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();

    let coeffs_dev = memcpy_htod(&coeffs);
    cuda_sync();

    let time = std::time::Instant::now();
    let cuda_result = cuda_eval_multilinear_in_monomial_basis(&coeffs_dev, &point);
    cuda_sync();
    println!(
        "CUDA eval_multilinear_in_monomial_basiss took {} ms",
        time.elapsed().as_millis()
    );
    let time = std::time::Instant::now();
    let expected_result = CoefficientListHost::new(coeffs).evaluate(&point);
    println!("CPU took {} ms", time.elapsed().as_millis());

    assert_eq!(cuda_result, expected_result);
}

#[test]
#[ignore]
pub fn test_cuda_eval_multilinear_in_lagrange_basis() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;
    cuda_engine::init::<F>(&[], 0);

    let rng = &mut StdRng::seed_from_u64(0);
    let n_vars = 20;
    let len = 1 << n_vars;
    let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
    let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();

    let coeffs_dev = memcpy_htod(&coeffs);
    cuda_sync();

    let time = std::time::Instant::now();
    let cuda_result = cuda_eval_multilinear_in_lagrange_basis(&coeffs_dev, &point);
    cuda_sync();
    println!(
        "CUDA eval_multilinear_in_lagrange_basiss took {} ms",
        time.elapsed().as_millis()
    );
    let time = std::time::Instant::now();
    let expected_result = MultilinearHost::new(coeffs).evaluate(&point);
    println!("CPU took {} ms", time.elapsed().as_millis());

    assert_eq!(cuda_result, expected_result);
}

#[test]
#[ignore]
pub fn test_cuda_eq_mle() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;
    cuda_engine::init::<F>(&[], 0);

    let rng = &mut StdRng::seed_from_u64(0);
    let n_vars = 18;
    let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<EF>>();

    let time = std::time::Instant::now();
    let cuda_result = cuda_eq_mle(&point);
    cuda_sync();
    println!("CUDA eq_mle took {} ms", time.elapsed().as_millis());
    let cuda_result = memcpy_dtoh(&cuda_result);
    cuda_sync();
    let time = std::time::Instant::now();
    let expected_result = MultilinearHost::eq_mle(&point).evals;
    println!("CPU took {} ms", time.elapsed().as_millis());

    assert_eq!(cuda_result, expected_result);
}

#[test]
#[ignore]
pub fn test_cuda_whir_fold() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;
    cuda_engine::init::<F>(&[], 0);

    let rng = &mut StdRng::seed_from_u64(0);
    let n_vars = 23;
    let folding_factor = 4;
    let folding_randomness = (0..folding_factor)
        .map(|_| rng.random())
        .collect::<Vec<EF>>();
    let coeffs = (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
    let coeffs_dev = memcpy_htod(&coeffs);
    let coeffs_dev = CoefficientListDevice::new(coeffs_dev);
    cuda_sync();

    let time = std::time::Instant::now();
    let cuda_result = cuda_whir_fold(&coeffs_dev.coeffs, &folding_randomness);
    cuda_sync();
    println!("CUDA whir folding took {} ms", time.elapsed().as_millis());
    let cuda_result = CoefficientListHost::new(memcpy_dtoh(&cuda_result));
    cuda_sync();
    let time = std::time::Instant::now();
    let expected_result = CoefficientListHost::new(coeffs).fold(&folding_randomness);
    println!("CPU took {} ms", time.elapsed().as_millis());

    assert!(cuda_result == expected_result);
}
