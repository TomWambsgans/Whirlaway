use crate::{
    SumcheckComputation, cuda_alloc, cuda_expanded_ntt, cuda_info, cuda_keccak256, cuda_ntt,
    cuda_restructure_evaluations, cuda_sum_over_hypercube, cuda_sync, memcpy_dtoh, memcpy_htod,
};
use algebra::{
    ntt::{expand_from_coeff, ntt_batch, restructure_evaluations},
    pols::{ArithmeticCircuit, MultilinearPolynomial},
    utils::{KeccakDigest, keccak256},
};
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_field::TwoAdicField;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

/*
To run all the tests here:

for test in test_cuda_hypercube_sum test_cuda_keccak test_cuda_expanded_ntt test_cuda_expanded_ntt test_cuda_ntt test_cuda_restructure_evaluations; do
  cargo test --release --package cuda-bindings --lib -- "tests::$test" --exact --nocapture --ignored
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
    super::init(&[sumcheck_computation.clone()], 0);
    println!("CUDA initialized in {} ms", time.elapsed().as_millis());
    let cuda = cuda_info();

    let rng = &mut StdRng::seed_from_u64(0);

    let multilinears = (0..n_multilinears)
        .map(|_| MultilinearPolynomial::<EF>::random(rng, n_vars))
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
            multiliner_dev
        })
        .collect::<Vec<_>>();
    let copy_duration = time.elapsed();

    let time = std::time::Instant::now();
    let cuda_sum = cuda_sum_over_hypercube::<F, EF>(
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
fn test_cuda_keccak() {
    let t = std::time::Instant::now();
    super::init::<KoalaBear>(&[], 0);
    println!("CUDA initialized in {} ms", t.elapsed().as_millis());

    let n_inputs = 10_000;
    let batch_size = 501;
    let input = (0..n_inputs * batch_size)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>();

    let time = std::time::Instant::now();
    let expected_result = (0..n_inputs)
        .into_par_iter()
        .map(|i| keccak256(&input[i * batch_size..(i + 1) * batch_size]))
        .collect::<Vec<KeccakDigest>>();
    println!("CPU keccak took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let input_dev = memcpy_htod(&input);
    cuda_sync();
    println!("CUDA memcpy_htod took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let output_dev = cuda_alloc::<KeccakDigest>(n_inputs as usize);
    cuda_keccak256(
        &input_dev.as_view(),
        batch_size,
        &mut output_dev.as_view_mut(),
    );
    cuda_sync();
    println!("CUDA keccak took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let dest = memcpy_dtoh(&output_dev);
    cuda_sync();
    println!("CUDA memcpy_dtoh took {} ms", time.elapsed().as_millis());
    assert!(dest == expected_result);

    let n_particular_hashes = 300;
    let particular_indexes = (0..n_particular_hashes)
        .map(|i| (i * i * 3 + i + 78) % n_inputs)
        .collect::<Vec<_>>();
    let mut particular_hashes = (0..n_particular_hashes)
        .map(|_| [KeccakDigest::default()])
        .collect::<Vec<_>>();

    let time = std::time::Instant::now();
    for i in 0..n_particular_hashes {
        cuda_info()
            .stream
            .memcpy_dtoh(
                &output_dev.slice(particular_indexes[i]..1 + particular_indexes[i]),
                &mut particular_hashes[i],
            )
            .unwrap();
    }
    cuda_sync();
    println!(
        "CUDA transfer of {} hashes from gpu to cpu took {} ms",
        n_particular_hashes,
        time.elapsed().as_millis()
    );

    for i in 0..n_particular_hashes {
        assert_eq!(
            particular_hashes[i][0],
            expected_result[particular_indexes[i]]
        );
    }
}

#[test]
#[ignore]
pub fn test_cuda_expanded_ntt() {
    super::init::<KoalaBear>(&[], 0);

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
    let cuda_result = cuda_expanded_ntt(&coeffs, expansion_factor);
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

    super::init::<F>(&[], 0);

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
pub fn test_cuda_restructure_evaluations() {
    const EXT_DEGREE: usize = 8;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, EXT_DEGREE>;
    let whir_folding_factor = 4;

    super::init::<F>(&[], whir_folding_factor);

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

// fn cuda_shared_memory() -> Result<usize, Box<dyn Error>> {
//     let dev = CudaContext::new(0)?;
//     Ok(dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK)? as usize)
// }

// fn cuda_constant_memory() -> Result<usize, Box<dyn Error>> {
//     let dev = CudaContext::new(0)?;
//     Ok(dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)? as usize)
// }
