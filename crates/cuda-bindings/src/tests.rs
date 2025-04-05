use std::error::Error;

use algebra::{
    ntt::expand_from_coeff,
    pols::{ArithmeticCircuit, MultilinearPolynomial, TransparentComputation},
};
use cudarc::driver::{CudaDevice, sys::CUdevice_attribute};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{cuda_batch_keccak, cuda_ntt, cuda_sum_over_hypercube};
use rayon::prelude::*;

#[test]
#[ignore]
fn test_cuda_hypercube_sum() {
    type F = KoalaBear;
    const EXT_DEGREE: usize = 8;
    type EF = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let n_multilinears = 5;
    let n_vars = 22;

    let composition: ArithmeticCircuit<F, usize> =
        ArithmeticCircuit::random(&mut StdRng::seed_from_u64(0), n_multilinears, 5);
    let composition = TransparentComputation::Generic(composition.fix_computation(true));
    let time = std::time::Instant::now();
    super::init::<F, EF>(&[&composition]);
    println!("CUDA initialized in {} ms", time.elapsed().as_millis());

    let rng = &mut StdRng::seed_from_u64(0);

    let multilinears = (0..n_multilinears)
        .map(|_| MultilinearPolynomial::<EF>::random(rng, n_vars))
        .collect::<Vec<_>>();

    let time = std::time::Instant::now();
    let expected_sum = (0..1 << n_vars)
        .into_par_iter()
        .map(|i| {
            composition.eval(
                &(0..n_multilinears)
                    .map(|j| multilinears[j].evals[i])
                    .collect::<Vec<_>>(),
            )
        })
        .sum::<EF>();
    println!("CPU hypercube sum took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let cuda_sum_u32 =
        cuda_sum_over_hypercube::<EXT_DEGREE, F, EF>(&composition, &multilinears, &[]);
    let cuda_sum: EF = unsafe { std::mem::transmute(cuda_sum_u32) };
    println!("CUDA hypercube sum took {} ms", time.elapsed().as_millis());

    assert_eq!(cuda_sum, expected_sum);
}

#[test]
fn test_cuda_keccak() {
    let t = std::time::Instant::now();
    super::init::<KoalaBear, KoalaBear>(&[]);
    println!("CUDA initialized in {} ms", t.elapsed().as_millis());

    let n_inputs = 1000_000;
    let input_length = 100;
    let input_packed_length = 111;
    let src_bytes = (0..n_inputs * input_packed_length)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>();

    let time = std::time::Instant::now();
    let expected_result = (0..n_inputs)
        .into_par_iter()
        .map(|i| {
            hash_keccak256(
                &src_bytes[i * input_packed_length..i * input_packed_length + input_length],
            )
        })
        .collect::<Vec<[u8; 32]>>();
    println!("CPU took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let dest = cuda_batch_keccak(&src_bytes, input_length, input_packed_length).unwrap();
    println!("CUDA took {} ms", time.elapsed().as_millis());
    assert_eq!(dest.len(), expected_result.len());
}

fn hash_keccak256(data: &[u8]) -> [u8; 32] {
    // TODO this function is duplicated elsewhere
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

#[test]
pub fn test_cuda_ntt() {
    super::init::<KoalaBear, KoalaBear>(&[]);

    const EXT_DEGREE: usize = 8;

    type F = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let rng = &mut StdRng::seed_from_u64(0);
    let log_len = 20;
    let len = 1 << log_len;
    let log_expension_factor: usize = 3;
    let expansion_factor = 1 << log_expension_factor;
    let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<F>>();

    println!(
        "number of field elements: {}, expension factor: {}",
        len, expansion_factor
    );
    let time = std::time::Instant::now();
    let cuda_result = cuda_ntt(&coeffs, expansion_factor);
    println!("CUDA NTT took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let expected_result = expand_from_coeff(&coeffs, expansion_factor);
    println!("CPU NTT took {} ms", time.elapsed().as_millis());

    assert_eq!(cuda_result, expected_result);
}

fn cuda_shared_memory() -> Result<usize, Box<dyn Error>> {
    let dev = CudaDevice::new(0)?;
    Ok(dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK)? as usize)
}

fn cuda_constant_memory() -> Result<usize, Box<dyn Error>> {
    let dev = CudaDevice::new(0)?;
    Ok(dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)? as usize)
}

#[test]
fn print_cuda_info() {
    super::init::<KoalaBear, KoalaBear>(&[]);

    let shared_memory = cuda_shared_memory().unwrap();
    println!("Shared memory per block: {} bytes", shared_memory);

    let constant_memory = cuda_constant_memory().unwrap();
    println!("Constant memory: {} bytes", constant_memory);
}
