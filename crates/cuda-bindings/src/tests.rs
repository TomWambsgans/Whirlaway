use std::error::Error;

use algebra::{ntt::expand_from_coeff, utils::expand_randomness};
use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig, sys::CUdevice_attribute};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{cuda_batch_keccak, cuda_info, cuda_ntt, init_cuda};
use rayon::prelude::*;

#[test]
fn test_cuda_hypercube_sum() {
    init_cuda::<KoalaBear>();
    let dev = &cuda_info().dev;

    const EXT_DEGREE: usize = 8;
    type EF = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let n_slices = 4;
    let n_vars = 24;

    let rng = &mut StdRng::seed_from_u64(0);

    let slices = (0..n_slices)
        .map(|_| (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<EF>>())
        .collect::<Vec<_>>();

    let batching_scalar: EF = rng.random();

    let batching_scalars = expand_randomness(batching_scalar, n_slices - 1);
    let batching_scalars_u32 = unsafe {
        std::slice::from_raw_parts(
            batching_scalars.as_ptr() as *const u32,
            EXT_DEGREE * (n_slices - 1),
        )
    };
    let batching_scalars_dev = dev.htod_copy(batching_scalars_u32.to_vec()).unwrap();

    let time = std::time::Instant::now();
    let expected_sum = (0..1 << n_vars)
        .into_par_iter()
        .map(|i| {
            slices[3][i]
                * (slices[0][i] * KoalaBear::new(11) * batching_scalars[0]
                    + slices[1][i] * KoalaBear::new(22) * batching_scalars[1]
                    + slices[2][i] * KoalaBear::new(33) * batching_scalars[2])
        })
        .sum::<EF>();
    println!("CPU hypercube sum took {} ms", time.elapsed().as_millis());

    let cfg = LaunchConfig {
        grid_dim: (64, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let slices_dev = slices
        .iter()
        .map(|slice| {
            dev.htod_sync_copy(unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u32, EXT_DEGREE * slice.len())
            })
            .unwrap()
        })
        .collect::<Vec<_>>();

    let slices_ptr_dev = dev
        .htod_sync_copy(
            &slices_dev
                .iter()
                .map(|slice_dev| *slice_dev.device_ptr())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let mut sums_dev = unsafe { dev.alloc::<u32>(EXT_DEGREE * 1 << n_vars).unwrap() };

    let mut res_dev = unsafe { dev.alloc::<u32>(EXT_DEGREE).unwrap() };

    let f_fold_prime_by_prime = dev.get_func("sumcheck", "sum_over_hypercube_ext").unwrap();
    let time = std::time::Instant::now();
    unsafe {
        f_fold_prime_by_prime.launch_cooperative(
            cfg,
            (
                &slices_ptr_dev,
                &mut sums_dev,
                &batching_scalars_dev,
                n_vars as u32,
                &mut res_dev,
            ),
        )
    }
    .unwrap();

    let res_u32: [u32; 8] = dev.sync_reclaim(res_dev).unwrap().try_into().unwrap();
    let result: EF = unsafe { std::mem::transmute(res_u32) };

    println!("CUDA hypercube sum took {} ms", time.elapsed().as_millis());

    assert_eq!(result, expected_sum);
}

#[test]
fn test_cuda_keccak() {
    let t = std::time::Instant::now();
    init_cuda::<KoalaBear>();
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
    init_cuda::<KoalaBear>();

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
    init_cuda::<KoalaBear>();

    let shared_memory = cuda_shared_memory().unwrap();
    println!("Shared memory per block: {} bytes", shared_memory);

    let constant_memory = cuda_constant_memory().unwrap();
    println!("Constant memory: {} bytes", constant_memory);
}
