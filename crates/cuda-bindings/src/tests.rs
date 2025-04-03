use std::error::Error;

use algebra::{ntt::expand_from_coeff, pols::MultilinearPolynomial};
use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig, sys::CUdevice_attribute};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{cuda_batch_keccak, cuda_info, cuda_ntt, init_cuda};
use rayon::prelude::*;


#[test]
fn test_cuda_fold_ext_by_ext() {
    init_cuda::<KoalaBear>();
    let dev = &cuda_info().dev;

    const EXT_DEGREE: usize = 8;
    type EF = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let n_slices = 382;
    let log_slices_len = 11;

    let rng = &mut StdRng::seed_from_u64(0);

    let slices = (0..n_slices)
        .map(|_| {
            (0..1 << log_slices_len)
                .map(|_| rng.random())
                .collect::<Vec<EF>>()
        })
        .collect::<Vec<_>>();

    let cfg = LaunchConfig {
        grid_dim: (7, 1, 1),
        block_dim: (3, 1, 1),
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

    let results_dev = (0..n_slices)
        .map(|_| unsafe {
            dev.alloc::<u32>(EXT_DEGREE * 1 << (log_slices_len - 1))
                .unwrap()
        })
        .collect::<Vec<_>>();

    let mut results_ptr_dev = dev
        .htod_sync_copy(
            &results_dev
                .iter()
                .map(|slice_dev| *slice_dev.device_ptr())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let scalar: EF = rng.random();
    let scalar_u32 = unsafe { std::mem::transmute::<_, [u32; EXT_DEGREE]>(scalar) };
    let scalar_dev = dev.htod_copy(scalar_u32.to_vec()).unwrap();

    let f_fold_prime_by_prime = dev.get_func("sumcheck", "fold_prime_by_prime").unwrap();
    unsafe {
        f_fold_prime_by_prime.launch(
            cfg,
            (
                &slices_ptr_dev,
                &mut results_ptr_dev,
                &scalar_dev,
                n_slices as u32,
                log_slices_len as u32,
            ),
        )
    }
    .unwrap();

    let result = results_dev
        .into_iter()
        .map(|res| unsafe {
            std::slice::from_raw_parts(
                dev.sync_reclaim(res).unwrap().as_ptr() as *const KoalaBear,
                1 << (log_slices_len - 1),
            )
        })
        .collect::<Vec<_>>();

    let expected_result = slices
        .iter()
        .map(|slice| {
            MultilinearPolynomial::new(slice.clone())
                .fix_variable(EF::from(scalar))
                .evals
        })
        .collect::<Vec<_>>();

    assert_eq!(result.len(), expected_result.len());
}


#[test]
fn test_cuda_fold_ext_by_prime() {
    init_cuda::<KoalaBear>();
    let dev = &cuda_info().dev;

    const EXT_DEGREE: usize = 8;
    type EF = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let n_slices = 382;
    let log_slices_len = 11;

    let rng = &mut StdRng::seed_from_u64(0);

    let slices = (0..n_slices)
        .map(|_| {
            (0..1 << log_slices_len)
                .map(|_| rng.random())
                .collect::<Vec<EF>>()
        })
        .collect::<Vec<_>>();

    let cfg = LaunchConfig {
        grid_dim: (7, 1, 1),
        block_dim: (3, 1, 1),
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

    let results_dev = (0..n_slices)
        .map(|_| unsafe {
            dev.alloc::<u32>(EXT_DEGREE * 1 << (log_slices_len - 1))
                .unwrap()
        })
        .collect::<Vec<_>>();

    let mut results_ptr_dev = dev
        .htod_sync_copy(
            &results_dev
                .iter()
                .map(|slice_dev| *slice_dev.device_ptr())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let scalar: KoalaBear = rng.random();
    let scalar_u32 = unsafe { std::mem::transmute::<_, u32>(scalar) };

    let f_fold_prime_by_prime = dev.get_func("sumcheck", "fold_prime_by_prime").unwrap();
    unsafe {
        f_fold_prime_by_prime.launch(
            cfg,
            (
                &slices_ptr_dev,
                &mut results_ptr_dev,
                scalar_u32 as u32,
                n_slices as u32,
                log_slices_len as u32,
            ),
        )
    }
    .unwrap();

    let result = results_dev
        .into_iter()
        .map(|res| unsafe {
            std::slice::from_raw_parts(
                dev.sync_reclaim(res).unwrap().as_ptr() as *const KoalaBear,
                1 << (log_slices_len - 1),
            )
        })
        .collect::<Vec<_>>();

    let expected_result = slices
        .iter()
        .map(|slice| {
            MultilinearPolynomial::new(slice.clone())
                .fix_variable(EF::from(scalar))
                .evals
        })
        .collect::<Vec<_>>();

    assert_eq!(result.len(), expected_result.len());
}

#[test]
fn test_cuda_fold_prime_by_ext() {
    init_cuda::<KoalaBear>();
    let dev = &cuda_info().dev;

    const EXT_DEGREE: usize = 8;
    type EF = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let n_slices = 382;
    let log_slices_len = 11;

    let rng = &mut StdRng::seed_from_u64(0);

    let slices = (0..n_slices)
        .map(|_| {
            (0..1 << log_slices_len)
                .map(|_| rng.random())
                .collect::<Vec<KoalaBear>>()
        })
        .collect::<Vec<_>>();

    let cfg = LaunchConfig {
        grid_dim: (7, 1, 1),
        block_dim: (3, 1, 1),
        shared_mem_bytes: 0,
    };

    let slices_dev = slices
        .iter()
        .map(|slice| {
            dev.htod_sync_copy(unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len())
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

    let results_dev = (0..n_slices)
        .map(|_| unsafe {
            dev.alloc::<u32>(EXT_DEGREE * 1 << (log_slices_len - 1))
                .unwrap()
        })
        .collect::<Vec<_>>();

    let mut results_ptr_dev = dev
        .htod_sync_copy(
            &results_dev
                .iter()
                .map(|slice_dev| *slice_dev.device_ptr())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let scalar: EF = rng.random();
    let scalar_u32 = unsafe { std::mem::transmute::<_, [u32; 8]>(scalar) };
    let scalar_dev = dev.htod_copy(scalar_u32.to_vec()).unwrap();

    let f_fold_prime_by_prime = dev.get_func("sumcheck", "fold_prime_by_prime").unwrap();
    unsafe {
        f_fold_prime_by_prime.launch(
            cfg,
            (
                &slices_ptr_dev,
                &mut results_ptr_dev,
                &scalar_dev,
                n_slices as u32,
                log_slices_len as u32,
            ),
        )
    }
    .unwrap();

    let result = results_dev
        .into_iter()
        .map(|res| unsafe {
            std::slice::from_raw_parts(
                dev.sync_reclaim(res).unwrap().as_ptr() as *const KoalaBear,
                1 << (log_slices_len - 1),
            )
        })
        .collect::<Vec<_>>();

    let expected_result = slices
        .iter()
        .map(|slice| {
            MultilinearPolynomial::new(slice.clone())
                .fix_variable(scalar)
                .evals
        })
        .collect::<Vec<_>>();

    assert_eq!(result.len(), expected_result.len());
}

#[test]
fn test_cuda_fold_prime_by_prime() {
    init_cuda::<KoalaBear>();
    let dev = &cuda_info().dev;

    let n_slices = 382;
    let log_slices_len = 11;

    let rng = &mut StdRng::seed_from_u64(0);

    let slices = (0..n_slices)
        .map(|_| {
            (0..1 << log_slices_len)
                .map(|_| rng.random())
                .collect::<Vec<KoalaBear>>()
        })
        .collect::<Vec<_>>();

    let cfg = LaunchConfig {
        grid_dim: (7, 1, 1),
        block_dim: (3, 1, 1),
        shared_mem_bytes: 0,
    };

    let slices_dev = slices
        .iter()
        .map(|slice| {
            dev.htod_sync_copy(unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len())
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

    let results_dev = (0..n_slices)
        .map(|_| unsafe { dev.alloc::<u32>(1 << (log_slices_len - 1)).unwrap() })
        .collect::<Vec<_>>();

    let mut results_ptr_dev = dev
        .htod_sync_copy(
            &results_dev
                .iter()
                .map(|slice_dev| *slice_dev.device_ptr())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let scalar: KoalaBear = rng.random();
    let scalar_u32 = unsafe { std::mem::transmute::<_, u32>(scalar) };

    let f_fold_prime_by_prime = dev.get_func("sumcheck", "fold_prime_by_prime").unwrap();
    unsafe {
        f_fold_prime_by_prime.launch(
            cfg,
            (
                &slices_ptr_dev,
                &mut results_ptr_dev,
                scalar_u32,
                n_slices as u32,
                log_slices_len as u32,
            ),
        )
    }
    .unwrap();

    let result = results_dev
        .into_iter()
        .map(|res| unsafe {
            std::slice::from_raw_parts(
                dev.sync_reclaim(res).unwrap().as_ptr() as *const KoalaBear,
                1 << (log_slices_len - 1),
            )
        })
        .collect::<Vec<_>>();

    let expected_result = slices
        .iter()
        .map(|slice| {
            MultilinearPolynomial::new(slice.clone())
                .fix_variable(scalar)
                .evals
        })
        .collect::<Vec<_>>();

    assert_eq!(result.len(), expected_result.len());
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
