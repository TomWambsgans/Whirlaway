#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod tests;

use std::sync::{Arc, OnceLock};

use cudarc::{
    driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

use p3_field::TwoAdicField;
use rayon::prelude::*;
use tracing::instrument;

const NTT_LOG_N_THREADS_PER_BLOCK: u32 = 8; // TODO this value is also hardcoded in ntt.cuda, this is ugly
const NTT_N_THREADS_PER_BLOCK: u32 = 1 << NTT_LOG_N_THREADS_PER_BLOCK;
const NTT_MAX_LOG_N_BLOCKS: u32 = 5; // Should be a power of two, Should be not too big to avoid CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE

pub struct CudaInfo {
    dev: Arc<CudaDevice>,
    twiddles: CudaSlice<u32>, // For now we restrain ourseleves to the 2-addic roots of unity in the prime fields, so each one is represented by a u32
    two_adicity: usize,
}

static CUDA_INFO: OnceLock<CudaInfo> = OnceLock::new();

fn cuda_info() -> &'static CudaInfo {
    CUDA_INFO.get().expect("CUDA not initialized")
}

#[instrument(name = "CUDA initialization", skip_all)]
pub fn init_cuda<F: TwoAdicField>() {
    let _ = CUDA_INFO.get_or_init(|| {
        let dev = CudaDevice::new(0).unwrap();

        let keccak_ptx_path = env!("PTX_KECCAK_PATH");
        let keccak_ptx = std::fs::read_to_string(keccak_ptx_path).expect("Failed to read PTX file");
        dev.load_ptx(Ptx::from_src(keccak_ptx), "keccak", &["batch_keccak256"])
            .unwrap();

        let keccak_ptx_path = env!("PTX_NTT_PATH");
        let keccak_ptx = std::fs::read_to_string(keccak_ptx_path).expect("Failed to read PTX file");
        dev.load_ptx(Ptx::from_src(keccak_ptx), "ntt", &["ntt"])
            .unwrap();

        let keccak_ptx_path = env!("PTX_SUMCHECK_PATH");
        let keccak_ptx = std::fs::read_to_string(keccak_ptx_path).expect("Failed to read PTX file");
        dev.load_ptx(
            Ptx::from_src(keccak_ptx),
            "sumcheck",
            &[
                "fold_prime_by_prime",
                "fold_prime_by_ext",
                "fold_ext_by_prime",
                "fold_ext_by_ext",
                "sum_over_hypercube_ext",
            ],
        )
        .unwrap();

        let twiddles = store_twiddles::<F>(&dev).unwrap();

        CudaInfo {
            dev,
            twiddles,
            two_adicity: F::TWO_ADICITY,
        }
    });
}

#[instrument(name = "pre-processing twiddles for CUDA", skip_all)]
fn store_twiddles<F: TwoAdicField>(dev: &Arc<CudaDevice>) -> Result<CudaSlice<u32>, DriverError> {
    assert!(F::bits() <= 32);
    let num_threads = rayon::current_num_threads().next_power_of_two();
    let mut all_twiddles = Vec::new();
    for i in 0..=F::TWO_ADICITY {
        // TODO only use the required twiddles (TWO_ADICITY may be larger than needed)
        let root = F::two_adic_generator(i);
        let twiddles = if (1 << i) <= num_threads {
            (0..1 << i)
                .into_iter()
                .map(|j| root.exp_u64(j as u64))
                .collect::<Vec<F>>()
        } else {
            let chunk_size = (1 << i) / num_threads;
            (0..num_threads)
                .into_par_iter()
                .map(|j| {
                    let mut start = root.exp_u64(j as u64 * chunk_size as u64);
                    let mut chunck = Vec::new();
                    for _ in 0..chunk_size {
                        chunck.push(start);
                        start = start * root;
                    }
                    chunck
                })
                .flatten()
                .collect()
        };
        all_twiddles.extend(twiddles);
    }

    let all_twiddles_u32 = unsafe {
        std::slice::from_raw_parts(all_twiddles.as_ptr() as *const u32, all_twiddles.len())
    }
    .to_vec();

    let all_twiddles_dev = dev.htod_copy(all_twiddles_u32).unwrap();
    dev.synchronize().unwrap();

    Ok(all_twiddles_dev)
}

pub fn cuda_batch_keccak(
    buff: &[u8],
    input_length: usize,
    input_packed_length: usize,
) -> Result<Vec<[u8; 32]>, DriverError> {
    assert!(buff.len() % input_packed_length == 0);
    assert!(input_length <= input_packed_length);

    let dev = &cuda_info().dev;
    let f = dev.get_func("keccak", "batch_keccak256").unwrap();

    let n_inputs = buff.len() / input_packed_length;
    let src_bytes_dev = dev.htod_sync_copy(buff)?;
    // dev.synchronize()?;
    let mut dest_dev = unsafe { dev.alloc::<u8>(32 * n_inputs)? };

    const NUM_THREADS: u32 = 256;
    let num_blocks = (n_inputs as u32).div_ceil(NUM_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(
            cfg,
            (
                &src_bytes_dev,
                n_inputs as u32,
                input_length as u32,
                input_packed_length as u32,
                &mut dest_dev,
            ),
        )
    }?;

    let mut res = dev.sync_reclaim(dest_dev)?;

    assert!(res.len() == 32 * n_inputs);

    let array_count = res.len() / 32;
    let capacity = res.capacity() / 32;
    let ptr = res.as_mut_ptr() as *mut [u8; 32];
    std::mem::forget(res);
    unsafe { Ok(Vec::from_raw_parts(ptr, array_count, capacity)) }
}

#[instrument(name = "CUDA NTT", skip_all)]
pub fn cuda_ntt<F: TwoAdicField>(coeffs: &[F], expansion_factor: usize) -> Vec<F> {
    // SAFETY: one should have called init_cuda::<F::PrimeSubfield>() before

    let cuda = cuda_info();

    assert!(coeffs.len().is_power_of_two());
    assert!(expansion_factor.is_power_of_two());

    let expanded_len = coeffs.len() * expansion_factor;
    let log_len = coeffs.len().trailing_zeros() as u32;
    let log_expension_factor = expansion_factor.trailing_zeros() as u32;

    // Because of `ntt_at_block_level` in ntt.cu, where each blocks handles 1 << (NTT_LOG_N_THREADS_PER_BLOCK + 1) elements
    assert!(
        log_len >= NTT_LOG_N_THREADS_PER_BLOCK + 1,
        "Cuda small NTT not suported (for now), use CPU instead"
    );

    let log_n_blocks = (log_len + log_expension_factor - NTT_LOG_N_THREADS_PER_BLOCK - 1)
        .min(NTT_MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;

    assert!(
        log_expension_factor + log_len <= cuda.two_adicity as u32,
        "NTT to big, TODO use the two addic unit roots from the extensioon field"
    );

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);
    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve

    let coeffs_u32 = unsafe {
        std::slice::from_raw_parts(
            coeffs.as_ptr() as *const u32,
            coeffs.len() * extension_degree,
        )
    };

    let coeffs_dev = cuda.dev.htod_sync_copy(coeffs_u32).unwrap();
    cuda.dev.synchronize().unwrap();

    let mut buff_dev = unsafe {
        cuda.dev
            .alloc::<u32>(expanded_len * extension_degree)
            .unwrap()
    };

    let mut result_dev = unsafe {
        cuda.dev
            .alloc::<u32>(expanded_len * extension_degree)
            .unwrap()
    };

    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (NTT_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: (NTT_N_THREADS_PER_BLOCK * 2) * (extension_degree as u32 + 1) * 4, // cf `ntt_at_block_level` in ntt.cu
    };

    let f_ntt = cuda.dev.get_func("ntt", "ntt").unwrap();
    unsafe {
        f_ntt.launch_cooperative(
            cfg,
            (
                &coeffs_dev,
                &mut buff_dev,
                &mut result_dev,
                log_len as u32,
                log_expension_factor as u32,
                &cuda.twiddles,
            ),
        )
    }
    .unwrap();

    let cuda_result_u32: Vec<u32> = cuda.dev.sync_reclaim(result_dev).unwrap();
    assert_eq!(cuda_result_u32.len(), expanded_len * extension_degree);
    assert_eq!(cuda_result_u32.capacity(), cuda_result_u32.len());

    let ptr = cuda_result_u32.as_ptr() as *mut F;
    // Prevent the original vector from being dropped
    let _ = std::mem::ManuallyDrop::new(cuda_result_u32);
    unsafe { Vec::from_raw_parts(ptr, expanded_len, expanded_len) }
}
