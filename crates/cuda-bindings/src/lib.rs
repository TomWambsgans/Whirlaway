#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod init;
use std::time::Duration;

use algebra::pols::{MultilinearPolynomial, TransparentComputation};
pub use init::*;

#[cfg(test)]
mod tests;

use cudarc::driver::{DevicePtr, DriverError, LaunchAsync, LaunchConfig};

use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use tracing::instrument;

// TODO this value is also hardcoded in ntt.cuda, this is ugly
const NTT_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const NTT_N_THREADS_PER_BLOCK: u32 = 1 << NTT_LOG_N_THREADS_PER_BLOCK;

// TODO same remark, avoid hardcoding
const SUMCHECK_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const SUMCHECK_N_THREADS_PER_BLOCK: u32 = 1 << SUMCHECK_LOG_N_THREADS_PER_BLOCK;

// Should be not too big to avoid CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
const MAX_LOG_N_BLOCKS: u32 = 5;

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

    let log_n_blocks =
        (log_len + log_expension_factor - NTT_LOG_N_THREADS_PER_BLOCK - 1).min(MAX_LOG_N_BLOCKS);
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

pub fn cuda_sum_over_hypercube<const EXT_DEGREE: usize, F: PrimeField32, EF: ExtensionField<F>>(
    composition: &TransparentComputation<F, EF>,
    multilinears: &[MultilinearPolynomial<EF>],
    batching_scalars: &[EF],
) -> ([u32; EXT_DEGREE], Duration) {
    // TODO return EF
    let dev = &cuda_info().dev;
    assert!(
        multilinears
            .iter()
            .all(|m| m.n_vars == multilinears[0].n_vars)
    );
    let n_vars = multilinears[0].n_vars as u32;

    let batching_scalars_u32 = if batching_scalars.is_empty() {
        // We put 1 dummy scalar because N_BATCHING_SCALARS is always N_BATCHING_SCALARS >= 1 (To avoid an nvcc error)
        &[0_u32; EXT_DEGREE]
    } else {
        unsafe {
            std::slice::from_raw_parts(
                batching_scalars.as_ptr() as *const u32,
                EXT_DEGREE * batching_scalars.len(),
            )
        }
    };
    let batching_scalars_dev = dev.htod_copy(batching_scalars_u32.to_vec()).unwrap();

    let log_n_blocks =
        (n_vars.saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK)).min(MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (SUMCHECK_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: batching_scalars.len() as u32 * EXT_DEGREE as u32 * 4, // cf: __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];
    };

    let time = std::time::Instant::now();
    // TODO this could be parallelized ?
    let multilinears_dev = multilinears
        .iter()
        .map(|multilinear| {
            dev.htod_sync_copy(unsafe {
                std::slice::from_raw_parts(
                    multilinear.evals.as_ptr() as *const u32,
                    EXT_DEGREE << n_vars,
                )
            })
            .unwrap()
        })
        .collect::<Vec<_>>();

    let multilinears_ptrs_dev = dev
        .htod_sync_copy(
            &multilinears_dev
                .iter()
                .map(|slice_dev| *slice_dev.device_ptr())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let copy_duration = time.elapsed();

    let mut sums_dev = unsafe { dev.alloc::<u32>(EXT_DEGREE << n_vars).unwrap() };

    let mut res_dev = unsafe { dev.alloc::<u32>(EXT_DEGREE).unwrap() };

    let module_name = format!("sumcheck_{:x}", composition.uuid());
    let f = dev
        .get_func(&module_name, "sum_over_hypercube_ext")
        .unwrap();
    unsafe {
        f.launch_cooperative(
            cfg,
            (
                &multilinears_ptrs_dev,
                &mut sums_dev,
                &batching_scalars_dev,
                n_vars as u32,
                &mut res_dev,
            ),
        )
    }
    .unwrap();

    let res_u32: [u32; EXT_DEGREE] = dev.sync_reclaim(res_dev).unwrap().try_into().unwrap();
    (res_u32, copy_duration)
}
