use std::time::Duration;

use algebra::pols::{MultilinearPolynomial, TransparentComputation};
use cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};

use p3_field::{ExtensionField, PrimeField32};

use crate::{MAX_LOG_N_BLOCKS, cuda_info};

// TODO avoid hardcoding
const SUMCHECK_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const SUMCHECK_N_THREADS_PER_BLOCK: u32 = 1 << SUMCHECK_LOG_N_THREADS_PER_BLOCK;

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
