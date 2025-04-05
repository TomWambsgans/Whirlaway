use std::time::Duration;

use algebra::pols::{MultilinearPolynomial, TransparentComputation};
use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};

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
    let cuda = cuda_info();
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
    let mut batching_scalars_dev = unsafe {
        cuda.stream
            .alloc::<u32>(batching_scalars_u32.len())
            .unwrap()
    };
    cuda.stream
        .memcpy_htod(batching_scalars_u32, &mut batching_scalars_dev)
        .unwrap();

    let log_n_blocks =
        (n_vars.saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK)).min(MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (SUMCHECK_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: batching_scalars.len() as u32 * EXT_DEGREE as u32 * 4, // cf: __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];
    };

    let time = std::time::Instant::now();

    let multilinears_dev = multilinears
        .iter()
        .map(|multilinear| {
            let mut multiliner_dev =
                unsafe { cuda.stream.alloc::<u32>(EXT_DEGREE << n_vars).unwrap() };
            cuda.stream
                .memcpy_htod(
                    unsafe {
                        std::slice::from_raw_parts(
                            multilinear.evals.as_ptr() as *const u32,
                            EXT_DEGREE << n_vars,
                        )
                    },
                    &mut multiliner_dev,
                )
                .unwrap();
            multiliner_dev
        })
        .collect::<Vec<_>>();

    let mut multilinears_ptrs_dev =
        unsafe { cuda.stream.alloc::<u64>(multilinears_dev.len()).unwrap() }; // TODO avoid hardcoding u64 (this is platform dependent)
    cuda.stream
        .memcpy_htod(
            &multilinears_dev
                .iter()
                .map(|slice_dev| slice_dev.device_ptr(&cuda.stream).0)
                .collect::<Vec<_>>(),
            &mut multilinears_ptrs_dev,
        )
        .unwrap();
    let copy_duration = time.elapsed();

    let mut sums_dev = unsafe { cuda.stream.alloc::<u32>(EXT_DEGREE << n_vars).unwrap() };

    let mut res_dev = unsafe { cuda.stream.alloc::<u32>(EXT_DEGREE).unwrap() };

    let module_name = format!("sumcheck_{:x}", composition.uuid());
    let f = cuda.get_function(&module_name, "sum_over_hypercube_ext");

    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&multilinears_ptrs_dev);
    launch_args.arg(&mut sums_dev);
    launch_args.arg(&batching_scalars_dev);
    launch_args.arg(&n_vars);
    launch_args.arg(&mut res_dev);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    let mut res_u32 = [0u32; EXT_DEGREE];
    cuda.stream.memcpy_dtoh(&res_dev, &mut res_u32).unwrap();
    cuda.stream.synchronize().unwrap();

    (res_u32, copy_duration)
}
