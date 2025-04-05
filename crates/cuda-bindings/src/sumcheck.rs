use algebra::pols::TransparentComputation;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};

use p3_field::{ExtensionField, PrimeField32};

use crate::{MAX_LOG_N_BLOCKS, cuda_info};

// TODO avoid hardcoding
const SUMCHECK_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const SUMCHECK_N_THREADS_PER_BLOCK: u32 = 1 << SUMCHECK_LOG_N_THREADS_PER_BLOCK;

pub fn cuda_sum_over_hypercube<const EXT_DEGREE: usize, F: PrimeField32, EF: ExtensionField<F>>(
    composition: &TransparentComputation<F, EF>,
    n_vars: u32,
    multilinears: &[CudaSlice<EF>],
    batching_scalars: &[EF],
) -> EF {
    // TODO return EF
    let cuda = cuda_info();
    assert!(multilinears.iter().all(|m| m.len() == 1 << n_vars),);

    let batching_scalars = if batching_scalars.is_empty() {
        // We put 1 dummy scalar because N_BATCHING_SCALARS is always N_BATCHING_SCALARS >= 1 (To avoid an nvcc error)
        &[EF::ZERO]
    } else {
        batching_scalars
    };
    let mut batching_scalars_dev =
        unsafe { cuda.stream.alloc::<EF>(batching_scalars.len()).unwrap() };
    cuda.stream
        .memcpy_htod(batching_scalars, &mut batching_scalars_dev)
        .unwrap();

    let log_n_blocks =
        (n_vars.saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK)).min(MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (SUMCHECK_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: batching_scalars.len() as u32 * EXT_DEGREE as u32 * 4, // cf: __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];
    };

    let mut multilinears_ptrs_dev =
        unsafe { cuda.stream.alloc::<u64>(multilinears.len()).unwrap() }; // TODO avoid hardcoding u64 (this is platform dependent)
    cuda.stream
        .memcpy_htod(
            &multilinears
                .iter()
                .map(|slice_dev| slice_dev.device_ptr(&cuda.stream).0)
                .collect::<Vec<_>>(),
            &mut multilinears_ptrs_dev,
        )
        .unwrap();

    let mut sums_dev = unsafe { cuda.stream.alloc::<EF>(1 << n_vars).unwrap() };

    let mut res_dev = unsafe { cuda.stream.alloc::<EF>(1).unwrap() };

    let module_name = format!("sumcheck_{:x}", composition.uuid());
    let f = cuda.get_function(&module_name, "sum_over_hypercube_ext");

    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&multilinears_ptrs_dev);
    launch_args.arg(&mut sums_dev);
    launch_args.arg(&batching_scalars_dev);
    launch_args.arg(&n_vars);
    launch_args.arg(&mut res_dev);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    let mut res = [EF::ZERO];
    cuda.stream.memcpy_dtoh(&res_dev, &mut res).unwrap();
    cuda.stream.synchronize().unwrap();

    res[0]
}
