use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use p3_field::{ExtensionField, Field};

use crate::{MAX_LOG_N_BLOCKS, SumcheckComputation, concat_pointers, cuda_info, memcpy_htod};

// TODO avoid hardcoding
const SUMCHECK_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const SUMCHECK_N_THREADS_PER_BLOCK: u32 = 1 << SUMCHECK_LOG_N_THREADS_PER_BLOCK;

pub fn cuda_sum_over_hypercube<F: Field, EF: ExtensionField<F>>(
    sumcheck_computation: &SumcheckComputation<F>,
    multilinears: &[CudaSlice<EF>],
    batching_scalars: &CudaSlice<EF>,
) -> EF {
    assert_eq!(batching_scalars.len(), sumcheck_computation.inner.len());
    let cuda = cuda_info();
    let n_vars = multilinears[0].len().trailing_zeros();
    assert!(multilinears.iter().all(|m| m.len() == 1 << n_vars),);

    let log_n_blocks =
        (n_vars.saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK)).min(MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;
    let ext_degree = (size_of::<EF>() / F::PrimeSubfield::bits().div_ceil(8)) as u32; // TODO this is ugly

    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (SUMCHECK_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: batching_scalars.len() as u32 * ext_degree * 4, // cf: __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];
    };

    let multilinears_ptrs_dev = concat_pointers(multilinears);

    let mut sums_dev = unsafe { cuda.stream.alloc::<EF>(1 << n_vars).unwrap() };

    let mut res_dev = unsafe { cuda.stream.alloc::<EF>(1).unwrap() };

    let module_name = format!("sumcheck_{:x}", sumcheck_computation.uuid());
    let f = cuda.get_function(&module_name, "sum_over_hypercube_ext");

    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&multilinears_ptrs_dev);
    launch_args.arg(&mut sums_dev);
    launch_args.arg(batching_scalars);
    launch_args.arg(&n_vars);
    launch_args.arg(&mut res_dev);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    let mut res = [EF::ZERO];
    cuda.stream.memcpy_dtoh(&res_dev, &mut res).unwrap();
    cuda.stream.synchronize().unwrap();

    res[0]
}

pub fn fold_ext_by_prime<F: Field, EF: ExtensionField<F>>(
    slices: &[CudaSlice<EF>],
    scalar: F,
) -> Vec<CudaSlice<EF>> {
    let log_slice_len = slices[0].len().trailing_zeros();
    assert!(log_slice_len >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << log_slice_len));
    let cuda = cuda_info();

    let slices_ptrs_dev = concat_pointers(slices);
    let res = (0..slices.len())
        .map(|_| unsafe { cuda.stream.alloc::<EF>(1 << (log_slice_len - 1)).unwrap() })
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);

    let log_n_blocks = ((log_slice_len - 1).saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK))
        .min(MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (SUMCHECK_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = cuda.get_function(&"sumcheck_folding", "fold_ext_by_prime");
    let n_slices = slices.len() as u32;
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&slices_ptrs_dev);
    launch_args.arg(&mut res_ptrs_dev);
    launch_args.arg(&scalar);
    launch_args.arg(&n_slices);
    launch_args.arg(&log_slice_len);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    res
}

pub fn fold_ext_by_ext<EF: Field>(slices: &[CudaSlice<EF>], scalar: EF) -> Vec<CudaSlice<EF>> {
    let log_slice_len = slices[0].len().trailing_zeros();
    assert!(log_slice_len >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << log_slice_len));
    let cuda = cuda_info();

    let slices_ptrs_dev = concat_pointers(slices);

    let res = (0..slices.len())
        .map(|_| unsafe { cuda.stream.alloc::<EF>(1 << (log_slice_len - 1)).unwrap() })
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);

    let log_n_blocks = ((log_slice_len - 1).saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK))
        .min(MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (SUMCHECK_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_slices = slices.len() as u32;
    let scalar_dev = memcpy_htod(&[scalar]);

    let f = cuda.get_function(&"sumcheck_folding", "fold_ext_by_ext");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&slices_ptrs_dev);
    launch_args.arg(&mut res_ptrs_dev);
    launch_args.arg(&scalar_dev);
    launch_args.arg(&n_slices);
    launch_args.arg(&log_slice_len);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    res
}
