use std::borrow::Borrow;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use cuda_engine::{SumcheckComputation, concat_pointers, cuda_info, memcpy_htod};
use p3_field::{ExtensionField, Field};

use crate::MAX_LOG_N_BLOCKS;

// TODO avoid hardcoding
const SUMCHECK_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const SUMCHECK_N_THREADS_PER_BLOCK: u32 = 1 << SUMCHECK_LOG_N_THREADS_PER_BLOCK;

pub fn cuda_sum_over_hypercube<F: Field, EF: ExtensionField<F>, ML: Borrow<CudaSlice<EF>>>(
    sumcheck_computation: &SumcheckComputation<F>,
    multilinears: &[ML], // in lagrange basis
    batching_scalars: &CudaSlice<EF>,
) -> EF {
    let multilinears: Vec<&CudaSlice<EF>> =
        multilinears.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert_eq!(batching_scalars.len(), sumcheck_computation.inner.len());
    let cuda = cuda_info();
    assert!(multilinears[0].len().is_power_of_two());
    let n_vars = multilinears[0].len().ilog2() as u32;
    assert!(multilinears.iter().all(|m| m.len() == 1 << n_vars as usize));

    let log_n_blocks =
        (n_vars.saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK)).min(MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;
    let ext_degree = (size_of::<EF>() / F::PrimeSubfield::bits().div_ceil(8)) as u32; // TODO this is ugly

    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (SUMCHECK_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: batching_scalars.len() as u32 * ext_degree * 4, // cf: __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];
    };

    let multilinears_ptrs_dev = concat_pointers(&multilinears);

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

pub fn fold_ext_by_prime<F: Field, EF: ExtensionField<F>, ML: Borrow<CudaSlice<EF>>>(
    slices: &[ML],
    scalar: F,
) -> Vec<CudaSlice<EF>> {
    let slices: Vec<&CudaSlice<EF>> = slices.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert!(slices[0].len().is_power_of_two());
    let n_vars = slices[0].len().ilog2() as u32;
    assert!(n_vars >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << n_vars as usize));
    let cuda = cuda_info();

    let slices_ptrs_dev = concat_pointers(&slices);
    let res = (0..slices.len())
        .map(|_| unsafe { cuda.stream.alloc::<EF>(1 << (n_vars - 1)).unwrap() })
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);

    let log_n_blocks =
        ((n_vars - 1).saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK)).min(MAX_LOG_N_BLOCKS);
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
    launch_args.arg(&n_vars);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    res
}

pub fn fold_ext_by_ext<EF: Field, ML: Borrow<CudaSlice<EF>>>(
    slices: &[ML],
    scalar: EF,
) -> Vec<CudaSlice<EF>> {
    let slices: Vec<&CudaSlice<EF>> = slices.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert!(slices[0].len().is_power_of_two());
    let n_vars = slices[0].len().ilog2() as u32;
    assert!(n_vars >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << n_vars as usize));
    let cuda = cuda_info();
    let slices_ptrs_dev = concat_pointers(&slices);

    let res = (0..slices.len())
        .map(|_| unsafe { cuda.stream.alloc::<EF>(1 << (n_vars - 1)).unwrap() })
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);

    let log_n_blocks =
        ((n_vars - 1).saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK)).min(MAX_LOG_N_BLOCKS);
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
    launch_args.arg(&n_vars);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    res
}
