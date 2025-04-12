use std::{any::TypeId, borrow::Borrow};

use cudarc::driver::{CudaSlice, PushKernelArg};

use cuda_engine::{
    CudaCall, SumcheckComputation, concat_pointers, cuda_alloc, cuda_sync, memcpy_dtoh, memcpy_htod,
};
use p3_field::{BasedVectorSpace, ExtensionField, Field};

use crate::{MAX_LOG_N_COOPERATIVE_BLOCKS, MAX_N_BLOCKS};

// TODO avoid hardcoding
const SUMCHECK_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const SUMCHECK_N_THREADS_PER_BLOCK: u32 = 1 << SUMCHECK_LOG_N_THREADS_PER_BLOCK;

/// Async
pub fn cuda_sum_over_hypercube_of_computation<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    ML: Borrow<CudaSlice<NF>>,
>(
    comp: &SumcheckComputation<F>,
    multilinears: &[ML], // in lagrange basis
    batching_scalars: &[EF],
) -> EF {
    if TypeId::of::<EF>() != TypeId::of::<NF>() {
        unimplemented!()
    }

    let multilinears: Vec<&CudaSlice<NF>> =
        multilinears.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert_eq!(batching_scalars.len(), comp.exprs.len());
    assert!(multilinears[0].len().is_power_of_two());
    let n_vars = multilinears[0].len().ilog2() as u32;
    assert!(multilinears.iter().all(|m| m.len() == 1 << n_vars as usize));

    let log_n_blocks =
        (n_vars.saturating_sub(SUMCHECK_LOG_N_THREADS_PER_BLOCK)).min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let n_blocks = 1 << log_n_blocks;
    let ext_degree = <EF as BasedVectorSpace<F>>::DIMENSION as u32;

    let multilinears_ptrs_dev = concat_pointers(&multilinears);

    let batching_scalars_dev = memcpy_htod(batching_scalars);
    let mut sums_dev = cuda_alloc::<EF>(1 << n_vars);

    let mut res_dev = cuda_alloc::<EF>(1);

    let module_name = format!("sumcheck_{:x}", comp.uuid());
    let mut call = CudaCall::new(&module_name, "sum_over_hypercube_ext")
        .blocks(n_blocks)
        .threads_per_block(SUMCHECK_N_THREADS_PER_BLOCK)
        .shared_mem_bytes(batching_scalars.len() as u32 * ext_degree * 4); // cf: __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];;
    call.arg(&multilinears_ptrs_dev);
    call.arg(&mut sums_dev);
    call.arg(&batching_scalars_dev);
    call.arg(&n_vars);
    call.arg(&mut res_dev);
    call.launch_cooperative();

    let res: [EF; 1] = memcpy_dtoh(&res_dev).try_into().unwrap();
    cuda_sync();

    res[0]
}

/// Async
pub fn cuda_fix_variable_in_small_field<
    F: Field,
    EF: ExtensionField<F>,
    ML: Borrow<CudaSlice<EF>>,
>(
    slices: &[ML],
    scalar: F,
) -> Vec<CudaSlice<EF>> {
    assert!(TypeId::of::<EF>() != TypeId::of::<F>(), "TODO");
    let slices: Vec<&CudaSlice<EF>> = slices.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert!(slices[0].len().is_power_of_two());
    let n_vars = slices[0].len().ilog2() as u32;
    assert!(n_vars >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << n_vars as usize));

    let slices_ptrs_dev = concat_pointers(&slices);
    let res = (0..slices.len())
        .map(|_| cuda_alloc::<EF>(1 << (n_vars - 1)))
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);

    let n_reps = (slices.len() as u32) << (n_vars - 1);
    let n_threads_per_block = n_reps.min(1 << SUMCHECK_LOG_N_THREADS_PER_BLOCK);
    let n_blocks = ((n_reps + n_threads_per_block - 1) / n_threads_per_block).min(MAX_N_BLOCKS);
    let n_slices = slices.len() as u32;
    let mut call = CudaCall::new("multilinear", "fold_ext_by_prime")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_block);
    call.arg(&slices_ptrs_dev);
    call.arg(&mut res_ptrs_dev);
    call.arg(&scalar);
    call.arg(&n_slices);
    call.arg(&n_vars);
    call.launch();

    res
}

/// Async
pub fn cuda_fix_variable_in_big_field<F: Field, EF: ExtensionField<F>, ML: Borrow<CudaSlice<F>>>(
    slices: &[ML],
    scalar: EF,
) -> Vec<CudaSlice<EF>> {
    assert!(F::bits() > 32, "TODO");
    let slices: Vec<&CudaSlice<F>> = slices.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert!(slices[0].len().is_power_of_two());
    let n_vars = slices[0].len().ilog2() as u32;
    assert!(n_vars >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << n_vars as usize));
    let slices_ptrs_dev = concat_pointers(&slices);

    let res = (0..slices.len())
        .map(|_| cuda_alloc::<EF>(1 << (n_vars - 1)))
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);

    let n_reps = (slices.len() as u32) << (n_vars - 1);
    let n_threads_per_block = n_reps.min(1 << SUMCHECK_LOG_N_THREADS_PER_BLOCK);
    let n_blocks = ((n_reps + n_threads_per_block - 1) / n_threads_per_block).min(MAX_N_BLOCKS);
    let n_slices = slices.len() as u32;
    let scalar_dev = memcpy_htod(&[scalar]);

    let mut call = CudaCall::new("multilinear", "fold_ext_by_ext")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_block);
    call.arg(&slices_ptrs_dev);
    call.arg(&mut res_ptrs_dev);
    call.arg(&scalar_dev);
    call.arg(&n_slices);
    call.arg(&n_vars);
    call.launch();

    res
}
