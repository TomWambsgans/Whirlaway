use std::{any::TypeId, borrow::Borrow};

use cudarc::driver::{CudaSlice, PushKernelArg};

use cuda_engine::{CudaCall, SumcheckComputation, concat_pointers, cuda_alloc, memcpy_htod};
use p3_field::{BasedVectorSpace, ExtensionField, Field};

use crate::{MAX_N_BLOCKS, cuda_dot_product, cuda_fold_sum, cuda_sum};

// TODO avoid hardcoding
const SUMCHECK_LOG_N_THREADS_PER_BLOCK: u32 = 10;
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
    eq_mle: Option<&CudaSlice<EF>>,
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
    assert_eq!(eq_mle.is_some(), comp.eq_mle_multiplier);

    let n_compute_units = comp.n_cuda_compute_units() as u32;
    let n_blocks =
        ((n_compute_units << n_vars).div_ceil(SUMCHECK_N_THREADS_PER_BLOCK)).min(MAX_N_BLOCKS);
    let ext_degree = <EF as BasedVectorSpace<F>>::DIMENSION as u32;

    let multilinears_ptrs_dev = concat_pointers(&multilinears);

    let batching_scalars_dev = memcpy_htod(batching_scalars);
    let mut sums_dev = cuda_alloc::<EF>((n_compute_units as usize) << n_vars);

    let module_name = format!("sumcheck_{:x}", comp.uuid());
    let mut call = CudaCall::new(&module_name, "sum_over_hypercube_ext")
        .blocks(n_blocks)
        .threads_per_block(SUMCHECK_N_THREADS_PER_BLOCK)
        .shared_mem_bytes(batching_scalars.len() as u32 * ext_degree * 4); // cf: __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];;
    call.arg(&multilinears_ptrs_dev);
    call.arg(&mut sums_dev);
    call.arg(&batching_scalars_dev);
    call.arg(&n_vars);
    call.arg(&n_compute_units);
    call.launch();

    let hypercube_evals = if n_compute_units == 1 {
        sums_dev
    } else {
        cuda_fold_sum(&sums_dev, n_compute_units as usize)
    };

    if comp.eq_mle_multiplier {
        let eq_mle = eq_mle.unwrap();
        cuda_dot_product(eq_mle, &hypercube_evals)
    } else {
        cuda_sum(hypercube_evals)
    }
}
