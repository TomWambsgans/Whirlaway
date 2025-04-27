use cuda_engine::{CudaCall, CudaFunctionInfo, concat_pointers, cuda_alloc, memcpy_htod};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::{ExtensionField, Field};
use std::borrow::Borrow;

/// Async
pub fn cuda_fold_rectangular_in_small_field<
    F: Field,
    EF: ExtensionField<F>,
    ML: Borrow<CudaSlice<EF>>,
>(
    slices: &[ML],
    scalars: &[F],
) -> Vec<CudaSlice<EF>> {
    cuda_fold_rectangular::<EF, F, EF, _>(slices, scalars)
}

/// Async
pub fn cuda_fold_rectangular_in_large_field<
    F: Field,
    EF: ExtensionField<F>,
    ML: Borrow<CudaSlice<F>>,
>(
    slices: &[ML],
    scalars: &[EF],
) -> Vec<CudaSlice<EF>> {
    cuda_fold_rectangular::<F, EF, EF, _>(slices, scalars)
}

/// Async
fn cuda_fold_rectangular<F1: Field, F2: Field, F3: Field, ML: Borrow<CudaSlice<F1>>>(
    slices: &[ML],
    scalars: &[F2],
) -> Vec<CudaSlice<F3>> {
    let slices: Vec<&CudaSlice<F1>> = slices.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert!(slices[0].len().is_power_of_two());
    let n_vars = slices[0].len().ilog2() as u32;
    assert!(n_vars >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << n_vars as usize));
    assert!(scalars.len().is_power_of_two());
    let log_n_scalars = scalars.len().ilog2() as u32;
    assert!(log_n_scalars <= n_vars);
    let scalars_dev = memcpy_htod(scalars);

    let slices_ptrs_dev = concat_pointers(&slices);
    let res = (0..slices.len())
        .map(|_| cuda_alloc::<F3>(1 << (n_vars - log_n_scalars)))
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);

    let n_slices = slices.len() as u32;
    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F1, F2>("multilinear.cu", "fold_rectangular"),
        (slices.len() as u32) << (n_vars - log_n_scalars),
    );
    call.arg(&slices_ptrs_dev);
    call.arg(&mut res_ptrs_dev);
    call.arg(&scalars_dev);
    call.arg(&n_slices);
    call.arg(&n_vars);
    call.arg(&log_n_scalars);
    call.launch();

    res
}
