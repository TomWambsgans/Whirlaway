use cuda_engine::{CudaCall, CudaFunctionInfo, concat_pointers, cuda_alloc};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::Field;
use std::borrow::Borrow;

// Async
pub fn cuda_air_columns_up<F: Field, S: Borrow<CudaSlice<F>>>(columns: &[S]) -> Vec<CudaSlice<F>> {
    cuda_air_columns_up_or_down(columns, true)
}

/// Async
pub fn cuda_air_columns_down<F: Field, S: Borrow<CudaSlice<F>>>(
    columns: &[S],
) -> Vec<CudaSlice<F>> {
    cuda_air_columns_up_or_down(columns, false)
}

// Async
fn cuda_air_columns_up_or_down<F: Field, S: Borrow<CudaSlice<F>>>(
    columns: &[S],
    up: bool,
) -> Vec<CudaSlice<F>> {
    let columns: Vec<&CudaSlice<F>> = columns.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    let n_vars = columns[0].len().ilog2() as u32;
    assert!(columns.iter().all(|c| c.len() == 1 << n_vars as usize));
    let column_ptrs = concat_pointers(&columns);
    let res = (0..columns.len())
        .map(|_| cuda_alloc::<F>(1 << n_vars as usize))
        .collect::<Vec<_>>();
    let res_ptrs = concat_pointers(&res);
    let func_name = if up {
        "multilinears_up"
    } else {
        "multilinears_down"
    };
    let n_columns = columns.len() as u32;
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", func_name),
        (columns.len() << n_vars) as u32,
    );
    call.arg(&column_ptrs);
    call.arg(&n_columns);
    call.arg(&n_vars);
    call.arg(&res_ptrs);
    call.launch();
    res
}
