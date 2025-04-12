use std::{any::TypeId, borrow::Borrow};

use cudarc::driver::{CudaSlice, PushKernelArg};

use p3_field::{ExtensionField, Field};

use cuda_engine::{
    CudaCall, concat_pointers, cuda_alloc, cuda_get_at_index, cuda_sync, memcpy_htod,
};

use crate::{MAX_LOG_N_COOPERATIVE_BLOCKS, MAX_N_BLOCKS, MAX_N_COOPERATIVE_BLOCKS};

const MULTILINEAR_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const MULTILINEAR_N_THREADS_PER_BLOCK: u32 = 1 << MULTILINEAR_LOG_N_THREADS_PER_BLOCK;

/// (+ reverse vars)
/// Async
pub fn cuda_monomial_to_lagrange_basis_rev<F: Field>(coeffs: &CudaSlice<F>) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;
    let log_n_blocks = (n_vars.saturating_sub(MULTILINEAR_LOG_N_THREADS_PER_BLOCK + 1))
        .min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let mut buff = cuda_alloc::<F>(coeffs.len());
    let mut result = cuda_alloc::<F>(coeffs.len());
    let mut call = CudaCall::new("multilinear", "monomial_to_lagrange_basis")
        .blocks(1 << log_n_blocks)
        .threads_per_block(MULTILINEAR_N_THREADS_PER_BLOCK);
    call.arg(coeffs);
    call.arg(&mut buff);
    call.arg(&mut result);
    call.arg(&n_vars);
    call.launch_cooperative();

    result
}

// Async
pub fn cuda_lagrange_to_monomial_basis<F: Field>(evals: &CudaSlice<F>) -> CudaSlice<F> {
    assert!(evals.len().is_power_of_two());
    let n_vars = evals.len().ilog2() as u32;
    let n_threads_per_blocks = MULTILINEAR_N_THREADS_PER_BLOCK.min(evals.len() as u32 / 2);
    let n_blocks = ((evals.len() as u32 / 2 + n_threads_per_blocks - 1) / n_threads_per_blocks)
        .min(MAX_N_COOPERATIVE_BLOCKS);
    let mut result = cuda_alloc::<F>(evals.len());
    let mut call = CudaCall::new("multilinear", "lagrange_to_monomial_basis")
        .blocks(n_blocks)
        .threads_per_block(MULTILINEAR_N_THREADS_PER_BLOCK);
    call.arg(evals);
    call.arg(&mut result);
    call.arg(&n_vars);
    call.launch_cooperative();

    result
}

// Async
pub fn cuda_eval_multilinear_in_monomial_basis<F: Field, EF: ExtensionField<F>>(
    coeffs: &CudaSlice<F>,
    point: &[EF],
) -> EF {
    cuda_eval_multilinear(coeffs, point, "eval_multilinear_in_monomial_basis")
}

// Async
pub fn cuda_eval_multilinear_in_lagrange_basis<F: Field, EF: ExtensionField<F>>(
    evals: &CudaSlice<F>,
    point: &[EF],
) -> EF {
    cuda_eval_multilinear(evals, point, "eval_multilinear_in_lagrange_basis")
}

// Async
fn cuda_eval_multilinear<F: Field, EF: ExtensionField<F>>(
    coeffs: &CudaSlice<F>,
    point: &[EF],
    function_name: &str,
) -> EF {
    if TypeId::of::<F>() != TypeId::of::<EF>() || F::bits() <= 32 {
        unimplemented!()
    }

    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;
    assert_eq!(n_vars, point.len() as u32);

    if n_vars == 0 {
        return EF::from(cuda_get_at_index(coeffs, 0));
    }

    let point_dev = memcpy_htod(&point);
    let mut buff = cuda_alloc::<EF>(coeffs.len() - 1);
    let log_n_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n_vars - 1);
    let log_n_blocks =
        ((n_vars - 1).saturating_sub(log_n_per_blocks)).min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let mut call = CudaCall::new("multilinear", function_name)
        .blocks(1 << log_n_blocks)
        .threads_per_block(1 << log_n_per_blocks);
    call.arg(coeffs);
    call.arg(&point_dev);
    call.arg(&n_vars);
    call.arg(&mut buff);
    call.launch_cooperative();

    cuda_get_at_index(&buff, coeffs.len() - 2)
}

// Async
pub fn cuda_eq_mle<F: Field>(point: &[F]) -> CudaSlice<F> {
    if point.len() == 0 {
        let res = memcpy_htod(&[F::ONE]);
        cuda_sync();
        return res;
    }
    let n_vars = point.len() as u32;
    let point_dev = memcpy_htod(&point);
    let mut res = cuda_alloc::<F>(1 << n_vars);
    let log_n_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n_vars - 1);
    let log_n_blocks =
        ((n_vars - 1).saturating_sub(log_n_per_blocks)).min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let mut call = CudaCall::new("multilinear", "eq_mle")
        .blocks(1 << log_n_blocks)
        .threads_per_block(1 << log_n_per_blocks);
    call.arg(&point_dev);
    call.arg(&n_vars);
    call.arg(&mut res);
    call.launch_cooperative();
    res
}

// Async
pub fn cuda_scale_slice_in_place<F: Field>(slice: &mut CudaSlice<F>, scalar: F) {
    assert!(F::bits() > 32, "TODO");
    let scalar = [scalar];
    let scalar_dev = memcpy_htod(&scalar);
    let n = slice.len() as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);
    let mut call = CudaCall::new("multilinear", "scale_ext_slice_in_place")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(slice);
    call.arg(&n);
    call.arg(&scalar_dev);
    call.launch();
}

// Async
pub fn cuda_scale_slice<F: Field, EF: ExtensionField<F>>(
    slice: &CudaSlice<F>,
    scalar: EF,
) -> CudaSlice<EF> {
    assert!(TypeId::of::<F>() != TypeId::of::<EF>(), "TODO");
    let scalar = [scalar];
    let scalar_dev = memcpy_htod(&scalar);
    let n = slice.len() as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);

    let mut res = cuda_alloc::<EF>(slice.len());
    let mut call = CudaCall::new("multilinear", "scale_prime_slice_by_ext")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(slice);
    call.arg(&n);
    call.arg(&scalar_dev);
    call.arg(&mut res);
    call.launch();
    res
}

// Async
pub fn cuda_add_slices<F: Field>(a: &CudaSlice<F>, b: &CudaSlice<F>) -> CudaSlice<F> {
    let n = a.len() as u32;
    assert_eq!(n, b.len() as u32);
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);
    let mut res = cuda_alloc::<F>(n as usize);
    let mut call = CudaCall::new("multilinear", "add_slices")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(a);
    call.arg(b);
    call.arg(&mut res);
    call.arg(&n);
    call.launch();
    res
}

// Async
pub fn cuda_add_assign_slices<F: Field>(a: &mut CudaSlice<F>, b: &CudaSlice<F>) {
    // a += b;
    let n = a.len() as u32;
    assert_eq!(n, b.len() as u32);
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);
    let mut call = CudaCall::new("multilinear", "add_assign_slices")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(a);
    call.arg(b);
    call.arg(&n);
    call.launch();
}

// Async
pub fn cuda_whir_fold<F: Field>(coeffs: &CudaSlice<F>, folding_randomness: &[F]) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;
    let folding_factor = folding_randomness.len() as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(coeffs.len() as u32);
    let n_blocks = (((coeffs.len() / 2) as u32 + n_threads_per_blocks - 1) / n_threads_per_blocks)
        .next_power_of_two()
        .min(MAX_N_COOPERATIVE_BLOCKS);
    let folding_randomness_dev = memcpy_htod(folding_randomness);
    let buff_size = (0..folding_factor - 1)
        .map(|i| 1 << (n_vars - i - 1))
        .sum::<usize>();
    let mut buff = cuda_alloc::<F>(buff_size);
    let mut res = cuda_alloc::<F>(coeffs.len() / (1 << folding_factor) as usize);
    let mut call = CudaCall::new("multilinear", "whir_fold")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(coeffs);
    call.arg(&n_vars);
    call.arg(&folding_factor);
    call.arg(&folding_randomness_dev);
    call.arg(&mut buff);
    call.arg(&mut res);
    call.launch_cooperative();
    res
}

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
    assert!(F::bits() <= 32);
    let columns: Vec<&CudaSlice<F>> = columns.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    let n_vars = columns[0].len().ilog2() as u32;
    assert!(columns.iter().all(|c| c.len() == 1 << n_vars as usize));
    let column_ptrs = concat_pointers(&columns);
    let res = (0..columns.len())
        .map(|_| cuda_alloc::<F>(1 << n_vars as usize))
        .collect::<Vec<_>>();
    let res_ptrs = concat_pointers(&res);
    let total = (columns.len() << n_vars) as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(total);
    let n_blocks = ((total + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);
    let func_name = if up {
        "multilinears_up"
    } else {
        "multilinears_down"
    };
    let n_columns = columns.len() as u32;
    let mut call = CudaCall::new("multilinear", func_name)
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(&column_ptrs);
    call.arg(&n_columns);
    call.arg(&n_vars);
    call.arg(&res_ptrs);
    call.launch();
    res
}

// Async
pub fn cuda_dot_product<F: Field>(a: &CudaSlice<F>, b: &CudaSlice<F>) -> F {
    assert!(F::bits() > 32, "TODO");
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_power_of_two());
    let log_len = a.len().ilog2() as u32;

    let n_threads_per_blocks = MULTILINEAR_N_THREADS_PER_BLOCK.min(a.len() as u32);
    let n_blocks = ((a.len() as u32 + n_threads_per_blocks - 1) / n_threads_per_blocks)
        .min(MAX_N_COOPERATIVE_BLOCKS);

    let buff = cuda_alloc::<F>(a.len());
    let mut call = CudaCall::new("multilinear", "dot_product")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(a);
    call.arg(b);
    call.arg(&buff);
    call.arg(&log_len);
    call.launch_cooperative();

    cuda_get_at_index(&buff, 0)
}

// Async
pub fn cuda_fold_sum<F: Field>(input: &CudaSlice<F>, sum_size: usize) -> CudaSlice<F> {
    assert!(F::bits() > 32, "TODO");
    assert!(
        sum_size <= 256,
        "CUDA implement is not optimized for large sum sizes"
    );
    assert!(input.len() % sum_size == 0);

    let output_len = input.len() / sum_size;

    let n_threads_per_blocks = MULTILINEAR_N_THREADS_PER_BLOCK.min(output_len as u32);
    let n_blocks =
        ((output_len as u32 + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);

    let mut output = cuda_alloc::<F>(output_len);
    let len_u32 = input.len() as u32;
    let sum_size_u32 = sum_size as u32;
    let mut call = CudaCall::new("multilinear", "fold_sum")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(input);
    call.arg(&mut output);
    call.arg(&len_u32);
    call.arg(&sum_size_u32);
    call.launch();

    output
}
