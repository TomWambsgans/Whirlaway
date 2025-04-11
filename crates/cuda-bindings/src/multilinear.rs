use std::{any::TypeId, borrow::Borrow};

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use p3_field::{ExtensionField, Field};

use cuda_engine::{
    concat_pointers, cuda_alloc, cuda_get_at_index, cuda_info, cuda_sync, memcpy_htod,
};

use crate::{MAX_LOG_N_BLOCKS, MAX_LOG_N_COOPERATIVE_BLOCKS};

const MULTILINEAR_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const MULTILINEAR_N_THREADS_PER_BLOCK: u32 = 1 << MULTILINEAR_LOG_N_THREADS_PER_BLOCK;

/// (+ reverse vars)
/// Async
pub fn cuda_monomial_to_lagrange_basis_rev<F: Field>(coeffs: &CudaSlice<F>) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;

    let cuda = cuda_info();

    let log_n_blocks = (n_vars.saturating_sub(MULTILINEAR_LOG_N_THREADS_PER_BLOCK + 1))
        .min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let n_blocks = 1 << log_n_blocks;

    let mut buff = cuda_alloc::<F>(coeffs.len());
    let mut result = cuda_alloc::<F>(coeffs.len());

    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (MULTILINEAR_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = cuda.get_function("multilinear", "monomial_to_lagrange_basis");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(coeffs);
    launch_args.arg(&mut buff);
    launch_args.arg(&mut result);
    launch_args.arg(&n_vars);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    result
}

// Async
pub fn cuda_lagrange_to_monomial_basis<F: Field>(evals: &CudaSlice<F>) -> CudaSlice<F> {
    assert!(evals.len().is_power_of_two());
    let n_vars = evals.len().ilog2() as u32;

    let cuda = cuda_info();

    let n_threads_per_blocks = MULTILINEAR_N_THREADS_PER_BLOCK.min(evals.len() as u32 / 2);
    let n_blocks = ((evals.len() as u32 / 2 + n_threads_per_blocks - 1) / n_threads_per_blocks)
        .min(1 << MAX_LOG_N_COOPERATIVE_BLOCKS);

    let mut result = cuda_alloc::<F>(evals.len());

    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = cuda.get_function("multilinear", "lagrange_to_monomial_basis");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(evals);
    launch_args.arg(&mut result);
    launch_args.arg(&n_vars);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

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

    let cuda = cuda_info();

    let point_dev = memcpy_htod(&point);
    let mut buff = cuda_alloc::<EF>(coeffs.len() - 1);

    let log_n_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n_vars - 1);
    let log_n_blocks =
        ((n_vars - 1).saturating_sub(log_n_per_blocks)).min(MAX_LOG_N_COOPERATIVE_BLOCKS);

    let cfg = LaunchConfig {
        grid_dim: (1 << log_n_blocks, 1, 1),
        block_dim: (1 << log_n_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = cuda.get_function("multilinear", function_name);
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(coeffs);
    launch_args.arg(&point_dev);
    launch_args.arg(&n_vars);
    launch_args.arg(&mut buff);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

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

    let cuda = cuda_info();

    let point_dev = memcpy_htod(&point);
    let mut res = cuda_alloc::<F>(1 << n_vars);

    let log_n_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n_vars - 1);
    let log_n_blocks =
        ((n_vars - 1).saturating_sub(log_n_per_blocks)).min(MAX_LOG_N_COOPERATIVE_BLOCKS);

    let cfg = LaunchConfig {
        grid_dim: (1 << log_n_blocks, 1, 1),
        block_dim: (1 << log_n_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = cuda.get_function("multilinear", "eq_mle");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&point_dev);
    launch_args.arg(&n_vars);
    launch_args.arg(&mut res);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    res
}

// Async
pub fn cuda_scale_slice_in_place<F: Field>(slice: &mut CudaSlice<F>, scalar: F) {
    assert!(F::bits() > 32, "TODO");
    let cuda = cuda_info();
    let scalar = [scalar];
    let scalar_dev = memcpy_htod(&scalar);
    let n = slice.len() as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks =
        ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(1 << MAX_LOG_N_BLOCKS);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };
    let f = cuda.get_function("multilinear", "scale_ext_slice_in_place");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(slice);
    launch_args.arg(&n);
    launch_args.arg(&scalar_dev);
    unsafe { launch_args.launch(cfg) }.unwrap();
}

// Async
pub fn cuda_scale_slice<F: Field, EF: ExtensionField<F>>(
    slice: &CudaSlice<F>,
    scalar: EF,
) -> CudaSlice<EF> {
    assert!(TypeId::of::<F>() != TypeId::of::<EF>(), "TODO");
    let cuda = cuda_info();
    let scalar = [scalar];
    let scalar_dev = memcpy_htod(&scalar);
    let n = slice.len() as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks =
        ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(1 << MAX_LOG_N_BLOCKS);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut res = cuda_alloc::<EF>(slice.len());
    let f = cuda.get_function("multilinear", "scale_prime_slice_by_ext");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(slice);
    launch_args.arg(&n);
    launch_args.arg(&scalar_dev);
    launch_args.arg(&mut res);
    unsafe { launch_args.launch(cfg) }.unwrap();
    res
}

// Async
pub fn cuda_add_slices<F: Field>(a: &CudaSlice<F>, b: &CudaSlice<F>) -> CudaSlice<F> {
    let cuda = cuda_info();
    let n = a.len() as u32;
    assert_eq!(n, b.len() as u32);
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks =
        ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(1 << MAX_LOG_N_BLOCKS);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut res = cuda_alloc::<F>(n as usize);
    let f = cuda.get_function("multilinear", "add_slices");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(a);
    launch_args.arg(b);
    launch_args.arg(&mut res);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }.unwrap();
    res
}

// Async
pub fn cuda_add_assign_slices<F: Field>(a: &mut CudaSlice<F>, b: &CudaSlice<F>) {
    // a += b;
    let cuda = cuda_info();
    let n = a.len() as u32;
    assert_eq!(n, b.len() as u32);
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks =
        ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(1 << MAX_LOG_N_BLOCKS);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };
    let f = cuda.get_function("multilinear", "add_assign_slices");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(a);
    launch_args.arg(b);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }.unwrap();
}

// Async
pub fn cuda_whir_fold<F: Field>(coeffs: &CudaSlice<F>, folding_randomness: &[F]) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;
    let folding_factor = folding_randomness.len() as u32;
    let cuda = cuda_info();
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(coeffs.len() as u32);
    let n_blocks = (((coeffs.len() / 2) as u32 + n_threads_per_blocks - 1) / n_threads_per_blocks)
        .next_power_of_two()
        .min(1 << MAX_LOG_N_COOPERATIVE_BLOCKS);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };
    let folding_randomness_dev = memcpy_htod(folding_randomness);
    let buff_size = (0..folding_factor - 1)
        .map(|i| 1 << (n_vars - i - 1))
        .sum::<usize>();
    let mut buff = cuda_alloc::<F>(buff_size);
    let mut res = cuda_alloc::<F>(coeffs.len() / (1 << folding_factor) as usize);
    let f = cuda.get_function("multilinear", "whir_fold");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(coeffs);
    launch_args.arg(&n_vars);
    launch_args.arg(&folding_factor);
    launch_args.arg(&folding_randomness_dev);
    launch_args.arg(&mut buff);
    launch_args.arg(&mut res);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();
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
    let cuda = cuda_info();
    let total = (columns.len() << n_vars) as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(total);
    let n_blocks =
        ((total + n_threads_per_blocks - 1) / n_threads_per_blocks).min(1 << MAX_LOG_N_BLOCKS);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads_per_blocks, 1, 1),
        shared_mem_bytes: 0,
    };
    let func_name = if up {
        "multilinears_up"
    } else {
        "multilinears_down"
    };
    let f = cuda.get_function("multilinear", func_name);
    let n_columns = columns.len() as u32;
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&column_ptrs);
    launch_args.arg(&n_columns);
    launch_args.arg(&n_vars);
    launch_args.arg(&res_ptrs);
    unsafe { launch_args.launch(cfg) }.unwrap();
    res
}
