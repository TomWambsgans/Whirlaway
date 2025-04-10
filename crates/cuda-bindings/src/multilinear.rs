use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use p3_field::Field;

use crate::MAX_LOG_N_BLOCKS;
use cuda_engine::{cuda_alloc, cuda_get_at_index, cuda_info, memcpy_htod};

const MULTILINEAR_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const MULTILINEAR_N_THREADS_PER_BLOCK: u32 = 1 << MULTILINEAR_LOG_N_THREADS_PER_BLOCK;

/// (+ reverse vars)
/// Async
pub fn cuda_monomial_to_lagrange_basis_rev<F: Field>(coeffs: &CudaSlice<F>) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;

    let cuda = cuda_info();

    let log_n_blocks = (n_vars
        .checked_sub(MULTILINEAR_LOG_N_THREADS_PER_BLOCK)
        .unwrap()
        - 1)
    .min(MAX_LOG_N_BLOCKS);
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
        .min(1 << MAX_LOG_N_BLOCKS);

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
pub fn cuda_eval_multilinear_in_monomial_basis<F: Field>(coeffs: &CudaSlice<F>, point: &[F]) -> F {
    cuda_eval_multilinear(coeffs, point, "eval_multilinear_in_monomial_basis")
}

// Async
pub fn cuda_eval_multilinear_in_lagrange_basis<F: Field>(evals: &CudaSlice<F>, point: &[F]) -> F {
    cuda_eval_multilinear(evals, point, "eval_multilinear_in_lagrange_basis")
}
// Async
fn cuda_eval_multilinear<F: Field>(coeffs: &CudaSlice<F>, point: &[F], function_name: &str) -> F {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;
    assert_eq!(n_vars, point.len() as u32);

    let cuda = cuda_info();

    let point_dev = memcpy_htod(&point);
    let mut buff = cuda_alloc::<F>(coeffs.len() - 1);

    let log_n_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n_vars - 1);
    let log_n_blocks = ((n_vars - 1).saturating_sub(log_n_per_blocks)).min(MAX_LOG_N_BLOCKS);

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
    let n_vars = point.len() as u32;

    let cuda = cuda_info();

    let point_dev = memcpy_htod(&point);
    let mut res = cuda_alloc::<F>(1 << n_vars);

    let log_n_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n_vars - 1);
    let log_n_blocks = ((n_vars - 1).saturating_sub(log_n_per_blocks)).min(MAX_LOG_N_BLOCKS);

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
    let f = cuda.get_function("multilinear", "scale_slice_in_place");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(slice);
    launch_args.arg(&n);
    launch_args.arg(&scalar_dev);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();
}

// Async
pub fn cuda_add_slices<F: Field>(a: &CudaSlice<F>, b: &CudaSlice<F>) -> CudaSlice<F> {
    let cuda = cuda_info();
    let n = a.len() as u32;
    assert_eq!(n, b.len() as u32);
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = (n + n_threads_per_blocks - 1) / n_threads_per_blocks;
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
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();
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
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();
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
        .min(1 << MAX_LOG_N_BLOCKS);
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
