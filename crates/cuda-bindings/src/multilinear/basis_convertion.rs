use crate::{
    MAX_LOG_N_COOPERATIVE_BLOCKS, MAX_N_COOPERATIVE_BLOCKS,
    multilinear::{MULTILINEAR_LOG_N_THREADS_PER_BLOCK, MULTILINEAR_N_THREADS_PER_BLOCK},
};
use cuda_engine::{CudaCall, cuda_alloc};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::Field;

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
#[cfg(test)]
mod tests {
    use super::*;
    use algebra::pols::*;
    use cuda_engine::*;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    pub fn test_cuda_monomial_to_lagrange_basis() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;

        cuda_init();

        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 20;
        let coeffs = (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
        let coeffs_dev = memcpy_htod(&coeffs);
        cuda_sync();
        let time = std::time::Instant::now();
        let cuda_result_dev = cuda_monomial_to_lagrange_basis_rev(&coeffs_dev);
        cuda_sync();
        println!(
            "CUDA lagrange_to_monomial_basis transform took {} ms",
            time.elapsed().as_millis()
        );
        let cuda_result = memcpy_dtoh(&cuda_result_dev);
        cuda_sync();
        let time = std::time::Instant::now();
        let expected_result = CoefficientListHost::new(coeffs)
            .reverse_vars()
            .to_lagrange_basis()
            .evals;
        println!(
            "CPU lagrange_to_monomial_basis transform took {} ms",
            time.elapsed().as_millis()
        );
        assert!(cuda_result == expected_result);
    }

    #[test]
    pub fn test_cuda_lagrange_to_monomial_basis() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;

        cuda_init();

        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 20;
        let evals = (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
        let evals_dev = memcpy_htod(&evals);
        cuda_sync();
        let time = std::time::Instant::now();
        let cuda_result_dev = cuda_lagrange_to_monomial_basis(&evals_dev);
        cuda_sync();
        println!(
            "CUDA lagrange_to_monomial_basis transform took {} ms",
            time.elapsed().as_millis()
        );
        let cuda_result = memcpy_dtoh(&cuda_result_dev);
        cuda_sync();
        let time = std::time::Instant::now();
        let expected_result = MultilinearHost::new(evals).to_monomial_basis().coeffs;
        println!(
            "CPU lagrange_to_monomial_basis transform took {} ms",
            time.elapsed().as_millis()
        );
        assert_eq!(cuda_result, expected_result);
    }
}
