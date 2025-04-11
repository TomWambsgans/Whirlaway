use cuda_engine::{cuda_info, memcpy_dtoh, memcpy_htod};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use p3_field::Field;
use tracing::instrument;

use crate::MAX_LOG_N_COOPERATIVE_BLOCKS;

// TODO this value is also hardcoded in ntt.cuda, this is ugly
const NTT_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const NTT_N_THREADS_PER_BLOCK: u32 = 1 << NTT_LOG_N_THREADS_PER_BLOCK;

pub fn cuda_expanded_ntt<F: Field>(coeffs: &CudaSlice<F>, expansion_factor: usize) -> CudaSlice<F> {
    // SAFETY: one should have called init_cuda::<F::PrimeSubfield>() before

    let cuda = cuda_info();

    assert!(coeffs.len().is_power_of_two());
    assert!(expansion_factor.is_power_of_two());

    let expanded_len = coeffs.len() * expansion_factor;
    let log_len = coeffs.len().trailing_zeros() as u32;
    let log_expension_factor = expansion_factor.trailing_zeros() as u32;

    let log_n_blocks = ((log_len + log_expension_factor)
        .saturating_sub(NTT_LOG_N_THREADS_PER_BLOCK + 1))
    .min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let n_blocks = 1 << log_n_blocks;

    assert!(
        log_expension_factor + log_len <= cuda.two_adicity as u32,
        "NTT to big"
    );

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let mut buff_dev = unsafe { cuda.stream.alloc::<F>(expanded_len).unwrap() };
    let mut result_dev = unsafe { cuda.stream.alloc::<F>(expanded_len).unwrap() };

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (NTT_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: (NTT_N_THREADS_PER_BLOCK * 2) * (extension_degree as u32 + 1) * 4, // cf `ntt_at_block_level` in ntt.cu
    };

    let f = cuda.get_function("ntt", "expanded_ntt");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(coeffs);
    launch_args.arg(&mut buff_dev);
    launch_args.arg(&mut result_dev);
    launch_args.arg(&log_len);
    launch_args.arg(&log_expension_factor);
    let twiddles = cuda.twiddles::<F::PrimeSubfield>();
    launch_args.arg(&twiddles);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    result_dev
}

#[instrument(name = "CUDA NTT", skip_all)]
pub fn cuda_ntt<F: Field>(coeffs: &[F], log_chunck_size: usize) -> Vec<F> {
    // SAFETY: one should have called init_cuda::<F::PrimeSubfield>() before

    let cuda = cuda_info();
    assert!(coeffs.len().is_power_of_two());

    let log_len = coeffs.len().trailing_zeros() as u32;

    let log_n_blocks =
        (log_len - NTT_LOG_N_THREADS_PER_BLOCK - 1).min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let n_blocks = 1 << log_n_blocks;

    assert!(log_len <= cuda.two_adicity as u32, "NTT to big");

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let mut coeffs_dev = memcpy_htod(coeffs);

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (NTT_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: (NTT_N_THREADS_PER_BLOCK * 2) * (extension_degree as u32 + 1) * 4, // cf `ntt_at_block_level` in ntt.cu
    };

    let twiddles = cuda.twiddles::<F::PrimeSubfield>();
    let f = cuda.get_function("ntt", "ntt_global");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&mut coeffs_dev);
    launch_args.arg(&log_len);
    launch_args.arg(&log_chunck_size);
    launch_args.arg(&twiddles);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    let cuda_result = memcpy_dtoh(&coeffs_dev);
    cuda.stream.synchronize().unwrap();

    cuda_result
}

pub fn cuda_restructure_evaluations<F: Field>(
    coeffs: &CudaSlice<F>,
    whir_folding_factor: usize,
) -> CudaSlice<F> {
    let cuda = cuda_info();

    assert!(coeffs.len().is_power_of_two());
    assert_eq!(whir_folding_factor, cuda.whir_folding_factor);
    let whir_folding_factor = whir_folding_factor as u32;

    let log_len = coeffs.len().trailing_zeros() as u32;

    let log_n_blocks =
        (log_len.saturating_sub(NTT_LOG_N_THREADS_PER_BLOCK + 1)).min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let n_blocks = 1 << log_n_blocks;

    assert!(log_len <= cuda.two_adicity as u32, "NTT to big");

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let mut result_dev = unsafe { cuda.stream.alloc::<F>(coeffs.len()).unwrap() };

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (NTT_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: (NTT_N_THREADS_PER_BLOCK * 2) * (extension_degree as u32 + 1) * 4, // cf `ntt_at_block_level` in ntt.cu
    };

    let twiddles = cuda.twiddles::<F::PrimeSubfield>();
    let correction_twiddles = cuda.correction_twiddles::<F::PrimeSubfield>();
    let f = cuda.get_function("ntt", "restructure_evaluations");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(coeffs);
    launch_args.arg(&mut result_dev);
    launch_args.arg(&log_len);
    launch_args.arg(&whir_folding_factor);
    launch_args.arg(&twiddles);
    launch_args.arg(&correction_twiddles);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    result_dev
}
