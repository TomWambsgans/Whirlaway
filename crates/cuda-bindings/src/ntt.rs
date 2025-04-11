use cuda_engine::{
    CudaCall, cuda_alloc, cuda_correction_twiddles, cuda_sync, cuda_twiddles,
    cuda_twiddles_two_adicity, memcpy_dtoh, memcpy_htod,
};
use cudarc::driver::{CudaSlice, PushKernelArg};

use p3_field::Field;
use tracing::instrument;

use crate::MAX_LOG_N_COOPERATIVE_BLOCKS;

// TODO this value is also hardcoded in ntt.cuda, this is ugly
const NTT_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const NTT_N_THREADS_PER_BLOCK: u32 = 1 << NTT_LOG_N_THREADS_PER_BLOCK;

pub fn cuda_expanded_ntt<F: Field>(coeffs: &CudaSlice<F>, expansion_factor: usize) -> CudaSlice<F> {
    // SAFETY: one should have called init_cuda::<F::PrimeSubfield>() before

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
        log_expension_factor + log_len <= cuda_twiddles_two_adicity::<F::PrimeSubfield>() as u32,
        "NTT to big"
    );

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let mut buff_dev = cuda_alloc::<F>(expanded_len);
    let mut result_dev = cuda_alloc::<F>(expanded_len);

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve

    let mut call = CudaCall::new("ntt", "expanded_ntt")
        .blocks(n_blocks)
        .threads_per_block(NTT_N_THREADS_PER_BLOCK)
        .shared_mem_bytes((NTT_N_THREADS_PER_BLOCK * 2) * (extension_degree as u32 + 1) * 4); // cf `ntt_at_block_level` in ntt.cu
    call.arg(coeffs);
    call.arg(&mut buff_dev);
    call.arg(&mut result_dev);
    call.arg(&log_len);
    call.arg(&log_expension_factor);
    let twiddles = cuda_twiddles::<F::PrimeSubfield>();
    call.arg(&twiddles);
    call.launch_cooperative();

    result_dev
}

#[instrument(name = "CUDA NTT", skip_all)]
pub fn cuda_ntt<F: Field>(coeffs: &[F], log_chunck_size: usize) -> Vec<F> {
    // SAFETY: one should have called init_cuda::<F::PrimeSubfield>() before

    assert!(coeffs.len().is_power_of_two());

    let log_len = coeffs.len().trailing_zeros() as u32;

    let log_n_blocks =
        (log_len - NTT_LOG_N_THREADS_PER_BLOCK - 1).min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let n_blocks = 1 << log_n_blocks;

    assert!(
        log_len <= cuda_twiddles_two_adicity::<F::PrimeSubfield>() as u32,
        "NTT to big"
    );

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let mut coeffs_dev = memcpy_htod(coeffs);

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve

    let twiddles = cuda_twiddles::<F::PrimeSubfield>();
    let mut call = CudaCall::new("ntt", "ntt_global")
        .blocks(n_blocks)
        .threads_per_block(NTT_N_THREADS_PER_BLOCK)
        .shared_mem_bytes((NTT_N_THREADS_PER_BLOCK * 2) * (extension_degree as u32 + 1) * 4); // cf `ntt_at_block_level` in ntt.cu
    call.arg(&mut coeffs_dev);
    call.arg(&log_len);
    call.arg(&log_chunck_size);
    call.arg(&twiddles);
    call.launch_cooperative();

    let cuda_result = memcpy_dtoh(&coeffs_dev);
    cuda_sync();

    cuda_result
}

pub fn cuda_restructure_evaluations<F: Field>(
    coeffs: &CudaSlice<F>,
    whir_folding_factor: usize,
) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    let correction_twiddles = cuda_correction_twiddles::<F::PrimeSubfield>(whir_folding_factor);

    let whir_folding_factor = whir_folding_factor as u32;

    let log_len = coeffs.len().trailing_zeros() as u32;

    let log_n_blocks =
        (log_len.saturating_sub(NTT_LOG_N_THREADS_PER_BLOCK + 1)).min(MAX_LOG_N_COOPERATIVE_BLOCKS);
    let n_blocks = 1 << log_n_blocks;

    assert!(
        log_len <= cuda_twiddles_two_adicity::<F::PrimeSubfield>() as u32,
        "NTT to big"
    );

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let mut result_dev = cuda_alloc::<F>(coeffs.len());

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve

    let twiddles = cuda_twiddles::<F::PrimeSubfield>();
    let mut launch_args = CudaCall::new("ntt", "restructure_evaluations")
        .blocks(n_blocks)
        .threads_per_block(NTT_N_THREADS_PER_BLOCK)
        .shared_mem_bytes((NTT_N_THREADS_PER_BLOCK * 2) * (extension_degree as u32 + 1) * 4); // cf `ntt_at_block_level` in ntt.cu;
    launch_args.arg(coeffs);
    launch_args.arg(&mut result_dev);
    launch_args.arg(&log_len);
    launch_args.arg(&whir_folding_factor);
    launch_args.arg(&twiddles);
    launch_args.arg(&correction_twiddles);
    launch_args.launch_cooperative();

    result_dev
}
