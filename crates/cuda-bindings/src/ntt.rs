use cudarc::driver::{LaunchConfig, PushKernelArg};

use p3_field::TwoAdicField;
use tracing::instrument;

use crate::{MAX_LOG_N_BLOCKS, cuda_info};

// TODO this value is also hardcoded in ntt.cuda, this is ugly
const NTT_LOG_N_THREADS_PER_BLOCK: u32 = 8;
const NTT_N_THREADS_PER_BLOCK: u32 = 1 << NTT_LOG_N_THREADS_PER_BLOCK;

#[instrument(name = "CUDA NTT", skip_all)]
pub fn cuda_ntt<F: TwoAdicField>(coeffs: &[F], expansion_factor: usize) -> Vec<F> {
    // SAFETY: one should have called init_cuda::<F::PrimeSubfield>() before

    let cuda = cuda_info();

    assert!(coeffs.len().is_power_of_two());
    assert!(expansion_factor.is_power_of_two());

    let expanded_len = coeffs.len() * expansion_factor;
    let log_len = coeffs.len().trailing_zeros() as u32;
    let log_expension_factor = expansion_factor.trailing_zeros() as u32;

    // Because of `ntt_at_block_level` in ntt.cu, where each blocks handles 1 << (NTT_LOG_N_THREADS_PER_BLOCK + 1) elements
    assert!(
        log_len >= NTT_LOG_N_THREADS_PER_BLOCK + 1,
        "Cuda small NTT not suported (for now), use CPU instead"
    );

    let log_n_blocks =
        (log_len + log_expension_factor - NTT_LOG_N_THREADS_PER_BLOCK - 1).min(MAX_LOG_N_BLOCKS);
    let n_blocks = 1 << log_n_blocks;

    assert!(
        log_expension_factor + log_len <= cuda.two_adicity as u32,
        "NTT to big, TODO use the two addic unit roots from the extensioon field"
    );

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let mut coeffs_dev = unsafe { cuda.stream.alloc(coeffs.len()).unwrap() };
    cuda.stream.memcpy_htod(coeffs, &mut coeffs_dev).unwrap();
    let mut buff_dev = unsafe { cuda.stream.alloc::<F>(expanded_len).unwrap() };
    let mut result_dev = unsafe { cuda.stream.alloc::<F>(expanded_len).unwrap() };

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (NTT_N_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: (NTT_N_THREADS_PER_BLOCK * 2) * (extension_degree as u32 + 1) * 4, // cf `ntt_at_block_level` in ntt.cu
    };

    let f = cuda.get_function("ntt", "ntt");
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&coeffs_dev);
    launch_args.arg(&mut buff_dev);
    launch_args.arg(&mut result_dev);
    launch_args.arg(&log_len);
    launch_args.arg(&log_expension_factor);
    let twiddles = cuda.twiddles::<F::PrimeSubfield>();
    launch_args.arg(&twiddles);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    let mut cuda_result = vec![F::ZERO; expanded_len];
    cuda.stream
        .memcpy_dtoh(&result_dev, &mut cuda_result)
        .unwrap();
    cuda.stream.synchronize().unwrap();

    cuda_result
}
