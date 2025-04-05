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
    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve

    let coeffs_u32 = unsafe {
        std::slice::from_raw_parts(
            coeffs.as_ptr() as *const u32,
            coeffs.len() * extension_degree,
        )
    };

    let mut coeffs_dev = unsafe { cuda.stream.alloc::<u32>(coeffs_u32.len()).unwrap() };
    cuda.stream
        .memcpy_htod(coeffs_u32, &mut coeffs_dev)
        .unwrap();
    cuda.dev.synchronize().unwrap();

    let mut buff_dev = unsafe {
        cuda.stream
            .alloc::<u32>(expanded_len * extension_degree)
            .unwrap()
    };

    let mut result_dev = unsafe {
        cuda.stream
            .alloc::<u32>(expanded_len * extension_degree)
            .unwrap()
    };

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
    launch_args.arg(&cuda.twiddles);
    unsafe { launch_args.launch_cooperative(cfg) }.unwrap();

    let mut cuda_result_u32 = vec![0u32; expanded_len * extension_degree];
    cuda.stream
        .memcpy_dtoh(&result_dev, &mut cuda_result_u32)
        .unwrap();
    cuda.stream.synchronize().unwrap();

    assert_eq!(cuda_result_u32.capacity(), cuda_result_u32.len());

    let ptr = cuda_result_u32.as_ptr() as *mut F;
    // Prevent the original vector from being dropped
    let _ = std::mem::ManuallyDrop::new(cuda_result_u32);
    unsafe { Vec::from_raw_parts(ptr, expanded_len, expanded_len) }
}
