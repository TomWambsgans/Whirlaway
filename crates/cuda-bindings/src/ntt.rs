use cuda_engine::{
    CudaCall, CudaFunctionInfo, MAX_THREADS_PER_COOPERATIVE_BLOCK, cuda_alloc, cuda_twiddles,
    cuda_twiddles_two_adicity,
};
use cudarc::driver::{CudaSlice, PushKernelArg};

use p3_field::Field;
use utils::extension_degree;

pub fn cuda_transpose<F: Field>(
    input: &CudaSlice<F>,
    log_rows: u32,
    log_cols: u32,
) -> CudaSlice<F> {
    assert!(input.len().is_power_of_two());
    assert_eq!(log_rows + log_cols, input.len().trailing_zeros() as u32);
    let mut result_dev = cuda_alloc::<F>(input.len());

    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("ntt/transpose.cu", "transpose"),
        1 << (log_rows + log_cols),
    );
    call.arg(input);
    call.arg(&mut result_dev);
    call.arg(&log_rows);
    call.arg(&log_cols);
    call.launch();

    result_dev
}

pub fn cuda_ntt<F: Field>(coeffs: &mut CudaSlice<F>, log_chunck_size: usize) {
    assert!(coeffs.len().is_power_of_two());

    let log_len = coeffs.len().trailing_zeros() as u32;

    assert!(
        log_len <= cuda_twiddles_two_adicity::<F::PrimeSubfield>() as u32,
        "NTT to big"
    );

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let twiddles = cuda_twiddles::<F::PrimeSubfield>();
    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F::PrimeSubfield, F>("ntt/ntt.cu", "ntt"),
        1 << (log_len - 1),
    )
    .shared_mem_bytes(
        (MAX_THREADS_PER_COOPERATIVE_BLOCK * 2) * (extension_degree::<F>() as u32 + 1) * 4,
    ); // cf `ntt_at_block_level` in ntt.cu
    call.arg(coeffs);
    call.arg(&log_len);
    call.arg(&log_chunck_size);
    call.arg(&twiddles);
    call.launch_cooperative();
}
