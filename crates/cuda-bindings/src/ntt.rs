use cuda_engine::{
    CudaCall, CudaFunctionInfo, cuda_alloc, cuda_alloc_zeros, cuda_twiddles,
    cuda_twiddles_two_adicity, max_ntt_log_size_at_block_level,
};
use cudarc::driver::{CudaSlice, PushKernelArg};

use p3_field::Field;

pub fn cuda_transpose<F: Field>(
    input: &CudaSlice<F>,
    log_rows: u32,
    log_cols: u32,
) -> CudaSlice<F> {
    assert!(input.len().is_power_of_two());
    assert_eq!(log_rows + log_cols, input.len().trailing_zeros());
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

pub fn cuda_reverse_bit_order_for_ntt<F: Field>(
    input: &CudaSlice<F>,
    expansion_factor: usize,
    log_chunck_size: usize,
) -> CudaSlice<F> {
    assert!(input.len().is_power_of_two());
    assert!(expansion_factor.is_power_of_two());
    let log_len_u32 = input.len().trailing_zeros();
    let log_chunck_size_u32 = log_chunck_size as u32;
    let log_expansion_factor_u32 = expansion_factor.ilog2() as u32;
    assert!(log_chunck_size_u32 <= log_len_u32 + log_expansion_factor_u32);
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("ntt/bit_reverse.cu", "reverse_bit_order_for_ntt"),
        input.len() as u32,
    );
    let mut result_dev = cuda_alloc_zeros::<F>(input.len() * expansion_factor);
    call.arg(input);
    call.arg(&mut result_dev);
    call.arg(&log_len_u32);
    call.arg(&log_expansion_factor_u32);
    call.arg(&log_chunck_size_u32);
    call.launch();
    result_dev
}

pub fn cuda_ntt_step<F: Field>(
    coeffs: &mut CudaSlice<F>,
    twiddles: &CudaSlice<F::PrimeSubfield>,
    log_len_u32: u32,
    step: u32,
) {
    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F::PrimeSubfield, F>("ntt/ntt.cu", "ntt_step"),
        1 << (log_len_u32 - 1),
    );
    call.arg(coeffs);
    call.arg(&log_len_u32);
    call.arg(&step);
    call.arg(twiddles);
    call.launch();
}

pub fn cuda_ntt_at_block_level<F: Field>(
    coeffs: &mut CudaSlice<F>,
    twiddles: &CudaSlice<F::PrimeSubfield>,
    log_chunck_size: u32,
    log_len_u32: u32,
) {
    let mut call = CudaCall::new(
        CudaFunctionInfo::ntt_at_block_level::<F>(),
        1 << (log_len_u32 - 1),
    )
    .max_log_threads_per_block(max_ntt_log_size_at_block_level::<F>() as u32 - 1)
    .shared_mem_bytes(
        ((1 << max_ntt_log_size_at_block_level::<F>())
            * (std::mem::size_of::<F>() + std::mem::size_of::<F::PrimeSubfield>())) as u32,
    );
    call.arg(coeffs);
    call.arg(&log_len_u32);
    call.arg(&log_chunck_size);
    call.arg(twiddles);
    call.launch();
}

// Async
pub fn cuda_ntt<F: Field>(coeffs: &mut CudaSlice<F>, log_chunck_size: usize) {
    assert!(coeffs.len().is_power_of_two());

    let log_len_u32 = coeffs.len().trailing_zeros();
    assert!(
        log_chunck_size <= cuda_twiddles_two_adicity::<F::PrimeSubfield>(),
        "NTT to big"
    );

    let twiddles = cuda_twiddles::<F::PrimeSubfield>();

    let block_log_chunck_size = log_chunck_size.min(max_ntt_log_size_at_block_level::<F>()) as u32;
    cuda_ntt_at_block_level(coeffs, &twiddles, block_log_chunck_size, log_len_u32);
    for step in block_log_chunck_size..log_chunck_size as u32 {
        cuda_ntt_step(coeffs, &twiddles, log_len_u32, step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_engine::{
        cuda_init, cuda_load_function, cuda_preprocess_twiddles, cuda_sync, memcpy_dtoh,
        memcpy_htod,
    };
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::{TwoAdicField, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::{Matrix, dense::RowMajorMatrix};
    use utils::{extension_degree, switch_endianness_vec};

    #[test]
    fn test_cuda_ntt() {
        for log_width in [0, 1, 2, 5, 12] {
            for log_len in [1, 3, 10, 14, 17] {
                if log_width > log_len {
                    continue;
                }
                test_cuda_ntt_helper::<KoalaBear>(log_len, log_width);
                test_cuda_ntt_helper::<BinomialExtensionField<KoalaBear, 4>>(log_len, log_width);
                test_cuda_ntt_helper::<BinomialExtensionField<KoalaBear, 8>>(log_len, log_width);
            }
        }
    }

    fn test_cuda_ntt_helper<F: TwoAdicField + Ord>(log_len: usize, log_width: usize) {
        cuda_init();
        cuda_load_function(CudaFunctionInfo::two_fields::<KoalaBear, F>(
            "ntt/ntt.cu",
            "ntt_step",
        ));
        cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<F>());
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "ntt/bit_reverse.cu",
            "reverse_bit_order_for_ntt",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "ntt/transpose.cu",
            "transpose",
        ));
        cuda_preprocess_twiddles::<KoalaBear>();

        let len = 1 << log_len;
        let coeffs = (0..len).map(|i| F::from_usize(i)).collect::<Vec<_>>();
        let coeffs_dev = memcpy_htod(&coeffs);
        cuda_sync();

        let time = std::time::Instant::now();
        let mut coeffs_dev = cuda_reverse_bit_order_for_ntt(&coeffs_dev, 1, log_len - log_width);
        cuda_ntt(&mut coeffs_dev, log_len - log_width);
        let res_dev = cuda_transpose(&coeffs_dev, log_width as u32, (log_len - log_width) as u32);
        cuda_sync();
        println!("CUDA ntt took: {} ms", time.elapsed().as_millis());
        let cuda_res = memcpy_dtoh(&res_dev);
        cuda_sync();

        let time = std::time::Instant::now();
        let cpu_res = Radix2DitParallel::<F>::default()
            .dft_batch(RowMajorMatrix::new(
                switch_endianness_vec(&coeffs),
                1 << log_width,
            ))
            // Get natural order of rows.
            .to_row_major_matrix()
            .values;
        println!("CPU ntt took: {} ms", time.elapsed().as_millis());

        assert!(
            cuda_res == cpu_res,
            "log_len = {log_len}, log_width = {log_width}, extension = {}",
            extension_degree::<F>()
        );
    }
}
