use cuda_engine::{
    CudaCall, CudaFunctionInfo, cuda_alloc, cuda_twiddles, max_ntt_log_size_at_block_level,
};
use cudarc::driver::{CudaSlice, PushKernelArg};

use p3_field::Field;

// Async
pub fn cuda_ntt_at_block_level<F: Field>(
    input: &CudaSlice<F>,
    output: &mut CudaSlice<F>,
    inner_log_len: usize,
    log_chunck_size: usize,
    on_rows: bool,
    final_twiddles: bool,
    final_transpositions: Vec<(usize, usize)>,
    log_whir_expansion_factor: Option<usize>,
    previous_internal_transposition: Option<(usize, usize)>,
    skip_last_internal_transposition: bool,
) {
    // TODO opimization: if final_transpositions.is_empty(), we could do the NTT in place

    assert_eq!(
        input.len() << log_whir_expansion_factor.unwrap_or(0),
        output.len()
    );
    assert!(input.len().is_power_of_two());
    if final_transpositions.len() > 3 {
        todo!("Add variables in the cuda kernel");
    }
    if let Some(log_whir_expansion_factor) = log_whir_expansion_factor {
        assert!(log_whir_expansion_factor != 0);
    }

    let log_len = output.len().trailing_zeros() as usize;
    let mut call = CudaCall::new(
        CudaFunctionInfo::ntt_at_block_level::<F>(),
        1 << (log_len - 1),
    )
    .max_log_threads_per_block(max_ntt_log_size_at_block_level::<F>() - 1);
    let log_len_u32 = log_len as u32;
    let inner_log_len_u32 = inner_log_len as u32;
    let log_chunck_size_u32 = log_chunck_size as u32;
    let n_final_transpositions_u32 = final_transpositions.len() as u32;
    let log_whir_expansion_factor_u32 = log_whir_expansion_factor.unwrap_or(0) as u32;
    let missed_previous_internal_transposition = previous_internal_transposition.is_some();
    let (previous_internal_transposition_log_rows, previous_internal_transposition_log_cols) =
        previous_internal_transposition.unwrap_or((0, 0));
    let (mut tr_row_0, mut tr_col_0, mut tr_row_1, mut tr_col_1, mut tr_row_2, mut tr_col_2) =
        Default::default();
    for ((row_u32, col_u32), (row, col)) in [
        (&mut tr_row_0, &mut tr_col_0),
        (&mut tr_row_1, &mut tr_col_1),
        (&mut tr_row_2, &mut tr_col_2),
    ]
    .into_iter()
    .zip(&final_transpositions)
    {
        *row_u32 = *row as u32;
        *col_u32 = *col as u32;
    }
    call.arg(input);
    call.arg(output);
    call.arg(&log_len_u32);
    call.arg(&inner_log_len_u32);
    call.arg(&log_chunck_size_u32);
    call.arg(&on_rows);
    call.arg(&final_twiddles);
    call.arg(cuda_twiddles::<F>(log_chunck_size));
    call.arg(&log_whir_expansion_factor_u32);
    call.arg(&n_final_transpositions_u32);
    call.arg(&tr_row_0);
    call.arg(&tr_col_0);
    call.arg(&tr_row_1);
    call.arg(&tr_col_1);
    call.arg(&tr_row_2);
    call.arg(&tr_col_2);
    call.arg(&missed_previous_internal_transposition);
    call.arg(&previous_internal_transposition_log_rows);
    call.arg(&previous_internal_transposition_log_cols);
    call.arg(&skip_last_internal_transposition);
    call.launch();
}

// Async
pub fn cuda_ntt<F: Field>(
    input: &CudaSlice<F>,
    log_chunck_size: usize,
    final_transpositions: Vec<(usize, usize)>,
    log_whir_expansion_factor: Option<usize>,
) -> CudaSlice<F> {
    assert!(log_chunck_size > 0);
    let expanded_len = input.len() << log_whir_expansion_factor.unwrap_or(0);
    let mut buffer = cuda_alloc::<F>(expanded_len);
    let mut output = cuda_alloc::<F>(expanded_len);
    cuda_ntt_helper(
        Some(input),
        &mut buffer,
        &mut output,
        log_chunck_size,
        final_transpositions,
        log_whir_expansion_factor,
        None,
    );
    output
}

// Async
pub fn cuda_ntt_helper<F: Field>(
    input: Option<&CudaSlice<F>>,
    buffer: &mut CudaSlice<F>,
    output: &mut CudaSlice<F>,
    log_chunck_size: usize,
    mut final_transpositions: Vec<(usize, usize)>,
    log_whir_expansion_factor: Option<usize>,
    previous_internal_transposition: Option<(usize, usize)>,
) {
    if let Some(input) = input {
        assert!(input.len().is_power_of_two());
        assert_eq!(
            input.len() << log_whir_expansion_factor.unwrap_or(0),
            output.len()
        );
    }

    assert_eq!(buffer.len(), output.len());
    assert!(buffer.len().is_power_of_two());

    let log_len = output.len().trailing_zeros() as usize;
    assert!(log_chunck_size <= log_len);

    if log_chunck_size == 0 {
        return;
    }

    let max_cuda_ntt_log_size = max_ntt_log_size_at_block_level::<F>();

    if log_chunck_size <= max_cuda_ntt_log_size {
        final_transpositions.reverse();
        cuda_ntt_at_block_level(
            input.unwrap_or(buffer),
            output,
            log_chunck_size,
            log_chunck_size,
            true,
            false,
            final_transpositions,
            log_whir_expansion_factor,
            previous_internal_transposition,
            false,
        );
    } else {
        cuda_ntt_at_block_level(
            input.unwrap_or(buffer),
            output,
            log_chunck_size,
            max_cuda_ntt_log_size,
            false,
            true,
            vec![],
            log_whir_expansion_factor,
            previous_internal_transposition,
            true,
        );
        final_transpositions.push((
            max_cuda_ntt_log_size,
            log_chunck_size - max_cuda_ntt_log_size,
        ));
        std::mem::swap(buffer, output);
        cuda_ntt_helper(
            None,
            buffer,
            output,
            log_chunck_size - max_cuda_ntt_log_size,
            final_transpositions,
            None,
            Some((
                log_chunck_size - max_cuda_ntt_log_size,
                max_cuda_ntt_log_size,
            )),
        );
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
    use p3_field::BasedVectorSpace;
    use p3_field::{TwoAdicField, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::{Matrix, dense::RowMajorMatrix};

    #[test]
    fn test_cuda_ntt() {
        for log_width in [1, 3, 15, 20] {
            for log_len in [3, 9, 14, 21] {
                if log_width >= log_len {
                    continue;
                }
                test_cuda_ntt_helper::<BinomialExtensionField<KoalaBear, 8>>(log_len, log_width);
            }
        }
    }

    fn test_cuda_ntt_helper<F: TwoAdicField + Ord>(log_len: usize, log_width: usize) {
        cuda_init();
        cuda_load_function(CudaFunctionInfo::ntt_at_block_level::<F>());
        cuda_preprocess_twiddles::<KoalaBear>(log_len - log_width);

        let len = 1 << log_len;
        let coeffs = (0..len).map(|i| F::from_usize(i)).collect::<Vec<_>>();
        let input = memcpy_htod(&&coeffs);
        // let mut coeffs_dev = memcpy_htod(&switch_endianness_vec(&coeffs));
        cuda_sync();

        let time = std::time::Instant::now();
        let cuda_res = cuda_ntt(&input, log_len - log_width, vec![], None);
        cuda_sync();
        println!("CUDA ntt took: {} ms", time.elapsed().as_millis());
        let cuda_res = memcpy_dtoh(&cuda_res);
        cuda_sync();

        let time = std::time::Instant::now();
        let cpu_res = Radix2DitParallel::<F>::default()
            .dft_batch(RowMajorMatrix::new(coeffs, 1 << (log_len - log_width)).transpose())
            // Get natural order of rows.
            .to_row_major_matrix()
            .transpose()
            .values;
        println!("CPU ntt took: {} ms", time.elapsed().as_millis());

        assert!(
            cuda_res == cpu_res,
            "log_len = {log_len}, log_width = {log_width}, extension = {}",
            F::DIMENSION
        );
    }
}
