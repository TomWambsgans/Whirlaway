use std::any::TypeId;

use cuda_engine::{
    CudaCall, MAX_THREADS_PER_COOPERATIVE_BLOCK, cuda_alloc, cuda_correction_twiddles,
    cuda_twiddles, cuda_twiddles_two_adicity,
};
use cudarc::driver::{CudaSlice, PushKernelArg};

use p3_field::{Field, extension::BinomialExtensionField};
use p3_koala_bear::KoalaBear;

// TODO this value is also hardcoded in ntt.cuda, this is ugly

pub fn cuda_expanded_ntt<F: Field>(coeffs: &CudaSlice<F>, expansion_factor: usize) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    assert!(expansion_factor.is_power_of_two());
    assert_eq!(
        TypeId::of::<F>(),
        TypeId::of::<BinomialExtensionField<KoalaBear, 8>>(),
        "TODO other fields"
    );

    let log_len = coeffs.len().trailing_zeros() as u32;
    let log_expension_factor = expansion_factor.trailing_zeros() as u32;

    let mut expanded = cuda_expand_with_twiddles(coeffs, expansion_factor);
    cuda_ntt(&mut expanded, log_len as usize);
    cuda_transpose(&expanded, log_len, log_expension_factor)
}

fn cuda_expand_with_twiddles<F: Field>(
    coeffs: &CudaSlice<F>,
    expansion_factor: usize,
) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    assert!(expansion_factor.is_power_of_two());
    assert_eq!(
        TypeId::of::<F>(),
        TypeId::of::<BinomialExtensionField<KoalaBear, 8>>(),
        "TODO other fields"
    );

    let expanded_len = coeffs.len() * expansion_factor;
    let log_len = coeffs.len().trailing_zeros() as u32;
    let log_expension_factor = expansion_factor.trailing_zeros() as u32;

    assert!(
        log_expension_factor + log_len <= cuda_twiddles_two_adicity::<F::PrimeSubfield>() as u32,
        "NTT to big"
    );

    let mut result_dev = cuda_alloc::<F>(expanded_len);
    let twiddles = cuda_twiddles::<F::PrimeSubfield>();

    let mut call = CudaCall::new(
        "ntt",
        "expand_with_twiddles",
        1 << (log_len + log_expension_factor),
    );
    call.arg(coeffs);
    call.arg(&mut result_dev);
    call.arg(&log_len);
    call.arg(&log_expension_factor);
    call.arg(&twiddles);
    call.launch();

    result_dev
}

fn cuda_transpose<F: Field>(input: &CudaSlice<F>, log_rows: u32, log_cols: u32) -> CudaSlice<F> {
    assert!(input.len().is_power_of_two());
    assert_eq!(log_rows + log_cols, input.len().trailing_zeros() as u32);
    assert_eq!(
        TypeId::of::<F>(),
        TypeId::of::<BinomialExtensionField<KoalaBear, 8>>(),
        "TODO other fields"
    );
    let mut result_dev = cuda_alloc::<F>(input.len());

    let mut call = CudaCall::new("ntt", "transpose", 1 << (log_rows + log_cols));
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

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve

    let twiddles = cuda_twiddles::<F::PrimeSubfield>();
    let mut call = CudaCall::new("ntt", "ntt_global", 1 << (log_len - 1)).shared_mem_bytes(
        (MAX_THREADS_PER_COOPERATIVE_BLOCK * 2) * (extension_degree as u32 + 1) * 4,
    ); // cf `ntt_at_block_level` in ntt.cu
    call.arg(coeffs);
    call.arg(&log_len);
    call.arg(&log_chunck_size);
    call.arg(&twiddles);
    call.launch_cooperative();
}

pub fn cuda_restructure_evaluations<F: Field>(
    coeffs: &CudaSlice<F>,
    whir_folding_factor: usize,
) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    let correction_twiddles = cuda_correction_twiddles::<F::PrimeSubfield>(whir_folding_factor);

    let whir_folding_factor = whir_folding_factor as u32;

    let log_len = coeffs.len().trailing_zeros() as u32;

    assert!(
        log_len <= cuda_twiddles_two_adicity::<F::PrimeSubfield>() as u32,
        "NTT to big"
    );

    assert_eq!(std::mem::size_of::<F>() % std::mem::size_of::<u32>(), 0);

    let mut result_dev = cuda_alloc::<F>(coeffs.len());

    let extension_degree = std::mem::size_of::<F>() / std::mem::size_of::<u32>(); // TODO improve

    let twiddles = cuda_twiddles::<F::PrimeSubfield>();
    let mut launch_args = CudaCall::new("ntt", "restructure_evaluations", 1 << (log_len - 1))
        .shared_mem_bytes(
            (MAX_THREADS_PER_COOPERATIVE_BLOCK * 2) * (extension_degree as u32 + 1) * 4,
        ); // cf `ntt_at_block_level` in ntt.cu;
    launch_args.arg(coeffs);
    launch_args.arg(&mut result_dev);
    launch_args.arg(&log_len);
    launch_args.arg(&whir_folding_factor);
    launch_args.arg(&twiddles);
    launch_args.arg(&correction_twiddles);
    launch_args.launch_cooperative();

    result_dev
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::ntt::*;
    use cuda_engine::*;
    use p3_field::TwoAdicField;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    pub fn test_cuda_expanded_ntt() {
        cuda_init();
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;

        cuda_preprocess_twiddles::<F>();

        let rng = &mut StdRng::seed_from_u64(0);
        for log_len in [5, 7, 13] {
            for log_expension_factor in [2, 3] {
                let len = 1 << log_len;
                let expansion_factor = 1 << log_expension_factor;
                let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();

                println!(
                    "number of field elements: {}, expension factor: {}",
                    len, expansion_factor
                );

                let time = std::time::Instant::now();
                let coeffs_dev = memcpy_htod(&coeffs);
                cuda_sync();
                println!("CUDA memcpy_htod took {} ms", time.elapsed().as_millis());

                let time = std::time::Instant::now();
                let cuda_result = cuda_expanded_ntt(&coeffs_dev, expansion_factor);
                cuda_sync();
                println!("CUDA NTT took {} ms", time.elapsed().as_millis());

                let time = std::time::Instant::now();
                let cuda_result = memcpy_dtoh(&cuda_result);
                cuda_sync();
                println!("CUDA memcpy_dtoh took {} ms", time.elapsed().as_millis());

                let time = std::time::Instant::now();
                let expected_result = expand_from_coeff::<F, EF>(&coeffs, expansion_factor);
                println!("CPU NTT took {} ms", time.elapsed().as_millis());

                assert!(cuda_result == expected_result);
            }
        }
    }

    #[test]
    pub fn test_cuda_ntt() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;

        cuda_init();
        cuda_preprocess_twiddles::<F>();

        let rng = &mut StdRng::seed_from_u64(0);
        for log_len in [2, 3, 5, 7, 13] {
            for log_chunck_size in [2, 3, 4, 5, 9, 13] {
                if log_chunck_size > log_len {
                    continue;
                }
                let len = 1 << log_len;
                let mut coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();
                let mut coeffs_dev = memcpy_htod(&coeffs);
                cuda_ntt(&mut coeffs_dev, log_chunck_size);
                let cuda_result = memcpy_dtoh(&coeffs_dev);
                cuda_sync();
                ntt_batch::<F, EF>(&mut coeffs, 1 << log_chunck_size);
                assert!(cuda_result == coeffs);
            }
        }
    }

    #[test]
    pub fn test_cuda_restructure_evaluations() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;
        let whir_folding_factor = 4;

        cuda_init();
        cuda_preprocess_all_twiddles::<F>(whir_folding_factor);

        let rng = &mut StdRng::seed_from_u64(0);
        for log_len in [4, 5, 13] {
            let len = 1 << log_len;
            let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();
            let coeffs_dev = memcpy_htod(&coeffs);
            cuda_sync();

            let time = std::time::Instant::now();
            let cuda_result = cuda_restructure_evaluations(&coeffs_dev, whir_folding_factor);
            cuda_sync();
            println!(
                "CUDA restructuraction took {} ms",
                time.elapsed().as_millis()
            );
            let time = std::time::Instant::now();
            let cuda_result = memcpy_dtoh(&cuda_result);
            cuda_sync();
            println!("CUDA memcpy_dtoh took {} ms", time.elapsed().as_millis());

            let time = std::time::Instant::now();
            let domain_gen_inv = F::two_adic_generator(log_len).inverse();
            let expected_result =
                restructure_evaluations::<F, EF>(coeffs, domain_gen_inv, whir_folding_factor);
            println!(
                "CPU restructuraction took {} ms",
                time.elapsed().as_millis()
            );

            assert!(cuda_result == expected_result);
        }
    }
}
