use cuda_engine::{CudaCall, concat_pointers, cuda_alloc, memcpy_htod};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::Field;
use std::borrow::Borrow;

// Async
pub fn cuda_whir_fold<F: Field>(coeffs: &CudaSlice<F>, folding_randomness: &[F]) -> CudaSlice<F> {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;
    let folding_factor = folding_randomness.len() as u32;
    let folding_randomness_dev = memcpy_htod(folding_randomness);
    let buff_size = (0..folding_factor - 1)
        .map(|i| 1 << (n_vars - i - 1))
        .sum::<usize>();
    let mut buff = cuda_alloc::<F>(buff_size);
    let mut res = cuda_alloc::<F>(coeffs.len() / (1 << folding_factor) as usize);
    let mut call = CudaCall::new("multilinear", "whir_fold", coeffs.len() as u32 / 2);
    call.arg(coeffs);
    call.arg(&n_vars);
    call.arg(&folding_factor);
    call.arg(&folding_randomness_dev);
    call.arg(&mut buff);
    call.arg(&mut res);
    call.launch_cooperative();
    res
}

// Async
pub fn cuda_air_columns_up<F: Field, S: Borrow<CudaSlice<F>>>(columns: &[S]) -> Vec<CudaSlice<F>> {
    cuda_air_columns_up_or_down(columns, true)
}

/// Async
pub fn cuda_air_columns_down<F: Field, S: Borrow<CudaSlice<F>>>(
    columns: &[S],
) -> Vec<CudaSlice<F>> {
    cuda_air_columns_up_or_down(columns, false)
}

// Async
fn cuda_air_columns_up_or_down<F: Field, S: Borrow<CudaSlice<F>>>(
    columns: &[S],
    up: bool,
) -> Vec<CudaSlice<F>> {
    assert!(F::bits() <= 32);
    let columns: Vec<&CudaSlice<F>> = columns.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    let n_vars = columns[0].len().ilog2() as u32;
    assert!(columns.iter().all(|c| c.len() == 1 << n_vars as usize));
    let column_ptrs = concat_pointers(&columns);
    let res = (0..columns.len())
        .map(|_| cuda_alloc::<F>(1 << n_vars as usize))
        .collect::<Vec<_>>();
    let res_ptrs = concat_pointers(&res);
    let func_name = if up {
        "multilinears_up"
    } else {
        "multilinears_down"
    };
    let n_columns = columns.len() as u32;
    let mut call = CudaCall::new("multilinear", func_name, (columns.len() << n_vars) as u32);
    call.arg(&column_ptrs);
    call.arg(&n_columns);
    call.arg(&n_vars);
    call.arg(&res_ptrs);
    call.launch();
    res
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
    pub fn test_cuda_whir_fold() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;
        cuda_init();

        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 23;
        let folding_factor = 4;
        let folding_randomness = (0..folding_factor)
            .map(|_| rng.random())
            .collect::<Vec<EF>>();
        let coeffs = (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
        let coeffs_dev = memcpy_htod(&coeffs);
        let coeffs_dev = CoefficientListDevice::new(coeffs_dev);
        cuda_sync();

        let time = std::time::Instant::now();
        let cuda_result = cuda_whir_fold(&coeffs_dev.coeffs, &folding_randomness);
        cuda_sync();
        println!("CUDA whir folding took {} ms", time.elapsed().as_millis());
        let cuda_result = CoefficientListHost::new(memcpy_dtoh(&cuda_result));
        cuda_sync();
        let time = std::time::Instant::now();
        let expected_result = CoefficientListHost::new(coeffs).fold(&folding_randomness);
        println!("CPU took {} ms", time.elapsed().as_millis());

        assert!(cuda_result == expected_result);
    }
}
