use cuda_engine::{CudaCall, CudaFunctionInfo, concat_pointers, cuda_alloc};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::{ExtensionField, Field};
use std::borrow::Borrow;
use utils::expanded_point_for_multilinear_monomial_evaluation;

use crate::cuda_linear_combination_at_row_level;

// Async
pub fn cuda_whir_fold<F: Field, EF: ExtensionField<F>>(
    coeffs: &CudaSlice<F>,
    folding_randomness: &[EF],
) -> CudaSlice<EF> {
    assert!(coeffs.len().is_power_of_two());
    assert!(
        folding_randomness.len() <= 6,
        "current implem is not optimized big folding factor"
    );
    let expanded_randomness =
        expanded_point_for_multilinear_monomial_evaluation(folding_randomness);
    cuda_linear_combination_at_row_level(&coeffs.as_view(), &expanded_randomness)
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
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", func_name),
        (columns.len() << n_vars) as u32,
    );
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
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "linear_combination_at_row_level",
        ));
        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 22;
        let folding_factor = 4;
        let folding_randomness = (0..folding_factor)
            .map(|_| rng.random())
            .collect::<Vec<EF>>();
        let coeffs = (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<F>>();
        let coeffs_dev = memcpy_htod(&coeffs);
        let coeffs_dev = CoefficientListDevice::new(coeffs_dev);
        cuda_sync();

        let time = std::time::Instant::now();
        let cuda_result = cuda_whir_fold(&coeffs_dev.coeffs, &folding_randomness);
        cuda_sync();
        println!("CUDA whir_folding took {} ms", time.elapsed().as_millis());
        let cuda_result = CoefficientListHost::new(memcpy_dtoh(&cuda_result));
        cuda_sync();
        let time = std::time::Instant::now();
        let expected_result = CoefficientListHost::new(coeffs).whir_fold(&folding_randomness);
        println!("CPU took {} ms", time.elapsed().as_millis());

        assert!(cuda_result == expected_result);
    }
}
