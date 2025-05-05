use cuda_engine::{CudaCall, CudaFunctionInfo, cuda_alloc};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::Field;

// Async
pub fn cuda_lagrange_to_monomial_basis<F: Field>(evals: &CudaSlice<F>) -> CudaSlice<F> {
    assert!(evals.len().is_power_of_two());
    let n_vars = evals.len().ilog2() as u32;
    let mut result = cuda_alloc::<F>(evals.len());
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "lagrange_to_monomial_basis"),
        1 << (n_vars - 1),
    );
    call.arg(evals);

    let mut buff;
    buff = cuda_alloc::<F>(evals.len());
    call.arg(&mut buff);

    call.arg(&mut result);
    call.arg(&n_vars);
    call.launch_cooperative();

    result
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
    pub fn test_lagrange_to_monomial_basis() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;

        cuda_init();
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "lagrange_to_monomial_basis",
        ));

        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 17;
        let coeffs = (0..1 << n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
        let coeffs_dev = memcpy_htod(&coeffs);
        cuda_sync();
        let time = std::time::Instant::now();
        let cuda_result_dev = cuda_lagrange_to_monomial_basis(&coeffs_dev);
        cuda_sync();
        println!(
            "CUDA lagrange_to_monomial_basis transform took {} ms",
            time.elapsed().as_millis()
        );
        let cuda_result = memcpy_dtoh(&cuda_result_dev);
        cuda_sync();
        let time = std::time::Instant::now();
        let expected_result = MultilinearHost::new(coeffs.clone())
            .to_monomial_basis()
            .coeffs;
        println!(
            "CPU lagrange_to_monomial_basis transform took {} ms",
            time.elapsed().as_millis()
        );
        assert!(cuda_result == expected_result);
    }
}
