use cuda_engine::{
    CudaCall, CudaFunctionInfo, cuda_alloc, cuda_get_at_index, cuda_sync, memcpy_htod,
};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::{ExtensionField, Field};

// Async
pub fn cuda_eval_multilinear_in_lagrange_basis<F: Field, EF: ExtensionField<F>>(
    coeffs: &CudaSlice<F>,
    point: &[EF],
) -> EF {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2();
    assert_eq!(n_vars, point.len() as u32);

    if n_vars == 0 {
        return EF::from(cuda_get_at_index(coeffs, 0));
    }

    let point_dev = memcpy_htod(point);
    let mut buff = cuda_alloc::<EF>(coeffs.len() - 1);

    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "eval_multilinear_in_lagrange_basis",
        ),
        1 << (n_vars - 1),
    );
    call.arg(coeffs);
    call.arg(&point_dev);
    call.arg(&n_vars);
    call.arg(&mut buff);
    call.launch_cooperative();

    cuda_get_at_index(&buff, buff.len() - 1)
}

// Async
pub fn cuda_eq_mle<F: Field>(point: &[F]) -> CudaSlice<F> {
    if point.is_empty() {
        let res = memcpy_htod(&[F::ONE]);
        cuda_sync();
        return res;
    }
    let n_vars = point.len() as u32;
    let point_dev = memcpy_htod(point);
    let mut res = cuda_alloc::<F>(1 << n_vars);
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "eq_mle"),
        1 << (n_vars - 1),
    );
    call.arg(&point_dev);
    call.arg(&n_vars);
    call.arg(&mut res);
    call.launch_cooperative();
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::pols::*;
    use cuda_engine::*;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{
        Rng, SeedableRng,
        distr::{Distribution, StandardUniform},
        rngs::StdRng,
    };

    #[test]
    pub fn test_cuda_eval_multilinear_in_lagrange_basis() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<KoalaBear, 8>;
        test_cuda_eval_multilinear_in_lagrange_basis_helper::<F, EF>();
        test_cuda_eval_multilinear_in_lagrange_basis_helper::<F, F>();
        test_cuda_eval_multilinear_in_lagrange_basis_helper::<EF, EF>();
    }

    fn test_cuda_eval_multilinear_in_lagrange_basis_helper<F: Field, EF: ExtensionField<F>>()
    where
        StandardUniform: Distribution<F>,
        StandardUniform: Distribution<EF>,
    {
        cuda_init();
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "eval_multilinear_in_lagrange_basis",
        ));
        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 20;
        let len = 1 << n_vars;
        let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<F>>();
        let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<EF>>();

        let coeffs_dev = memcpy_htod(&coeffs);
        cuda_sync();

        let time = std::time::Instant::now();
        let cuda_result = cuda_eval_multilinear_in_lagrange_basis(&coeffs_dev, &point);
        cuda_sync();
        println!(
            "CUDA eval_multilinear_in_lagrange_basis took {} ms",
            time.elapsed().as_millis()
        );
        let time = std::time::Instant::now();
        let expected_result = MultilinearHost::new(coeffs).evaluate(&point);
        println!("CPU took {} ms", time.elapsed().as_millis());

        assert_eq!(cuda_result, expected_result);
    }

    #[test]
    pub fn test_cuda_eq_mle() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;
        cuda_init();
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "eq_mle",
        ));
        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 3;
        let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<EF>>();

        let time = std::time::Instant::now();
        let cuda_result = cuda_eq_mle(&point);
        cuda_sync();
        println!("CUDA eq_mle took {} ms", time.elapsed().as_millis());
        let cuda_result = memcpy_dtoh(&cuda_result);
        cuda_sync();
        let time = std::time::Instant::now();
        let expected_result = MultilinearHost::eq_mle(&point).evals;
        println!("CPU took {} ms", time.elapsed().as_millis());

        assert_eq!(cuda_result, expected_result);
    }
}
