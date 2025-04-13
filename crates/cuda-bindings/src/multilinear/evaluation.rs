use cuda_engine::{
    CudaCall, concat_pointers, cuda_alloc, cuda_get_at_index, cuda_sync, memcpy_htod,
};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::{ExtensionField, Field};
use std::{any::TypeId, borrow::Borrow};

// Async
pub fn cuda_eval_multilinear_in_monomial_basis<F: Field, EF: ExtensionField<F>>(
    coeffs: &CudaSlice<F>,
    point: &[EF],
) -> EF {
    cuda_eval_multilinear(coeffs, point, "eval_multilinear_in_monomial_basis")
}

// Async
pub fn cuda_eval_multilinear_in_lagrange_basis<F: Field, EF: ExtensionField<F>>(
    evals: &CudaSlice<F>,
    point: &[EF],
) -> EF {
    cuda_eval_multilinear(evals, point, "eval_multilinear_in_lagrange_basis")
}

// Async
fn cuda_eval_multilinear<F: Field, EF: ExtensionField<F>>(
    coeffs: &CudaSlice<F>,
    point: &[EF],
    function_name: &str,
) -> EF {
    if TypeId::of::<F>() != TypeId::of::<EF>() || F::bits() <= 32 {
        unimplemented!()
    }

    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2() as u32;
    assert_eq!(n_vars, point.len() as u32);

    if n_vars == 0 {
        return EF::from(cuda_get_at_index(coeffs, 0));
    }

    let point_dev = memcpy_htod(&point);
    let mut buff = cuda_alloc::<EF>(coeffs.len() - 1);

    let mut call = CudaCall::new("multilinear", function_name, 1 << (n_vars - 1));
    call.arg(coeffs);
    call.arg(&point_dev);
    call.arg(&n_vars);
    call.arg(&mut buff);
    call.launch_cooperative();

    cuda_get_at_index(&buff, coeffs.len() - 2)
}

/// Async
pub fn cuda_fix_variable_in_small_field<
    F: Field,
    EF: ExtensionField<F>,
    ML: Borrow<CudaSlice<EF>>,
>(
    slices: &[ML],
    scalar: F,
) -> Vec<CudaSlice<EF>> {
    assert!(TypeId::of::<EF>() != TypeId::of::<F>(), "TODO");
    let slices: Vec<&CudaSlice<EF>> = slices.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert!(slices[0].len().is_power_of_two());
    let n_vars = slices[0].len().ilog2() as u32;
    assert!(n_vars >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << n_vars as usize));

    let slices_ptrs_dev = concat_pointers(&slices);
    let res = (0..slices.len())
        .map(|_| cuda_alloc::<EF>(1 << (n_vars - 1)))
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);

    let n_slices = slices.len() as u32;
    let mut call = CudaCall::new(
        "multilinear",
        "fold_ext_by_prime",
        (slices.len() as u32) << (n_vars - 1),
    );
    call.arg(&slices_ptrs_dev);
    call.arg(&mut res_ptrs_dev);
    call.arg(&scalar);
    call.arg(&n_slices);
    call.arg(&n_vars);
    call.launch();

    res
}

/// Async
pub fn cuda_fix_variable_in_big_field<F: Field, EF: ExtensionField<F>, ML: Borrow<CudaSlice<F>>>(
    slices: &[ML],
    scalar: EF,
) -> Vec<CudaSlice<EF>> {
    assert!(F::bits() > 32, "TODO");
    let slices: Vec<&CudaSlice<F>> = slices.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    assert!(slices[0].len().is_power_of_two());
    let n_vars = slices[0].len().ilog2() as u32;
    assert!(n_vars >= 1);
    assert!(slices.iter().all(|s| s.len() == 1 << n_vars as usize));
    let slices_ptrs_dev = concat_pointers(&slices);

    let res = (0..slices.len())
        .map(|_| cuda_alloc::<EF>(1 << (n_vars - 1)))
        .collect::<Vec<_>>();
    let mut res_ptrs_dev = concat_pointers(&res);
    let n_slices = slices.len() as u32;
    let scalar_dev = memcpy_htod(&[scalar]);
    let mut call = CudaCall::new(
        "multilinear",
        "fold_ext_by_ext",
        (slices.len() as u32) << (n_vars - 1),
    );
    call.arg(&slices_ptrs_dev);
    call.arg(&mut res_ptrs_dev);
    call.arg(&scalar_dev);
    call.arg(&n_slices);
    call.arg(&n_vars);
    call.launch();

    res
}

// Async
pub fn cuda_eq_mle<F: Field>(point: &[F]) -> CudaSlice<F> {
    if point.len() == 0 {
        let res = memcpy_htod(&[F::ONE]);
        cuda_sync();
        return res;
    }
    let n_vars = point.len() as u32;
    let point_dev = memcpy_htod(&point);
    let mut res = cuda_alloc::<F>(1 << n_vars);
    let mut call = CudaCall::new("multilinear", "eq_mle", 1 << (n_vars - 1));
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
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    pub fn test_cuda_eval_multilinear_in_monomial_basis() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;
        cuda_init();

        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 20;
        let len = 1 << n_vars;
        let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
        let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();

        let coeffs_dev = memcpy_htod(&coeffs);
        cuda_sync();

        let time = std::time::Instant::now();
        let cuda_result = cuda_eval_multilinear_in_monomial_basis(&coeffs_dev, &point);
        cuda_sync();
        println!(
            "CUDA eval_multilinear_in_monomial_basiss took {} ms",
            time.elapsed().as_millis()
        );
        let time = std::time::Instant::now();
        let expected_result = CoefficientListHost::new(coeffs).evaluate(&point);
        println!("CPU took {} ms", time.elapsed().as_millis());

        assert_eq!(cuda_result, expected_result);
    }

    #[test]
    pub fn test_cuda_eval_multilinear_in_lagrange_basis() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;
        cuda_init();

        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 20;
        let len = 1 << n_vars;
        let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
        let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();

        let coeffs_dev = memcpy_htod(&coeffs);
        cuda_sync();

        let time = std::time::Instant::now();
        let cuda_result = cuda_eval_multilinear_in_lagrange_basis(&coeffs_dev, &point);
        cuda_sync();
        println!(
            "CUDA eval_multilinear_in_lagrange_basiss took {} ms",
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

    #[test]
    fn test_cuda_fix_variable_in_small_field() {
        const EXT_DEGREE: usize = 8;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;
        cuda_init();

        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 5;
        let n_slices = 3;
        let slices = (0..n_slices)
            .map(|_| MultilinearHost::<EF>::random(rng, n_vars))
            .collect::<Vec<_>>();
        let slices_dev = slices
            .iter()
            .map(|multilinear| MultilinearDevice::new(memcpy_htod(&multilinear.evals)))
            .collect::<Vec<_>>();
        let scalar: F = rng.random();

        let time = std::time::Instant::now();
        let cuda_result = cuda_fix_variable_in_small_field(&slices_dev, scalar);
        cuda_sync();
        println!(
            "CUDA fix_variable_in_small_field took {} ms",
            time.elapsed().as_millis()
        );
        let time = std::time::Instant::now();
        let expected_result = slices
            .iter()
            .map(|multilinear| multilinear.fix_variable_in_small_field(scalar))
            .collect::<Vec<_>>();
        println!("CPU took {} ms", time.elapsed().as_millis());
        let retrieved = cuda_result
            .iter()
            .map(|multilinear| MultilinearHost::new(memcpy_dtoh(multilinear)))
            .collect::<Vec<_>>();
        cuda_sync();
        assert_eq!(retrieved.len(), expected_result.len());
        assert!(retrieved == expected_result);
    }
}
