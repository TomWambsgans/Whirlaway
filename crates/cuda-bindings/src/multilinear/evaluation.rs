use cuda_engine::{
    CudaCall, CudaFunctionInfo, LOG_MAX_THREADS_PER_BLOCK, cuda_alloc, cuda_get_at_index,
    cuda_sync, memcpy_htod, shared_memory,
};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::{ExtensionField, Field};
use utils::log2_down;

// Async
pub fn cuda_eval_multilinear_in_lagrange_basis<F1: Field, F2: Field, ResField: ExtensionField<F1>>(
    coeffs: &CudaSlice<F1>,
    point: &[F2],
) -> ResField {
    assert!(coeffs.len().is_power_of_two());
    let n_vars = coeffs.len().ilog2();
    assert_eq!(n_vars, point.len() as u32);

    if n_vars == 0 {
        return ResField::from(cuda_get_at_index(coeffs, 0));
    }

    let point_dev = memcpy_htod(point);
    let mut buff = cuda_alloc::<ResField>(coeffs.len() - 1);

    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F1, F2>(
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

fn eq_mle_steps_within_shared_memory<F: Field>(n_vars: usize) -> usize {
    assert!(shared_memory() > n_vars * size_of::<F>());
    let max_steps_within_shared_memory =
        log2_down((shared_memory() - n_vars * size_of::<F>()) / size_of::<F>())
            .min(LOG_MAX_THREADS_PER_BLOCK);
    max_steps_within_shared_memory.min(n_vars)
}

// Async
pub fn cuda_eq_mle_start<F: Field>(point: &CudaSlice<F>, res: &mut CudaSlice<F>, steps: usize) {
    assert!(steps <= point.len());
    assert!(!point.is_empty());
    let n_vars = point.len() as u32;

    let steps_within_shared_memory = eq_mle_steps_within_shared_memory::<F>(point.len());

    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "eq_mle_start"),
        1 << steps_within_shared_memory,
    )
    .shared_mem_bytes(((1 << steps_within_shared_memory) + point.len()) * size_of::<F>());

    let steps_u32 = steps as u32;
    let steps_within_shared_memory_u32 = steps_within_shared_memory as u32;

    call.arg(point);
    call.arg(&n_vars);
    call.arg(&steps_within_shared_memory_u32);
    call.arg(&steps_u32);
    call.arg(res);
    call.launch();
}

// Async
pub fn cuda_eq_mle_steps<F: Field>(
    point: &CudaSlice<F>,
    res: &mut CudaSlice<F>,
    start_step: usize,
    additional_steps: usize,
) {
    assert!(start_step + additional_steps <= point.len());
    assert!(!point.is_empty());
    let n_vars = point.len() as u32;

    assert!(shared_memory() > point.len() * size_of::<F>());

    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "eq_mle_steps"),
        1 << start_step,
    )
    .shared_mem_bytes(point.len() * size_of::<F>());

    let start_step_u32 = start_step as u32;
    let additional_steps_u32 = additional_steps as u32;

    call.arg(point);
    call.arg(&n_vars);
    call.arg(&start_step_u32);
    call.arg(&additional_steps_u32);
    call.arg(res);
    call.launch();
}

// each cuda thread will have to store 2^MAX_EQ_MLE_STEPS_PER_ITER field_elements (type F) in its registers
const MAX_EQ_MLE_STEPS_PER_ITER: usize = 5; // 2^5 x 32 = 1024 bytes = 256 registers of 32 bits

// Async
pub fn cuda_eq_mle<F: Field>(point: &[F]) -> CudaSlice<F> {
    if point.is_empty() {
        let res = memcpy_htod(&[F::ONE]);
        cuda_sync();
        return res;
    }
    let point = memcpy_htod(point);

    let eq_mle_steps_within_shared_memory = eq_mle_steps_within_shared_memory::<F>(point.len());

    let mut start_step =
        (eq_mle_steps_within_shared_memory + MAX_EQ_MLE_STEPS_PER_ITER).min(point.len());

    let mut res = cuda_alloc::<F>(1 << point.len());

    cuda_eq_mle_start(&point, &mut res, start_step);

    let mut missing_steps = point.len() - start_step;
    while missing_steps > 0 {
        let additional_steps = missing_steps.min(MAX_EQ_MLE_STEPS_PER_ITER);
        cuda_eq_mle_steps(&point, &mut res, start_step, additional_steps);
        start_step += additional_steps;
        missing_steps -= additional_steps;
    }

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
        let cuda_result: EF = cuda_eval_multilinear_in_lagrange_basis(&coeffs_dev, &point);
        cuda_sync();
        println!(
            "CUDA eval_multilinear_in_lagrange_basis took {} ms",
            time.elapsed().as_millis()
        );
        let time = std::time::Instant::now();
        let expected_result = MultilinearHost::new(coeffs).evaluate_in_large_field(&point);
        println!("CPU took {} ms", time.elapsed().as_millis());

        assert_eq!(cuda_result, expected_result);
    }

    #[test]
    pub fn test_cuda_eq_mle() {
        const EXT_DEGREE: usize = 4;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, EXT_DEGREE>;
        cuda_init();
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "eq_mle_start",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "eq_mle_steps",
        ));
        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 19;
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

        assert!(cuda_result == expected_result);
    }
}
