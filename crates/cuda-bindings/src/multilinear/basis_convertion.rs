use cuda_engine::{
    CudaCall, CudaFunctionInfo, LOG_MAX_THREADS_PER_BLOCK, cuda_alloc, shared_memory,
};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::Field;
use utils::log2_down;

const MAX_STEPS_PER_KERNEL_LAGRANGE_TO_MONOMIAL: usize = 5;

// Async
pub fn cuda_lagrange_to_monomial_basis<F: Field>(mut input: &CudaSlice<F>) -> CudaSlice<F> {
    assert!(input.len().is_power_of_two());

    let mut output = cuda_alloc::<F>(input.len());
    let n_vars = input.len().ilog2() as usize;

    let steps_in_shared_memory = n_vars
        .min(LOG_MAX_THREADS_PER_BLOCK + 1)
        .min(log2_down(shared_memory() / size_of::<F>()));

    let mut remaining_initial_steps = n_vars - steps_in_shared_memory;
    let mut previous_steps = 0;
    while remaining_initial_steps > 0 {
        let steps_to_perform =
            remaining_initial_steps.min(MAX_STEPS_PER_KERNEL_LAGRANGE_TO_MONOMIAL);
        cuda_lagrange_to_monomial_basis_steps(input, &mut output, previous_steps, steps_to_perform);
        previous_steps += steps_to_perform;
        remaining_initial_steps -= steps_to_perform;
        input = unsafe { &*(&output as *const CudaSlice<F>) }
    }

    cuda_lagrange_to_monomial_basis_end(&input, &mut output, steps_in_shared_memory);

    output
}

// Async
fn cuda_lagrange_to_monomial_basis_end<F: Field>(
    input: &CudaSlice<F>,
    output: &mut CudaSlice<F>,
    missing_steps: usize,
) {
    assert!(missing_steps > 0);
    assert_eq!(input.len(), output.len());
    assert!(input.len().is_power_of_two());
    let n_vars = input.len().ilog2();
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "lagrange_to_monomial_basis_end"),
        1 << (n_vars - 1),
    )
    .max_log_threads_per_block(missing_steps - 1)
    .shared_mem_bytes((1 << missing_steps) * std::mem::size_of::<F>());
    let missing_steps_u32 = missing_steps as u32;
    call.arg(input);
    call.arg(output);
    call.arg(&n_vars);
    call.arg(&missing_steps_u32);
    call.launch();
}

// Async
fn cuda_lagrange_to_monomial_basis_steps<F: Field>(
    input: &CudaSlice<F>,
    output: &mut CudaSlice<F>,
    previous_steps: usize,
    steps_to_perform: usize,
) {
    assert_eq!(input.len(), output.len());
    assert!(input.len().is_power_of_two());
    assert!(steps_to_perform > 0);
    let n_vars = input.len().ilog2();
    assert!(previous_steps + steps_to_perform <= n_vars as usize);
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "lagrange_to_monomial_basis_steps"),
        1 << (n_vars - steps_to_perform as u32),
    );
    let previous_steps_u32 = previous_steps as u32;
    let steps_to_perform_u32 = steps_to_perform as u32;
    call.arg(input);
    call.arg(output);
    call.arg(&n_vars);
    call.arg(&previous_steps_u32);
    call.arg(&steps_to_perform_u32);
    call.launch();
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
            "lagrange_to_monomial_basis_end",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<EF>(
            "multilinear.cu",
            "lagrange_to_monomial_basis_steps",
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
