use cuda_engine::{
    CudaCall, CudaFunctionInfo, concat_pointers, cuda_alloc, cuda_get_at_index, memcpy_dtoh,
    memcpy_htod,
};
use cudarc::driver::{CudaSlice, CudaView, PushKernelArg};
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use std::borrow::Borrow;

use crate::cuda_eq_mle;

// Async
pub fn cuda_dot_product<F: Field, EF: ExtensionField<F>>(
    a: &CudaSlice<F>,
    b: &CudaSlice<EF>,
) -> EF {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_power_of_two());
    let log_len = a.len().ilog2() as u32;
    let buff = cuda_alloc::<EF>(a.len());
    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F, EF>("multilinear.cu", "dot_product"),
        a.len(),
    );
    call.arg(a);
    call.arg(b);
    call.arg(&buff);
    call.arg(&log_len);
    call.launch_cooperative();

    cuda_get_at_index(&buff, 0)
}

// Async
pub fn cuda_scale_slice_in_place<F: Field>(slice: &mut CudaSlice<F>, scalar: F) {
    let scalar = [scalar];
    let scalar_dev = memcpy_htod(&scalar);
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "scale_in_place"),
        slice.len(),
    );
    let n = slice.len() as u32;
    call.arg(slice);
    call.arg(&n);
    call.arg(&scalar_dev);
    call.launch();
}

// Async
pub fn cuda_add_slices<F: Field, S: Borrow<CudaSlice<F>>>(slices: &[S]) -> CudaSlice<F> {
    let n_slices = slices.len() as u32;
    let len = slices[0].borrow().len() as u32;
    assert!(
        slices
            .iter()
            .all(|slice| slice.borrow().len() == len as usize)
    );
    let mut res = cuda_alloc::<F>(len as usize);
    let slices_ptrs = concat_pointers(slices);
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "add_slices"),
        len as usize,
    );
    call.arg(&slices_ptrs);
    call.arg(&mut res);
    call.arg(&n_slices);
    call.arg(&len);
    call.launch();
    res
}

// Async
pub fn cuda_add_assign_slices<F: Field>(a: &mut CudaSlice<F>, b: &CudaSlice<F>) {
    // a += b;
    assert_eq!(a.len(), b.len());
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "add_assign_slices"),
        a.len(),
    );
    let n = a.len() as u32;
    call.arg(a);
    call.arg(b);
    call.arg(&n);
    call.launch();
}

// Async
pub fn cuda_piecewise_sum<F: Field>(input: &CudaSlice<F>, sum_size: usize) -> CudaSlice<F> {
    assert!(
        sum_size <= 128,
        "current CUDA implementation is not optimized for large sum sizes"
    );
    assert!(input.len() % sum_size == 0);
    let output_len = input.len() / sum_size;
    let mut output = cuda_alloc::<F>(output_len);
    let len_u32 = input.len() as u32;
    let sum_size_u32 = sum_size as u32;
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "piecewise_sum"),
        output_len,
    );
    call.arg(input);
    call.arg(&mut output);
    call.arg(&len_u32);
    call.arg(&sum_size_u32);
    call.launch();
    output
}

// Async
pub fn cuda_linear_combination_at_row_level<F: Field, EF: ExtensionField<F>>(
    input: &CudaView<F>,
    scalars: &[EF],
) -> CudaSlice<EF> {
    assert!(
        scalars.len() <= 64,
        "current CUDA implementation is not optimized for a large linear combination"
    );
    assert!(input.len() % scalars.len() == 0);
    let output_len = input.len() / scalars.len();
    let mut output = cuda_alloc::<EF>(output_len);
    let len_u32 = input.len() as u32;
    let n_scalars_u32 = scalars.len() as u32;
    let scalars_dev = memcpy_htod(scalars);
    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F, EF>("multilinear.cu", "linear_combination_at_row_level"),
        output_len,
    );
    call.arg(input);
    call.arg(&mut output);
    call.arg(&scalars_dev);
    call.arg(&len_u32);
    call.arg(&n_scalars_u32);
    call.launch();
    output
}

// Async
pub fn cuda_linear_combination<F: Field, EF: ExtensionField<F>, S: Borrow<CudaSlice<F>>>(
    inputs: &[S],
    scalars: &[EF],
) -> CudaSlice<EF> {
    assert!(
        scalars.len() <= 512,
        "current CUDA implementation is not optimized for a large linear combination"
    );
    assert_eq!(inputs.len(), scalars.len());
    let len = inputs[0].borrow().len();
    assert!(inputs.iter().all(|input| input.borrow().len() == len));
    let mut output = cuda_alloc::<EF>(len);
    let len_u32 = len as u32;
    let n_scalars_u32 = scalars.len() as u32;
    let scalars_dev = memcpy_htod(scalars);
    let inputs_ptr = concat_pointers(inputs);
    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F, EF>("multilinear.cu", "linear_combination"),
        len,
    );
    call.arg(&inputs_ptr);
    call.arg(&mut output);
    call.arg(&scalars_dev);
    call.arg(&len_u32);
    call.arg(&n_scalars_u32);
    call.launch();
    output
}

// Async
pub fn cuda_repeat_slice_from_outside<F: Field>(
    input: &CudaSlice<F>,
    n_repetitions: usize,
) -> CudaSlice<F> {
    assert!(
        n_repetitions <= 128,
        "current CUDA implementation is not optimized for a large repetitions"
    );
    cuda_repeat_slice(input, n_repetitions, true)
}

// Async
pub fn cuda_repeat_slice_from_inside<F: Field>(
    input: &CudaSlice<F>,
    n_repetitions: usize,
) -> CudaSlice<F> {
    cuda_repeat_slice(input, n_repetitions, false)
}

// Async
fn cuda_repeat_slice<F: Field>(
    input: &CudaSlice<F>,
    n_repetitions: usize,
    outside: bool,
) -> CudaSlice<F> {
    let mut output = cuda_alloc::<F>(input.len() * n_repetitions);
    let len_u32 = input.len() as u32;
    let n_repetitions_u32 = n_repetitions as u32;
    let func_name = if outside {
        "repeat_slice_from_outside"
    } else {
        "repeat_slice_from_inside"
    };
    let n_ops = if outside {
        len_u32
    } else {
        len_u32 * n_repetitions_u32
    };
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", func_name),
        n_ops as usize,
    );
    call.arg(input);
    call.arg(&mut output);
    call.arg(&len_u32);
    call.arg(&n_repetitions_u32);
    call.launch();
    output
}

// Async
pub fn cuda_sum<F: Field>(terms: CudaSlice<F>) -> F {
    // we take owneship of `terms` because it will be aletered
    assert!(terms.len().is_power_of_two());
    if terms.len() == 1 {
        return cuda_get_at_index(&terms, 0);
    }
    let log_len = terms.len().ilog2() as u32;
    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>("multilinear.cu", "sum_in_place"),
        terms.len() / 2,
    );
    call.arg(&terms);
    call.arg(&log_len);
    call.launch_cooperative();
    cuda_get_at_index(&terms, 0)
}

// Async
pub fn cuda_eval_mixed_tensor<F: Field, EF: ExtensionField<F>>(
    terms: &CudaSlice<EF>,
    point: &[EF],
) -> Vec<Vec<F>> {
    assert!(terms.len().is_power_of_two());
    let n_vars = terms.len().ilog2() as u32;
    assert_eq!(point.len(), n_vars as usize);
    let log_n_tasks_per_thread = n_vars.min(5); // TODO find the best value

    let eq_mle = cuda_eq_mle(point);

    let ext_degree = <EF as BasedVectorSpace<F>>::DIMENSION;

    let n_ops = terms.len() >> log_n_tasks_per_thread;

    let mut buffer = cuda_alloc::<F>(n_ops * ext_degree.pow(2));
    let mut result = cuda_alloc::<F>(ext_degree.pow(2));

    let mut call = CudaCall::new(
        CudaFunctionInfo::two_fields::<F, EF>("tensor_algebra.cu", "tensor_algebra_dot_product"),
        n_ops,
    );
    call.arg(&eq_mle);
    call.arg(terms);
    call.arg(&mut buffer);
    call.arg(&mut result);
    call.arg(&n_vars);
    call.arg(&log_n_tasks_per_thread);
    call.launch_cooperative();

    let retrieved_result = memcpy_dtoh(&result);
    assert_eq!(retrieved_result.len(), ext_degree.pow(2));
    let mut final_result = vec![vec![F::ZERO; ext_degree as usize]; ext_degree as usize];
    for i in 0..ext_degree {
        for j in 0..ext_degree {
            final_result[i as usize][j as usize] = retrieved_result[i * ext_degree + j];
        }
    }
    final_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::pols::MultilinearHost;
    use cuda_engine::*;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rayon::prelude::*;

    #[test]
    fn test_cuda_dot_product() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 8>;

        cuda_init();
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "dot_product",
        ));

        for log_len in [1, 3, 11, 15] {
            println!("Testing CUDA dot product with len = {}", 1 << log_len);
            let rng = &mut StdRng::seed_from_u64(0);
            let a = (0..1 << log_len).map(|_| rng.random()).collect::<Vec<F>>();
            let b = (0..1 << log_len).map(|_| rng.random()).collect::<Vec<EF>>();
            let a_dev = memcpy_htod(&a);
            let b_dev = memcpy_htod(&b);
            cuda_sync();

            let time = std::time::Instant::now();
            let res_cuda = cuda_dot_product(&a_dev, &b_dev);
            cuda_sync();
            println!("CUDA time: {:?} ms", time.elapsed().as_millis());

            let time = std::time::Instant::now();
            let res_cpu = (0..1 << log_len)
                .into_par_iter()
                .map(|i| b[i] * a[i])
                .sum::<EF>();
            println!("CPU time: {:?} ms", time.elapsed().as_millis());

            assert!(res_cuda == res_cpu);
        }
    }

    #[test]
    fn test_cuda_eval_mixed_tensor() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 8>;
        cuda_init();
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "tensor_algebra.cu",
            "tensor_algebra_dot_product",
        ));
        let n_vars = 20;
        let rng = &mut StdRng::seed_from_u64(0);
        let terms = MultilinearHost::random(rng, n_vars);
        let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<EF>>();
        let terms_dev = memcpy_htod(&terms.evals);
        cuda_sync();

        let time = std::time::Instant::now();
        let res_cuda = cuda_eval_mixed_tensor::<F, EF>(&terms_dev, &point);
        cuda_sync();
        println!("CUDA time: {:?} ms", time.elapsed().as_millis());

        let time = std::time::Instant::now();
        let res_cpu = terms.eval_mixed_tensor::<F>(&point);
        println!("CPU time: {:?} ms", time.elapsed().as_millis());

        assert_eq!(res_cuda, res_cpu.data);
    }

    #[test]
    fn test_cuda_piecewise_sum() {
        let rng = &mut StdRng::seed_from_u64(0);
        type F = BinomialExtensionField<KoalaBear, 8>;
        cuda_init();
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "piecewise_sum",
        ));
        for log_len in [1, 3, 11, 20] {
            for sum_size in [1, 2, 4, 64] {
                let len = 1 << log_len;
                if sum_size > len {
                    continue;
                }
                let input = (0..len).map(|_| rng.random()).collect::<Vec<F>>();
                let input_dev = memcpy_htod(&input);
                cuda_sync();
                let time = std::time::Instant::now();
                let cuda_res = cuda_piecewise_sum(&input_dev, sum_size);
                cuda_sync();
                println!("CUDA time: {:?} ms", time.elapsed().as_millis());
                let cuda_res = memcpy_dtoh(&cuda_res);
                cuda_sync();
                let time = std::time::Instant::now();
                let output_len = len / sum_size;
                let cpu_res = (0..output_len)
                    .into_par_iter()
                    .map(|i| {
                        let mut sum = input[i];
                        for j in 1..sum_size {
                            sum += input[i + output_len * j];
                        }
                        sum
                    })
                    .collect::<Vec<F>>();
                println!("CPU time: {:?} ms", time.elapsed().as_millis());
                assert!(cuda_res == cpu_res);
            }
        }
    }

    #[test]
    fn test_linear_combination_at_row_level() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<KoalaBear, 8>;
        cuda_init();
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "linear_combination_at_row_level",
        ));
        let rng = &mut StdRng::seed_from_u64(0);
        for log_len in [1, 3, 11, 20] {
            for n_scalars in [1, 2, 4, 64] {
                let len = 1 << log_len;
                if n_scalars > len {
                    continue;
                }
                let input = (0..len).map(|_| rng.random()).collect::<Vec<F>>();
                let input_dev = memcpy_htod(&input);
                cuda_sync();
                let scalars = (0..n_scalars).map(|_| rng.random()).collect::<Vec<EF>>();
                let time = std::time::Instant::now();
                let cuda_res = cuda_linear_combination_at_row_level(&input_dev.as_view(), &scalars);
                cuda_sync();
                println!("CUDA time: {:?} ms", time.elapsed().as_millis());
                let cuda_res = memcpy_dtoh(&cuda_res);
                cuda_sync();
                let time = std::time::Instant::now();
                let output_len = len / n_scalars;
                let cpu_res = (0..output_len)
                    .into_par_iter()
                    .map(|i| {
                        let mut sum = EF::ZERO;
                        for j in 0..n_scalars {
                            sum += scalars[j] * input[i * n_scalars + j];
                        }
                        sum
                    })
                    .collect::<Vec<EF>>();
                println!("CPU time: {:?} ms", time.elapsed().as_millis());
                assert!(cuda_res == cpu_res);
            }
        }
    }

    #[test]
    fn test_cuda_linear_combination() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 8>;
        cuda_init();
        cuda_load_function(CudaFunctionInfo::two_fields::<F, EF>(
            "multilinear.cu",
            "linear_combination",
        ));
        let rng = &mut StdRng::seed_from_u64(0);
        cuda_init();
        for len in [1, 11, 251, 700051] {
            for n_scalars in [1, 2, 7, 64] {
                if n_scalars > len {
                    continue;
                }
                let inputs = (0..n_scalars)
                    .map(|_| (0..len).map(|_| rng.random()).collect::<Vec<F>>())
                    .collect::<Vec<Vec<F>>>();
                let inputs_dev = inputs
                    .iter()
                    .map(|input| memcpy_htod(input))
                    .collect::<Vec<CudaSlice<F>>>();
                cuda_sync();
                let scalars = (0..n_scalars).map(|_| rng.random()).collect::<Vec<EF>>();
                let time = std::time::Instant::now();
                let cuda_res = cuda_linear_combination(&inputs_dev, &scalars);
                cuda_sync();
                println!("CUDA time: {:?} ms", time.elapsed().as_millis());
                let cuda_res = memcpy_dtoh(&cuda_res);
                cuda_sync();
                let time = std::time::Instant::now();
                let cpu_res = (0..len)
                    .into_par_iter()
                    .map(|i| {
                        let mut sum = EF::ZERO;
                        for j in 0..n_scalars {
                            sum += scalars[j] * inputs[j][i];
                        }
                        sum
                    })
                    .collect::<Vec<EF>>();
                println!("CPU time: {:?} ms", time.elapsed().as_millis());
                assert!(cuda_res == cpu_res);
            }
        }
    }

    #[test]
    fn test_cuda_sum_in_place() {
        cuda_init();
        let rng = &mut StdRng::seed_from_u64(0);
        type F = BinomialExtensionField<KoalaBear, 8>;
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "sum_in_place",
        ));
        for log_len in [1, 3, 11, 20] {
            let len = 1 << log_len;
            let input = (0..len).map(|_| rng.random()).collect::<Vec<F>>();
            let input_dev = memcpy_htod(&input);
            cuda_sync();
            let time = std::time::Instant::now();
            let cuda_res = cuda_sum(input_dev);
            cuda_sync();
            println!("CUDA time: {:?} ms", time.elapsed().as_millis());
            cuda_sync();
            let time = std::time::Instant::now();
            let cpu_res = input.into_par_iter().sum::<F>();
            println!("CPU time: {:?} ms", time.elapsed().as_millis());
            assert_eq!(cuda_res, cpu_res);
        }
    }
}
