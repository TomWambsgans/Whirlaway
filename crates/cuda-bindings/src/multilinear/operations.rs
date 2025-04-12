use crate::{
    MAX_N_BLOCKS, MAX_N_COOPERATIVE_BLOCKS,
    multilinear::{MULTILINEAR_LOG_N_THREADS_PER_BLOCK, MULTILINEAR_N_THREADS_PER_BLOCK},
};
use cuda_engine::{CudaCall, cuda_alloc, cuda_get_at_index, memcpy_htod};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::{ExtensionField, Field};
use std::any::TypeId;

// Async
pub fn cuda_dot_product<F: Field>(a: &CudaSlice<F>, b: &CudaSlice<F>) -> F {
    assert!(F::bits() > 32, "TODO");
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_power_of_two());
    let log_len = a.len().ilog2() as u32;

    let n_threads_per_blocks = MULTILINEAR_N_THREADS_PER_BLOCK.min(a.len() as u32);
    let n_blocks = ((a.len() as u32 + n_threads_per_blocks - 1) / n_threads_per_blocks)
        .min(MAX_N_COOPERATIVE_BLOCKS);

    let buff = cuda_alloc::<F>(a.len());
    let mut call = CudaCall::new("multilinear", "dot_product")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(a);
    call.arg(b);
    call.arg(&buff);
    call.arg(&log_len);
    call.launch_cooperative();

    cuda_get_at_index(&buff, 0)
}

// Async
pub fn cuda_scale_slice_in_place<F: Field>(slice: &mut CudaSlice<F>, scalar: F) {
    assert!(F::bits() > 32, "TODO");
    let scalar = [scalar];
    let scalar_dev = memcpy_htod(&scalar);
    let n = slice.len() as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);
    let mut call = CudaCall::new("multilinear", "scale_ext_slice_in_place")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(slice);
    call.arg(&n);
    call.arg(&scalar_dev);
    call.launch();
}

// Async
pub fn cuda_scale_slice<F: Field, EF: ExtensionField<F>>(
    slice: &CudaSlice<F>,
    scalar: EF,
) -> CudaSlice<EF> {
    assert!(TypeId::of::<F>() != TypeId::of::<EF>(), "TODO");
    let scalar = [scalar];
    let scalar_dev = memcpy_htod(&scalar);
    let n = slice.len() as u32;
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);

    let mut res = cuda_alloc::<EF>(slice.len());
    let mut call = CudaCall::new("multilinear", "scale_prime_slice_by_ext")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(slice);
    call.arg(&n);
    call.arg(&scalar_dev);
    call.arg(&mut res);
    call.launch();
    res
}

// Async
pub fn cuda_add_slices<F: Field>(a: &CudaSlice<F>, b: &CudaSlice<F>) -> CudaSlice<F> {
    let n = a.len() as u32;
    assert_eq!(n, b.len() as u32);
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);
    let mut res = cuda_alloc::<F>(n as usize);
    let mut call = CudaCall::new("multilinear", "add_slices")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(a);
    call.arg(b);
    call.arg(&mut res);
    call.arg(&n);
    call.launch();
    res
}

// Async
pub fn cuda_add_assign_slices<F: Field>(a: &mut CudaSlice<F>, b: &CudaSlice<F>) {
    // a += b;
    let n = a.len() as u32;
    assert_eq!(n, b.len() as u32);
    let n_threads_per_blocks = MULTILINEAR_LOG_N_THREADS_PER_BLOCK.min(n);
    let n_blocks = ((n + n_threads_per_blocks - 1) / n_threads_per_blocks).min(MAX_N_BLOCKS);
    let mut call = CudaCall::new("multilinear", "add_assign_slices")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(a);
    call.arg(b);
    call.arg(&n);
    call.launch();
}

// Async
pub fn cuda_fold_sum<F: Field>(input: &CudaSlice<F>, sum_size: usize) -> CudaSlice<F> {
    assert!(F::bits() > 32, "TODO");
    assert!(
        sum_size <= 256,
        "CUDA implement is not optimized for large sum sizes"
    );
    assert!(input.len() % sum_size == 0);

    let output_len = input.len() / sum_size;

    let n_threads_per_blocks = MULTILINEAR_N_THREADS_PER_BLOCK.min(output_len as u32);
    let n_blocks = ((output_len as u32).div_ceil(n_threads_per_blocks)).min(MAX_N_BLOCKS);

    let mut output = cuda_alloc::<F>(output_len);
    let len_u32 = input.len() as u32;
    let sum_size_u32 = sum_size as u32;
    let mut call = CudaCall::new("multilinear", "fold_sum")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(input);
    call.arg(&mut output);
    call.arg(&len_u32);
    call.arg(&sum_size_u32);
    call.launch();

    output
}

// Async
pub fn cuda_sum<F: Field>(terms: CudaSlice<F>) -> F {
    // we take owneship of `terms` because it will be aletered
    assert!(F::bits() > 32, "TODO");
    assert!(terms.len().is_power_of_two());

    if terms.len() == 1 {
        return cuda_get_at_index(&terms, 0);
    }

    let log_len = terms.len().ilog2() as u32;

    let n_threads_per_blocks = MULTILINEAR_N_THREADS_PER_BLOCK.min((terms.len() / 2) as u32);
    let n_blocks =
        ((terms.len() as u32 / 2).div_ceil(n_threads_per_blocks)).min(MAX_N_COOPERATIVE_BLOCKS);
    let mut call = CudaCall::new("multilinear", "sum_in_place")
        .blocks(n_blocks)
        .threads_per_block(n_threads_per_blocks);
    call.arg(&terms);
    call.arg(&log_len);
    call.launch_cooperative();
    cuda_get_at_index(&terms, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_engine::*;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rayon::prelude::*;

    #[test]
    fn test_cuda_dot_product() {
        cuda_init();
        type F = BinomialExtensionField<KoalaBear, 8>;
        for log_len in [1, 3, 11, 15] {
            println!("Testing CUDA dot product with len = {}", 1 << log_len);
            let rng = &mut StdRng::seed_from_u64(0);
            let a = (0..1 << log_len).map(|_| rng.random()).collect::<Vec<F>>();
            let b = (0..1 << log_len).map(|_| rng.random()).collect::<Vec<F>>();
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
                .map(|i| a[i] * b[i])
                .sum::<F>();
            println!("CPU time: {:?} ms", time.elapsed().as_millis());

            assert!(res_cuda == res_cpu);
        }
    }

    #[test]
    fn test_cuda_fold_sum() {
        cuda_init();
        let rng = &mut StdRng::seed_from_u64(0);
        type F = BinomialExtensionField<KoalaBear, 8>;
        for log_len in [1, 3, 11, 20] {
            for sum_size in [1, 2, 4, 128] {
                let len = 1 << log_len;
                if sum_size > len {
                    continue;
                }
                let input = (0..len).map(|_| rng.random()).collect::<Vec<F>>();
                let input_dev = memcpy_htod(&input);
                cuda_sync();
                let time = std::time::Instant::now();
                let cuda_res = cuda_fold_sum(&input_dev, sum_size);
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
    fn test_cuda_sum_in_place() {
        cuda_init();
        let rng = &mut StdRng::seed_from_u64(0);
        type F = BinomialExtensionField<KoalaBear, 8>;
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
