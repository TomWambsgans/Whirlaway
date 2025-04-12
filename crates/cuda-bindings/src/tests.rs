use cuda_engine::{cuda_init, cuda_sync, memcpy_dtoh, memcpy_htod};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::{cuda_dot_product, cuda_fold_sum};

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
