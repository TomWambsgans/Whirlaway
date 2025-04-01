use std::error::Error;

use algebra::ntt::ntt_batch;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, sys::CUdevice_attribute};
use p3_field::{PrimeCharacteristicRing, TwoAdicField, extension::BinomialExtensionField};
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{cuda_batch_keccak, init_cuda};
use rayon::prelude::*;

#[test]
fn test_cuda_keccak() {
    let t = std::time::Instant::now();
    init_cuda().unwrap();
    println!("CUDA initialized in {} ms", t.elapsed().as_millis());

    let n_inputs = 10_000;
    let input_length = 1000;
    let input_packed_length = 1111;
    let src_bytes = (0..n_inputs * input_packed_length)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>();

    let t = std::time::Instant::now();
    let dest = cuda_batch_keccak(&src_bytes, input_length, input_packed_length).unwrap();
    println!("CUDA took {} ms", t.elapsed().as_millis());
    for i in 0..n_inputs {
        assert_eq!(
            hash_keccak256(
                &src_bytes[i * input_packed_length..i * input_packed_length + input_length]
            ),
            dest[i]
        );
    }
}

fn hash_keccak256(data: &[u8]) -> [u8; 32] {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

#[test]
pub fn test_cuda_monty_field() {
    init_cuda().unwrap();

    const EXT_DEGREE: usize = 8;

    type F = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let rng = &mut StdRng::seed_from_u64(0);
    for _ in 0..1000 {
        let a: F = rng.random();
        let b: F = rng.random();

        let dev = crate::get_cuda_device();
        let f_add = dev.get_func("ntt", "test_add").unwrap();
        let f_mul = dev.get_func("ntt", "test_mul").unwrap();
        let f_sub = dev.get_func("ntt", "test_sub").unwrap();

        let a_bytes = unsafe { std::mem::transmute::<_, [u32; EXT_DEGREE]>(a) };
        let b_bytes = unsafe { std::mem::transmute::<_, [u32; EXT_DEGREE]>(b) };
        let a_bytes_dev = dev.htod_copy(a_bytes.to_vec()).unwrap();
        let b_bytes_dev = dev.htod_copy(b_bytes.to_vec()).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut res_dev = dev.alloc_zeros::<u32>(EXT_DEGREE).unwrap();
        unsafe { f_add.launch(cfg, (&a_bytes_dev, &b_bytes_dev, &mut res_dev)) }.unwrap();
        let res: [u32; EXT_DEGREE] = dev.sync_reclaim(res_dev).unwrap().try_into().unwrap();
        let res = unsafe { std::mem::transmute::<_, F>(res) };
        assert_eq!(res, a + b);

        let mut res_dev = dev.alloc_zeros::<u32>(EXT_DEGREE).unwrap();
        unsafe { f_mul.launch(cfg, (&a_bytes_dev, &b_bytes_dev, &mut res_dev)) }.unwrap();
        let res: [u32; EXT_DEGREE] = dev.sync_reclaim(res_dev).unwrap().try_into().unwrap();
        let res = unsafe { std::mem::transmute::<_, F>(res) };
        assert_eq!(res, a * b);

        let mut res_dev = dev.alloc_zeros::<u32>(EXT_DEGREE).unwrap();
        unsafe { f_sub.launch(cfg, (&a_bytes_dev, &b_bytes_dev, &mut res_dev)) }.unwrap();
        let res: [u32; EXT_DEGREE] = dev.sync_reclaim(res_dev).unwrap().try_into().unwrap();
        let res = unsafe { std::mem::transmute::<_, F>(res) };
        assert_eq!(res, a - b);
    }
}

#[test]
pub fn test_ntt_at_block_level() {
    init_cuda().unwrap();

    const EXT_DEGREE: usize = 8;

    type F = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let rng = &mut StdRng::seed_from_u64(0);
    let dev = crate::get_cuda_device();
    let f_ntt_at_block_level = dev.get_func("ntt", "test_ntt_at_block_level").unwrap();
    let log_threads_per_block = 8;
    let threads_per_block = 1 << log_threads_per_block;
    let n_blocks = 10;
    let mut buff = (0..2 * threads_per_block * n_blocks)
        .map(|_| rng.random())
        .collect::<Vec<F>>();
    let mut expected_ntt = buff.clone();
    ntt_batch(&mut expected_ntt, 2 * threads_per_block);

    let twiddles = (0..threads_per_block * 2)
        .map(|i| KoalaBear::two_adic_generator(1 + log_threads_per_block).exp_u64(i as u64))
        .collect::<Vec<KoalaBear>>();
    // cast twiddles as a vec of u32 of length threads_per_block * EXT_DEGREE, using unsafe
    let twiddles_u32 = unsafe {
        std::slice::from_raw_parts(twiddles.as_ptr() as *const u32, threads_per_block * 2)
    }
    .to_vec();

    for arr in buff.chunks_mut(2 * threads_per_block) {
        for i in 0..log_threads_per_block {
            for j in 0..1 << i {
                let arr = &mut arr[j * (1 << (log_threads_per_block + 1 - i))
                    ..(j + 1) * (1 << (log_threads_per_block + 1 - i))];
                // e0, o0, e1, o1, e2, o2, e3, o3 ... becomes e0, e1, e2, e3, ..., o0, o1, o2, o3 ...
                let arr_copy = arr.to_vec();
                for k in 0..(1 << (log_threads_per_block - i)) {
                    arr[k] = arr_copy[k * 2];
                    arr[k + (1 << (log_threads_per_block - i))] = arr_copy[k * 2 + 1];
                }
            }
        }
    }

    let buff_u32 = unsafe {
        std::slice::from_raw_parts(
            buff.as_ptr() as *const u32,
            2 * threads_per_block * n_blocks * EXT_DEGREE,
        )
    }
    .to_vec();
    let twiddles_dev = dev.htod_copy(twiddles_u32).unwrap();
    let mut buff_dev = dev.htod_copy(buff_u32).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (n_blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe { f_ntt_at_block_level.launch(cfg, (&mut buff_dev, &twiddles_dev)) }.unwrap();

    let cuda_ntt_u32: Vec<u32> = dev.sync_reclaim(buff_dev).unwrap();
    let cuda_ntt = unsafe {
        std::slice::from_raw_parts(
            cuda_ntt_u32.as_ptr() as *const F,
            2 * threads_per_block * n_blocks,
        )
    }
    .to_vec();
    assert_eq!(cuda_ntt, expected_ntt);
}

#[test]
pub fn test_ntt() {
    init_cuda().unwrap();

    const EXT_DEGREE: usize = 8;

    type F = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let rng = &mut StdRng::seed_from_u64(0);
    let dev = crate::get_cuda_device();
    let f_ntt = dev.get_func("ntt", "test_ntt").unwrap();
    let log_threads_per_block = 8;
    let threads_per_block = 1 << log_threads_per_block;
    let n_blocks = 1 << 5;
    let log_len = log_threads_per_block + 4;
    let len = 1 << log_len;
    let log_expension_factor = (threads_per_block * 2 * n_blocks / len as usize).ilog2() as usize;
    let expansion_factor = 1 << log_expension_factor;
    let coeffs = (0..len).map(|_| rng.random()).collect::<Vec<F>>();

    println!("number of field elements: {}, expension factor: {}", len, expansion_factor);

    let expanded_size = coeffs.len() * expansion_factor;
    let root = KoalaBear::two_adic_generator(log_len + log_expension_factor);
    let mut expected_result = Vec::new();
    expected_result.extend_from_slice(&coeffs);
    expected_result.extend((1..expansion_factor).flat_map(|i| {
        let root_i = root.exp_u64(i as u64);
        coeffs
            .iter()
            .enumerate()
            .map(move |(j, coeff)| *coeff * root_i.exp_u64(j as u64))
    }));
    ntt_batch(&mut expected_result, coeffs.len());
    // expected_result = bit_reverse_order(&expected_result, log_len);

    let mut all_twiddles = Vec::new();

    for i in 0..=log_len + log_expension_factor {
        let root = KoalaBear::two_adic_generator(i);
        let twiddles = (0..1 << i)
            .map(|j| root.exp_u64(j as u64))
            .collect::<Vec<KoalaBear>>();
        all_twiddles.extend(twiddles);
    }

    let all_twiddles_u32 = unsafe {
        std::slice::from_raw_parts(all_twiddles.as_ptr() as *const u32, all_twiddles.len())
    }
    .to_vec();

    let all_twiddles_dev = dev.htod_copy(all_twiddles_u32).unwrap();

    let coeffs_u32 =
        unsafe { std::slice::from_raw_parts(coeffs.as_ptr() as *const u32, len * EXT_DEGREE) }
            .to_vec();

    let time = std::time::Instant::now();
    let coeffs_dev = dev.htod_copy(coeffs_u32).unwrap();
    dev.synchronize().unwrap();
    println!(
        "CPU -> GPU DATA transfer took {} ms",
        time.elapsed().as_millis()
    );

    let mut output_dev = dev.alloc_zeros::<u32>(expanded_size * EXT_DEGREE).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (n_blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: ((threads_per_block * 2) * (EXT_DEGREE + 1) * 4) as u32,
    };

    let time = std::time::Instant::now();
    unsafe {
        f_ntt.launch_cooperative(
            cfg,
            (
                &coeffs_dev,
                &mut output_dev,
                log_len as u32,
                log_expension_factor as u32,
                &all_twiddles_dev,
            ),
        )
    }
    .unwrap();
    dev.synchronize().unwrap();
    println!("CUDA kernel took {} ms", time.elapsed().as_millis());

    let time = std::time::Instant::now();
    let cuda_result_u32: Vec<u32> = dev.sync_reclaim(output_dev).unwrap();
    println!(
        "GPU -> CPU DATA transfer took {} ms",
        time.elapsed().as_millis()
    );
    let cuda_result =
        unsafe { std::slice::from_raw_parts(cuda_result_u32.as_ptr() as *const F, expanded_size) }
            .to_vec();
    assert!(cuda_result == expected_result);
}

#[test]
pub fn test_batch_reverse_bit_order() {
    init_cuda().unwrap();

    const EXT_DEGREE: usize = 8;

    type F = BinomialExtensionField<KoalaBear, EXT_DEGREE>;

    let rng = &mut StdRng::seed_from_u64(0);
    let dev = crate::get_cuda_device();
    let f_test_batch_reverse_bit_order =
        dev.get_func("ntt", "test_batch_reverse_bit_order").unwrap();
    let bits = 16;
    let n_batch: usize = 32;
    let coeffs = (0..n_batch * (1 << bits))
        .map(|_| rng.random())
        .collect::<Vec<F>>();

    let time = std::time::Instant::now();
    let expected_result = bit_reverse_order(&coeffs, bits);
    println!("CPU took {} ms", time.elapsed().as_millis());

    let coeffs_u32 = unsafe {
        std::slice::from_raw_parts(coeffs.as_ptr() as *const u32, coeffs.len() * EXT_DEGREE)
    }
    .to_vec();

    let mut coeffs_u32_dev = dev.htod_copy(coeffs_u32).unwrap();
    dev.synchronize().unwrap();

    let cfg = LaunchConfig {
        grid_dim: ((n_batch * (1 << bits)) as u32 / 512, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let time = std::time::Instant::now();
    unsafe {
        f_test_batch_reverse_bit_order.launch(
            cfg,
            (
                &mut coeffs_u32_dev,
                bits as u32,
                (n_batch * (1 << bits)) as u32,
            ),
        )
    }
    .unwrap();
    dev.synchronize().unwrap();
    println!("CUDA took {} ms", time.elapsed().as_millis());

    let cuda_result_u32: Vec<u32> = dev.sync_reclaim(coeffs_u32_dev).unwrap();
    let cuda_result = unsafe {
        std::slice::from_raw_parts(cuda_result_u32.as_ptr() as *const F, n_batch * (1 << bits))
    }
    .to_vec();
    assert_eq!(cuda_result.len(), expected_result.len());
    assert_eq!(cuda_result, expected_result);
}

fn cuda_shared_memory() -> Result<usize, Box<dyn Error>> {
    let dev = CudaDevice::new(0)?;
    Ok(dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK)? as usize)
}

fn cuda_constant_memory() -> Result<usize, Box<dyn Error>> {
    let dev = CudaDevice::new(0)?;
    Ok(dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)? as usize)
}

#[test]
fn print_cuda_info() {
    init_cuda().unwrap();

    let shared_memory = cuda_shared_memory().unwrap();
    println!("Shared memory per block: {} bytes", shared_memory);

    let constant_memory = cuda_constant_memory().unwrap();
    println!("Constant memory: {} bytes", constant_memory);
}

fn bit_reverse_order<T: Clone + Send + Copy>(arr: &[T], bits: usize) -> Vec<T> {
    assert!(arr.len() % (1 << bits) == 0);
    let mut res = arr.to_vec();

    // Process and update each chunk in parallel using chunks_mut
    res.par_chunks_mut(1 << bits).for_each(|chunk| {
        // Create a temporary copy for the computations
        let mut new_chunk = vec![chunk[0].clone(); chunk.len()];

        for i in 0_u32..(1 << bits) {
            let rev_i = i.reverse_bits() >> (32 - bits);
            new_chunk[i as usize] = chunk[rev_i as usize].clone();
        }

        // Copy back the results
        chunk.copy_from_slice(&new_chunk);
    });

    res
}
