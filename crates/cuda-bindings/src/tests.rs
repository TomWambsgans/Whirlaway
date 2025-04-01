use std::error::Error;

use algebra::ntt::ntt_batch;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, sys::CUdevice_attribute};
use p3_field::{PrimeCharacteristicRing, TwoAdicField, extension::BinomialExtensionField};
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{cuda_batch_keccak, init_cuda};

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

        let mut res_dev = unsafe { dev.alloc::<u32>(EXT_DEGREE).unwrap() };
        unsafe { f_add.launch(cfg, (&a_bytes_dev, &b_bytes_dev, &mut res_dev)) }.unwrap();
        let res: [u32; EXT_DEGREE] = dev.sync_reclaim(res_dev).unwrap().try_into().unwrap();
        let res = unsafe { std::mem::transmute::<_, F>(res) };
        assert_eq!(res, a + b);

        let mut res_dev = unsafe { dev.alloc::<u32>(EXT_DEGREE).unwrap() };
        unsafe { f_mul.launch(cfg, (&a_bytes_dev, &b_bytes_dev, &mut res_dev)) }.unwrap();
        let res: [u32; EXT_DEGREE] = dev.sync_reclaim(res_dev).unwrap().try_into().unwrap();
        let res = unsafe { std::mem::transmute::<_, F>(res) };
        assert_eq!(res, a * b);

        let mut res_dev = unsafe { dev.alloc::<u32>(EXT_DEGREE).unwrap() };
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
