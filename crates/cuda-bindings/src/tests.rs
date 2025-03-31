use cudarc::driver::{LaunchAsync, LaunchConfig};
use p3_field::extension::BinomialExtensionField;
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
