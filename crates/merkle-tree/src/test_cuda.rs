use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

#[test]
fn test_cuda_keccak() -> Result<(), DriverError> {
    let ptx_path = env!("PTX_KECCAK_PATH");
    let ptx_content = std::fs::read_to_string(ptx_path).expect("Failed to read PTX file");

    let dev = CudaDevice::new(0)?;

    dev.load_ptx(Ptx::from_src(ptx_content), "keccak", &["batch_keccak256"])?;

    // and then retrieve the function with `get_func`
    let f = dev.get_func("keccak", "batch_keccak256").unwrap();

    let n_inputs = 10_000;
    let input_length = 1000;
    let input_packed_length = 1111;
    let src_bytes = (0..n_inputs * input_packed_length)
        .map(|i| (i % 256) as u8)
        .collect::<Vec<u8>>();

    let t = std::time::Instant::now();
    let src_bytes_dev = dev.htod_copy(src_bytes.clone())?;
    dev.synchronize()?;
    println!("Copy to device took {:?}", t.elapsed());
    let mut dest_dev = unsafe { dev.alloc::<u8>(32 * n_inputs)? };

    const NUM_THREADS: u32 = 256;
    let num_blocks = (n_inputs as u32).div_ceil(NUM_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(
            cfg,
            (
                &src_bytes_dev,
                n_inputs as u32,
                input_length as u32,
                input_packed_length as u32,
                &mut dest_dev,
            ),
        )
    }?;

    let dest = dev.sync_reclaim(dest_dev)?;

    for i in 0..n_inputs {
        assert_eq!(
            hash_keccak256(
                &src_bytes[i * input_packed_length..i * input_packed_length + input_length]
            ),
            &dest[i * 32..(i + 1) * 32]
        );
    }

    Ok(())
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
