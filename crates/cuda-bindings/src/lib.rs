#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod tests;

use std::sync::{Arc, OnceLock};

use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

static CUDA_DEVICE: OnceLock<Arc<CudaDevice>> = OnceLock::new();

pub fn init_cuda() -> Result<(), DriverError> {
    let ptx_path = env!("PTX_KECCAK_PATH");
    let ptx_content = std::fs::read_to_string(ptx_path).expect("Failed to read PTX file");
    let dev = CudaDevice::new(0)?;
    dev.load_ptx(Ptx::from_src(ptx_content), "keccak", &["batch_keccak256"])?;
    CUDA_DEVICE
        .set(dev)
        .expect("CUDA device already initialized");
    Ok(())
}

fn get_cuda_device() -> Arc<CudaDevice> {
    CUDA_DEVICE
        .get()
        .expect("CUDA device not initialized")
        .clone()
}

pub fn cuda_batch_keccak(
    buff: &[u8],
    input_length: usize,
    input_packed_length: usize,
) -> Result<Vec<[u8; 32]>, DriverError> {
    assert!(buff.len() % input_packed_length == 0);
    assert!(input_length <= input_packed_length);

    let dev = get_cuda_device();
    let f = dev.get_func("keccak", "batch_keccak256").unwrap();

    let n_inputs = buff.len() / input_packed_length;
    let src_bytes_dev = dev.htod_copy(buff.to_vec())?;
    dev.synchronize()?;
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

    let mut res = dev.sync_reclaim(dest_dev)?;

    assert!(res.len() == 32 * n_inputs);

    let array_count = res.len() / 32;
    let capacity = res.capacity() / 32;
    let ptr = res.as_mut_ptr() as *mut [u8; 32];
    std::mem::forget(res);
    unsafe { Ok(Vec::from_raw_parts(ptr, array_count, capacity)) }
}
