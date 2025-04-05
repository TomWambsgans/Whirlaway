use cudarc::driver::{DriverError, LaunchAsync, LaunchConfig};

use crate::cuda_info;

const NUM_THREADS: u32 = 256;

pub fn cuda_batch_keccak(
    buff: &[u8],
    input_length: usize,
    input_packed_length: usize,
) -> Result<Vec<[u8; 32]>, DriverError> {
    assert!(buff.len() % input_packed_length == 0);
    assert!(input_length <= input_packed_length);

    let dev = &cuda_info().dev;
    let f = dev.get_func("keccak", "batch_keccak256").unwrap();

    let n_inputs = buff.len() / input_packed_length;
    let src_bytes_dev = dev.htod_sync_copy(buff)?;
    // dev.synchronize()?;
    let mut dest_dev = unsafe { dev.alloc::<u8>(32 * n_inputs)? };

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
