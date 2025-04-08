use cudarc::driver::{DriverError, LaunchConfig, PushKernelArg};

use crate::{cuda_info, memcpy_htod};

const NUM_THREADS: u32 = 256;

pub fn cuda_batch_keccak(
    buff: &[u8],
    input_length: u32,
    input_packed_length: u32,
) -> Result<Vec<[u8; 32]>, DriverError> {
    assert!(buff.len() as u32 % input_packed_length == 0);
    assert!(input_length <= input_packed_length);

    let cuda = &cuda_info();
    let f = cuda.get_function("keccak", "batch_keccak256");

    let n_inputs: u32 = buff.len() as u32 / input_packed_length;
    let src_bytes_dev = memcpy_htod(buff);
    let mut dest_dev = unsafe { cuda.stream.alloc::<u8>(32 * n_inputs as usize)? };

    let num_blocks = (n_inputs as u32).div_ceil(NUM_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(&src_bytes_dev);
    launch_args.arg(&n_inputs);
    launch_args.arg(&input_length);
    launch_args.arg(&input_packed_length);
    launch_args.arg(&mut dest_dev);
    unsafe { launch_args.launch(cfg) }?;

    let mut res = vec![0u8; 32 * n_inputs as usize];
    cuda.stream.memcpy_dtoh(&dest_dev, &mut res)?;
    cuda.stream.synchronize()?;

    let array_count = res.len() / 32;
    let capacity = res.capacity() / 32;
    let ptr = res.as_mut_ptr() as *mut [u8; 32];
    std::mem::forget(res);
    unsafe { Ok(Vec::from_raw_parts(ptr, array_count, capacity)) }
}
