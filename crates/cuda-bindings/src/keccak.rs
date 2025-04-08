use algebra::utils::KeccakDigest;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::cuda_info;

const NUM_THREADS: u32 = 256;

pub fn cuda_keccak256(
    buff: &CudaSlice<u8>,
    input_length: u32,
    input_packed_length: u32,
) -> CudaSlice<KeccakDigest> {
    assert!(buff.len() as u32 % input_packed_length == 0);
    assert!(input_length <= input_packed_length);

    let cuda = &cuda_info();
    let f = cuda.get_function("keccak", "batch_keccak256");

    let n_inputs: u32 = buff.len() as u32 / input_packed_length;
    let mut res_dev = unsafe { cuda.stream.alloc::<KeccakDigest>(n_inputs as usize) }.unwrap();

    let num_blocks = (n_inputs as u32).div_ceil(NUM_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(buff);
    launch_args.arg(&n_inputs);
    launch_args.arg(&input_length);
    launch_args.arg(&input_packed_length);
    launch_args.arg(&mut res_dev);
    unsafe { launch_args.launch(cfg) }.unwrap();

    res_dev
}
