use algebra::utils::KeccakDigest;
use cudarc::driver::{CudaView, CudaViewMut, DeviceRepr, LaunchConfig, PushKernelArg};

use crate::cuda_info;

const NUM_THREADS: u32 = 256;

pub fn cuda_keccak256<T: DeviceRepr>(
    input: &CudaView<T>,
    batch_size: usize,
    output: &mut CudaViewMut<KeccakDigest>,
) {
    assert!(input.len() % batch_size == 0);
    let n_inputs = input.len() / batch_size;
    assert_eq!(n_inputs, output.len());
    let input_length = (batch_size * std::mem::size_of::<T>()) as u32;
    let input_packed_length = input_length;

    let cuda = &cuda_info();
    let f = cuda.get_function("keccak", "batch_keccak256");

    let num_blocks = (n_inputs as u32).div_ceil(NUM_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch_args = cuda.stream.launch_builder(&f);
    launch_args.arg(input);
    launch_args.arg(&n_inputs);
    launch_args.arg(&input_length);
    launch_args.arg(&input_packed_length);
    launch_args.arg(output);
    unsafe { launch_args.launch(cfg) }.unwrap();
}
