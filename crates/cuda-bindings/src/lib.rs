#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod tests;

mod keccak;
use cudarc::driver::DeviceRepr;
pub use keccak::*;

mod ntt;
pub use ntt::*;

mod sumcheck;
pub use sumcheck::*;

mod init;
pub use init::*;

// Should not be too big to avoid CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
const MAX_LOG_N_BLOCKS: u32 = 5;

pub use cudarc::driver::CudaSlice;

pub fn memcpy_htod<T: DeviceRepr>(src: &[T]) -> CudaSlice<T> {
    let cuda = cuda_info();
    let mut dst = unsafe { cuda.stream.alloc(src.len()).unwrap() };
    cuda.stream.memcpy_htod(src, &mut dst).unwrap();
    dst
}

pub fn memcpy_dtoh<T: DeviceRepr + Default + Clone>(src: &CudaSlice<T>) -> Vec<T> {
    let cuda = cuda_info();
    let mut dst = vec![T::default(); src.len()];
    cuda.stream.memcpy_dtoh(src, &mut dst).unwrap();
    dst
}

pub fn cuda_sync() {
    cuda_info().stream.synchronize().unwrap();
}
