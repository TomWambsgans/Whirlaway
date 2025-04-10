#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use std::{borrow::Borrow, ops::Range};

use cudarc::driver::DevicePtr;

#[cfg(test)]
mod tests;

mod keccak;
pub use keccak::*;

mod ntt;
pub use ntt::*;

mod sumcheck;
pub use sumcheck::*;

mod init;
pub use init::*;

mod multilinear;
pub use multilinear::*;

pub mod cuda_pols;
pub use cuda_pols::*;

pub use cudarc::driver::{CudaSlice, DeviceRepr};

// Should not be too big to avoid CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
const MAX_LOG_N_BLOCKS: u32 = 5;

pub enum VecOrCudaSlice<T> {
    Vec(Vec<T>),
    Cuda(CudaSlice<T>),
}

impl<T: DeviceRepr + Clone + Default> VecOrCudaSlice<T> {
    pub fn index(&self, idx: usize) -> T {
        // This function is Async !
        match self {
            VecOrCudaSlice::Vec(v) => v[idx].clone(),
            VecOrCudaSlice::Cuda(slice) => cuda_get_at_index(slice, idx),
        }
    }

    pub fn slice(&self, range: Range<usize>) -> Vec<T> {
        match self {
            // This function is Async !
            VecOrCudaSlice::Vec(v) => v[range].to_vec(),
            VecOrCudaSlice::Cuda(slice) => {
                let cuda = cuda_info();
                let mut dst = vec![T::default(); range.end - range.start];
                cuda.stream
                    .memcpy_dtoh(&slice.slice(range), &mut dst)
                    .unwrap();
                dst
            }
        }
    }
}

pub fn cuda_sync() {
    cuda_info().stream.synchronize().unwrap();
}

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

pub fn concat_pointers<T: DeviceRepr, S: Borrow<CudaSlice<T>>>(slices: &[S]) -> CudaSlice<u64> {
    let cuda = cuda_info();
    memcpy_htod(
        &slices
            .iter()
            .map(|slice_dev| slice_dev.borrow().device_ptr(&cuda.stream).0)
            .collect::<Vec<u64>>(), // TODO avoid hardcoding u64 (this is platform dependent)
    )
}

pub fn cuda_alloc<T: DeviceRepr>(size: usize) -> CudaSlice<T> {
    unsafe { cuda_info().stream.alloc(size).unwrap() }
}

pub fn cuda_get_at_index<T: DeviceRepr + Default>(slice: &CudaSlice<T>, idx: usize) -> T {
    let cuda = cuda_info();
    let mut dst = [T::default()];
    cuda.stream
        .memcpy_dtoh(&slice.slice(idx..idx + 1), &mut dst)
        .unwrap();
    dst.into_iter().next().unwrap()
}
