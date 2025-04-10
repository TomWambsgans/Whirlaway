use std::{borrow::Borrow, ops::Range};

use crate::cuda_info;
use cudarc::driver::DevicePtr;
use cudarc::driver::{CudaSlice, DeviceRepr};

pub enum HostOrDeviceBuffer<T> {
    Host(Vec<T>),
    Device(CudaSlice<T>),
}

impl<T: DeviceRepr + Clone + Default> HostOrDeviceBuffer<T> {
    /// Async
    pub fn index(&self, idx: usize) -> T {
        match self {
            HostOrDeviceBuffer::Host(v) => v[idx].clone(),
            HostOrDeviceBuffer::Device(slice) => cuda_get_at_index(slice, idx),
        }
    }

    // Async
    pub fn slice(&self, range: Range<usize>) -> Vec<T> {
        match self {
            HostOrDeviceBuffer::Host(v) => v[range].to_vec(),
            HostOrDeviceBuffer::Device(slice) => {
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

/// Async
pub fn memcpy_htod<T: DeviceRepr>(src: &[T]) -> CudaSlice<T> {
    let cuda = cuda_info();
    let mut dst = unsafe { cuda.stream.alloc(src.len()).unwrap() };
    cuda.stream.memcpy_htod(src, &mut dst).unwrap();
    dst
}

/// Async
pub fn memcpy_dtoh<T: DeviceRepr + Default + Clone>(src: &CudaSlice<T>) -> Vec<T> {
    let cuda = cuda_info();
    let mut dst = vec![T::default(); src.len()];
    cuda.stream.memcpy_dtoh(src, &mut dst).unwrap();
    dst
}

/// Async
pub fn concat_pointers<T: DeviceRepr, S: Borrow<CudaSlice<T>>>(slices: &[S]) -> CudaSlice<u64> {
    let cuda = cuda_info();
    memcpy_htod(
        &slices
            .iter()
            .map(|slice_dev| slice_dev.borrow().device_ptr(&cuda.stream).0)
            .collect::<Vec<u64>>(), // TODO avoid hardcoding u64 (this is platform dependent)
    )
}

/// Async
pub fn cuda_alloc<T: DeviceRepr>(size: usize) -> CudaSlice<T> {
    unsafe { cuda_info().stream.alloc(size).unwrap() }
}

/// Async
pub fn cuda_get_at_index<T: DeviceRepr + Default>(slice: &CudaSlice<T>, idx: usize) -> T {
    let cuda = cuda_info();
    let mut dst = [T::default()];
    cuda.stream
        .memcpy_dtoh(&slice.slice(idx..idx + 1), &mut dst)
        .unwrap();
    dst.into_iter().next().unwrap()
}
