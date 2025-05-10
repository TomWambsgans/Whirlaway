use cudarc::driver::{CudaSlice, DeviceRepr};
use cudarc::driver::{DevicePtr, ValidAsZeroBits};
use std::borrow::Borrow;

use crate::{CudaEngine, cuda_engine, memcpy_htod};

pub struct CudaPtr<T: DeviceRepr> {
    pub ptr: u64, // TODO avoid hardcoding u64 (this is platform dependent)
    pub _phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceRepr> CudaPtr<T> {
    pub fn from(slice: &CudaSlice<T>, engine: &CudaEngine) -> Self {
        Self {
            ptr: slice.device_ptr(&engine.stream).0,
            _phantom: std::marker::PhantomData,
        }
    }
}

unsafe impl<T: DeviceRepr> DeviceRepr for CudaPtr<T> {}
unsafe impl<T: DeviceRepr> ValidAsZeroBits for CudaPtr<T> {}

/// Async
pub fn concat_pointers<T: DeviceRepr, S: Borrow<CudaSlice<T>>>(
    slices: &[S],
) -> CudaSlice<CudaPtr<T>> {
    let cuda = cuda_engine();
    memcpy_htod(
        &slices
            .iter()
            .map(|slice_dev| CudaPtr::from(slice_dev.borrow(), &cuda))
            .collect::<Vec<_>>(),
    )
}
