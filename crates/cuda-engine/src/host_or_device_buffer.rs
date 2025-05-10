use cudarc::driver::{CudaSlice, DeviceRepr};
use std::{hash::Hash, ops::Range};
use utils::default_hash;

use crate::{cuda_engine, cuda_get_at_index, cuda_sync, memcpy_dtoh};

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
                let cuda = cuda_engine();
                let mut dst = vec![T::default(); range.end - range.start];
                cuda.stream
                    .memcpy_dtoh(&slice.slice(range), &mut dst)
                    .unwrap();
                dst
            }
        }
    }
}

impl<T: DeviceRepr + Default + Clone + Hash> HostOrDeviceBuffer<T> {
    /// Debug purpose
    /// Sync
    pub fn hash(&self) -> u64 {
        match self {
            Self::Host(host_buff) => default_hash(host_buff),
            Self::Device(dev_buff) => {
                let host_buff = memcpy_dtoh(dev_buff);
                cuda_sync();
                default_hash(&host_buff)
            }
        }
    }
}
