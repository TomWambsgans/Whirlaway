use std::any::TypeId;
use std::hash::Hash;
use std::{borrow::Borrow, ops::Range};

use crate::{CudaFunctionInfo, cuda_engine, try_get_cuda_engine};
use cudarc::driver::{
    CudaFunction, DevicePtr, DevicePtrMut, LaunchArgs, LaunchConfig, ValidAsZeroBits,
};
use cudarc::driver::{CudaSlice, DeviceRepr};
use p3_field::Field;
use utils::default_hash;

// TODO Avoid hardcoding : This is GPU dependent
pub const LOG_MAX_THREADS_PER_COOPERATIVE_BLOCK: u32 = 8;
pub const LOG_MAX_THREADS_PER_BLOCK: u32 = 8;

pub const MAX_THREADS_PER_COOPERATIVE_BLOCK: u32 = 1 << LOG_MAX_THREADS_PER_COOPERATIVE_BLOCK;
pub const MAX_THREADS_PER_BLOCK: u32 = 1 << LOG_MAX_THREADS_PER_BLOCK;

pub const MAX_COOPERATIVE_BLOCKS: u32 = 1 << 4;
pub const MAX_BLOCKS: u32 = 1 << 14;

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

/// Does nothing if cuda has not been initialized
pub fn cuda_sync() {
    if let Some(cuda) = try_get_cuda_engine() {
        cuda.stream.synchronize().unwrap()
    }
}

/// Async
pub fn memcpy_htod<T: DeviceRepr>(src: &[T]) -> CudaSlice<T> {
    let cuda = cuda_engine();
    let mut dst = unsafe { cuda.stream.alloc(src.len()).unwrap() };
    cuda.stream.memcpy_htod(src, &mut dst).unwrap();
    dst
}

/// Async
pub fn memcpy_dtoh<T: DeviceRepr + Default + Clone, Src: DevicePtr<T>>(src: &Src) -> Vec<T> {
    let cuda = cuda_engine();
    let mut dst = vec![T::default(); src.len()];
    cuda.stream.memcpy_dtoh(src, &mut dst).unwrap();
    dst
}

// Async
pub fn clone_dtod<T: DeviceRepr, Src: DevicePtr<T>>(src: &Src) -> CudaSlice<T> {
    cuda_engine().stream.clone_dtod(src).unwrap()
}

// Async
pub fn memcpy_dtod<T: DeviceRepr, Dst: DevicePtrMut<T>>(src: &CudaSlice<T>, dst: &mut Dst) {
    cuda_engine().stream.memcpy_dtod(src, dst).unwrap();
}

/// Async
pub fn memcpy_dtod_to<T: DeviceRepr, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
    src: &Src,
    dst: &mut Dst,
) {
    cuda_engine().stream.memcpy_dtod(src, dst).unwrap();
}

/// Async
pub fn concat_pointers<T: DeviceRepr, S: Borrow<CudaSlice<T>>>(slices: &[S]) -> CudaSlice<u64> {
    let cuda = cuda_engine();
    memcpy_htod(
        &slices
            .iter()
            .map(|slice_dev| slice_dev.borrow().device_ptr(&cuda.stream).0)
            .collect::<Vec<u64>>(), // TODO avoid hardcoding u64 (this is platform dependent)
    )
}

/// Async
pub fn cuda_alloc<T: DeviceRepr>(size: usize) -> CudaSlice<T> {
    unsafe { cuda_engine().stream.alloc(size).unwrap() }
}

/// Async
pub fn cuda_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(size: usize) -> CudaSlice<T> {
    cuda_engine().stream.alloc_zeros(size).unwrap()
}

/// Async
pub fn cuda_get_at_index<T: DeviceRepr + Default>(slice: &CudaSlice<T>, idx: usize) -> T {
    assert!(idx < slice.len());
    let cuda = cuda_engine();
    let mut dst = [T::default()];
    cuda.stream
        .memcpy_dtoh(&slice.slice(idx..idx + 1), &mut dst)
        .unwrap();
    dst.into_iter().next().unwrap()
}

pub fn cuda_twiddles_two_adicity<F: Field>() -> usize {
    let n = cuda_twiddles::<F>().len();
    assert!((n + 1).is_power_of_two());
    (n + 1).ilog2() as usize - 1
}

pub fn cuda_twiddles<F: Field>() -> CudaSlice<F> {
    unsafe {
        std::mem::transmute(
            cuda_engine()
                .twiddles
                .read()
                .unwrap()
                .get(&TypeId::of::<F>())
                .unwrap_or_else(|| {
                    panic!(
                        "twiddles have not been preprocessed for : {}",
                        std::any::type_name::<F>()
                    )
                })
                .clone(),
        )
    }
}

#[derive(derive_more::Deref, derive_more::DerefMut)]
pub struct CudaCall<'a> {
    _function: &'static CudaFunction,
    #[deref]
    #[deref_mut]
    pub args: LaunchArgs<'a>,
    max_log_threads_per_block: Option<u32>,
    n_ops: u32,
    shared_mem_bytes: u32,
    info: CudaFunctionInfo,
}

impl CudaCall<'_> {
    pub fn new(info: CudaFunctionInfo, n_ops: u32) -> Self {
        let cuda = cuda_engine();
        let guard = cuda.functions.read().unwrap();
        let function = guard
            .get(&info)
            .unwrap_or_else(|| panic!("{info:?} not found"));

        // we never overwite a function so it can be safely statically borrowed (TODO avoid unsafe)
        let function =
            unsafe { std::mem::transmute::<&CudaFunction, &'static CudaFunction>(function) };

        let args = cuda.stream.launch_builder(function);
        Self {
            _function: function,
            args,
            n_ops,
            shared_mem_bytes: 0,
            max_log_threads_per_block: None,
            info,
        }
    }

    pub fn total_n_threads(&self, cooperative: bool) -> u32 {
        let launch_config = self.launch_config(cooperative);
        launch_config.block_dim.0 * launch_config.grid_dim.0
    }

    pub fn shared_mem_bytes(mut self, n: u32) -> Self {
        self.shared_mem_bytes = n;
        self
    }

    pub fn max_log_threads_per_block(mut self, n: u32) -> Self {
        assert!(n <= LOG_MAX_THREADS_PER_BLOCK);
        self.max_log_threads_per_block = Some(n);
        self
    }

    fn launch_config(&self, cooperative: bool) -> LaunchConfig {
        assert!(self.n_ops > 0);
        let max_log_threads = if cooperative {
            LOG_MAX_THREADS_PER_COOPERATIVE_BLOCK // also harcoded in ntt.cu
        } else {
            LOG_MAX_THREADS_PER_BLOCK
        }
        .min(self.max_log_threads_per_block.unwrap_or(u32::MAX));
        let max_blocks = if cooperative {
            MAX_COOPERATIVE_BLOCKS
        } else {
            MAX_BLOCKS
        }; // TODO why only 16 cooperative blocks? Sometimes it works with 32 wtf
        let log_threads = max_log_threads.min(self.n_ops.next_power_of_two().ilog2());
        let blocks = max_blocks.min(self.n_ops.div_ceil(1 << log_threads));
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (1 << log_threads, 1, 1),
            shared_mem_bytes: self.shared_mem_bytes,
        }
    }

    pub fn launch(mut self) {
        unsafe { self.args.launch(self.launch_config(false)) }
            .unwrap_or_else(|e| panic!("{:?} failed with: {}", self.info, e));
    }

    pub fn launch_cooperative(mut self) {
        unsafe { self.args.launch_cooperative(self.launch_config(true)) }
            .unwrap_or_else(|e| panic!("{:?} failed with: {}", self.info, e));
    }
}
