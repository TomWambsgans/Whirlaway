use crate::{CudaFunctionInfo, cuda_engine, try_get_cuda_engine};
use cudarc::driver::{
    CudaFunction, DevicePtr, DevicePtrMut, LaunchArgs, LaunchConfig, ValidAsZeroBits,
};
use cudarc::driver::{CudaSlice, DeviceRepr};
use utils::log2_up;

// TODO Avoid hardcoding : This is GPU dependent
pub const LOG_MAX_THREADS_PER_BLOCK: usize = 8;

pub const MAX_THREADS_PER_BLOCK: usize = 1 << LOG_MAX_THREADS_PER_BLOCK;

pub const MAX_BLOCKS: usize = 1 << 14;

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
pub fn cuda_alloc<T: DeviceRepr>(size: usize) -> CudaSlice<T> {
    unsafe { cuda_engine().stream.alloc(size).unwrap() }
}

/// Async
pub fn cuda_alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(size: usize) -> CudaSlice<T> {
    cuda_engine().stream.alloc_zeros(size).unwrap()
}

/// Async
pub fn cuda_get_at_index<T: DeviceRepr + Default>(slice: &CudaSlice<T>, idx: usize) -> T {
    let mut dst = [T::default()];
    cuda_engine()
        .stream
        .memcpy_dtoh(&slice.slice(idx..idx + 1), &mut dst)
        .unwrap();
    dst.into_iter().next().unwrap()
}

/// Async
pub fn cuda_set_at_index<T: DeviceRepr>(slice: &mut CudaSlice<T>, idx: usize, value: T) {
    cuda_engine()
        .stream
        .memcpy_htod(&[value], &mut slice.slice_mut(idx..idx + 1))
        .unwrap();
}

#[derive(derive_more::Deref, derive_more::DerefMut)]
pub struct CudaCall<'a> {
    _function: &'static CudaFunction,
    #[deref]
    #[deref_mut]
    pub args: LaunchArgs<'a>,
    max_log_threads_per_block: Option<usize>,
    n_ops: usize,
    shared_mem_bytes: usize,
    info: CudaFunctionInfo,
}

impl CudaCall<'_> {
    pub fn new(info: CudaFunctionInfo, n_ops: usize) -> Self {
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

    pub fn total_n_threads(&self) -> usize {
        let launch_config = self.launch_config();
        (launch_config.block_dim.0 * launch_config.grid_dim.0) as usize
    }

    pub fn shared_mem_bytes(mut self, n: usize) -> Self {
        self.shared_mem_bytes = n;
        self
    }

    pub fn max_log_threads_per_block(mut self, n: usize) -> Self {
        assert!(n <= LOG_MAX_THREADS_PER_BLOCK);
        self.max_log_threads_per_block = Some(n);
        self
    }

    fn launch_config(&self) -> LaunchConfig {
        assert!(self.n_ops > 0);
        let max_log_threads =
            LOG_MAX_THREADS_PER_BLOCK.min(self.max_log_threads_per_block.unwrap_or(usize::MAX));
        let log_threads = max_log_threads.min(log2_up(self.n_ops));
        let blocks = MAX_BLOCKS.min(self.n_ops.div_ceil(1 << log_threads));
        LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (1 << log_threads, 1, 1),
            shared_mem_bytes: self.shared_mem_bytes as u32,
        }
    }

    pub fn launch(mut self) {
        unsafe { self.args.launch(self.launch_config()) }
            .unwrap_or_else(|e| panic!("{:?} failed with: {}", self.info, e));
    }
}
