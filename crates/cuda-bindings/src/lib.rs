#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod keccak;
pub use keccak::*;

mod ntt;
pub use ntt::*;

mod sumcheck;
pub use sumcheck::*;

mod multilinear;
pub use multilinear::*;

pub use cudarc::driver::{CudaSlice, DeviceRepr};

// Should not be too big to avoid CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
const MAX_LOG_N_BLOCKS: u32 = 8;
const MAX_LOG_N_COOPERATIVE_BLOCKS: u32 = 5;
