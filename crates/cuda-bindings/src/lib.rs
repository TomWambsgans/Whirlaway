#![cfg_attr(not(test), warn(unused_crate_dependencies))]

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

// Should not be too big to avoid CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
const MAX_LOG_N_BLOCKS: u32 = 5;
