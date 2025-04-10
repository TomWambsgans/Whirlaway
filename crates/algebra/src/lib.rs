#![cfg_attr(not(test), warn(unused_crate_dependencies))]

pub mod ntt;
pub mod pols;
pub mod tensor_algebra;

#[cfg(test)]
mod tests;
