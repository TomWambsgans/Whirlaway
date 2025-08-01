#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod prove;
pub mod table;
mod uni_skip_utils;
mod utils;
mod verify;

#[derive(Clone, Debug)]
pub struct AirSettings {
    pub univariate_skips: usize,
}

impl AirSettings {
    pub const fn new(univariate_skips: usize) -> Self {
        Self { univariate_skips }
    }
}
