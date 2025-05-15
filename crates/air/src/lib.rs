#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod builder;
mod prove;
mod setup;
mod table;
mod uni_skip_utils;
mod utils;
mod verify;

#[cfg(test)]
mod test;

pub use builder::*;
use whir::parameters::{FoldingFactor, SoundnessType};

#[derive(Clone, Debug)]
pub struct AirSettings {
    pub security_bits: usize,
    pub whir_soudness_type: SoundnessType,
    pub whir_folding_factor: FoldingFactor,
    pub whir_log_inv_rate: usize,
    pub univariate_skips: usize,
}

impl AirSettings {
    pub fn new(
        security_bits: usize,
        whir_soudness_type: SoundnessType,
        whir_folding_factor: FoldingFactor,
        whir_log_inv_rate: usize,
        univariate_skips: usize,
    ) -> Self {
        Self {
            security_bits,
            whir_soudness_type,
            whir_folding_factor,
            whir_log_inv_rate,
            univariate_skips,
        }
    }
}
