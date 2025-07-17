#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod prove;
pub mod table;
mod uni_skip_utils;
mod utils;
mod verify;

const WHIR_POW_BITS: usize = 16;

use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};

#[derive(Clone, Debug)]
pub struct AirSettings {
    pub security_bits: usize,
    pub whir_soudness_type: SecurityAssumption,
    pub whir_folding_factor: FoldingFactor,
    pub whir_log_inv_rate: usize,
    pub univariate_skips: usize,
    pub whir_initial_domain_reduction_factor: usize,
}

impl AirSettings {
    pub const fn new(
        security_bits: usize,
        whir_soudness_type: SecurityAssumption,
        whir_folding_factor: FoldingFactor,
        whir_log_inv_rate: usize,
        univariate_skips: usize,
        whir_initial_domain_reduction_factor: usize,
    ) -> Self {
        Self {
            security_bits,
            whir_soudness_type,
            whir_folding_factor,
            whir_log_inv_rate,
            univariate_skips,
            whir_initial_domain_reduction_factor,
        }
    }
}
