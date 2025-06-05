#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod prove;
pub mod table;
mod uni_skip_utils;
mod utils;
mod verify;

#[cfg(test)]
mod test;

type ByteHash = Blake3;
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
const WHIR_POW_BITS: usize = 16;

use p3_blake3::Blake3;
use p3_keccak::KeccakF;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use whir_p3::{
    fiat_shamir::duplex_sponge::DuplexSponge,
    parameters::{FoldingFactor, errors::SecurityAssumption},
};

type MyPerm = KeccakF;
type MyU = u8;
const MY_PERM_WIDTH: usize = 200;
type MySponge = DuplexSponge<MyU, MyPerm, MY_PERM_WIDTH, 136>;

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
