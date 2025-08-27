use std::marker::PhantomData;

use pcs::WhirBatchPcs;
use utils::*;
use vm::{EF, F};
use whir_p3::whir::config::{FoldingFactor, SecurityAssumption, WhirConfigBuilder};
pub(crate) fn build_batch_pcs()
-> WhirBatchPcs<F, EF, EF, MyMerkleHash, MyMerkleCompress, MY_DIGEST_ELEMS> {
    let base_pcs = WhirConfigBuilder {
        folding_factor: FoldingFactor::ConstantFromSecondRound(7, 4),
        soundness_type: SecurityAssumption::CapacityBound,
        merkle_hash: build_merkle_hash(),
        merkle_compress: build_merkle_compress(),
        pow_bits: 16,
        max_num_variables_to_send_coeffs: 6,
        rs_domain_initial_reduction_factor: 5,
        security_level: 128,
        starting_log_inv_rate: 1,
        base_field: PhantomData::<F>,
        extension_field: PhantomData::<EF>,
    };

    let extension_pcs = WhirConfigBuilder {
        folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
        soundness_type: SecurityAssumption::CapacityBound,
        merkle_hash: build_merkle_hash(),
        merkle_compress: build_merkle_compress(),
        pow_bits: 16,
        max_num_variables_to_send_coeffs: 6,
        rs_domain_initial_reduction_factor: 2,
        security_level: 128,
        starting_log_inv_rate: 1,
        base_field: PhantomData::<EF>,
        extension_field: PhantomData::<EF>,
    };

    WhirBatchPcs(base_pcs, extension_pcs)
}
