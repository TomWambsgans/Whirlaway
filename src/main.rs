#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod examples;

use air::AirSettings;
use examples::poseidon2::{SupportedField, prove_poseidon2_with};
use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};

const SECURITY_BITS: usize = 128;

fn main() {
    let (log_n_rows, log_inv_rate) = (18, 1);
    let benchmark = prove_poseidon2_with(
        SupportedField::KoalaBear,
        log_n_rows,
        AirSettings::new(
            SECURITY_BITS,
            SecurityAssumption::CapacityBound,
            FoldingFactor::ConstantFromSecondRound(6, 4),
            log_inv_rate,
            3,
            4,
        ),
        true,
    );
    println!("\n{benchmark}");
}
