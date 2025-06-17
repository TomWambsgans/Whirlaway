#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod examples;

use air::AirSettings;
use examples::poseidon2::SupportedField;
use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};

const SECURITY_BITS: usize = 100; // (temporary)

fn main() {
    let (log_n_rows, log_inv_rate) = (17, 1);
    let benchmark = SupportedField::KoalaBear.prove_poseidon2_with(
        log_n_rows,
        AirSettings::new(
            SECURITY_BITS,
            SecurityAssumption::CapacityBound,
            FoldingFactor::ConstantFromSecondRound(7, 4),
            log_inv_rate,
            2,
            4,
        ),
        0,
        true,
    );
    println!("\n{benchmark}");
}
