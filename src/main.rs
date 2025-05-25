#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod examples;

use air::AirSettings;
use examples::poseidon2::prove_poseidon2_koala_bear;
use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};

const SECURITY_BITS: usize = 100; // (temporary)

fn main() {
    for (log_n_rows, log_inv_rate) in [(15, 1)] {
        let benchmark = prove_poseidon2_koala_bear(
            log_n_rows,
            AirSettings::new(
                SECURITY_BITS,
                SecurityAssumption::CapacityBound,
                FoldingFactor::ConstantFromSecondRound(6, 4),
                log_inv_rate,
                2,
                3, // TODO
            ),
            true,
        );
        println!("\n{}", benchmark.to_string());
    }
}
