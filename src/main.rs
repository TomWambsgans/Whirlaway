#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod examples;

use air::AirSettings;
use examples::poseidon2::prove_poseidon2_with;
use utils::SupportedField;
use whir::parameters::{FoldingFactor, SoundnessType};

const USE_CUDA: bool = true;
const SECURITY_BITS: usize = 128;

fn main() {
    for (log_n_rows, log_inv_rate) in [(17, 3)] {
        let benchmark = prove_poseidon2_with(
            SupportedField::KoalaBear,
            log_n_rows,
            false,
            AirSettings::new(
                SECURITY_BITS,
                SoundnessType::ProvableList,
                FoldingFactor::Constant(4),
                log_inv_rate,
                4,
            ),
            USE_CUDA,
            true,
        );
        println!("\n{}", benchmark.to_string());
    }
}
