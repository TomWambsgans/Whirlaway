#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod examples;

use examples::poseidon2_koala_bear::prove_poseidon2;
use pcs::WhirParameters;
use whir::parameters::SoundnessType;

const USE_CUDA: bool = true;
const SECURITY_BITS: usize = 128;

fn main() {
    for (log_n_rows, log_inv_rate) in [(15, 4), (16, 3), (17, 2), (17, 1)] {
        let benchmark = prove_poseidon2(
            log_n_rows,
            WhirParameters::standard(
                SoundnessType::ProvableList,
                SECURITY_BITS,
                log_inv_rate,
                USE_CUDA,
            ),
            false,
        );
        println!("{}", benchmark.to_string());
    }
}
