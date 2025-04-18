#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod bench_field;
mod examples;

use examples::poseidon2_koala_bear::prove_poseidon2;
use pcs::WhirParameters;
use whir::parameters::SoundnessType;

const USE_CUDA: bool = true;
const SECURITY_BITS: usize = 128;

fn main() {
    let benchmark = prove_poseidon2(
        17,
        WhirParameters::standard(SoundnessType::ProvableList, SECURITY_BITS, 2, USE_CUDA),
        true,
    );
    println!("\n{}", benchmark.to_string());
}

#[test]
fn benchmark() {
    for soundness_type in [SoundnessType::ProvableList, SoundnessType::ConjectureList] {
        for (log_n_rows, log_inv_rate) in [(15, 4), (16, 3), (17, 2)] {
            let benchmark = prove_poseidon2(
                log_n_rows,
                WhirParameters::standard(soundness_type, SECURITY_BITS, log_inv_rate, USE_CUDA),
                false,
            );
            println!("{}", benchmark.to_string());
        }
    }
}
