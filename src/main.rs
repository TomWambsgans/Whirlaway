#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod bench_field;
mod examples;

use examples::poseidon2_koala_bear::prove_poseidon2;
use pcs::WhirParameters;

fn main() {
    prove_poseidon2(10, WhirParameters::standard(128, 4));
}
