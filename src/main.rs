#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod bench_field;
mod examples;

use examples::poseidon2_koala_bear::prove_poseidon2;

const USE_CUDA: bool = true;

fn main() {
    prove_poseidon2(16, 128, 3, USE_CUDA);
}
