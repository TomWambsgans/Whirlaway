#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod bench_field;
mod examples;

use examples::poseidon2_koala_bear::prove_poseidon2;
use pcs::WhirParameters;

fn main() {
    cuda_bindings::init_cuda().unwrap();
    prove_poseidon2(14, WhirParameters::standard(128, 4, false));
}
