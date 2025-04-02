#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(test)]
mod bench_field;
mod examples;

use examples::poseidon2_koala_bear::prove_poseidon2;
use p3_koala_bear::KoalaBear;
use pcs::WhirParameters;

const USE_CUDA: bool = true;

fn main() {
    if USE_CUDA {
        cuda_bindings::init_cuda::<KoalaBear>();
    }
    prove_poseidon2(13, WhirParameters::standard(128, 3, USE_CUDA));
}
