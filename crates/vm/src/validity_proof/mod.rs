use p3_koala_bear::GenericPoseidon2LinearLayersKoalaBear;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants};
use rand::{SeedableRng, rngs::StdRng};

use crate::F;

pub mod prove_execution;
pub mod verify_execution;

type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS_16: usize = 20;
const PARTIAL_ROUNDS_24: usize = 23;

fn build_poseidon_16_air() -> Poseidon2Air<
    F,
    LinearLayers,
    16,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS_16,
> {
    let constants = RoundConstants::<F, 16, HALF_FULL_ROUNDS, PARTIAL_ROUNDS_16>::from_rng(
        &mut StdRng::seed_from_u64(0),
    );

    Poseidon2Air::<
        F,
        LinearLayers,
        16,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS_16,
    >::new(constants.clone())
}

fn build_poseidon_24_air() -> Poseidon2Air<
    F,
    LinearLayers,
    24,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS_24,
> {
    let constants = RoundConstants::<F, 24, HALF_FULL_ROUNDS, PARTIAL_ROUNDS_24>::from_rng(
        &mut StdRng::seed_from_u64(0),
    );
    Poseidon2Air::<
        F,
        LinearLayers,
        24,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS_24,
    >::new(constants.clone())
}
