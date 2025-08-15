use hashsig::{
    inc_encoding::target_sum::*,
    signature::{SignatureScheme, generalized_xmss::*},
    symmetric::{
        message_hash::top_level_poseidon::*, prf::shake_to_field::*, tweak_hash::poseidon::*,
    },
};
use rand::Rng;

const LOG_LIFETIME: usize = 12;

const DIMENSION: usize = 64;
const BASE: usize = 8;
const FINAL_LAYER: usize = 77;
const TARGET_SUM: usize = 375;

const PARAMETER_LEN: usize = 5;
const TWEAK_LEN_FE: usize = 2;
const MSG_LEN_FE: usize = 9;
const RAND_LEN_FE: usize = 6;
const HASH_LEN_FE: usize = 7;

const CAPACITY: usize = 9;

const POS_OUTPUT_LEN_PER_INV_FE: usize = 15;
const POS_INVOCATIONS: usize = 1;
const POS_OUTPUT_LEN_FE: usize = POS_OUTPUT_LEN_PER_INV_FE * POS_INVOCATIONS;

type MH = TopLevelPoseidonMessageHash<
    POS_OUTPUT_LEN_PER_INV_FE,
    POS_INVOCATIONS,
    POS_OUTPUT_LEN_FE,
    DIMENSION,
    BASE,
    FINAL_LAYER,
    TWEAK_LEN_FE,
    MSG_LEN_FE,
    PARAMETER_LEN,
    RAND_LEN_FE,
>;
type TH = PoseidonTweakHash<PARAMETER_LEN, HASH_LEN_FE, TWEAK_LEN_FE, CAPACITY, DIMENSION>;
type PRF = ShakePRFtoF<HASH_LEN_FE>;
type IE = TargetSumEncoding<MH, TARGET_SUM>;

pub type T = GeneralizedXMSSSignatureScheme<PRF, IE, TH, LOG_LIFETIME>;

#[test]
pub fn test_real_xmms() {
    let mut rng = rand::rng();
    let epoch = 15;
    let activation_epoch = 10;
    let num_active_epochs = 12;

    // Generate a key pair
    let (pk, sk) = T::key_gen(&mut rng, activation_epoch, num_active_epochs);

    // Sample random test message
    let message = rng.random();

    // Sign the message
    let signature = T::sign(&mut rng, &sk, epoch, &message);

    // Ensure signing was successful
    assert!(
        signature.is_ok(),
        "Signing failed: {:?}. Epoch was {:?}",
        signature.err(),
        epoch
    );

    // Verify the signature
    let signature = signature.unwrap();
    let is_valid = T::verify(&pk, epoch, &message, &signature);
    assert!(
        is_valid,
        "Signature verification failed. . Epoch was {:?}",
        epoch
    );
}
