mod air;
mod columns;
mod constants;
mod generation;

use ::air::{AirBuilder, AirExpr};
use algebra::pols::MultilinearPolynomial;
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_matrix::Matrix;
use pcs::{BatchSettings, RingSwitch, WhirPCS, WhirParameters};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};

use std::{borrow::Borrow, time::Instant};

use {
    air::{Poseidon2Air, write_constraints},
    columns::{Poseidon2Cols, num_cols},
    constants::RoundConstants,
    generation::generate_trace_rows,
};

const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;
const COLS: usize =
    num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

type F = KoalaBear;
type EF = BinomialExtensionField<KoalaBear, 8>;

#[test]
fn test_poseidon2() {
    prove_poseidon2(4, WhirParameters::standard(100, 2, false));
}

pub fn prove_poseidon2(log_n_rows: usize, whir_params: WhirParameters) {
    let n_rows = 1 << log_n_rows;
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let t = Instant::now();

    let rng = &mut StdRng::seed_from_u64(0);
    let constants = RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(rng);
    let poseidon_air = Poseidon2Air::<
        F,
        GenericPoseidon2LinearLayersKoalaBear,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >::new(constants.clone());

    let mut air_builder = AirBuilder::<F, COLS>::new();
    let (up, down) = air_builder.vars();

    type Columns = Poseidon2Cols<
        AirExpr<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >;
    let columns_up = Borrow::<Columns>::borrow(&up[..]);
    let _columns_down = Borrow::<Columns>::borrow(&down[..]);

    write_constraints(&poseidon_air, &mut air_builder, &columns_up);

    let inputs = (0..n_rows)
        .map(|_| {
            (0..WIDTH)
                .map(|_| rng.random())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let witness_matrix = generate_trace_rows::<
        F,
        GenericPoseidon2LinearLayersKoalaBear,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(inputs, &constants)
    .transpose();

    let witness = witness_matrix
        .rows()
        .map(|col| MultilinearPolynomial::new(col.collect()))
        .collect::<Vec<_>>();

    let table = air_builder.build();
    // table.check_validity(&witness);

    let batch =
        BatchSettings::<F, EF, RingSwitch<F, EF, WhirPCS<EF>>>::new(COLS, log_n_rows, &whir_params);

    let mut batch_prover = batch.clone();

    let mut fs_prover = FsProver::new();
    let batch_witness = batch_prover.commit(&mut fs_prover, witness);
    table.prove(&mut fs_prover, &mut batch_prover, &batch_witness.polys);
    batch_prover.prove(batch_witness, &mut fs_prover);
    let proof_size = fs_prover.transcript_len();

    let prover_time = t.elapsed();
    let time = Instant::now();

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    let mut batch_verifier = batch.clone();
    let commitment = batch_verifier.parse_commitment(&mut fs_verifier).unwrap();
    table
        .verify(&mut fs_verifier, &mut batch_verifier, log_n_rows)
        .unwrap();
    batch_verifier
        .verify(&mut fs_verifier, &commitment)
        .unwrap();
    let verifier_time = time.elapsed();

    println!();
    println!(
        "Security level: {} bits ({})",
        whir_params.security_level, whir_params.soundness_type
    );
    println!("Proved {} poseidon2 hashes in {:?}", n_rows, prover_time);
    println!("Proof size: {:.1} KiB", proof_size as f64 / 1024.0);
    println!("Verification: {:?}", verifier_time);
}
