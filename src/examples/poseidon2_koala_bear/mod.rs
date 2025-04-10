use ::air::{AirBuilder, AirExpr};
use algebra::pols::MultilinearPolynomial;
use cuda_bindings::SumcheckComputation;
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_matrix::Matrix;
use pcs::{PCS, RingSwitch, WhirPCS, WhirParameters};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{borrow::Borrow, time::Instant};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use {
    air::{Poseidon2Air, write_constraints},
    columns::{Poseidon2Cols, num_cols},
    constants::RoundConstants,
    generation::generate_trace_rows,
};

mod air;
mod columns;
mod constants;
mod generation;

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
    prove_poseidon2(4, 100, 2, false);
}

pub fn prove_poseidon2(log_n_rows: usize, security_bits: usize, log_inv_rate: usize, cuda: bool) {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let whir_params = WhirParameters::standard(security_bits, log_inv_rate, cuda);
    let n_rows = 1 << log_n_rows;

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

    let mut air_builder = AirBuilder::<F, COLS>::new(log_n_rows);
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
    // println!("Constraints degree: {}", table.constraint_degree());
    // table.check_validity(&witness);

    if cuda {
        let sumcheck_computations = SumcheckComputation {
            inner: table.constraints.clone(),
            n_multilinears: table.n_columns * 2 + 1,
            eq_mle_multiplier: true,
        };
        cuda_bindings::init(
            &[sumcheck_computations],
            whir_params.folding_factor.as_constant().unwrap(), // TODO handle ConstantFromSecondRound
        );
    }

    let pcs = RingSwitch::<F, EF, WhirPCS<F, EF>>::new(
        log_n_rows + table.log_n_witness_columns(),
        &whir_params,
    );

    let t = Instant::now();
    let mut fs_prover = FsProver::new();
    table.prove(&mut fs_prover, &pcs, &witness, cuda);
    let proof_size = fs_prover.transcript_len();

    let prover_time = t.elapsed();
    let time = Instant::now();

    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    table.verify(&mut fs_verifier, &pcs, log_n_rows).unwrap();
    let verifier_time = time.elapsed();

    println!();
    println!(
        "Security level: {} bits ({})",
        whir_params.security_level, whir_params.soundness_type
    );
    println!(
        "Proved {} poseidon2 hashes in {:?} ({} / s)",
        n_rows,
        prover_time,
        (n_rows as f64 / prover_time.as_secs_f64()).round() as usize
    );
    println!("Proof size: {:.1} KiB", proof_size as f64 / 1024.0);
    println!("Verification: {:?}", verifier_time);
}
