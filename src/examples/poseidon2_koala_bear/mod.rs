use ::air::{AirBuilder, AirExpr};
use algebra::pols::MultilinearHost;
use arithmetic_circuit::ArithmeticCircuit;
use cuda_engine::{
    SumcheckComputation, cuda_init, cuda_preprocess_all_twiddles,
    cuda_preprocess_sumcheck_computation,
};
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_matrix::Matrix;
use pcs::{PCS, RingSwitch, WhirPCS, WhirParameters};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{
    borrow::Borrow,
    time::{Duration, Instant},
};
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

#[cfg(test)]
mod tests {
    use super::*;
    use whir::parameters::SoundnessType;

    #[test]
    fn test_poseidon2() {
        prove_poseidon2(
            4,
            WhirParameters::standard(SoundnessType::ProvableList, 100, 2, false),
            false,
        );
    }
}

#[derive(Clone)]
pub struct Poseidon2Benchmark {
    pub log_n_rows: usize,
    pub whir_params: WhirParameters,
    pub prover_time: Duration,
    pub verifier_time: Duration,
    pub proof_size: usize,
}

impl ToString for Poseidon2Benchmark {
    fn to_string(&self) -> String {
        let mut res = String::new();
        res += &format!(
            "Security level: {} bits ({}), starting rate: 1/{}\n",
            self.whir_params.security_level,
            self.whir_params.soundness_type,
            1 << self.whir_params.starting_log_inv_rate
        );
        let n_rows = 1 << self.log_n_rows;
        res += &format!(
            "Proved {} poseidon2 hashes in {:.3} s ({} / s)\n",
            n_rows,
            self.prover_time.as_millis() as f64 / 1000.0,
            (n_rows as f64 / self.prover_time.as_secs_f64()).round() as usize
        );
        res += &format!("Proof size: {:.1} KiB\n", self.proof_size as f64 / 1024.0);
        res += &format!("Verification: {} ms\n", self.verifier_time.as_millis());
        res
    }
}

pub fn prove_poseidon2(
    log_n_rows: usize,
    whir_params: WhirParameters,
    display_logs: bool,
) -> Poseidon2Benchmark {
    if display_logs {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }

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
        .map(|col| MultilinearHost::new(col.collect()))
        .collect::<Vec<_>>();

    let table = air_builder.build();
    // println!("Constraints degree: {}", table.constraint_degree());
    // table.check_validity(&witness);

    if whir_params.cuda {
        let constraint_sumcheck_computations = SumcheckComputation::<F> {
            exprs: &table.constraints,
            n_multilinears: table.n_columns * 2 + 1,
            eq_mle_multiplier: true,
        };
        let prod_sumcheck = SumcheckComputation::<F> {
            exprs: &[
                (ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(1)).fix_computation(false)
            ],
            n_multilinears: 2,
            eq_mle_multiplier: false,
        };
        let inner_air_sumcheck = SumcheckComputation::<F> {
            exprs: &[(ArithmeticCircuit::Node(4)
                * ((ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(2))
                    + (ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(3))))
            .fix_computation(false)],
            n_multilinears: 5,
            eq_mle_multiplier: false,
        };
        cuda_init();
        cuda_preprocess_sumcheck_computation(&constraint_sumcheck_computations);
        cuda_preprocess_sumcheck_computation(&prod_sumcheck);
        cuda_preprocess_sumcheck_computation(&inner_air_sumcheck);
        cuda_preprocess_all_twiddles::<F>(whir_params.folding_factor.as_constant().unwrap()); // TODO handle ConstantFromSecondRound
    }

    let pcs = RingSwitch::<F, EF, WhirPCS<F, EF>>::new(
        log_n_rows + table.log_n_witness_columns(),
        &whir_params,
    );

    let t = Instant::now();
    let mut fs_prover = FsProver::new();
    table.prove(&mut fs_prover, &pcs, witness, whir_params.cuda);
    let proof_size = fs_prover.transcript_len();

    let prover_time = t.elapsed();
    let time = Instant::now();
    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    table.verify(&mut fs_verifier, &pcs, log_n_rows).unwrap();
    let verifier_time = time.elapsed();

    Poseidon2Benchmark {
        log_n_rows,
        whir_params,
        prover_time,
        verifier_time,
        proof_size,
    }
}
