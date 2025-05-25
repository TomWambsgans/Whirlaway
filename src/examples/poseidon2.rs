use ::air::AirSettings;
use air::table::AirTable;
use algebra::Multilinear;
use colored::Colorize;
use fiat_shamir::{FsProver, FsVerifier, get_total_grinding_time, reset_total_grinding_time};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_matrix::Matrix;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::{Duration, Instant};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::parameters::FoldingFactor;

const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;
type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;

#[cfg(test)]
mod tests {

    use whir_p3::parameters::errors::SecurityAssumption;

    use super::*;

    #[test]
    fn test_poseidon2() {
        let settings = AirSettings::new(
            100,
            SecurityAssumption::CapacityBound,
            FoldingFactor::Constant(4),
            2,
            3,
            1,
        );
        prove_poseidon2_koala_bear(7, settings.clone(), false);
    }
}

#[derive(Clone, Debug)]
pub struct Poseidon2Benchmark {
    pub log_n_rows: usize,
    pub settings: AirSettings,
    pub prover_time: Duration,
    pub verifier_time: Duration,
    pub proof_size: usize,
    pub total_grinding_time: Duration,
}

impl ToString for Poseidon2Benchmark {
    fn to_string(&self) -> String {
        let mut res = String::new();
        res += &format!(
            "Security level: {} bits ({:?}), starting rate: 1/{}, folding factor: {}\n",
            self.settings.security_bits,
            self.settings.whir_soudness_type,
            1 << self.settings.whir_log_inv_rate,
            match self.settings.whir_folding_factor {
                FoldingFactor::Constant(f) => format!("{}", f),
                FoldingFactor::ConstantFromSecondRound(first, then) =>
                    format!("1st: {} then {}", first, then),
            }
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

        res += &format!(
            "\nTotal grinding time: {:.3} s\n",
            self.total_grinding_time.as_millis() as f64 / 1000.0
        )
        .blue()
        .to_string();

        res
    }
}

pub fn prove_poseidon2_koala_bear(
    log_n_rows: usize,
    settings: AirSettings,
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
    reset_total_grinding_time();

    let n_rows = 1 << log_n_rows;

    let rng = &mut StdRng::seed_from_u64(0);
    let constants = RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(rng);
    let poseidon_air = Poseidon2Air::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >::new(constants.clone());

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
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(inputs, &constants, 0)
    .transpose();

    let witness = witness_matrix
        .rows()
        .map(|col| Multilinear::new(col.collect()))
        .collect::<Vec<_>>();

    let table = AirTable::<'_, F, EF, _>::new(
        poseidon_air,
        log_n_rows,
        settings.univariate_skips,
        vec![],
        3,
    );
    // println!("Constraints degree: {}", table.constraint_degree());
    // table.check_validity(&witness);

    let t = Instant::now();
    let mut fs_prover = FsProver::new();
    let whir_proof = table.prove(&settings, &mut fs_prover, witness);
    let proof_size = fs_prover.transcript_len() + whir_proof.narg_string().len();

    let prover_time = t.elapsed();
    let time = Instant::now();
    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    table
        .verify(&settings, &mut fs_verifier, log_n_rows, whir_proof)
        .unwrap();
    let verifier_time = time.elapsed();

    Poseidon2Benchmark {
        log_n_rows,
        settings,
        prover_time,
        verifier_time,
        proof_size,
        total_grinding_time: get_total_grinding_time(),
    }
}
