use ::air::{AirBuilder, AirSettings, ConstraintVariable};
use algebra::pols::MultilinearHost;
use arithmetic_circuit::ArithmeticCircuit;
use colored::Colorize;
use fiat_shamir::{FsProver, FsVerifier, get_total_grinding_time, reset_total_grinding_time};
use generation::generate_vectorized_trace_rows;
use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_field::{
    ExtensionField, PrimeCharacteristicRing, PrimeField32, TwoAdicField,
    extension::BinomialExtensionField,
};
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_matrix::Matrix;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
};
use std::time::{Duration, Instant};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use utils::{MyExtensionField, SupportedField};
use vectorized::{VectorizedPoseidon2Air, write_vectorized_constraints};
use whir::parameters::FoldingFactor;
use {columns::num_cols, constants::RoundConstants};

pub(crate) mod air;
pub(crate) mod columns;
pub(crate) mod constants;
pub(crate) mod generation;
pub(crate) mod vectorized;

#[cfg(test)]
mod tests {
    use whir::parameters::SoundnessType;

    use super::*;

    #[test]
    fn test_poseidon2() {
        let settings = AirSettings::new(
            100,
            SoundnessType::ProvableList,
            FoldingFactor::Constant(4),
            2,
            3,
        );
        prove_poseidon2_baby_bear(7, false, settings.clone(), false, false);
        prove_poseidon2_koala_bear(7, false, settings.clone(), false, false);
    }
}

#[derive(Clone, Debug)]
pub struct Poseidon2Benchmark {
    pub log_n_rows: usize,
    pub vector_len: usize,
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
        let n_poseidon2 = n_rows * self.vector_len;
        res += &format!(
            "Proved {} poseidon2 hashes in {:.3} s ({} / s)\n",
            n_poseidon2,
            self.prover_time.as_millis() as f64 / 1000.0,
            (n_poseidon2 as f64 / self.prover_time.as_secs_f64()).round() as usize
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

pub fn prove_poseidon2_with(
    field: SupportedField,
    log_n_rows: usize,
    vectorized: bool,
    settings: AirSettings,
    cuda: bool,
    display_logs: bool,
) -> Poseidon2Benchmark {
    match field {
        SupportedField::KoalaBear => {
            prove_poseidon2_koala_bear(log_n_rows, vectorized, settings, cuda, display_logs)
        }
        SupportedField::BabyBear => {
            prove_poseidon2_baby_bear(log_n_rows, vectorized, settings, cuda, display_logs)
        }
    }
}

fn prove_poseidon2_koala_bear(
    log_n_rows: usize,
    vectorized: bool,
    settings: AirSettings,
    cuda: bool,
    display_logs: bool,
) -> Poseidon2Benchmark {
    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type WhirF = BinomialExtensionField<F, 8>;
    type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;

    const WIDTH: usize = 16;
    const SBOX_DEGREE: u64 = 3;
    const SBOX_REGISTERS: usize = 0;
    const HALF_FULL_ROUNDS: usize = 4;
    const PARTIAL_ROUNDS: usize = 20;
    if vectorized {
        const VECTOR_LEN: usize = 6;
        const COLS: usize = VECTOR_LEN
            * num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

        prove_poseidon2::<
            F,
            EF,
            WhirF,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            COLS,
            VECTOR_LEN,
        >(log_n_rows, settings, cuda, display_logs)
    } else {
        const COLS: usize =
            num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
        prove_poseidon2::<
            F,
            EF,
            WhirF,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            COLS,
            1,
        >(log_n_rows, settings, cuda, display_logs)
    }
}

fn prove_poseidon2_baby_bear(
    log_n_rows: usize,
    vectorized: bool,
    settings: AirSettings,
    cuda: bool,
    display_logs: bool,
) -> Poseidon2Benchmark {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type WhirF = BinomialExtensionField<F, 8>;
    type LinearLayers = GenericPoseidon2LinearLayersBabyBear;

    const WIDTH: usize = 16;
    const SBOX_DEGREE: u64 = 7;
    const SBOX_REGISTERS: usize = 1;
    const HALF_FULL_ROUNDS: usize = 4;
    const PARTIAL_ROUNDS: usize = 13;

    if vectorized {
        const VECTOR_LEN: usize = 3;
        const COLS: usize = VECTOR_LEN
            * num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

        prove_poseidon2::<
            F,
            EF,
            WhirF,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            COLS,
            VECTOR_LEN,
        >(log_n_rows, settings, cuda, display_logs)
    } else {
        const COLS: usize =
            num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
        prove_poseidon2::<
            F,
            EF,
            WhirF,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            COLS,
            1,
        >(log_n_rows, settings, cuda, display_logs)
    }
}

fn prove_poseidon2<
    F: TwoAdicField + PrimeField32,
    EF: ExtensionField<F>,
    WhirF: ExtensionField<F>
        + MyExtensionField<EF>
        + ExtensionField<<WhirF as PrimeCharacteristicRing>::PrimeSubfield>
        + TwoAdicField
        + Ord,
    LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>
        + GenericPoseidon2LinearLayers<ArithmeticCircuit<F, ConstraintVariable>, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const COLS: usize,
    const VECTOR_LEN: usize,
>(
    log_n_rows: usize,
    settings: AirSettings,
    cuda: bool,
    display_logs: bool,
) -> Poseidon2Benchmark
where
    StandardUniform: Distribution<F>,
    <WhirF as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
    EF: ExtensionField<<WhirF as PrimeCharacteristicRing>::PrimeSubfield>,
{
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
    let poseidon_air = VectorizedPoseidon2Air::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >::new(constants.clone());

    let mut air_builder = AirBuilder::<F, COLS>::new(log_n_rows);

    write_vectorized_constraints(&poseidon_air, &mut air_builder);

    let inputs = (0..n_rows * VECTOR_LEN)
        .map(|_| {
            (0..WIDTH)
                .map(|_| rng.random())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let witness_matrix = generate_vectorized_trace_rows::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >(inputs, &constants)
    .transpose();

    let witness = witness_matrix
        .rows()
        .map(|col| MultilinearHost::new(col.collect()))
        .collect::<Vec<_>>();

    let table = air_builder.build(settings.univariate_skips);
    // println!("Constraints degree: {}", table.constraint_degree());
    // table.check_validity(&witness);

    if cuda {
        table.cuda_setup::<EF, WhirF>(&settings);
    }

    let t = Instant::now();
    let mut fs_prover = FsProver::new(cuda);
    table.prove::<EF, WhirF>(&settings, &mut fs_prover, witness, cuda);
    let proof_size = fs_prover.transcript_len();

    let prover_time = t.elapsed();
    let time = Instant::now();
    let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
    table
        .verify::<EF, WhirF>(&settings, &mut fs_verifier, log_n_rows)
        .unwrap();
    let verifier_time = time.elapsed();

    Poseidon2Benchmark {
        log_n_rows,
        vector_len: VECTOR_LEN,
        settings,
        prover_time,
        verifier_time,
        proof_size,
        total_grinding_time: get_total_grinding_time(),
    }
}
