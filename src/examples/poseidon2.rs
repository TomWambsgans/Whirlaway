use ::air::AirSettings;
use air::table::AirTable;
use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_challenger::HashChallenger;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField, extension::BinomialExtensionField};
use p3_keccak::Keccak256Hash;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_matrix::Matrix;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows, num_cols};
use p3_uni_stark::SymbolicExpression;
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
};
use std::fmt;
use std::time::{Duration, Instant};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator, parameters::FoldingFactor,
    poly::evals::EvaluationsList,
};

type MyChallenger = HashChallenger<u8, Keccak256Hash, 32>;

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
        SupportedField::KoalaBear.prove_poseidon2_with(7, settings.clone(), 3, false);
        SupportedField::BabyBear.prove_poseidon2_with(7, settings.clone(), 8, false);
    }
}

#[derive(Clone, Debug)]
pub struct Poseidon2Benchmark {
    pub log_n_rows: usize,
    pub settings: AirSettings,
    pub prover_time: Duration,
    pub verifier_time: Duration,
    pub proof_size: usize,
}

impl fmt::Display for Poseidon2Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Security level: {} bits ({:?}), starting rate: 1/{}, folding factor: {}",
            self.settings.security_bits,
            self.settings.whir_soudness_type,
            1 << self.settings.whir_log_inv_rate,
            match self.settings.whir_folding_factor {
                FoldingFactor::Constant(factor) => format!("{factor}"),
                FoldingFactor::ConstantFromSecondRound(first, then) =>
                    format!("1st: {first} then {then}"),
            }
        )?;
        let n_rows = 1 << self.log_n_rows;
        writeln!(
            f,
            "Proved {} poseidon2 hashes in {:.3} s ({} / s)",
            n_rows,
            self.prover_time.as_millis() as f64 / 1000.0,
            (n_rows as f64 / self.prover_time.as_secs_f64()).round() as usize
        )?;
        writeln!(f, "Proof size: {:.1} KiB", self.proof_size as f64 / 1024.0)?;
        writeln!(f, "Verification: {} ms", self.verifier_time.as_millis())
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum SupportedField {
    KoalaBear,
    BabyBear,
}

impl SupportedField {
    pub fn prove_poseidon2_with(
        &self,
        log_n_rows: usize,
        settings: AirSettings,
        n_preprocessed_columns: usize,
        display_logs: bool,
    ) -> Poseidon2Benchmark {
        match self {
            SupportedField::KoalaBear => prove_poseidon2_koala_bear(
                log_n_rows,
                settings,
                n_preprocessed_columns,
                display_logs,
            ),
            SupportedField::BabyBear => prove_poseidon2_baby_bear(
                log_n_rows,
                settings,
                n_preprocessed_columns,
                display_logs,
            ),
        }
    }
}

fn prove_poseidon2_koala_bear(
    log_n_rows: usize,
    settings: AirSettings,
    n_preprocessed_columns: usize,
    display_logs: bool,
) -> Poseidon2Benchmark {
    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;

    const WIDTH: usize = 16;
    const SBOX_DEGREE: u64 = 3;
    const SBOX_REGISTERS: usize = 0;
    const HALF_FULL_ROUNDS: usize = 4;
    const PARTIAL_ROUNDS: usize = 20;
    const COLS: usize =
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

    prove_poseidon2::<
        F,
        EF,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        COLS,
    >(log_n_rows, settings, n_preprocessed_columns, display_logs)
}

fn prove_poseidon2_baby_bear(
    log_n_rows: usize,
    settings: AirSettings,
    n_preprocessed_columns: usize,
    display_logs: bool,
) -> Poseidon2Benchmark {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type LinearLayers = GenericPoseidon2LinearLayersBabyBear;

    const WIDTH: usize = 16;
    const SBOX_DEGREE: u64 = 7;
    const SBOX_REGISTERS: usize = 1;
    const HALF_FULL_ROUNDS: usize = 4;
    const PARTIAL_ROUNDS: usize = 13;
    const COLS: usize =
        num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

    prove_poseidon2::<
        F,
        EF,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        COLS,
    >(log_n_rows, settings, n_preprocessed_columns, display_logs)
}

fn prove_poseidon2<
    F,
    EF,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const COLS: usize,
>(
    log_n_rows: usize,
    settings: AirSettings,
    n_preprocessed_columns: usize,
    display_logs: bool,
) -> Poseidon2Benchmark
where
    StandardUniform: Distribution<F>,
    EF: ExtensionField<F> + TwoAdicField,
    F: TwoAdicField + PrimeField64,
    StandardUniform: Distribution<EF>,
    LinearLayers: GenericPoseidon2LinearLayers<F, WIDTH>
        + GenericPoseidon2LinearLayers<EF, WIDTH>
        + GenericPoseidon2LinearLayers<SymbolicExpression<F>, WIDTH>,
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

    let inputs: Vec<[F; WIDTH]> = (0..n_rows)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();

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

    let mut witness = witness_matrix
        .rows()
        .map(|col| EvaluationsList::new(col.collect()))
        .collect::<Vec<_>>();
    let preprocessed_columns = witness.drain(..n_preprocessed_columns).collect::<Vec<_>>();

    let table = AirTable::<F, EF, _>::new(
        poseidon_air,
        log_n_rows,
        settings.univariate_skips,
        preprocessed_columns,
        3,
    );
    // println!("Constraints degree: {}", table.constraint_degree());
    // table.check_validity(&witness);

    let t = Instant::now();

    let whir_params = table.build_whir_params(&settings);
    let mut domainsep: DomainSeparator<EF, F, u8> = DomainSeparator::new("🐎", false);
    domainsep.commit_statement(&whir_params);
    domainsep.add_whir_proof(&whir_params);
    let mut prover_state = domainsep.to_prover_state(MyChallenger::new(vec![], Keccak256Hash));

    table.prove(&settings, &mut prover_state, witness);
    let proof_size = prover_state.narg_string().len();

    let prover_time = t.elapsed();
    let time = Instant::now();

    let mut verifier_state = domainsep.to_verifier_state(
        prover_state.narg_string(),
        MyChallenger::new(vec![], Keccak256Hash),
    );

    table
        .verify(&settings, &mut verifier_state, log_n_rows)
        .unwrap();
    let verifier_time = time.elapsed();

    Poseidon2Benchmark {
        log_n_rows,
        settings,
        prover_time,
        verifier_time,
        proof_size,
    }
}
