use ::air::AirSettings;
use air::table::AirTable;
use algebra::Multilinear;
use fiat_shamir::{FsProver, FsVerifier, get_total_grinding_time, reset_total_grinding_time};
use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_field::{
    ExtensionField, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
    extension::BinomialExtensionField,
};
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
use whir_p3::parameters::FoldingFactor;

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
        prove_poseidon2_with(SupportedField::KoalaBear, 7, settings.clone(), false);
        prove_poseidon2_with(SupportedField::BabyBear, 7, settings.clone(), false);
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
        writeln!(f, "Verification: {} ms", self.verifier_time.as_millis())?;
        writeln!(
            f,
            "\nTotal grinding time: {:.3} s",
            self.total_grinding_time.as_millis() as f64 / 1000.0
        )
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum SupportedField {
    KoalaBear,
    BabyBear,
}

pub fn prove_poseidon2_with(
    field: SupportedField,
    log_n_rows: usize,
    settings: AirSettings,
    display_logs: bool,
) -> Poseidon2Benchmark {
    match field {
        SupportedField::KoalaBear => prove_poseidon2_koala_bear(log_n_rows, settings, display_logs),
        SupportedField::BabyBear => prove_poseidon2_baby_bear(log_n_rows, settings, display_logs),
    }
}

fn prove_poseidon2_koala_bear(
    log_n_rows: usize,
    settings: AirSettings,
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
    >(log_n_rows, settings, display_logs)
}

fn prove_poseidon2_baby_bear(
    log_n_rows: usize,
    settings: AirSettings,
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
    >(log_n_rows, settings, display_logs)
}

// pub struct GenericPoseidon2LinearLayersGoldilcoks;

// impl<A: Algebra<Goldilocks>> GenericPoseidon2LinearLayers<A, 8>
//     for GenericPoseidon2LinearLayersGoldilcoks
// {
//     fn internal_linear_layer(state: &mut [A; 8]) {
//         Poseidon2InternalLayerGoldilocks::new_from_constants(
//             new_goldilocks_array(HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS).to_vec(),
//         )
//         .permute_state(state);
//     }
// }

// const fn new_goldilocks_array<const N: usize>(input: [u64; N]) -> [Goldilocks; N] {
//     let mut output = [Goldilocks::ZERO; N];
//     let mut i = 0;
//     while i < N {
//         output[i] = unsafe { std::mem::transmute::<u64, Goldilocks>(input[i]) };
//         i += 1;
//     }
//     output
// }

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
    display_logs: bool,
) -> Poseidon2Benchmark
where
    StandardUniform: Distribution<F>,
    EF: ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield>
        + ExtensionField<F>
        + TwoAdicField
        + Ord,
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield> + TwoAdicField + PrimeField64,
    <F as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
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
