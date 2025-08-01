use ::air::AirSettings;
use air::table::AirTable;
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeField64};
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear, Poseidon2KoalaBear};
use p3_matrix::Matrix;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_strict_usize;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::fmt;
use std::marker::PhantomData;
use std::time::{Duration, Instant};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use utils::PF;
use whir_p3::dft::EvalsDft;
use whir_p3::fiat_shamir::prover::ProverState;
use whir_p3::fiat_shamir::verifier::VerifierState;
use whir_p3::whir::config::{FoldingFactor, SecurityAssumption, WhirConfigBuilder};

// Koalabear
type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

// Koalabear
type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;
type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

// BabyBear
// type F = BabyBear;
// type EF = BinomialExtensionField<F, 4>;
// type LinearLayers = GenericPoseidon2LinearLayersBabyBear;
// const SBOX_DEGREE: u64 = 7;
// const SBOX_REGISTERS: usize = 1;
// const HALF_FULL_ROUNDS: usize = 4;
// const PARTIAL_ROUNDS: usize = 13;

const WIDTH: usize = 16;

#[derive(Clone, Debug)]
pub struct Poseidon2Benchmark {
    pub log_n_rows: usize,
    pub prover_time: Duration,
    pub verifier_time: Duration,
    pub proof_size: f64, // in bytes
}

impl fmt::Display for Poseidon2Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n_rows = 1 << self.log_n_rows;
        writeln!(
            f,
            "Proved {} poseidon2 hashes in {:.3} s ({} / s)",
            n_rows,
            self.prover_time.as_millis() as f64 / 1000.0,
            (n_rows as f64 / self.prover_time.as_secs_f64()).round() as usize
        )?;
        writeln!(f, "Proof size: {:.1} KiB", self.proof_size / 1024.0)?;
        writeln!(f, "Verification: {} ms", self.verifier_time.as_millis())
    }
}

pub fn prove_poseidon2(
    log_n_rows: usize,
    settings: AirSettings,
    folding_factor: FoldingFactor,
    log_inv_rate: usize,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    security_level: usize,
    rs_domain_initial_reduction_factor: usize,
    max_num_variables_to_send_coeffs: usize,
    n_preprocessed_columns: usize,
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

    let mut rng = StdRng::seed_from_u64(0);
    let constants =
        RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(&mut rng);

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
        .map(|col| col.collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let preprocessed_columns = witness.drain(..n_preprocessed_columns).collect::<Vec<_>>();

    let table = AirTable::<EF, _>::new(
        poseidon_air,
        log_n_rows,
        settings.univariate_skips,
        preprocessed_columns,
        3,
    );

    let poseidon16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let poseidon24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let merkle_hash = MerkleHash::new(poseidon24);

    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    let t = Instant::now();

    let challenger = MyChallenger::new(poseidon16);

    let mut prover_state = ProverState::new(challenger.clone());

    let pcs = WhirConfigBuilder {
        folding_factor,
        soundness_type,
        merkle_hash,
        merkle_compress,
        pow_bits,
        max_num_variables_to_send_coeffs,
        rs_domain_initial_reduction_factor,
        security_level,
        starting_log_inv_rate: log_inv_rate,
        base_field: PhantomData::<PF<EF>>,
        extension_field: PhantomData::<EF>,
    };

    let ext_dim = <EF as BasedVectorSpace<PF<EF>>>::DIMENSION;
    let dft = EvalsDft::new(
        1 << (table.log_n_witness_columns() + log_n_rows + log_inv_rate
            - log2_strict_usize(ext_dim)),
    );

    table.prove(&settings, &mut prover_state, witness, &pcs, &dft);
    // let proof_size = prover_state.narg_string().len();

    let prover_time = t.elapsed();
    let time = Instant::now();

    let mut verifier_state = VerifierState::new(prover_state.proof_data().to_vec(), challenger);

    table
        .verify(&settings, &mut verifier_state, log_n_rows, &pcs)
        .unwrap();
    let verifier_time = time.elapsed();

    let proof_size = prover_state.proof_data().len() as f64 * (F::ORDER_U64 as f64).log2() / 8.0;

    Poseidon2Benchmark {
        log_n_rows,
        prover_time,
        verifier_time,
        proof_size,
    }
}

#[cfg(test)]
mod tests {
    use whir_p3::whir::config::{FoldingFactor, SecurityAssumption};

    use super::*;

    #[test]
    fn test_prove_poseidon2() {
        let benchmark = prove_poseidon2(
            13,
            AirSettings::new(5),
            FoldingFactor::ConstantFromSecondRound(5, 3),
            2,
            SecurityAssumption::CapacityBound,
            13,
            128,
            1,
            5,
            7,
            false,
        );
        println!("\n{benchmark}");
    }
}
