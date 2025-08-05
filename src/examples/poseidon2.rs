use air::table::AirTable;
use air::witness::AirWitness;
use p3_air::BaseAir;
use p3_field::PrimeField64;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows};
use p3_symmetric::Permutation;
use p3_util::log2_ceil_usize;
use pcs::{multi_commit, multi_open, parse_multi_commitment, verify_multi_commitment};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::fmt;
use std::marker::PhantomData;
use std::time::{Duration, Instant};
use utils::{
    HALF_FULL_ROUNDS_16, HALF_FULL_ROUNDS_24, PARTIAL_ROUNDS_16, PARTIAL_ROUNDS_24,
    build_merkle_compress, build_merkle_hash, build_poseidon16, build_poseidon24,
    build_prover_state, build_verifier_state, init_tracing, padd_with_zero_to_next_power_of_two,
};
use whir_p3::dft::EvalsDft;
use whir_p3::whir::config::{FoldingFactor, SecurityAssumption, WhirConfigBuilder};

const EXTENSION_DEGREE: usize = 8;
type F = KoalaBear;
type EF = BinomialExtensionField<F, EXTENSION_DEGREE>;
type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;

#[derive(Clone, Debug)]
pub struct Poseidon2Benchmark {
    pub log_n_poseidons_16: usize,
    pub log_n_poseidons_24: usize,
    pub prover_time: Duration,
    pub verifier_time: Duration,
    pub proof_size: f64, // in bytes
}

impl fmt::Display for Poseidon2Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Proved {} poseidon2-16 and {} poseidon2-24 in {:.3} s ({} / s)",
            1 << self.log_n_poseidons_16,
            1 << self.log_n_poseidons_24,
            self.prover_time.as_millis() as f64 / 1000.0,
            (((1 << self.log_n_poseidons_16) + (1 << self.log_n_poseidons_24)) as f64
                / self.prover_time.as_secs_f64())
            .round() as usize
        )?;
        writeln!(f, "Proof size: {:.1} KiB", self.proof_size / 1024.0)?;
        writeln!(f, "Verification: {} ms", self.verifier_time.as_millis())
    }
}

pub fn prove_poseidon2(
    log_n_poseidons_16: usize,
    log_n_poseidons_24: usize,
    univariate_skips: usize,
    folding_factor: FoldingFactor,
    log_inv_rate: usize,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    security_level: usize,
    rs_domain_initial_reduction_factor: usize,
    max_num_variables_to_send_coeffs: usize,
    display_logs: bool,
) -> Poseidon2Benchmark {
    if display_logs {
        init_tracing();
    }

    let n_poseidons_16 = 1 << log_n_poseidons_16;
    let n_poseidons_24 = 1 << log_n_poseidons_24;

    let constants_16 = RoundConstants::<F, 16, HALF_FULL_ROUNDS_16, PARTIAL_ROUNDS_16>::from_rng(
        &mut StdRng::seed_from_u64(0),
    );
    let constants_24 = RoundConstants::<F, 24, HALF_FULL_ROUNDS_24, PARTIAL_ROUNDS_24>::from_rng(
        &mut StdRng::seed_from_u64(0),
    );

    let poseidon_air_16 = Poseidon2Air::<
        F,
        LinearLayers,
        16,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS_16,
        PARTIAL_ROUNDS_16,
    >::new(constants_16.clone());
    let poseidon_air_24 = Poseidon2Air::<
        F,
        LinearLayers,
        24,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS_24,
        PARTIAL_ROUNDS_24,
    >::new(constants_24.clone());
    let n_columns_16 = poseidon_air_16.width();
    let n_columns_24 = poseidon_air_24.width();
    let log_table_area_16 = log_n_poseidons_16 + log2_ceil_usize(n_columns_16);
    let log_table_area_24 = log_n_poseidons_24 + log2_ceil_usize(n_columns_24);

    let mut rng = StdRng::seed_from_u64(0);
    let inputs_16: Vec<[F; 16]> = (0..n_poseidons_16).map(|_| Default::default()).collect();
    let inputs_24: Vec<[F; 24]> = (0..n_poseidons_24)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();

    let witness_matrix_16 = generate_trace_rows::<
        F,
        LinearLayers,
        16,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS_16,
        PARTIAL_ROUNDS_16,
    >(inputs_16, &constants_16, 0);

    let witness_matrix_24 = generate_trace_rows::<
        F,
        LinearLayers,
        24,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS_24,
        PARTIAL_ROUNDS_24,
    >(inputs_24, &constants_24, 0);

    assert_eq!(
        &witness_matrix_16.values[n_columns_16 - 16..n_columns_16],
        build_poseidon16().permute(witness_matrix_16.values[1..17].try_into().unwrap())
    );
    assert_eq!(
        &witness_matrix_24.values[n_columns_24 - 24..n_columns_24],
        build_poseidon24().permute(witness_matrix_24.values[1..25].try_into().unwrap())
    );

    let witness_matrix_16_transposed = witness_matrix_16.transpose();
    let witness_matrix_24_transposed = witness_matrix_24.transpose();

    let witness_columns_16 = (0..n_columns_16)
        .map(|row| {
            witness_matrix_16_transposed.values[row * n_poseidons_16..(row + 1) * n_poseidons_16]
                .to_vec()
        })
        .collect::<Vec<_>>();
    let witness_columns_24 = (0..n_columns_24)
        .map(|row| {
            witness_matrix_24_transposed.values[row * n_poseidons_24..(row + 1) * n_poseidons_24]
                .to_vec()
        })
        .collect::<Vec<_>>();
    let column_groups_16 = vec![0..n_columns_16];
    let column_groups_24 = vec![0..n_columns_24];
    let witness_16 = AirWitness::new(&witness_columns_16, &column_groups_16);
    let witness_24 = AirWitness::new(&witness_columns_24, &column_groups_24);

    let table_16 = AirTable::<EF, _>::new(poseidon_air_16, univariate_skips);
    let table_24 = AirTable::<EF, _>::new(poseidon_air_24, univariate_skips);

    let t = Instant::now();

    let mut prover_state = build_prover_state();

    let pcs = WhirConfigBuilder {
        folding_factor,
        soundness_type,
        merkle_hash: build_merkle_hash(),
        merkle_compress: build_merkle_compress(),
        pow_bits,
        max_num_variables_to_send_coeffs,
        rs_domain_initial_reduction_factor,
        security_level,
        starting_log_inv_rate: log_inv_rate,
        base_field: PhantomData::<F>,
        extension_field: PhantomData::<EF>,
    };

    // let pcs = RingSwitching::<F, EF, _, EXTENSION_DEGREE>::new(pcs);
    let dft = EvalsDft::new(
        1 << (log2_ceil_usize(n_columns_24)
            + log_n_poseidons_16.max(log_n_poseidons_24)
            + log_inv_rate
            - pcs.folding_factor.at_round(0)),
    );

    let commited_trace_polynomial_16 =
        padd_with_zero_to_next_power_of_two(&witness_columns_16.concat());
    let commited_trace_polynomial_24 =
        padd_with_zero_to_next_power_of_two(&witness_columns_24.concat());

    let commitment_witness = multi_commit(
        &pcs,
        &[commited_trace_polynomial_16, commited_trace_polynomial_24],
        &dft,
        &mut prover_state,
    );

    let evaluations_remaining_to_prove_16 = table_16.prove(&mut prover_state, witness_16);
    let evaluations_remaining_to_prove_24 = table_24.prove(&mut prover_state, witness_24);

    multi_open(
        &pcs,
        &dft,
        &mut prover_state,
        commitment_witness,
        &[
            evaluations_remaining_to_prove_16,
            evaluations_remaining_to_prove_24,
        ],
    );

    let prover_time = t.elapsed();
    let time = Instant::now();

    let mut verifier_state = build_verifier_state(&prover_state);

    let multi_parsed_commitment = parse_multi_commitment(
        &pcs,
        &mut verifier_state,
        vec![log_table_area_16, log_table_area_24],
    )
    .unwrap();

    let evaluations_remaining_to_verify_16 = table_16
        .verify(&mut verifier_state, log_n_poseidons_16, &column_groups_16)
        .unwrap();
    let evaluations_remaining_to_verify_24 = table_24
        .verify(&mut verifier_state, log_n_poseidons_24, &column_groups_24)
        .unwrap();

    verify_multi_commitment(
        &pcs,
        &mut verifier_state,
        &multi_parsed_commitment,
        &[
            evaluations_remaining_to_verify_16,
            evaluations_remaining_to_verify_24,
        ],
    )
    .unwrap();

    let verifier_time = time.elapsed();

    let proof_size = prover_state.proof_data().len() as f64 * (F::ORDER_U64 as f64).log2() / 8.0;

    Poseidon2Benchmark {
        log_n_poseidons_16,
        log_n_poseidons_24,
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
            12,
            4,
            FoldingFactor::ConstantFromSecondRound(5, 3),
            2,
            SecurityAssumption::CapacityBound,
            13,
            128,
            1,
            5,
            false,
        );
        println!("\n{benchmark}");
    }
}
