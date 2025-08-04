use air::table::AirTable;
use air::witness::AirWitness;
use multi_pcs::pcs::PCS;
use p3_air::BaseAir;
use p3_field::PrimeField64;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows};
use p3_util::log2_ceil_usize;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::fmt;
use std::marker::PhantomData;
use std::time::{Duration, Instant};
use utils::{
    build_merkle_compress, build_merkle_hash, build_prover_state, build_verifier_state,
    init_tracing, padd_with_zero_to_next_power_of_two,
};
use whir_p3::dft::EvalsDft;
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::whir::config::{FoldingFactor, SecurityAssumption, WhirConfigBuilder};

const EXTENSION_DEGREE: usize = 8;
type F = KoalaBear;
type EF = BinomialExtensionField<F, EXTENSION_DEGREE>;
type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

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
    univariate_skips: usize,
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
        init_tracing();
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

    let n_commited_columns = poseidon_air.width() - n_preprocessed_columns;

    let witness_columns = (0..poseidon_air.width())
        .map(|row| witness_matrix.values[row * n_rows..(row + 1) * n_rows].to_vec())
        .collect::<Vec<_>>();
    let column_groups = vec![
        0..n_preprocessed_columns,
        n_preprocessed_columns..poseidon_air.width(),
    ];
    let witness = AirWitness::new(&witness_columns, &column_groups);

    let table = AirTable::<EF, _>::new(poseidon_air, univariate_skips);

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
        1 << (log2_ceil_usize(n_commited_columns) + log_n_rows + log_inv_rate
            - pcs.folding_factor.at_round(0)),
    );

    let commited_trace_polynomial =
        padd_with_zero_to_next_power_of_two(&witness_columns[n_preprocessed_columns..].concat());
    let pcs_witness = pcs.commit(&dft, &mut prover_state, &commited_trace_polynomial);
    let evaluations_remaining_to_prove = table.prove(&mut prover_state, witness);
    pcs.open(
        &dft,
        &mut prover_state,
        &[evaluations_remaining_to_prove[1].clone()],
        pcs_witness,
        &commited_trace_polynomial,
    );

    let prover_time = t.elapsed();
    let time = Instant::now();

    let mut verifier_state = build_verifier_state(&prover_state);
    let parsed_commitment = pcs
        .parse_commitment(
            &mut verifier_state,
            log_n_rows + log2_ceil_usize(n_commited_columns),
        )
        .unwrap();
    let evaluations_remaining_to_verify = table
        .verify(&mut verifier_state, log_n_rows, &column_groups)
        .unwrap();
    assert_eq!(
        padd_with_zero_to_next_power_of_two(&witness_columns[..n_preprocessed_columns].concat())
            .evaluate(&evaluations_remaining_to_verify[0].point),
        evaluations_remaining_to_verify[0].value
    );
    pcs.verify(
        &mut verifier_state,
        &parsed_commitment,
        &[evaluations_remaining_to_verify[1].clone()],
    )
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
            4,
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
