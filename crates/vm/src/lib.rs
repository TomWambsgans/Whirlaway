use std::{marker::PhantomData, time::Instant};

use ::air::{table::AirTable, witness::AirWitness};
use multi_pcs::pcs::PCS;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use utils::{
    PF, build_merkle_compress, build_merkle_hash, build_prover_state, build_verifier_state,
    init_tracing, padd_with_zero_to_next_power_of_two,
};
use whir_p3::{
    dft::EvalsDft,
    poly::evals::EvaluationsList,
    whir::config::{FoldingFactor, SecurityAssumption, WhirConfigBuilder},
};

use crate::{
    air::VMAir, compiler::compile_program, precompiles::PRECOMPILES, runner::execute_bytecode,
    tracer::get_execution_trace,
};

pub mod air;
pub mod bytecode;
pub mod compiler;
pub mod instruction_encoder;
pub mod lang;
pub mod parser;
pub mod precompiles;
pub mod recursion;
pub mod runner;
pub mod tracer;

#[cfg(test)]
mod test;

const DIMENSION: usize = 8;
type F = KoalaBear;
type EF = BinomialExtensionField<F, DIMENSION>;

pub const N_AIR_COLUMNS: usize = 19;
pub const N_INSTRUCTION_FIELDS: usize = 15;
pub const N_INSTRUCTION_FIELDS_IN_AIR: usize = N_INSTRUCTION_FIELDS - PRECOMPILES.len();
pub const N_MEMORY_VALUE_COLUMNS: usize = 3; // virtual (lookup into memory, with logup*)
pub const N_COMMITTED_COLUMNS: usize = 5;

pub fn compile_and_run(program: &str, public_input: &[F], private_input: &[F]) {
    let bytecode = compile_program(program);
    execute_bytecode(&bytecode, &public_input, private_input);
}

pub fn compile_and_prove_execution(program: &str, public_input: &[F], private_input: &[F]) {
    let bytecode = compile_program(program);

    let time = Instant::now();
    let execution_result = execute_bytecode(&bytecode, &public_input, private_input);
    let trace = get_execution_trace(&bytecode, &execution_result);
    println!("Witness generation took: {:?}", time.elapsed());

    let log_n_rows = log2_strict_usize(trace[0].len());
    assert!(trace.iter().all(|col| col.len() == (1 << log_n_rows)));
    let mut prover_state = build_prover_state::<EF>();

    let pcs = WhirConfigBuilder {
        folding_factor: FoldingFactor::ConstantFromSecondRound(7, 4),
        soundness_type: SecurityAssumption::CapacityBound,
        merkle_hash: build_merkle_hash(),
        merkle_compress: build_merkle_compress(),
        pow_bits: 16,
        max_num_variables_to_send_coeffs: 6,
        rs_domain_initial_reduction_factor: 5,
        security_level: 128,
        starting_log_inv_rate: 1,
        base_field: PhantomData::<F>,
        extension_field: PhantomData::<EF>,
    };
    let dft = EvalsDft::default();

    let column_groups = vec![
        0..N_INSTRUCTION_FIELDS_IN_AIR,
        N_INSTRUCTION_FIELDS_IN_AIR..N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS,
        N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS
            ..N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS + N_COMMITTED_COLUMNS,
    ];
    let witness = AirWitness::<PF<EF>>::new(&trace, &column_groups);
    let table = AirTable::<EF, _>::new(VMAir, 4);
    table.check_trace_validity(&witness).unwrap();

    let time = Instant::now();
    // 1) Commit
    let commited_trace_polynomial = padd_with_zero_to_next_power_of_two(
        &trace[N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS..].concat(),
    );
    let pcs_witness = pcs.commit(&dft, &mut prover_state, &commited_trace_polynomial);

    // 2) PIOP
    let evaluations_remaining_to_prove = table.prove(&mut prover_state, witness);

    // 3) Open
    pcs.open(
        &dft,
        &mut prover_state,
        &[evaluations_remaining_to_prove[2].clone()],
        pcs_witness,
        &commited_trace_polynomial,
    );
    println!("Validity proof took: {:?}", time.elapsed());

    let mut verifier_state = build_verifier_state(&prover_state);

    let parsed_commitment = pcs
        .parse_commitment(
            &mut verifier_state,
            log_n_rows + log2_ceil_usize(N_COMMITTED_COLUMNS),
        )
        .unwrap();
    let evaluations_remaining_to_verify = table
        .verify(&mut verifier_state, log_n_rows, &column_groups)
        .unwrap();

    assert_eq!(
        padd_with_zero_to_next_power_of_two(&trace[..N_INSTRUCTION_FIELDS_IN_AIR].concat())
            .evaluate(&evaluations_remaining_to_verify[0].point),
        evaluations_remaining_to_verify[0].value
    );
    assert_eq!(
        padd_with_zero_to_next_power_of_two(
            &trace[N_INSTRUCTION_FIELDS_IN_AIR..N_INSTRUCTION_FIELDS_IN_AIR + 3].concat()
        )
        .evaluate(&evaluations_remaining_to_verify[1].point),
        evaluations_remaining_to_verify[1].value
    );
    pcs.verify(
        &mut verifier_state,
        &parsed_commitment,
        &[evaluations_remaining_to_verify[2].clone()],
    )
    .unwrap();
}
