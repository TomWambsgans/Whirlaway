use std::{marker::PhantomData, time::Instant};

use ::air::table::AirTable;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use p3_util::log2_strict_usize;
use utils::{build_merkle_compress, build_merkle_hash, build_prover_state, build_verifier_state, init_tracing};
use whir_p3::{
    dft::EvalsDft,
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

pub fn compile_and_run(program: &str, public_input: &[F], private_input: &[F]) {
    let bytecode = compile_program(program);
    execute_bytecode(&bytecode, &public_input, private_input);
}

pub fn compile_and_prove_execution(program: &str, public_input: &[F], private_input: &[F]) {
    let bytecode = compile_program(program);

    let time = Instant::now();
    let execution_result = execute_bytecode(&bytecode, &public_input, private_input);
    let mut trace = get_execution_trace(&bytecode, &execution_result);
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

    let preprocessed_columns = trace
        .drain(..N_INSTRUCTION_FIELDS_IN_AIR)
        .collect::<Vec<_>>();

    let table = AirTable::<EF, _>::new(VMAir, log_n_rows, 3, preprocessed_columns);
    table.check_trace_validity(&trace).unwrap();
    
    let time = Instant::now();
    table.prove(&mut prover_state, trace, &pcs, &dft);
    println!("Validity proof took: {:?}", time.elapsed());

    let mut verifier_state = build_verifier_state(&prover_state);

    table.verify(&mut verifier_state, &pcs).unwrap();
}
