use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;

use crate::{
    compiler::compile_program,
    precompiles::PRECOMPILES,
    runner::{ExecutionResult, execute_bytecode},
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

pub fn compile_and_run(
    program: &str,
    public_input: &[F],
    private_input: &[F],
) -> (Vec<Vec<F>>, ExecutionResult) {
    let bytecode = compile_program(program);
    let execution_result = execute_bytecode(&bytecode, &public_input, private_input);
    let trace = get_execution_trace(&bytecode, &execution_result);
    (trace, execution_result)
}
