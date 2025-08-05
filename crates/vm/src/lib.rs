use std::ops::Range;

use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;

use crate::{compiler::compile_program, precompiles::PRECOMPILES, runner::execute_bytecode};

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
mod validity_proof;
pub use validity_proof::*;

#[cfg(test)]
mod test;

const UNIVARIATE_SKIPS: usize = 4;

const DIMENSION: usize = 8;
type F = KoalaBear;
type EF = BinomialExtensionField<F, DIMENSION>;

const N_AIR_COLUMNS: usize = 19;
const N_INSTRUCTION_FIELDS: usize = 15;
const N_INSTRUCTION_FIELDS_IN_AIR: usize = N_INSTRUCTION_FIELDS - PRECOMPILES.len();
const N_MEMORY_VALUE_COLUMNS: usize = 3; // virtual (lookup into memory, with logup*)
const N_COMMITTED_EXEC_COLUMNS: usize = 5;
const COL_INDEX_MEM_ADDRESS_A: usize = 16;
const COL_INDEX_MEM_ADDRESS_C: usize = 18;
const ZERO_VEC_PTR: usize = 0; // convention (vectorized pointer of size 1, pointing to 8 zeros)
const POSEIDON_16_NULL_HASH_PTR: usize = 1; // convention (vectorized pointer of size 2, = the 16 elements of poseidon_16(0))
const POSEIDON_24_NULL_HASH_PTR: usize = 3; // convention (vectorized pointer of size 1, = the last 8 elements of poseidon_24(0))
const PUBLIC_INPUT_START: usize = 4 * 8; // normal pointer

const COLUMN_GROUPS_EXEC: [Range<usize>; 3] = [
    0..N_INSTRUCTION_FIELDS_IN_AIR,
    N_INSTRUCTION_FIELDS_IN_AIR..N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS,
    N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS
        ..N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS + N_COMMITTED_EXEC_COLUMNS,
];

pub fn compile_and_run(program: &str, public_input: &[F], private_input: &[F]) {
    let bytecode = compile_program(program);
    execute_bytecode(&bytecode, &public_input, private_input);
}
