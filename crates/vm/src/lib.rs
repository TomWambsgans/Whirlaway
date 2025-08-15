use std::ops::Range;

use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;

use crate::{compiler::compile_program, precompiles::PRECOMPILES, runner::execute_bytecode};

pub mod air;
pub mod bytecode;
pub mod compiler;
mod dot_product_air;
pub mod instruction_encoder;
pub mod lang;
pub mod parser;
pub mod precompiles;
pub mod recursion;
pub mod runner;
mod validity_proof;
pub use validity_proof::*;

#[cfg(test)]
mod test;

const UNIVARIATE_SKIPS: usize = 4;

const DIMENSION: usize = 8;
type F = KoalaBear;
type EF = BinomialExtensionField<F, DIMENSION>;

const N_INSTRUCTION_COLUMNS: usize = 15;
const N_COMMITTED_EXEC_COLUMNS: usize = 5;
const N_MEMORY_VALUE_COLUMNS: usize = 3; // virtual (lookup into memory, with logup*)
const N_EXEC_COLUMNS: usize = N_COMMITTED_EXEC_COLUMNS + N_MEMORY_VALUE_COLUMNS;
const N_INSTRUCTION_COLUMNS_IN_AIR: usize = N_INSTRUCTION_COLUMNS - PRECOMPILES.len();
const N_EXEC_AIR_COLUMNS: usize = N_INSTRUCTION_COLUMNS_IN_AIR + N_EXEC_COLUMNS;
const N_TOTAL_COLUMNS: usize = N_INSTRUCTION_COLUMNS + N_EXEC_COLUMNS;

// Instruction columns
const COL_INDEX_OPERAND_A: usize = 0;
const COL_INDEX_OPERAND_B: usize = 1;
const COL_INDEX_OPERAND_C: usize = 2;
const COL_INDEX_FLAG_A: usize = 3;
const COL_INDEX_FLAG_B: usize = 4;
const COL_INDEX_FLAG_C: usize = 5;
const COL_INDEX_ADD: usize = 6;
const COL_INDEX_MUL: usize = 7;
const COL_INDEX_DEREF: usize = 8;
const COL_INDEX_JUZ: usize = 9;
const COL_INDEX_AUX: usize = 10;
const COL_INDEX_POSEIDON_16: usize = 11;
const COL_INDEX_POSEIDON_24: usize = 12;
const COL_INDEX_DOT_PRODUCT: usize = 13;
const COL_INDEX_MULTILINEAR_EVAL: usize = 14;

// Execution columns
const COL_INDEX_MEM_VALUE_A: usize = 15; // virtual with logup*
const COL_INDEX_MEM_VALUE_B: usize = 16; // virtual with logup*
const COL_INDEX_MEM_VALUE_C: usize = 17; // virtual with logup*
const COL_INDEX_PC: usize = 18;
const COL_INDEX_FP: usize = 19;
const COL_INDEX_MEM_ADDRESS_A: usize = 20;
const COL_INDEX_MEM_ADDRESS_B: usize = 21;
const COL_INDEX_MEM_ADDRESS_C: usize = 22;

const ZERO_VEC_PTR: usize = 0; // convention (vectorized pointer of size 1, pointing to 8 zeros)
const POSEIDON_16_NULL_HASH_PTR: usize = 2; // convention (vectorized pointer of size 2, = the 16 elements of poseidon_16(0))
const POSEIDON_24_NULL_HASH_PTR: usize = 4; // convention (vectorized pointer of size 1, = the last 8 elements of poseidon_24(0))
const PUBLIC_INPUT_START: usize = 5 * 8; // normal pointer

fn exec_column_groups() -> Vec<Range<usize>> {
    [
        (0..N_INSTRUCTION_COLUMNS_IN_AIR)
            .map(|i| i..i + 1)
            .collect::<Vec<_>>(),
        vec![
            N_INSTRUCTION_COLUMNS_IN_AIR..N_INSTRUCTION_COLUMNS_IN_AIR + N_MEMORY_VALUE_COLUMNS,
            N_INSTRUCTION_COLUMNS_IN_AIR + N_MEMORY_VALUE_COLUMNS
                ..N_INSTRUCTION_COLUMNS_IN_AIR + N_MEMORY_VALUE_COLUMNS + 1, // pc
            N_INSTRUCTION_COLUMNS_IN_AIR + N_MEMORY_VALUE_COLUMNS + 1
                ..N_INSTRUCTION_COLUMNS_IN_AIR + N_MEMORY_VALUE_COLUMNS + 2, // fp
            N_INSTRUCTION_COLUMNS_IN_AIR + N_MEMORY_VALUE_COLUMNS + 2
                ..N_INSTRUCTION_COLUMNS_IN_AIR + N_MEMORY_VALUE_COLUMNS + N_COMMITTED_EXEC_COLUMNS,
        ],
    ]
    .concat()
}

pub fn compile_and_run(program: &str, public_input: &[F], private_input: &[F]) {
    let bytecode = compile_program(program);
    execute_bytecode(&bytecode, &public_input, private_input);
}

pub trait InAirColumnIndex {
    fn index_in_air(self) -> usize;
}

impl InAirColumnIndex for usize {
    fn index_in_air(self) -> usize {
        if self < N_INSTRUCTION_COLUMNS_IN_AIR {
            self
        } else {
            assert!(self >= N_INSTRUCTION_COLUMNS);
            assert!(self < N_INSTRUCTION_COLUMNS + N_EXEC_COLUMNS);
            self - PRECOMPILES.len()
        }
    }
}
