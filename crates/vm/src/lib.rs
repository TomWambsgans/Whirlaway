use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;

pub mod bytecode;
pub mod compiler;
pub mod lang;
pub mod parser;
pub mod runner;

#[cfg(test)]
mod examples;

const DIMENSION: usize = 8;
type F = KoalaBear;
type EF = BinomialExtensionField<F, DIMENSION>;

const AIR_COLUMNS_PER_OPCODE: usize = 1; // TODO
const PROGRAM_ENDING_ZEROS: usize = 8; // Every program ends with at least 8 zeros, useful for creating an "empty" pointer
