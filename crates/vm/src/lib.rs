use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;

pub mod bytecode;
pub mod compiler;
pub mod lang;
pub mod parser;
pub mod runner;

#[cfg(test)]
mod test;

const DIMENSION: usize = 8;
type F = KoalaBear;
type EF = BinomialExtensionField<F, DIMENSION>;

const AIR_COLUMNS_PER_OPCODE: usize = 1; // TODO
