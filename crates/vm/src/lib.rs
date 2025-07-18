use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};

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
type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;
#[cfg(test)]
type MyChallenger = p3_challenger::DuplexChallenger<F, Poseidon16, 16, 8>;
#[cfg(test)]
type MerkleHash = p3_symmetric::PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
#[cfg(test)]
type MerkleCompress = p3_symmetric::TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression

const FIELD_ELEMENTS_PER_OPCODE: usize = 13;
