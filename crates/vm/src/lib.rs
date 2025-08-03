use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use rand::{SeedableRng as _, rngs::StdRng};
use utils::{Poseidon16, Poseidon24};

use crate::{compiler::compile_program, runner::execute_bytecode};

pub mod air;
pub mod bytecode;
pub mod compiler;
pub mod lang;
pub mod parser;
pub mod precompiles;
pub mod recursion;
pub mod runner;

#[cfg(test)]
mod test;

const DIMENSION: usize = 8;
type F = KoalaBear;
type EF = BinomialExtensionField<F, DIMENSION>;

pub fn compile_and_run(program: &str, public_input: &[F], private_input: &[F]) {
    let bytecode = compile_program(program);

    let poseidon_16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let poseidon_24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(0));

    execute_bytecode(
        &bytecode,
        &public_input,
        private_input,
        &poseidon_16,
        &poseidon_24,
    );
}
