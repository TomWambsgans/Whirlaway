use p3_koala_bear::Poseidon2KoalaBear;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    compiler::compile_to_low_level_bytecode, parser::parse_program, runner::execute_bytecode,
};

#[test]
fn compile_aggregate_program() {
    let program_str = include_str!("aggregate.vm");
    let parsed_program = parse_program(program_str).unwrap();
    let compiled = compile_to_low_level_bytecode(parsed_program).unwrap();
    println!("Compiled Program:\n\n{}", compiled.to_string());

    let mut rng =  StdRng::seed_from_u64(0);
    let poseidon_16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut rng);
    let poseidon_24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut rng);

    let public_input = vec![];

    execute_bytecode(&compiled, &public_input, poseidon_16, poseidon_24);
}
