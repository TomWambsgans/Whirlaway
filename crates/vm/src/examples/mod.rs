use p3_koala_bear::Poseidon2KoalaBear;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    F, compiler::compile_to_low_level_bytecode, parser::parse_program, runner::execute_bytecode,
};

fn compile_and_run(program: &str, public_input: Vec<F>) {
    let parsed_program = parse_program(program).unwrap();
    let compiled = compile_to_low_level_bytecode(parsed_program).unwrap();
    println!("Compiled Program:\n\n{}", compiled.to_string());

    let mut rng = StdRng::seed_from_u64(0);
    let poseidon_16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut rng);
    let poseidon_24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut rng);

    execute_bytecode(&compiled, &public_input, poseidon_16, poseidon_24);
}

#[test]
fn tes_aggregate_program() {
    let program_str = include_str!("aggregate.vm");
    compile_and_run(program_str, vec![]);
}


#[test]
fn test_mini_program_0() {
    let program = r#"
    fn main() {
        for i in 0..5 {
            for j in i..10 {
                print(i, j);
            }
        }
        return;
    }
   "#;
    compile_and_run(program, vec![]);
}

#[test]
fn test_mini_program_1() {
    let program = r#"
    fn main() {
        for i in 0..5 {
            j = i - 10;
            for k in j..10 {
                if k == i {
                    print(i);
                } else {
                    func(i, k);
                }
            }
        }
        return;
    }

    fn func(a, b) {
        for i in a..b {
            print(i);
        }
    }
   "#;
    compile_and_run(program, vec![]);
}