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
fn test_fibonacci_program() {
    // a program to check the value of the 30th Fibonacci number (832040)
    let program = r#"
    fn main() {
        fibonacci(0, 1, 0, 30);
        return;
    }

    fn fibonacci(a, b, i, n) {
        if i == n {
            print(a);
            return;
        }
        new_a = b;
        new_b = a + b;
        new_i = i + 1;
        fibonacci(new_a, new_b, new_i, n);
        return;
    }
   "#;
    compile_and_run(program, vec![]);
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
    const N = 10;

    fn main() {
        arr = malloc(N);
        fill_array(arr);
        print_array(arr);
        return;
    }

    fn fill_array(arr) {
        for i in 0..N {
            if i == 0 {
                arr[i] = 10;
            } else {
                if i == 1 {
                    arr[i] = 20;
                } else {
                    if i == 2 {
                        arr[i] = 30;
                    } else {
                        i_plus_one = i + 1;
                        arr[i] = i_plus_one;
                    }
                }
            }
        }
        return;
    }

    fn print_array(arr) {
        for i in 0..N {
            arr_i = arr[i];
            print(arr_i);
        }
        return;
    }
   "#;
    compile_and_run(program, vec![]);
}
