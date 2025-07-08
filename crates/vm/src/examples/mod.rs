use p3_koala_bear::Poseidon2KoalaBear;
use p3_symmetric::Permutation;
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    F, compiler::compile_to_low_level_bytecode, parser::parse_program, runner::execute_bytecode,
};

fn compile_and_run(program: &str, public_input: &[F]) {
    let parsed_program = parse_program(program).unwrap();
    let compiled = compile_to_low_level_bytecode(parsed_program).unwrap();
    println!("Compiled Program:\n\n{}", compiled.to_string());

    let poseidon_16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let poseidon_24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut StdRng::seed_from_u64(0));

    execute_bytecode(&compiled, &public_input, poseidon_16, poseidon_24);
}

#[test]
fn tes_aggregate_program() {
    let program_str = include_str!("aggregate.vm");
    compile_and_run(program_str, &[]);
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
    compile_and_run(program, &[]);
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
    compile_and_run(program, &[]);
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
    compile_and_run(program, &[]);
}

#[test]
fn test_mini_program_2() {
    let program = r#"
    fn main() {
        for i in 0..10 {
            for j in i..10 {
                for k in j..10 {
                    sum, prod = compute_sum_and_product(i, j, k);
                    if sum == 10 {
                        print(i, j, k, prod);
                    }
                }
            }
        }
        return;
    }

    fn compute_sum_and_product(a, b, c) -> 2 {
        s1 = a + b;
        sum = s1 + c;
        p1 = a * b;
        product = p1 * c;
        return sum, product;
    }
   "#;
    compile_and_run(program, &[]);
}

#[test]
fn test_mini_program_3() {
    let program = r#"
    fn main() {
        a = public_input_start / 8;
        b = a + 1;
        c, d = poseidon16(a, b);

        c_shifted = c * 8;
        d_shifted = d * 8;

        for i in 0..8 {
            cc = c_shifted[i];
            print(cc);
        }
        for i in 0..8 {
            dd = d_shifted[i];
            print(dd);
        }
        return;
    }
   "#;
    let mut public_input: [F; 16] = (0..16)
        .map(|i| F::new(i))
        .collect::<Vec<F>>()
        .try_into()
        .unwrap();
    compile_and_run(program, &public_input);

    let mut rng = StdRng::seed_from_u64(0);
    let poseidon_16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut rng);
    poseidon_16.permute_mut(&mut public_input);
    dbg!(public_input);
}

#[test]
fn test_mini_program_4() {
    let program = r#"
    fn main() {
        a = public_input_start / 8;
        b = a + 1;
        c = a + 2;
        d, e, f = poseidon24(a, b, c);

        arr = malloc(3);
        arr[0] = d;
        arr[1] = e;
        arr[2] = f;

        for i in 0..3 {
            v = arr[i];
            v_shifted = v * 8;
            for j in 0..8 {
                vv = v_shifted[j];
                print(vv);
            }
        }
        return;
    }
   "#;
    let mut public_input: [F; 24] = (0..24)
        .map(|i| F::new(i))
        .collect::<Vec<F>>()
        .try_into()
        .unwrap();
    compile_and_run(program, &public_input);

    let poseidon_24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    poseidon_24.permute_mut(&mut public_input);
    dbg!(public_input);
}

#[test]
fn test_mini_program_5() {
    let program = r#"
    fn main() {
        a = memory[public_input_start];
        print(a);
        for i in 0..a {
            print(i);
        }
        return;
    }
   "#;
    let public_input = vec![F::new(10)];
    compile_and_run(program, &public_input);
}
