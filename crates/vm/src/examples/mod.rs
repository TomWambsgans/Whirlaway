use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::Poseidon2KoalaBear;
use p3_symmetric::Permutation;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    F, compiler::compile_to_low_level_bytecode, parser::parse_program, runner::execute_bytecode,
};

fn compile_and_run(program: &str, public_input: &[F], private_input: &[F]) {
    let parsed_program = parse_program(program).unwrap();
    let compiled = compile_to_low_level_bytecode(parsed_program).unwrap();
    // println!("Compiled Program:\n\n{}", compiled.to_string());

    let poseidon_16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let poseidon_24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut StdRng::seed_from_u64(0));

    execute_bytecode(
        &compiled,
        &public_input,
        private_input,
        poseidon_16,
        poseidon_24,
    );
}

#[test]
fn tes_aggregate_program() {
    let program_str = include_str!("aggregate.vm");
    compile_and_run(program_str, &[], &[]);
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
    compile_and_run(program, &[], &[]);
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
    compile_and_run(program, &[], &[]);
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
    compile_and_run(program, &[], &[]);
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
    compile_and_run(program, &[], &[]);
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
    compile_and_run(program, &public_input, &[]);

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
    compile_and_run(program, &public_input, &[]);

    let poseidon_24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    poseidon_24.permute_mut(&mut public_input);
    dbg!(public_input);
}

// #[test]
// fn test_mini_program_5() {
//     let program = r#"
//     fn main() {
//         const N = 10;

//         for i in 0..N {
//             ef = public_input_start + 1;
//             a = public_input_start + i;
//             b = a + 1;
//             assert_ext b = a * ef; // TODO
//         }

//     }
//    "#;

//     let mut rng = StdRng::seed_from_u64(0);
//     let ef: EF = rng.random();
//     let mut public_input = Vec::new();
//     for pow in 0..10 {
//         let exp = ef.exp_u64(pow);
//         public_input.extend_from_slice(exp.as_basis_coefficients_slice());
//     }

//     compile_and_run(program, &public_input);
// }

#[test]
fn test_verify_merkle_path() {
    let program = r#"
    const HEIGHT = 8;

    fn main() {
        private_input_start = public_input_start + 24; // leaf + root + "neighbours_are_left bits"
        thing_to_hash = public_input_start / 8;
        claimed_merkle_root = thing_to_hash + 1;
        neighbours_are_left = public_input_start + 16;
        proof = private_input_start / 8;
        merkle_root = merkle_step(0, HEIGHT, thing_to_hash, neighbours_are_left, proof);
        print_chunk_of_8(merkle_root);
        assert_ext merkle_root == claimed_merkle_root;
        return;
    }

    fn merkle_step(step, height, thing_to_hash, neighbours_are_left, proof) -> 1 {
        if step == height {
            return thing_to_hash;
        }
        neighbour_is_left = neighbours_are_left[0];
        neighbour = proof;

        if neighbour_is_left == 1 {
            hashed, trash = poseidon16(neighbour, thing_to_hash);
        } else {
            hashed, trash = poseidon16(thing_to_hash, neighbour);
        }

        next_step = step + 1;
        next_neighbours_are_left = neighbours_are_left + 1;
        next_proof = proof + 1;
        res = merkle_step(next_step, height, hashed, next_neighbours_are_left, next_proof);
        return res;
    }

    fn print_chunk_of_8(arr) {
        reindexed_arr = arr * 8;
        for i in 0..8 {
            arr_i = reindexed_arr[i];
            print(arr_i);
        }
        return;
    }
   "#;

    let mut rng = StdRng::seed_from_u64(0);
    let height = 8;
    let leaf = random_digest(&mut rng);
    let neighbour_is_left = (0..height).map(|_| rng.random()).collect::<Vec<bool>>();

    let mut private_input = vec![];

    let mut to_hash = leaf.clone();
    for i in 0..height {
        let neighbour = random_digest(&mut rng);
        if neighbour_is_left[i] {
            to_hash = poseidon_compress(neighbour, to_hash);
        } else {
            to_hash = poseidon_compress(to_hash, neighbour);
        }
        private_input.extend(neighbour);
    }

    let merkle_root = to_hash;
    dbg!(&merkle_root[0]);

    let mut public_input = leaf.to_vec();
    public_input.extend(merkle_root);
    for i in 0..height {
        if neighbour_is_left[i] {
            public_input.push(F::ONE);
        } else {
            public_input.push(F::ZERO);
        }
    }

    compile_and_run(program, &public_input, &private_input);

    dbg!(&merkle_root);
}

fn poseidon_compress(a: [F; 8], b: [F; 8]) -> [F; 8] {
    let poseidon_16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let mut buff = [F::ZERO; 16];
    buff[..8].copy_from_slice(&a);
    buff[8..].copy_from_slice(&b);
    poseidon_16.permute_mut(&mut buff);
    let mut res = [F::ZERO; 8];
    res.copy_from_slice(&buff[..8]);
    res
}

fn random_digest<R: Rng>(rng: &mut R) -> [F; 8] {
    (0..8)
        .map(|_| rng.random())
        .collect::<Vec<F>>()
        .try_into()
        .unwrap()
}
