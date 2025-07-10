use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::Poseidon2KoalaBear;
use p3_symmetric::Permutation;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::poseidon16_kb;
use xmss::{WotsSecretKey, XMSS_MERKLE_HEIGHT, XmssSecretKey, random_message};

use crate::{
    F, compiler::compile_to_low_level_bytecode, parser::parse_program, runner::execute_bytecode,
};

fn compile_and_run(program: &str, public_input: &[F], private_input: &[F]) {
    let parsed_program = parse_program(program).unwrap();
    let compiled = compile_to_low_level_bytecode(parsed_program).unwrap();
    println!("Compiled Program:\n\n{}", compiled.to_string());

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
        are_left = public_input_start + 16;
        neighbours = private_input_start / 8;
        merkle_root = merkle_step(0, HEIGHT, thing_to_hash, are_left, neighbours);
        print_chunk_of_8(merkle_root);
        assert_eq_ext(merkle_root, claimed_merkle_root);
        return;
    }

    fn assert_eq_ext(a, b) {
        // a and b both pointers in the memory of chunk of 8 field elements
        a_shifted = a * 8;
        b_shifted = b * 8;
        for i in 0..8 {
            a_i = a_shifted[i];
            b_i = b_shifted[i];
            assert a_i == b_i;
        }
        return;
    }

    fn merkle_step(step, height, thing_to_hash, are_left, neighbours) -> 1 {
        if step == height {
            return thing_to_hash;
        }
        is_left = are_left[0];

        if is_left == 1 {
            hashed, trash = poseidon16(thing_to_hash, neighbours);
        } else {
            hashed, trash = poseidon16(neighbours, thing_to_hash);
        }

        next_step = step + 1;
        next_are_left = are_left + 1;
        next_neighbours = neighbours + 1;
        res = merkle_step(next_step, height, hashed, next_are_left, next_neighbours);
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
    let is_left = (0..height).map(|_| rng.random()).collect::<Vec<bool>>();

    let mut private_input = vec![];

    let mut to_hash = leaf.clone();
    for i in 0..height {
        let neighbour = random_digest(&mut rng);
        if is_left[i] {
            to_hash = poseidon16_kb(to_hash, neighbour).0;
        } else {
            to_hash = poseidon16_kb(neighbour, to_hash).0;
        }
        private_input.extend(neighbour);
    }

    let merkle_root = to_hash;
    dbg!(&merkle_root[0]);

    let mut public_input = leaf.to_vec();
    public_input.extend(merkle_root);
    for i in 0..height {
        if is_left[i] {
            public_input.push(F::ONE);
        } else {
            public_input.push(F::ZERO);
        }
    }

    compile_and_run(program, &public_input, &private_input);

    dbg!(&merkle_root);
}

fn random_digest<R: Rng>(rng: &mut R) -> [F; 8] {
    (0..8)
        .map(|_| rng.random())
        .collect::<Vec<F>>()
        .try_into()
        .unwrap()
}

#[test]
fn test_verify_wots_signature() {
    // Public input: wots public key hash | message
    // Private input: signature
    let program = r#"

    const N_CHAINS = 64;
    const CHAIN_LENGTH = 8;

    fn main() {
        private_input_start = public_input_start + 72; // wots public key hash + message: 8 + 64 = 72
        wots_public_key_hash = public_input_start / 8;
        message = public_input_start + 8;
        signature = private_input_start / 8;
        wots_public_key_recovered = recover_wots_public_key(message, signature);
        wots_public_key_hash_recovered = hash_wots_public_key(wots_public_key_recovered);
        assert_eq_ext(wots_public_key_hash, wots_public_key_hash_recovered);
        print_chunk_of_8(wots_public_key_hash);
        return;
    }

    fn recover_wots_public_key(message, signature) -> 1 {
        // message: pointer
        // signature: vectorized pointer
        // return a pointer of vectorized pointers

        public_key = malloc(N_CHAINS);
        for i in 0..N_CHAINS {
            msg_i = message[i];
            n_hash_iter = CHAIN_LENGTH - msg_i;
            signature_i = signature + i;
            pk_i = hash_chain(signature_i, n_hash_iter);
            public_key[i] = pk_i;
        }
        return public_key;
    }

    fn hash_chain(thing_to_hash, n_iter) -> 1 {
        if n_iter == 0 {
            return thing_to_hash;
        }
        hashed, trash = poseidon16(thing_to_hash, pointer_to_zero_vector);
        n_iter_minus_one = n_iter - 1;
        res = hash_chain(hashed, n_iter_minus_one);
        return res;
    }

    fn hash_wots_public_key(public_key) -> 1 {
        hashes = malloc(33); // N_CHAINS / 2 + 1
        hashes[0] = pointer_to_zero_vector;
        for i in 0..32 {
            two_i = 2 * i;
            two_i_plus_one = two_i + 1;
            a = public_key[two_i];
            b = public_key[two_i_plus_one];
            c = hashes[i];
            next, trash1, trash2 = poseidon24(a, b, c);
            i_plus_one = i + 1;
            hashes[i_plus_one] = next;
        }
        res = hashes[32];
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

    fn assert_eq_ext(a, b) {
        // a and b both pointers in the memory of chunk of 8 field elements
        a_shifted = a * 8;
        b_shifted = b * 8;
        for i in 0..8 {
            a_i = a_shifted[i];
            b_i = b_shifted[i];
            assert a_i == b_i;
        }
        return;
    }
   "#;

    let mut rng = StdRng::seed_from_u64(0);
    let mesage = random_message(&mut rng);
    let wots_secret_key = WotsSecretKey::random(&mut rng);
    let signature = wots_secret_key.sign(&mesage);
    let public_key_hashed = wots_secret_key.public_key().hash();

    let mut public_input = public_key_hashed.to_vec();
    public_input.extend(mesage.iter().map(|&x| F::new(x as u32)));

    let private_input = signature
        .0
        .iter()
        .flat_map(|digest| digest.to_vec())
        .collect::<Vec<F>>();

    compile_and_run(program, &public_input, &private_input);

    dbg!(&public_key_hashed);
}

#[test]
fn test_verify_xmss_signature() {
    // Public input: xmss public key | message
    // Private input: xmss signature = wots signatue (N_CHAINS x 8) + merkle proof (neighbours: XMSS_MERKLE_HEIGHT x 8 + is_left: XMSS_MERKLE_HEIGHT)
    let program = r#"

    const N_CHAINS = 64;
    const CHAIN_LENGTH = 8;
    const XMSS_MERKLE_HEIGHT = 5;

    fn main() {
        private_input_start = public_input_start + 72; // wots public key hash + message: 8 + 64 = 72
        xmss_public_key = public_input_start / 8;
        message = public_input_start + 8;
        wots_signature = private_input_start / 8;
        merkle_path = wots_signature + N_CHAINS;
        xmss_public_key_recovered = verify_xmss(message, wots_signature, merkle_path);
        assert_eq_ext(xmss_public_key, xmss_public_key);
        print_chunk_of_8(xmss_public_key);
        return;
    }

    fn verify_xmss(message, wots_signature, merkle_path) -> 1 {
        // merkle_path: vectorized pointer to neighbours: XMSS_MERKLE_HEIGHT x 8 followed by is_left: XMSS_MERKLE_HEIGHT
        wots_public_key = recover_wots_public_key(message, wots_signature);
        wots_public_key_hash = hash_wots_public_key(wots_public_key);
        are_left_vec = merkle_path + XMSS_MERKLE_HEIGHT;
        are_left = are_left_vec * 8;
        merkle_root = merkle_step(0, XMSS_MERKLE_HEIGHT, wots_public_key_hash, are_left, merkle_path);
        return merkle_root;
    }

    fn recover_wots_public_key(message, signature) -> 1 {
        // message: pointer
        // signature: vectorized pointer
        // return a pointer of vectorized pointers

        public_key = malloc(N_CHAINS);
        for i in 0..N_CHAINS {
            msg_i = message[i];
            n_hash_iter = CHAIN_LENGTH - msg_i;
            signature_i = signature + i;
            pk_i = hash_chain(signature_i, n_hash_iter);
            public_key[i] = pk_i;
        }
        return public_key;
    }

    fn hash_chain(thing_to_hash, n_iter) -> 1 {
        if n_iter == 0 {
            return thing_to_hash;
        }
        hashed, trash = poseidon16(thing_to_hash, pointer_to_zero_vector);
        n_iter_minus_one = n_iter - 1;
        res = hash_chain(hashed, n_iter_minus_one);
        return res;
    }

    fn hash_wots_public_key(public_key) -> 1 {
        hashes = malloc(33); // N_CHAINS / 2 + 1
        hashes[0] = pointer_to_zero_vector;
        for i in 0..32 {
            two_i = 2 * i;
            two_i_plus_one = two_i + 1;
            a = public_key[two_i];
            b = public_key[two_i_plus_one];
            c = hashes[i];
            next, trash1, trash2 = poseidon24(a, b, c);
            i_plus_one = i + 1;
            hashes[i_plus_one] = next;
        }
        res = hashes[32];
        return res; 
    }

    fn merkle_step(step, height, thing_to_hash, are_left, neighbours) -> 1 {
        if step == height {
            return thing_to_hash;
        }
        is_left = are_left[0];

        if is_left == 1 {
            hashed, trash = poseidon16(thing_to_hash, neighbours);
        } else {
            hashed, trash = poseidon16(neighbours, thing_to_hash);
        }

        next_step = step + 1;
        next_are_left = are_left + 1;
        next_neighbours = neighbours + 1;
        res = merkle_step(next_step, height, hashed, next_are_left, next_neighbours);
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

    fn assert_eq_ext(a, b) {
        // a and b both pointers in the memory of chunk of 8 field elements
        a_shifted = a * 8;
        b_shifted = b * 8;
        for i in 0..8 {
            a_i = a_shifted[i];
            b_i = b_shifted[i];
            assert a_i == b_i;
        }
        return;
    }
   "#;

    let mut rng = StdRng::seed_from_u64(0);
    let mesage = random_message(&mut rng);
    let xmss_secret_key: XmssSecretKey = XmssSecretKey::random(&mut rng);
    let index = rng.random_range(0..1 << XMSS_MERKLE_HEIGHT);
    let signature = xmss_secret_key.sign(&mesage, index);

    let mut public_input = xmss_secret_key.public_key().root.to_vec();
    public_input.extend(mesage.iter().map(|&x| F::new(x as u32)));

    let mut private_input = signature
        .wots_signature
        .0
        .iter()
        .flat_map(|digest| digest.to_vec())
        .collect::<Vec<F>>();
    private_input.extend(
        signature
            .merkle_proof
            .iter()
            .flat_map(|(_, neighbour)| *neighbour),
    );
    private_input.extend(
        signature
            .merkle_proof
            .iter()
            .map(|(is_left, _)| F::new(*is_left as u32)),
    );

    compile_and_run(program, &public_input, &private_input);

    dbg!(xmss_secret_key.public_key().root); 
}
