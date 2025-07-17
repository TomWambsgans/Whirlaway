use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_symmetric::Permutation;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::poseidon16_kb;
use whir_p3::fiat_shamir::{domain_separator::DomainSeparator, verifier::VerifierState};
use xmss::{WotsSecretKey, XMSS_MERKLE_HEIGHT, XmssSecretKey, random_message};

use crate::{
    EF, F, MyChallenger, Poseidon16, Poseidon24, compiler::compile_to_low_level_bytecode,
    parser::parse_program, runner::execute_bytecode,
};

fn compile_and_run(program: &str, public_input: &[F], private_input: &[F]) {
    let parsed_program = parse_program(program).unwrap();
    let compiled = compile_to_low_level_bytecode(parsed_program).unwrap();
    println!("Compiled Program:\n\n{}", compiled.to_string());

    let poseidon_16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let poseidon_24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(0));

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
        fibonacci(b, a + b, i + 1, n);
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
            for j in i..2*i*(2+1) {
                print(i, j);
                if i == 4 {
                    if j == 7 {
                        break;
                    }
                }
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
    let poseidon_16 = Poseidon16::new_from_rng_128(&mut rng);
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

    let poseidon_24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(0));
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
            hashed, _ = poseidon16(thing_to_hash, neighbours);
        } else {
            hashed, _ = poseidon16(neighbours, thing_to_hash);
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
        hashed, _ = poseidon16(thing_to_hash, pointer_to_zero_vector);
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
            next, _1, _2 = poseidon24(a, b, c);
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
        assert_eq_ext(xmss_public_key, xmss_public_key_recovered);
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
        hashed, _ = poseidon16(thing_to_hash, pointer_to_zero_vector);
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
            next, _1, _2 = poseidon24(a, b, c);
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
            hashed, _ = poseidon16(thing_to_hash, neighbours);
        } else {
            hashed, _ = poseidon16(neighbours, thing_to_hash);
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

#[test]
fn test_aggregate_xmss_signatures() {
    const N_PUBLIC_KEYS: usize = 10;

    // Public input: message (N_CHAINS) | xmss public keys (N_PUBLIC_KEYS x 8) | bitfield (N_PUBLIC_KEYS)
    // Private input: xmss_signatures, alligned by 8
    // each xmss signature (N_CHAINS x 8 + XMSS_MERKLE_HEIGHT x 9) = wots signatue (N_CHAINS x 8) + merkle proof (neighbours: XMSS_MERKLE_HEIGHT x 8 + is_left: XMSS_MERKLE_HEIGHT)
    let program = r#"

    const N_PUBLIC_KEYS = 10;
    const N_CHAINS = 64;
    const CHAIN_LENGTH = 8;
    const XMSS_MERKLE_HEIGHT = 5;
    const XMSS_SIGNATURE_SIZE_ROUNDED_UP = 70; // number of chuncks of 8

    const VERIF_SUCCESSFUL = 1;

    fn main() {
        private_input_start = public_input_start + 160;
        message = public_input_start;
        xmss_public_keys = (public_input_start + N_CHAINS) / 8;
        bitfield = public_input_start + 144; // message + N_PUBLIC_KEYS x 8

        bitfield_counter = malloc(N_PUBLIC_KEYS + 1);
        bitfield_counter[0] = 0;

        for i in 0..N_PUBLIC_KEYS {
            if bitfield[i] == 1 {
                xmss_public_key = xmss_public_keys + i;
                wots_signature_offset = private_input_start / 8;
                wots_signature_shift = bitfield_counter[i] * XMSS_SIGNATURE_SIZE_ROUNDED_UP;
                wots_signature = wots_signature_offset + wots_signature_shift;
                merkle_path = wots_signature + N_CHAINS;
                xmss_public_key_recovered = verify_xmss(message, wots_signature, merkle_path);
                assert_eq_ext(xmss_public_key, xmss_public_key_recovered);
                print(VERIF_SUCCESSFUL);

                bitfield_counter[i + 1] = bitfield_counter[i] + 1;
            } else {
                bitfield_counter[i + 1] = bitfield_counter[i];
            }
        }
        return;
    }

    fn verify_xmss(message, wots_signature, merkle_path) -> 1 {
        // merkle_path: vectorized pointer to neighbours: XMSS_MERKLE_HEIGHT x 8 followed by is_left: XMSS_MERKLE_HEIGHT
        wots_public_key = recover_wots_public_key(message, wots_signature);
        wots_public_key_hash = hash_wots_public_key(wots_public_key);
        merkle_root = merkle_step(0, XMSS_MERKLE_HEIGHT, wots_public_key_hash, (merkle_path + XMSS_MERKLE_HEIGHT) * 8, merkle_path);
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
            pk_i = hash_chain(signature + i, n_hash_iter);
            public_key[i] = pk_i;
        }
        return public_key;
    }

    fn hash_chain(thing_to_hash, n_iter) -> 1 {
        if n_iter == 0 {
            return thing_to_hash;
        }
        hashed, _ = poseidon16(thing_to_hash, pointer_to_zero_vector);
        res = hash_chain(hashed, n_iter - 1);
        return res;
    }

    fn hash_wots_public_key(public_key) -> 1 {
        hashes = malloc((N_CHAINS / 2) + 1);
        hashes[0] = pointer_to_zero_vector;
        for i in 0..32 {
            next, _, _ = poseidon24(public_key[2 * i], public_key[(2 * i) + 1], hashes[i]);
            hashes[i + 1] = next;
        }
        res = hashes[32];
        return res; 
    }

    fn merkle_step(step, height, thing_to_hash, are_left, neighbours) -> 1 {
        if step == height {
            return thing_to_hash;
        }
        if are_left[0] == 1 {
            hashed, _ = poseidon16(thing_to_hash, neighbours);
        } else {
            hashed, _ = poseidon16(neighbours, thing_to_hash);
        }

        res = merkle_step(step + 1, height, hashed, are_left + 1, neighbours + 1);
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

    let bitfield = (0..N_PUBLIC_KEYS)
        .map(|_| rng.random())
        .collect::<Vec<bool>>();
    assert!(bitfield.iter().filter(|&&x| x).count() >= 3, "change seed");

    let xmss_secret_keys = (0..N_PUBLIC_KEYS)
        .map(|_| XmssSecretKey::random(&mut rng))
        .collect::<Vec<_>>();
    let mut xmss_public_keys = vec![];
    for xmss_secret_key in &xmss_secret_keys {
        xmss_public_keys.push(xmss_secret_key.public_key().root.to_vec());
    }

    let signatures = bitfield
        .iter()
        .enumerate()
        .filter_map(|(i, &bit)| {
            if bit {
                let index = rng.random_range(0..1 << XMSS_MERKLE_HEIGHT);
                Some(xmss_secret_keys[i].sign(&mesage, index))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut public_input = mesage.iter().map(|&x| F::new(x as u32)).collect::<Vec<F>>();
    for pk in xmss_public_keys {
        public_input.extend(pk);
    }
    for &bit in &bitfield {
        if bit {
            public_input.push(F::ONE);
        } else {
            public_input.push(F::ZERO);
        }
    }

    let mut private_input = vec![];
    for signature in &signatures {
        private_input.extend(
            signature
                .wots_signature
                .0
                .iter()
                .flat_map(|digest| digest.to_vec()),
        );
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
        // add padding to align with 8
        let padding = vec![F::ZERO; (8 - (private_input.len() % 8)) % 8];
        private_input.extend(padding);
    }

    compile_and_run(program, &public_input, &private_input);
}

#[test]
fn test_product_extension_field() {
    let program = r#"

    const W = 3; // in the extension field, X^8 = 3

    fn main() {
        a = public_input_start;
        b = public_input_start + 8;
        sum = add_extension(a, b);
        prod = mul_extension(a, b);

        for i in 0..8 {
            print(sum[i]);
        }

        for i in 0..8 {
            print(prod[i]);
        }

        real_sum = public_input_start + 16;
        real_prod = public_input_start + 24;

        for i in 0..8 {
            assert sum[i] == real_sum[i];
            assert prod[i] == real_prod[i];
        }
        
        return;
    }

    fn mul_extension(a, b) -> 1 {
        // a, b and c are pointers

        a0 = a[0];
        a1 = a[1];
        a2 = a[2];
        a3 = a[3];
        a4 = a[4];
        a5 = a[5];
        a6 = a[6];
        a7 = a[7];

        b0 = b[0];
        b1 = b[1];
        b2 = b[2];
        b3 = b[3];
        b4 = b[4];
        b5 = b[5];
        b6 = b[6];
        b7 = b[7];

        c0 = (a0 * b0) + W * ((a1 * b7) + (a2 * b6) + (a3 * b5) + (a4 * b4) + (a5 * b3) + (a6 * b2) + (a7 * b1));
        c1 = (a1 * b0) + (a0 * b1) + W * ((a2 * b7) + (a3 * b6) + (a4 * b5) + (a5 * b4) + (a6 * b3) + (a7 * b2));
        c2 = (a2 * b0) + (a1 * b1) + (a0 * b2) + W * ((a3 * b7) + (a4 * b6) + (a5 * b5) + (a6 * b4) + (a7 * b3));
        c3 = (a3 * b0) + (a2 * b1) + (a1 * b2) + (a0 * b3) + W * ((a4 * b7) + (a5 * b6) + (a6 * b5) + (a7 * b4));
        c4 = (a4 * b0) + (a3 * b1) + (a2 * b2) + (a1 * b3) + (a0 * b4) + W * ((a5 * b7) + (a6 * b6) + (a7 * b5));
        c5 = (a5 * b0) + (a4 * b1) + (a3 * b2) + (a2 * b3) + (a1 * b4) + (a0 * b5) + W * ((a6 * b7) + (a7 * b6));
        c6 = (a6 * b0) + (a5 * b1) + (a4 * b2) + (a3 * b3) + (a2 * b4) + (a1 * b5) + (a0 * b6) + W * (a7 * b7);
        c7 = (a7 * b0) + (a6 * b1) + (a5 * b2) + (a4 * b3) + (a3 * b4) + (a2 * b5) + (a1 * b6) + (a0 * b7);

        res = malloc(8);
        res[0] = c0;
        res[1] = c1;
        res[2] = c2;
        res[3] = c3;
        res[4] = c4;
        res[5] = c5;
        res[6] = c6;
        res[7] = c7;

        return res;
    }

    fn add_extension(a, b) -> 1 {
        res = malloc(8);
        res[0] = a[0] + b[0];
        res[1] = a[1] + b[1];
        res[2] = a[2] + b[2];
        res[3] = a[3] + b[3];
        res[4] = a[4] + b[4];
        res[5] = a[5] + b[5];
        res[6] = a[6] + b[6];
        res[7] = a[7] + b[7];
        return res;
    }
   "#;

    let mut rng = StdRng::seed_from_u64(0);
    let a: EF = rng.random();
    let b: EF = rng.random();

    let mut public_input = Vec::<F>::new();
    public_input.extend(a.as_basis_coefficients_slice());
    public_input.extend(b.as_basis_coefficients_slice());
    public_input.extend((a + b).as_basis_coefficients_slice());
    public_input.extend((a * b).as_basis_coefficients_slice());

    compile_and_run(program, &public_input, &[]);

    dbg!(a + b);
    dbg!(a * b);
}

#[test]
fn test_fiat_shamir() {
    let program = r#"

    const W = 3; // in the extension field, X^8 = 3

    fn main() {
        start = public_input_start;
        n = start[0];
        transcript = start + 1;

        fs_state = fiat_shamir_new(transcript);

        all_states = malloc(n + 1);
        all_states[0] = fs_state;
        for i in 0..n {
            fs_state = all_states[i];
            new_fs_state, _ = fiat_shamir_receive_base_field(fs_state);
            all_states[i + 1] = new_fs_state;
        }

        final_state = all_states[n];
        fiat_shamir_print_state(final_state);

        expected_state = start + n + 1;

        left = final_state[1] * 8;
        for i in 0..8 {
            assert left[i] == expected_state[i];
        }
        right = final_state[2] * 8;
        for i in 0..8 {
            assert right[i] == expected_state[i + 8];
        }

        return;
    }

    // FIAT SHAMIR layout:
    // 0 -> transcript
    // 1 -> vectorized pointer to first half of sponge state
    // 2 -> vectorized pointer to second half of sponge state
    // 3 -> input_buffer_size
    // 4 -> input_buffer (vectorized pointer)
    // 5 -> output_buffer_size

    fn fiat_shamir_new(transcript) -> 1 {
        // transcript is a (normal) pointer

        // TODO domain separator
        fs_state = malloc(6);
        fs_state[0] = transcript;
        fs_state[1] = pointer_to_zero_vector; // first half of sponge state
        fs_state[2] = pointer_to_zero_vector; // second half of sponge state
        fs_state[3] = 0; // input buffer size
        allocated = malloc_vec(1);
        fs_state[4] = allocated; // input buffer (vectorized pointer)
        fs_state[5] = 8; // output buffer size

        return fs_state;
    }

    fn fiat_shamir_receive_base_field(fs_state) -> 2 {
        new_fs_state = malloc(6);
        transcript_ptr = fs_state[0]; 
        value = transcript_ptr[0]; 
        input_buffer_size = fs_state[3];
        input_buffer = fs_state[4];
        input_buffer_ptr = input_buffer * 8;
        input_buffer_ptr[input_buffer_size] = value;

        if input_buffer_size == 7 {
            // duplexing
            l, r = poseidon16(input_buffer, fs_state[2]);
            new_fs_state[0] = transcript_ptr + 1;
            new_fs_state[1] = l;
            new_fs_state[2] = r;
            new_fs_state[3] = 0; // reset input buffer size
            allocated = malloc_vec(1);
            new_fs_state[4] = allocated;
            new_fs_state[5] = 8; // reset output buffer size

            return new_fs_state, value;
        }

        new_fs_state[0] = transcript_ptr + 1;
        new_fs_state[1] = fs_state[1];
        new_fs_state[2] = fs_state[2];
        new_fs_state[3] = input_buffer_size + 1;
        new_fs_state[4] = fs_state[4];
        new_fs_state[5] = 0; // "Any buffered output is now invalid."
        return new_fs_state, value;
    }

    

    fn fiat_shamir_print_state(fs_state) {
        left = fs_state[1] * 8;
        for i in 0..8 {
            print(left[i]);
        }
        right = fs_state[2] * 8;
        for i in 0..8 {
            print(right[i]);
        }
        return;
    }

    
   "#;
    let n = 100;

    let poseidon16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let mut rng = StdRng::seed_from_u64(0);
    let challenger = MyChallenger::new(poseidon16);
    let proof_data = (0..n).map(|_| rng.random()).collect::<Vec<F>>();
    let mut verifier_state = VerifierState::new(
        &DomainSeparator::<EF, F>::new(vec![]),
        proof_data.clone(),
        challenger,
    );
    for _ in 0..n {
        let _ = verifier_state.next_base_scalars_const::<1>();
    }

    let mut public_input = vec![F::from_usize(n)];
    public_input.extend(proof_data);
    public_input.extend(verifier_state.challenger().sponge_state.to_vec());

    let private_input = vec![];

    compile_and_run(program, &public_input, &private_input);

    dbg!(verifier_state.challenger().sponge_state);
}
