use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_symmetric::Permutation;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::{
    build_challenger, build_poseidon16, build_poseidon24, build_prover_state, build_verifier_state,
    poseidon16_kb,
};

use whir_p3::fiat_shamir::verifier::VerifierState;
use xmss::{WotsSecretKey, XMSS_MERKLE_HEIGHT, XmssSecretKey, random_message};

use crate::{EF, F, compile_and_run};

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
fn test_edge_case_0() {
    let program = r#"
    fn main() {
        a = malloc(1);
        a[0] = 0;
        for i in 0..1 {
            x = 1 + a[i];
        }
        for i in 0..1 {
            y = 1 + a[i];
        }
        return;
    }
   "#;
    compile_and_run(program, &[], &[]);
}

#[test]
fn test_unroll() {
    // a program to check the value of the 30th Fibonacci number (832040)
    let program = r#"
    fn main() {
        for i in 0..5 unroll {
            for j in i..2*i unroll {
                print(i, j);
            }
        }
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
        c = poseidon16(a, b);

        c_shifted = c * 8;
        d_shifted = (c + 1) * 8;

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

    build_poseidon16().permute_mut(&mut public_input);
    dbg!(public_input);
}

#[test]
fn test_mini_program_4() {
    let program = r#"
    fn main() {
        a = public_input_start / 8;
        c = a + 2;
        f = poseidon24(a, c);

        f_shifted = f * 8;
        for j in 0..8 {
            print(f_shifted[j]);
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

    build_poseidon24().permute_mut(&mut public_input);
    dbg!(&public_input[16..]);
}

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
            hashed = poseidon16(thing_to_hash, neighbours);
        } else {
            hashed = poseidon16(neighbours, thing_to_hash);
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

        public_key = malloc_vec(N_CHAINS);
        for i in 0..N_CHAINS {
            msg_i = message[i];
            n_hash_iter = CHAIN_LENGTH - msg_i;
            signature_i = signature + i;
            pk_i = hash_chain(signature_i, n_hash_iter);
            copy_vector(pk_i, public_key + i);
        }
        return public_key;
    }

    fn hash_chain(thing_to_hash, n_iter) -> 1 {
        if n_iter == 0 {
            return thing_to_hash;
        }
        hashed = poseidon16(thing_to_hash, pointer_to_zero_vector);
        n_iter_minus_one = n_iter - 1;
        res = hash_chain(hashed, n_iter_minus_one);
        return res;
    }

    fn hash_wots_public_key(public_key) -> 1 {
        hashes = malloc(N_CHAINS / 2 + 1);
        hashes[0] = pointer_to_zero_vector;
        for i in 0..N_CHAINS / 2 {
            next = poseidon24(public_key + 2 * i, hashes[i]);
            hashes[i + 1] = next;
        }
        res = hashes[N_CHAINS / 2];
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

    fn copy_vector(a, b) {
        // a and b both pointers in the memory of chunk of 8 field elements
        a_shifted = a * 8;
        b_shifted = b * 8;
        for i in 0..8 {
            a_i = a_shifted[i];
            b_shifted[i] = a_i;
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

        public_key = malloc_vec(N_CHAINS);
        for i in 0..N_CHAINS {
            msg_i = message[i];
            n_hash_iter = CHAIN_LENGTH - msg_i;
            signature_i = signature + i;
            pk_i = hash_chain(signature_i, n_hash_iter);
            copy_vector(pk_i, public_key + i);
        }
        return public_key;
    }

    fn hash_chain(thing_to_hash, n_iter) -> 1 {
        if n_iter == 0 {
            return thing_to_hash;
        }
        hashed = poseidon16(thing_to_hash, pointer_to_zero_vector);
        n_iter_minus_one = n_iter - 1;
        res = hash_chain(hashed, n_iter_minus_one);
        return res;
    }

    fn hash_wots_public_key(public_key) -> 1 {
        hashes = malloc(N_CHAINS / 2 + 1);
        hashes[0] = pointer_to_zero_vector;
        for i in 0..N_CHAINS / 2 {
            next = poseidon24(public_key + 2 * i, hashes[i]);
            hashes[i + 1] = next;
        }
        res = hashes[N_CHAINS / 2];
        return res; 
    }

    fn merkle_step(step, height, thing_to_hash, are_left, neighbours) -> 1 {
        if step == height {
            return thing_to_hash;
        }
        is_left = are_left[0];

        if is_left == 1 {
            hashed = poseidon16(thing_to_hash, neighbours);
        } else {
            hashed = poseidon16(neighbours, thing_to_hash);
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

    fn copy_vector(a, b) {
        // a and b both pointers in the memory of chunk of 8 field elements
        a_shifted = a * 8;
        b_shifted = b * 8;
        for i in 0..8 {
            a_i = a_shifted[i];
            b_shifted[i] = a_i;
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

        public_key = malloc_vec(N_CHAINS);
        for i in 0..N_CHAINS {
            msg_i = message[i];
            n_hash_iter = CHAIN_LENGTH - msg_i;
            signature_i = signature + i;
            pk_i = hash_chain(signature_i, n_hash_iter);
            copy_vector(pk_i, public_key + i);
        }
        return public_key;
    }

    fn hash_chain(thing_to_hash, n_iter) -> 1 {
        if n_iter == 0 {
            return thing_to_hash;
        }
        hashed = poseidon16(thing_to_hash, pointer_to_zero_vector);
        n_iter_minus_one = n_iter - 1;
        res = hash_chain(hashed, n_iter_minus_one);
        return res;
    }

    fn hash_wots_public_key(public_key) -> 1 {
        hashes = malloc(N_CHAINS / 2 + 1);
        hashes[0] = pointer_to_zero_vector;
        for i in 0..N_CHAINS / 2 {
            next = poseidon24(public_key + 2 * i, hashes[i]);
            hashes[i + 1] = next;
        }
        res = hashes[N_CHAINS / 2];
        return res; 
    }

    fn merkle_step(step, height, thing_to_hash, are_left, neighbours) -> 1 {
        if step == height {
            return thing_to_hash;
        }
        is_left = are_left[0];

        if is_left == 1 {
            hashed = poseidon16(thing_to_hash, neighbours);
        } else {
            hashed = poseidon16(neighbours, thing_to_hash);
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

    fn copy_vector(a, b) {
        // a and b both pointers in the memory of chunk of 8 field elements
        a_shifted = a * 8;
        b_shifted = b * 8;
        for i in 0..8 {
            a_i = a_shifted[i];
            b_shifted[i] = a_i;
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
        sum = malloc(8);
        prod = malloc(8);
        custom_add_extension(a, b, sum);
        custom_mul_extension(a, b, prod);

        real_sum = public_input_start + 16;
        real_prod = public_input_start + 24;

        for i in 0..8 {
            assert sum[i] == real_sum[i];
            assert prod[i] == real_prod[i];
        }
        
        return;
    }

    fn custom_mul_extension(a, b, c) {
        // a, b and c are pointers
        // c = a * b

        c[0] = (a[0] * b[0]) + W * ((a[1] * b[7]) + (a[2] * b[6]) + (a[3] * b[5]) + (a[4] * b[4]) + (a[5] * b[3]) + (a[6] * b[2]) + (a[7] * b[1]));
        c[1] = (a[1] * b[0]) + (a[0] * b[1]) + W * ((a[2] * b[7]) + (a[3] * b[6]) + (a[4] * b[5]) + (a[5] * b[4]) + (a[6] * b[3]) + (a[7] * b[2]));
        c[2] = (a[2] * b[0]) + (a[1] * b[1]) + (a[0] * b[2]) + W * ((a[3] * b[7]) + (a[4] * b[6]) + (a[5] * b[5]) + (a[6] * b[4]) + (a[7] * b[3]));
        c[3] = (a[3] * b[0]) + (a[2] * b[1]) + (a[1] * b[2]) + (a[0] * b[3]) + W * ((a[4] * b[7]) + (a[5] * b[6]) + (a[6] * b[5]) + (a[7] * b[4]));
        c[4] = (a[4] * b[0]) + (a[3] * b[1]) + (a[2] * b[2]) + (a[1] * b[3]) + (a[0] * b[4]) + W * ((a[5] * b[7]) + (a[6] * b[6]) + (a[7] * b[5]));
        c[5] = (a[5] * b[0]) + (a[4] * b[1]) + (a[3] * b[2]) + (a[2] * b[3]) + (a[1] * b[4]) + (a[0] * b[5]) + W * ((a[6] * b[7]) + (a[7] * b[6]));
        c[6] = (a[6] * b[0]) + (a[5] * b[1]) + (a[4] * b[2]) + (a[3] * b[3]) + (a[2] * b[4]) + (a[1] * b[5]) + (a[0] * b[6]) + W * (a[7] * b[7]);
        c[7] = (a[7] * b[0]) + (a[6] * b[1]) + (a[5] * b[2]) + (a[4] * b[3]) + (a[3] * b[4]) + (a[2] * b[5]) + (a[1] * b[6]) + (a[0] * b[7]);

        return;
    }

    fn custom_add_extension(a, b, c) {
        // a, b and c are pointers
        // c = a + b
        for i in 0..8 unroll {
            c[i] = a[i] + b[i];
        }
        return;
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
}

#[test]
fn test_fiat_shamir_complete() {
    let program = r#"

    fn main() {
        start = public_input_start;
        n = start[0];
        transcript = start + (2 * n) + 1;

        fs_state = fiat_shamir_new(transcript);

        all_states = malloc(n + 1);
        all_states[0] = fs_state;

        for i in 0..n {
            is_sample = start[(2 * i) + 1];
            n_elements = start[(2 * i) + 2];
            fs_state = all_states[i];
            if is_sample == 1 {
                new_fs_state, _ = fiat_shamir_sample_base_field(fs_state, n_elements);
            } else {
                new_fs_state, _ = fs_receive_base_field(fs_state, n_elements);
            }
            all_states[i + 1] = new_fs_state;
        }

        final_state = all_states[n];
        fiat_shamir_print_state(final_state);

        expected_state = final_state[0];

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
        fs_state[5] = 0; // output buffer size

        return fs_state;
    }

    fn less_than_8(a) -> 1 {
        if a * (a - 1) * (a - 2) * (a - 3) * (a - 4) * (a - 5) * (a - 6) * (a - 7) == 0 {
            return 1; // a < 8
        }
        return 0; // a >= 8
    }

    fn fiat_shamir_sample_base_field(fs_state, n) -> 2 {
        // return the updated fs_state, and a pointer to n field elements
        res = malloc(n);
        new_fs_state = fiat_shamir_sample_base_field_helper(fs_state, n, res);
        return new_fs_state, res;
    }

    fn fiat_shamir_sample_base_field_helper(fs_state, n, res) -> 1 {
        // return the updated fs_state
        // fill res with n field elements

        output_buffer_size = fs_state[5];
        output_buffer_ptr = fs_state[2] * 8;

        for i in 0..n {
            if output_buffer_size - i == 0 {
                break;
            }
            res[i] = output_buffer_ptr[7 - i];
        }

        finished = less_than_8(output_buffer_size - n);
        if finished == 1 {
            // no duplexing
            new_fs_state = malloc(6);
            new_fs_state[0] = fs_state[0];
            new_fs_state[1] = fs_state[1];
            new_fs_state[2] = fs_state[2];
            new_fs_state[3] = fs_state[3];
            new_fs_state[4] = fs_state[4];
            new_fs_state[5] = output_buffer_size - n;
            return new_fs_state;
        }

        // duplexing
        input_buffer_size = fs_state[3];
        input_buffer = fs_state[4];
        input_buffer_ptr = input_buffer * 8;
        l_ptr = 8 * fs_state[1];

        for i in input_buffer_size..8 {
            input_buffer_ptr[i] = l_ptr[i];
        }

        l_r = poseidon16(input_buffer, fs_state[2]);
        new_fs_state = malloc(6);
        new_fs_state[0] = fs_state[0];
        new_fs_state[1] = l_r;
        new_fs_state[2] = l_r + 1;
        new_fs_state[3] = 0; // reset input buffer size
        allocated = malloc_vec(1);
        new_fs_state[4] = allocated; // input buffer
        new_fs_state[5] = 8; // output_buffer_size

        remaining = n - output_buffer_size;
        if remaining == 0 {
            print(5);

            return new_fs_state;
        }

        shifted_res = res + output_buffer_size;
        final_res = fiat_shamir_sample_base_field_helper(new_fs_state, remaining, shifted_res);
        return final_res;

    }

    fn fs_receive_base_field(fs_state, n) -> 2 {
        // return the updated fs_state, and a pointer to n field elements

        transcript_ptr = fs_state[0];
        input_buffer_size = fs_state[3];
        input_buffer = fs_state[4];
        input_buffer_ptr = input_buffer * 8;

        for i in 0..n {
            input_buffer_ptr[input_buffer_size + i] = transcript_ptr[i];

            if input_buffer_size + i == 7 {
                break;
            }
        }

        finished = less_than_8(input_buffer_size + n);
        if finished == 1 {
            // no duplexing
            new_fs_state = malloc(6);
            new_fs_state[0] = transcript_ptr + n;
            new_fs_state[1] = fs_state[1];
            new_fs_state[2] = fs_state[2];
            new_fs_state[3] = input_buffer_size + n;
            new_fs_state[4] = fs_state[4];
            new_fs_state[5] = 0; // "Any buffered output is now invalid."
            return new_fs_state, transcript_ptr;
        }

        steps_done = 8 - input_buffer_size;

        // duplexing
        l_r = poseidon16(input_buffer, fs_state[2]);
        new_fs_state = malloc(6);
        new_fs_state[0] = transcript_ptr + steps_done;
        new_fs_state[1] = l_r;
        new_fs_state[2] = l_r + 1;
        new_fs_state[3] = 0; // reset input buffer size
        allocated = malloc_vec(1);
        new_fs_state[4] = allocated; // input buffer
        new_fs_state[5] = 8; // output_buffer_size

        remaining = n - steps_done;
        if remaining == 0 {
            return new_fs_state, transcript_ptr;
        }
        // continue reading
        final_fs_state, _ = fs_receive_base_field(new_fs_state, remaining);
        return final_fs_state, transcript_ptr;
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
    let n = 1000;

    let mut rng = StdRng::seed_from_u64(0);
    let challenger = build_challenger();

    let mut public_input = vec![F::from_usize(n)];
    let mut proof_data = vec![];
    let mut is_samples = vec![];
    let mut sizes = vec![];

    for _ in 0..n {
        let is_sample: bool = rng.random();
        let size = rng.random_range(1..30);
        is_samples.push(is_sample);
        sizes.push(size);
        if !is_sample {
            proof_data.extend((0..size).map(|_| rng.random()).collect::<Vec<F>>());
        }
    }

    let mut verifier_state = VerifierState::<F, EF, _>::new(proof_data.clone(), challenger);

    for (is_sample, size) in is_samples.iter().zip(&sizes) {
        if *is_sample {
            for _ in 0..*size {
                let _ = verifier_state.sample_bits(1);
            }
        } else {
            let _ = verifier_state.next_base_scalars_vec(*size);
        }
    }

    // public_input.extend(sizes.iter().map(|&x| F::from_usize(x)));
    public_input.extend(
        is_samples
            .iter()
            .zip(&sizes)
            .flat_map(|(&is_sample, &size)| vec![F::from_bool(is_sample), F::from_usize(size)]),
    );
    public_input.extend(proof_data.clone());
    public_input.extend(verifier_state.challenger().sponge_state.to_vec());

    dbg!(verifier_state.challenger().sponge_state);

    compile_and_run(program, &public_input, &[]);
}

#[test]
fn test_fiat_shamir_simple() {
    let program = r#"

    const F_BITS = 31;

    fn main() {
        start = public_input_start;
        n = start[0]; // n is assumed to be a multiple of 8 for padding

        fs_state = fs_new((start + (3 * n) + 8) / 8);

        all_states = malloc(n + 1);
        all_states[0] = fs_state;

        for i in 0..n {
            is_sample = start[(3 * i) + 1];
            n_elements = start[(3 * i) + 2];
            pow_bits = start[(3 * i) + 3];
            fs_state = all_states[i];
            if is_sample == 1 {
                fs_state_2, _ = fs_sample(fs_state, n_elements);
            } else {
                fs_state_2, _ = fs_receive(fs_state, n_elements);
            }
            print(pow_bits);
            fs_state_3 = fs_grinding(fs_state_2, pow_bits);
            all_states[i + 1] = fs_state_3;
        }

        final_state = all_states[n];
        fs_print_state(final_state);

        expected_state = final_state[0] * 8;

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
    // 0 -> transcript (vectorized pointer)
    // 1 -> vectorized pointer to first half of sponge state
    // 2 -> vectorized pointer to second half of sponge state
    // 3 -> output_buffer_size

    fn fs_new(transcript) -> 1 {
        // transcript is a (vectorized) pointer
        // TODO domain separator
        fs_state = malloc(4);
        fs_state[0] = transcript;
        fs_state[1] = pointer_to_zero_vector; // first half of sponge state
        fs_state[2] = pointer_to_zero_vector; // second half of sponge state
        fs_state[3] = 0; // output buffer size

        return fs_state;
    }

    fn fs_grinding(fs_state, bits) -> 1 {
        // WARNING: should not be called 2 times in a row without duplexing in between
        transcript_ptr = fs_state[0] * 8;
        l_ptr = fs_state[1] * 8;
        
        new_l = malloc_vec(1);
        new_l_ptr = new_l * 8;
        new_l_ptr[0] = transcript_ptr[0];
        for i in 1..8 unroll {
            new_l_ptr[i] = l_ptr[i];
        }

        l_r_updated = poseidon16(new_l, fs_state[2]);
        new_fs_state = malloc(4);
        new_fs_state[0] = fs_state[0] + 1; // read one 1 chunk of 8 field elements (7 are useless)
        new_fs_state[1] = l_r_updated;
        new_fs_state[2] = l_r_updated + 1;
        new_fs_state[3] = 7; // output_buffer_size

        l_updated_ptr = l_r_updated * 8;
        sampled = l_updated_ptr[7];
        sampled_bits = checked_decompose_bits(sampled);
        for i in 0..bits {
            assert sampled_bits[i] == 0;
        }
        return new_fs_state;
    }

    fn checked_decompose_bits(a) -> 1 {
        // return a pointer to bits of a
        bits = decompose_bits(a); // hint

        for i in 0..F_BITS unroll {
            assert bits[i] * (1 - bits[i]) == 0;
        }
        sums = malloc(F_BITS);
        sums[0] = bits[0];
        for i in 1..F_BITS unroll {
            sums[i] = sums[i - 1] + bits[i] * 2**i;
        }
        assert a == sums[F_BITS - 1];
        return bits;
    }

    fn less_than_8(a) -> 1 {
        if a * (a - 1) * (a - 2) * (a - 3) * (a - 4) * (a - 5) * (a - 6) * (a - 7) == 0 {
            return 1; // a < 8
        }
        return 0; // a >= 8
    }

    fn fs_sample(fs_state, n) -> 2 {
        // return the updated fs_state, and a pointer to n field elements
        res = malloc(n);
        new_fs_state = fs_sample_helper(fs_state, n, res);
        return new_fs_state, res;
    }

    fn fs_sample_helper(fs_state, n, res) -> 1 {
        // return the updated fs_state
        // fill res with n field elements

        output_buffer_size = fs_state[3];
        output_buffer_ptr = fs_state[1] * 8;

        for i in 0..n {
            if output_buffer_size - i == 0 {
                break;
            }
            res[i] = output_buffer_ptr[output_buffer_size - 1 - i];
        }

        finished = less_than_8(output_buffer_size - n);
        if finished == 1 {
            // no duplexing
            new_fs_state = malloc(4);
            new_fs_state[0] = fs_state[0];
            new_fs_state[1] = fs_state[1];
            new_fs_state[2] = fs_state[2];
            new_fs_state[3] = output_buffer_size - n;
            return new_fs_state;
        }

        // duplexing
        l_r = poseidon16(fs_state[1], fs_state[2]);
        new_fs_state = malloc(4);
        new_fs_state[0] = fs_state[0];
        new_fs_state[1] = l_r;
        new_fs_state[2] = l_r + 1;
        new_fs_state[3] = 8; // output_buffer_size

        remaining = n - output_buffer_size;
        if remaining == 0 {
            return new_fs_state;
        }

        shifted_res = res + output_buffer_size;
        final_res = fs_sample_helper(new_fs_state, remaining, shifted_res);
        return final_res;

    }

    fn fs_hint(fs_state, n) -> 2 {
        // return the updated fs_state, and a vectorized pointer to n chunk of 8 field elements

        res = fs_state[0];
        new_fs_state = malloc(4);
        new_fs_state[0] = res + n;
        new_fs_state[1] = fs_state[1];
        new_fs_state[2] = fs_state[2];
        new_fs_state[3] = fs_state[3];
        return new_fs_state, res; 
    }

    fn fs_receive(fs_state, n) -> 2 {
        // return the updated fs_state, and a vectorized pointer to n chunk of 8 field elements

        res = fs_state[0];
        final_fs_state = fs_observe(fs_state, n);
        return final_fs_state, res;
    }

    fn fs_observe(fs_state, n) -> 1 {
        // observe n chunk of 8 field elements from the transcript
        // and return the updated fs_state
        // duplexing
        l_r = poseidon16(fs_state[0], fs_state[2]);
        new_fs_state = malloc(4);
        new_fs_state[0] = fs_state[0] + 1;
        new_fs_state[1] = l_r;
        new_fs_state[2] = l_r + 1;
        new_fs_state[3] = 8; // output_buffer_size

        if n == 1 {
            return new_fs_state;
        } else {
            final_fs_state = fs_observe(new_fs_state, n - 1);
            return final_fs_state;
        }
    }

    fn fs_print_state(fs_state) {
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
    let n = 8 * 1; // must be a multiple of 8 for padding

    let mut rng = StdRng::seed_from_u64(0);

    let mut public_input = vec![];
    let mut is_samples = vec![];
    let mut sizes = vec![];
    let mut grinding_bits = vec![];

    let mut prover_state = build_prover_state::<EF>();

    for _ in 0..n {
        let is_sample: bool = rng.random();
        let size = rng.random_range(1..18);
        is_samples.push(is_sample);
        sizes.push(size);
        if is_sample {
            for _ in 0..size {
                let _ = prover_state.sample_bits(1);
            }
        } else {
            let random_data = (0..size * 8).map(|_| rng.random()).collect::<Vec<F>>();
            let _ = prover_state.add_base_scalars(&random_data);
        }
        let pow_bits = rng.random_range(1..10);
        grinding_bits.push(pow_bits);
        prover_state.pow_grinding(pow_bits);
    }

    let proof_data = prover_state.proof_data().to_vec();

    let mut verifier_state = build_verifier_state(&prover_state);

    for ((is_sample, size), pow_bits) in is_samples.iter().zip(&sizes).zip(&grinding_bits) {
        if *is_sample {
            for _ in 0..*size {
                let _ = verifier_state.sample_bits(1);
            }
        } else {
            let _ = verifier_state.next_base_scalars_vec(*size * 8);
        }
        verifier_state.check_pow_grinding(*pow_bits).unwrap();
    }

    public_input.push(F::from_usize(n));
    public_input.extend(is_samples.iter().zip(&sizes).zip(&grinding_bits).flat_map(
        |((&is_sample, &size), &pow_bits)| {
            vec![
                F::from_bool(is_sample),
                F::from_usize(size),
                F::from_usize(pow_bits),
            ]
        },
    ));
    public_input.extend(vec![F::ZERO; 7]); // padding
    public_input.extend(proof_data.clone());
    public_input.extend(verifier_state.challenger().sponge_state.to_vec());

    dbg!(verifier_state.challenger().sponge_state);

    compile_and_run(program, &public_input, &[]);
}
