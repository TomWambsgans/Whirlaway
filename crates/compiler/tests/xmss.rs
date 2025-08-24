use compiler::*;
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::Permutation;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::get_poseidon16;

use vm::*;
use xmss::{WotsSecretKey, find_randomness_for_wots_encoding};

#[test]
fn test_verify_merkle_path() {
    let program = r#"
    const HEIGHT = 8;

    fn main() {
        public_input_start_ = public_input_start;
        private_input_start = public_input_start_[0];
        thing_to_hash = (public_input_start + 8) / 8;
        claimed_merkle_root = thing_to_hash + 1;
        are_left = (public_input_start + 8) + 16;
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

        hashed = malloc_vec(2);
        if is_left == 1 {
            poseidon16(thing_to_hash, neighbours, hashed);
        } else {
            poseidon16(neighbours, thing_to_hash, hashed);
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
    let leaf: [F; 8] = rng.random();
    let is_left = (0..height).map(|_| rng.random()).collect::<Vec<bool>>();

    let mut private_input = vec![];

    let mut to_hash = leaf.clone();
    let poseidon16 = get_poseidon16();
    for i in 0..height {
        let neighbour: [F; 8] = rng.random();
        if is_left[i] {
            to_hash = poseidon16.permute([to_hash, neighbour].concat().try_into().unwrap())[0..8]
                .try_into()
                .unwrap();
        } else {
            to_hash = poseidon16.permute([neighbour, to_hash].concat().try_into().unwrap())[0..8]
                .try_into()
                .unwrap();
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
    public_input.insert(
        0,
        F::from_usize((public_input.len() + 8 + PUBLIC_INPUT_START).next_power_of_two()),
    );
    public_input.splice(1..1, F::zero_vec(7));

    compile_and_run(program, &public_input, &private_input);

    dbg!(&merkle_root);
}

#[test]
fn test_wots_encode() {
    // Public input: message_hash | randomness | encoding
    let program = r#"
    const V = 68;
    const W = 4;

    fn main() {
        message_hash = public_input_start / 8;
        randomness = message_hash + 1;
        expected_encoding = public_input_start + 16;
        encoding = wots_encoding(message_hash, randomness);
        for i in 0..V unroll {
            assert encoding[i] == expected_encoding[i];
        }
        return;
    }

    fn wots_encoding(message_hash, randomness) -> 1 {
        // both arguments are vectorized pointers (of length 1)
        // return a normal pointer (of length V)
        compressed = malloc_vec(2);
        poseidon16(message_hash, randomness, compressed);
        compressed_ptr = compressed * 8;
        bits = decompose_bits(compressed_ptr[0], compressed_ptr[1], compressed_ptr[2], compressed_ptr[3], compressed_ptr[4], compressed_ptr[5]);
        flipped_bits = malloc(186);
        for i in 0..186 unroll {
            flipped_bits[i] = 1 - bits[i];
        }
        zero = 0;
        for i in 0..186 unroll {
            zero = flipped_bits[i] * bits[i]; // TODO remove the use of auxiliary var (currently it generates 2 instructions instead of 1)
        }
        res = malloc(12 * 6);
        for i in 0..6 unroll {
            for j in 0..12 unroll {
                res[i * 12 + j] = bits[i * 31 + j * 2] + 2 * bits[i * 31 + j * 2 + 1];
            }
        }

        // we need to check that the (hinted) bit decomposition is valid

        for i in 0..6 unroll {
            powers_scaled_w = malloc(12);
            for j in 0..12 unroll {
                powers_scaled_w[j] = res[i*12 + j] * W**j;
            }
            powers_scaled_sum_w = malloc(11);
            powers_scaled_sum_w[0] = powers_scaled_w[0] + powers_scaled_w[1];
            for j in 1..11 unroll {
                powers_scaled_sum_w[j] = powers_scaled_sum_w[j - 1] + powers_scaled_w[j + 1];
            }

            powers_scaled_2 = malloc(7);
            for j in 0..7 unroll {
                powers_scaled_2[j] = bits[31 * i + 24 + j] * 2**(24 + j);
            }
            powers_scaled_sum_2 = malloc(6);
            powers_scaled_sum_2[0] = powers_scaled_2[0] + powers_scaled_2[1];
            for j in 1..6 unroll {
                powers_scaled_sum_2[j] = powers_scaled_sum_2[j - 1] + powers_scaled_2[j + 1];
            }

            assert powers_scaled_sum_w[10] + powers_scaled_sum_2[5] == compressed_ptr[i];
        }

        return res;
    }
   "#;

    let mut rng = StdRng::seed_from_u64(0);
    let message_hash: [F; 8] = rng.random();
    let (randomness, encoding) = find_randomness_for_wots_encoding(&message_hash, &mut rng);

    let mut public_input = message_hash.to_vec();
    public_input.extend(randomness.to_vec());
    public_input.extend(encoding.iter().map(|&x| F::from_u8(x)));

    compile_and_run(program, &public_input, &[]);
}

#[test]
fn test_verify_wots_signature() {
    // Public input: wots public_key_hash | message_hash
    // Private input: signature = randomness | chain_tips
    let program = r#"

    const V = 68;
    const W = 4;

    fn main() {
        public_input_start_ = public_input_start;
        private_input_start = public_input_start_[0];
        wots_public_key_hash = (public_input_start + 8) / 8;
        message_hash = wots_public_key_hash + 1;
        signature = private_input_start / 8;
        wots_public_key_hash_recovered = wots_recover_pub_key_hashed(message_hash, signature);
        assert_eq_ext(wots_public_key_hash, wots_public_key_hash_recovered);
        // print_chunk_of_8(wots_public_key_hash_recovered);
        return;
    }

    fn wots_recover_pub_key_hashed(message_hash, signature) -> 1 {
        // message_hash: vectorized pointers (of length 1)
        // signature: vectorized pointer = randomness | chain_tips
        // return a vectorized pointer (of length 1), the hashed public key
        
        randomness = signature;
        chain_tips = signature + 1;

        // 1) We encode message_hash + randomness into the d-th layer of the hypercube

        compressed = malloc_vec(2);
        poseidon16(message_hash, randomness, compressed);
        compressed_ptr = compressed * 8;
        bits = decompose_bits(compressed_ptr[0], compressed_ptr[1], compressed_ptr[2], compressed_ptr[3], compressed_ptr[4], compressed_ptr[5]);
        flipped_bits = malloc(186);
        for i in 0..186 unroll {
            flipped_bits[i] = 1 - bits[i];
        }
        zero = 0;
        for i in 0..186 unroll {
            zero = flipped_bits[i] * bits[i]; // TODO remove the use of auxiliary var (currently it generates 2 instructions instead of 1)
        }
        encoding = malloc(12 * 6);
        for i in 0..6 unroll {
            for j in 0..12 unroll {
                encoding[i * 12 + j] = bits[i * 31 + j * 2] + 2 * bits[i * 31 + j * 2 + 1];
            }
        }

        // we need to check that the (hinted) bit decomposition is valid

        for i in 0..6 unroll {
            powers_scaled_w = malloc(12);
            for j in 0..12 unroll {
                powers_scaled_w[j] = encoding[i*12 + j] * W**j;
            }
            powers_scaled_sum_w = malloc(11);
            powers_scaled_sum_w[0] = powers_scaled_w[0] + powers_scaled_w[1];
            for j in 1..11 unroll {
                powers_scaled_sum_w[j] = powers_scaled_sum_w[j - 1] + powers_scaled_w[j + 1];
            }

            powers_scaled_2 = malloc(7);
            for j in 0..7 unroll {
                powers_scaled_2[j] = bits[31 * i + 24 + j] * 2**(24 + j);
            }
            powers_scaled_sum_2 = malloc(6);
            powers_scaled_sum_2[0] = powers_scaled_2[0] + powers_scaled_2[1];
            for j in 1..6 unroll {
                powers_scaled_sum_2[j] = powers_scaled_sum_2[j - 1] + powers_scaled_2[j + 1];
            }

            assert powers_scaled_sum_w[10] + powers_scaled_sum_2[5] == compressed_ptr[i];
        }

        // FANCY OPTIMIZATION IDEA:
        // We know for sure that each number n of the encoding is < W. We can use a JUMP to a + n * b (a, b two constants).

        public_key = malloc_vec(V * 2);
        for i in 0..V / 2 unroll {
            if encoding[2 * i] == 2 {
                poseidon16(pointer_to_zero_vector, chain_tips + 2 * i, public_key + 4 * i);
            } else {
                if encoding[2 * i] == 1 {
                    chain_hash_2 = malloc_vec(2);
                    poseidon16(pointer_to_zero_vector, chain_tips +  2 * i, chain_hash_2);
                    poseidon16(pointer_to_zero_vector, chain_hash_2 + 1, public_key + 4 * i);
                } else {
                    if encoding[2 * i] == 3 {
                        // TODO single-instruction copy (trick based on DOT_PRODUCT precompile)
                        public_key_ptr = (public_key + 4 * i + 1) * 8;
                        chain_tips_ptr = (chain_tips + 2 * i) * 8;
                        for j in 0..8 unroll {
                            public_key_ptr[j] = chain_tips_ptr[j];
                        }
                    } else {
                        // encoding[2 * i] == 0
                        chain_hash_1 = malloc_vec(2);
                        chain_hash_2 = malloc_vec(2);
                        poseidon16(pointer_to_zero_vector, chain_tips + 2 * i, chain_hash_1);
                        poseidon16(pointer_to_zero_vector, chain_hash_1 + 1, chain_hash_2);
                        poseidon16(pointer_to_zero_vector, chain_hash_2 + 1, public_key + 4 * i);
                    }
                }
            }
        }

        for i in 0..V / 2 unroll {
            if encoding[2 * i + 1] == 2 {
                poseidon16(chain_tips + 2 * i + 1, pointer_to_zero_vector, public_key + 4 * i + 2);
            } else {
                if encoding[2 * i + 1] == 1 {
                    chain_hash_2 = malloc_vec(2);
                    poseidon16(chain_tips +  2 * i + 1, pointer_to_zero_vector, chain_hash_2);
                    poseidon16(chain_hash_2, pointer_to_zero_vector, public_key + (4 * i) + 2);
                } else {
                    if encoding[2 * i + 1] == 3 {
                        // TODO single-instruction copy (trick based on DOT_PRODUCT precompile)
                        public_key_ptr = (public_key + 4 * i + 2) * 8;
                        chain_tips_ptr = (chain_tips + 2 * i + 1) * 8;
                        for j in 0..8 unroll {
                            public_key_ptr[j] = chain_tips_ptr[j];
                        }
                    } else {
                        // encoding[2 * i + 1] == 0
                        chain_hash_1 = malloc_vec(2);
                        chain_hash_2 = malloc_vec(2);
                        poseidon16(chain_tips + 2 * i + 1, pointer_to_zero_vector, chain_hash_1);
                        poseidon16(chain_hash_1, pointer_to_zero_vector, chain_hash_2);
                        poseidon16(chain_hash_2, pointer_to_zero_vector, public_key + 4 * i + 2);
                    }
                }
            }
        }


        public_key_hashed = malloc_vec(V / 2);
        poseidon24(public_key + 1, pointer_to_zero_vector, public_key_hashed);


        for i in 1..V / 2 unroll {
            poseidon24(public_key + (4 * i + 1), public_key_hashed + (i - 1), public_key_hashed + i);
        }

        return public_key_hashed + (V / 2 - 1);
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
        for i in 0..8 unroll {
            a_i = a_shifted[i];
            b_i = b_shifted[i];
            assert a_i == b_i;
        }
        return;
    }

   "#;

    let mut rng = StdRng::seed_from_u64(0);
    let message_hash: [F; 8] = rng.random();
    let wots_secret_key = WotsSecretKey::random(&mut rng);
    let signature = wots_secret_key.sign(&message_hash, &mut rng);
    let public_key_hashed = wots_secret_key.public_key().hash();
    // dbg!(public_key_hashed);

    let mut public_input = public_key_hashed.to_vec();
    public_input.extend(message_hash);

    let mut private_input = signature.randomness.to_vec();
    private_input.extend(
        signature
            .chain_tips
            .iter()
            .flat_map(|digest| digest.to_vec()),
    );

    public_input.insert(
        0,
        F::from_usize((public_input.len() + 8 + PUBLIC_INPUT_START).next_power_of_two()),
    );
    public_input.splice(1..1, F::zero_vec(7));

    compile_and_run(program, &public_input, &private_input);
}

// #[test]
// fn test_verify_xmss_signature() {
//     // Public input: xmss public key | message
//     // Private input: xmss signature = wots signatue (N_CHAINS x 8) + merkle proof (neighbours: XMSS_MERKLE_HEIGHT x 8 + is_left: XMSS_MERKLE_HEIGHT)
//     let program = r#"

//     const N_CHAINS = 64;
//     const CHAIN_LENGTH = 8;
//     const XMSS_MERKLE_HEIGHT = 5;

//     fn main() {
//         public_input_start_ = public_input_start;
//         private_input_start = public_input_start_[0];
//         xmss_public_key = (public_input_start + 8) / 8;
//         message = public_input_start + 16;
//         wots_signature = private_input_start / 8;
//         merkle_path = wots_signature + N_CHAINS;
//         xmss_public_key_recovered = verify_xmss(message, wots_signature, merkle_path);
//         assert_eq_ext(xmss_public_key, xmss_public_key_recovered);
//         print_chunk_of_8(xmss_public_key);
//         return;
//     }

//     fn verify_xmss(message, wots_signature, merkle_path) -> 1 {
//         // merkle_path: vectorized pointer to neighbours: XMSS_MERKLE_HEIGHT x 8 followed by is_left: XMSS_MERKLE_HEIGHT
//         wots_public_key = recover_wots_public_key(message, wots_signature);
//         wots_public_key_hash = hash_wots_public_key(wots_public_key);
//         are_left_vec = merkle_path + XMSS_MERKLE_HEIGHT;
//         are_left = are_left_vec * 8;
//         merkle_root = merkle_step(0, XMSS_MERKLE_HEIGHT, wots_public_key_hash, are_left, merkle_path);
//         return merkle_root;
//     }

//     fn recover_wots_public_key(message, signature) -> 1 {
//         // message: pointer
//         // signature: vectorized pointer
//         // return a pointer of vectorized pointers

//         public_key = malloc_vec(N_CHAINS);
//         for i in 0..N_CHAINS {
//             msg_i = message[i];
//             n_hash_iter = CHAIN_LENGTH - msg_i;
//             signature_i = signature + i;
//             pk_i = hash_chain(signature_i, n_hash_iter);
//             copy_vector(pk_i, public_key + i);
//         }
//         return public_key;
//     }

//     fn hash_chain(thing_to_hash, n_iter) -> 1 {
//         if n_iter == 0 {
//             return thing_to_hash;
//         }
//         hashed = poseidon16(thing_to_hash, pointer_to_zero_vector);
//         n_iter_minus_one = n_iter - 1;
//         res = hash_chain(hashed, n_iter_minus_one);
//         return res;
//     }

//     fn hash_wots_public_key(public_key) -> 1 {
//         hashes = malloc(N_CHAINS / 2 + 1);
//         hashes[0] = pointer_to_zero_vector;
//         for i in 0..N_CHAINS / 2 {
//             next = poseidon24(public_key + 2 * i, hashes[i]);
//             hashes[i + 1] = next;
//         }
//         res = hashes[N_CHAINS / 2];
//         return res;
//     }

//     fn merkle_step(step, height, thing_to_hash, are_left, neighbours) -> 1 {
//         if step == height {
//             return thing_to_hash;
//         }
//         is_left = are_left[0];

//         if is_left == 1 {
//             hashed = poseidon16(thing_to_hash, neighbours);
//         } else {
//             hashed = poseidon16(neighbours, thing_to_hash);
//         }

//         next_step = step + 1;
//         next_are_left = are_left + 1;
//         next_neighbours = neighbours + 1;
//         res = merkle_step(next_step, height, hashed, next_are_left, next_neighbours);
//         return res;
//     }

//     fn print_chunk_of_8(arr) {
//         reindexed_arr = arr * 8;
//         for i in 0..8 {
//             arr_i = reindexed_arr[i];
//             print(arr_i);
//         }
//         return;
//     }

//     fn assert_eq_ext(a, b) {
//         // a and b both pointers in the memory of chunk of 8 field elements
//         a_shifted = a * 8;
//         b_shifted = b * 8;
//         for i in 0..8 {
//             a_i = a_shifted[i];
//             b_i = b_shifted[i];
//             assert a_i == b_i;
//         }
//         return;
//     }

//     fn copy_vector(a, b) {
//         // a and b both pointers in the memory of chunk of 8 field elements
//         a_shifted = a * 8;
//         b_shifted = b * 8;
//         for i in 0..8 {
//             a_i = a_shifted[i];
//             b_shifted[i] = a_i;
//         }
//         return;
//     }
//    "#;

//     let mut rng = StdRng::seed_from_u64(0);
//     let mesage = random_message(&mut rng);
//     let xmss_secret_key: XmssSecretKey =
//         XmssSecretKey::random(&mut rng, &get_poseidon16(), &get_poseidon24());
//     let index = rng.random_range(0..1 << XMSS_MERKLE_HEIGHT);
//     let signature = xmss_secret_key.sign(&mesage, index, &get_poseidon16());

//     let mut public_input = xmss_secret_key.public_key().root.to_vec();
//     public_input.extend(mesage.iter().map(|&x| F::new(x as u32)));

//     let mut private_input = signature
//         .wots_signature
//         .0
//         .iter()
//         .flat_map(|digest| digest.to_vec())
//         .collect::<Vec<F>>();
//     private_input.extend(
//         signature
//             .merkle_proof
//             .iter()
//             .flat_map(|(_, neighbour)| *neighbour),
//     );
//     private_input.extend(
//         signature
//             .merkle_proof
//             .iter()
//             .map(|(is_left, _)| F::new(*is_left as u32)),
//     );

//     public_input.insert(
//         0,
//         F::from_usize((public_input.len() + 8 + PUBLIC_INPUT_START).next_power_of_two()),
//     );
//     public_input.splice(1..1, F::zero_vec(7));

//     compile_and_run(program, &public_input, &private_input);

//     dbg!(xmss_secret_key.public_key().root);
// }

// #[test]
// fn test_aggregate_xmss_signatures() {
//     const N_PUBLIC_KEYS: usize = 10;

//     // Public input: message (N_CHAINS) | xmss public keys (N_PUBLIC_KEYS x 8) | bitfield (N_PUBLIC_KEYS)
//     // Private input: xmss_signatures, alligned by 8
//     // each xmss signature (N_CHAINS x 8 + XMSS_MERKLE_HEIGHT x 9) = wots signatue (N_CHAINS x 8) + merkle proof (neighbours: XMSS_MERKLE_HEIGHT x 8 + is_left: XMSS_MERKLE_HEIGHT)
//     let program = r#"

//     const N_PUBLIC_KEYS = 10;
//     const N_CHAINS = 64;
//     const CHAIN_LENGTH = 8;
//     const XMSS_MERKLE_HEIGHT = 5;
//     const XMSS_SIGNATURE_SIZE_ROUNDED_UP = 70; // number of chuncks of 8

//     const VERIF_SUCCESSFUL = 1;

//     fn main() {
//         public_input_start_ = public_input_start;
//         private_input_start = public_input_start_[0];
//         message = public_input_start + 8;
//         xmss_public_keys = (public_input_start + N_CHAINS + 8) / 8;
//         bitfield = public_input_start + 8 + 144; // message + N_PUBLIC_KEYS x 8

//         bitfield_counter = malloc(N_PUBLIC_KEYS + 1);
//         bitfield_counter[0] = 0;

//         for i in 0..N_PUBLIC_KEYS {
//             if bitfield[i] == 1 {
//                 xmss_public_key = xmss_public_keys + i;
//                 wots_signature_offset = private_input_start / 8;
//                 wots_signature_shift = bitfield_counter[i] * XMSS_SIGNATURE_SIZE_ROUNDED_UP;
//                 wots_signature = wots_signature_offset + wots_signature_shift;
//                 merkle_path = wots_signature + N_CHAINS;
//                 xmss_public_key_recovered = verify_xmss(message, wots_signature, merkle_path);
//                 assert_eq_ext(xmss_public_key, xmss_public_key_recovered);
//                 print(VERIF_SUCCESSFUL);

//                 bitfield_counter[i + 1] = bitfield_counter[i] + 1;
//             } else {
//                 bitfield_counter[i + 1] = bitfield_counter[i];
//             }
//         }
//         return;
//     }

//     fn verify_xmss(message, wots_signature, merkle_path) -> 1 {
//         // merkle_path: vectorized pointer to neighbours: XMSS_MERKLE_HEIGHT x 8 followed by is_left: XMSS_MERKLE_HEIGHT
//         wots_public_key = recover_wots_public_key(message, wots_signature);
//         wots_public_key_hash = hash_wots_public_key(wots_public_key);
//         merkle_root = merkle_step(0, XMSS_MERKLE_HEIGHT, wots_public_key_hash, (merkle_path + XMSS_MERKLE_HEIGHT) * 8, merkle_path);
//         return merkle_root;
//     }

//     fn recover_wots_public_key(message, signature) -> 1 {
//         // message: pointer
//         // signature: vectorized pointer
//         // return a pointer of vectorized pointers

//         public_key = malloc_vec(N_CHAINS);
//         for i in 0..N_CHAINS {
//             msg_i = message[i];
//             n_hash_iter = CHAIN_LENGTH - msg_i;
//             signature_i = signature + i;
//             pk_i = hash_chain(signature_i, n_hash_iter);
//             copy_vector(pk_i, public_key + i);
//         }
//         return public_key;
//     }

//     fn hash_chain(thing_to_hash, n_iter) -> 1 {
//         if n_iter == 0 {
//             return thing_to_hash;
//         }
//         hashed = poseidon16(thing_to_hash, pointer_to_zero_vector);
//         n_iter_minus_one = n_iter - 1;
//         res = hash_chain(hashed, n_iter_minus_one);
//         return res;
//     }

//     fn hash_wots_public_key(public_key) -> 1 {
//         hashes = malloc(N_CHAINS / 2 + 1);
//         hashes[0] = pointer_to_zero_vector;
//         for i in 0..N_CHAINS / 2 {
//             next = poseidon24(public_key + 2 * i, hashes[i]);
//             hashes[i + 1] = next;
//         }
//         res = hashes[N_CHAINS / 2];
//         return res;
//     }

//     fn merkle_step(step, height, thing_to_hash, are_left, neighbours) -> 1 {
//         if step == height {
//             return thing_to_hash;
//         }
//         is_left = are_left[0];

//         if is_left == 1 {
//             hashed = poseidon16(thing_to_hash, neighbours);
//         } else {
//             hashed = poseidon16(neighbours, thing_to_hash);
//         }

//         next_step = step + 1;
//         next_are_left = are_left + 1;
//         next_neighbours = neighbours + 1;
//         res = merkle_step(next_step, height, hashed, next_are_left, next_neighbours);
//         return res;
//     }

//     fn print_chunk_of_8(arr) {
//         reindexed_arr = arr * 8;
//         for i in 0..8 {
//             arr_i = reindexed_arr[i];
//             print(arr_i);
//         }
//         return;
//     }

//     fn assert_eq_ext(a, b) {
//         // a and b both pointers in the memory of chunk of 8 field elements
//         a_shifted = a * 8;
//         b_shifted = b * 8;
//         for i in 0..8 {
//             a_i = a_shifted[i];
//             b_i = b_shifted[i];
//             assert a_i == b_i;
//         }
//         return;
//     }

//     fn copy_vector(a, b) {
//         // a and b both pointers in the memory of chunk of 8 field elements
//         a_shifted = a * 8;
//         b_shifted = b * 8;
//         for i in 0..8 {
//             a_i = a_shifted[i];
//             b_shifted[i] = a_i;
//         }
//         return;
//     }
//    "#;

//     let mut rng = StdRng::seed_from_u64(0);
//     let mesage = random_message(&mut rng);

//     let bitfield = (0..N_PUBLIC_KEYS)
//         .map(|_| rng.random())
//         .collect::<Vec<bool>>();
//     assert!(bitfield.iter().filter(|&&x| x).count() >= 3, "change seed");

//     let xmss_secret_keys = (0..N_PUBLIC_KEYS)
//         .map(|_| XmssSecretKey::random(&mut rng, &get_poseidon16(), &get_poseidon24()))
//         .collect::<Vec<_>>();
//     let mut xmss_public_keys = vec![];
//     for xmss_secret_key in &xmss_secret_keys {
//         xmss_public_keys.push(xmss_secret_key.public_key().root.to_vec());
//     }

//     let signatures = bitfield
//         .iter()
//         .enumerate()
//         .filter_map(|(i, &bit)| {
//             if bit {
//                 let index = rng.random_range(0..1 << XMSS_MERKLE_HEIGHT);
//                 Some(xmss_secret_keys[i].sign(&mesage, index, &get_poseidon16()))
//             } else {
//                 None
//             }
//         })
//         .collect::<Vec<_>>();

//     let mut public_input = mesage.iter().map(|&x| F::new(x as u32)).collect::<Vec<F>>();
//     for pk in xmss_public_keys {
//         public_input.extend(pk);
//     }
//     for &bit in &bitfield {
//         if bit {
//             public_input.push(F::ONE);
//         } else {
//             public_input.push(F::ZERO);
//         }
//     }

//     let mut private_input = vec![];
//     for signature in &signatures {
//         private_input.extend(
//             signature
//                 .wots_signature
//                 .0
//                 .iter()
//                 .flat_map(|digest| digest.to_vec()),
//         );
//         private_input.extend(
//             signature
//                 .merkle_proof
//                 .iter()
//                 .flat_map(|(_, neighbour)| *neighbour),
//         );
//         private_input.extend(
//             signature
//                 .merkle_proof
//                 .iter()
//                 .map(|(is_left, _)| F::new(*is_left as u32)),
//         );
//         // add padding to align with 8
//         let padding = vec![F::ZERO; (8 - (private_input.len() % 8)) % 8];
//         private_input.extend(padding);
//     }

//     public_input.insert(
//         0,
//         F::from_usize((public_input.len() + 8 + PUBLIC_INPUT_START).next_power_of_two()),
//     );
//     public_input.splice(1..1, F::zero_vec(7));

//     compile_and_run(program, &public_input, &private_input);
// }
