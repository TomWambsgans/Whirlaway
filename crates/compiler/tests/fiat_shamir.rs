use compiler::*;
use p3_field::PrimeCharacteristicRing;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::{build_challenger, build_prover_state, build_verifier_state};

use vm::*;
use whir_p3::fiat_shamir::verifier::VerifierState;

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

        l_r = malloc_vec(2);
        poseidon16(input_buffer, fs_state[2], l_r);
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
        l_r = malloc_vec(2);
        poseidon16(input_buffer, fs_state[2], l_r);
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

        l_r_updated = malloc_vec(2);
        poseidon16(new_l, fs_state[2], l_r_updated);
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
        l_r = malloc_vec(2);
        poseidon16(fs_state[1], fs_state[2], l_r);
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
        l_r = malloc_vec(2);
        poseidon16(fs_state[0], fs_state[2], l_r);
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
