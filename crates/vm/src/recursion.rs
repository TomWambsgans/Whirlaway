use crate::*;
use p3_field::BasedVectorSpace;
use rand::{Rng, rngs::StdRng};
use whir_p3::{
    dft::*,
    fiat_shamir::domain_separator::*,
    parameters::{errors::*, *},
    poly::{evals::*, multilinear::*},
    whir::{
        committer::{reader::*, writer::*},
        parameters::*,
        prover::*,
        statement::{weights::*, *},
        verifier::*,
    },
};

#[test]
pub fn test_whir_verif() {
    run_whir_verif();
}

pub fn run_whir_verif() {
    let program = r#"

    // 1 OOD QUERY PER ROUND, 0 GRINDING

    const W = 3; // in the extension field, X^8 = 3

    const N_VARS = 25;
    const LOG_INV_RATE = 1; 
    const N_ROUNDS = 3;

    const FOLDING_FACTOR_0 = 7;
    const FOLDING_FACTOR_1 = 4;
    const FOLDING_FACTOR_2 = 4;

    const RS_REDUCTION_FACTOR_0 = 5;
    const RS_REDUCTION_FACTOR_1 = 5;
    const RS_REDUCTION_FACTOR_2 = 5;

    fn main() {
        transcript_start = public_input_start / 8;
        fs_state = fs_new(transcript_start);

        fs_state_1, root_0, ood_point_0, ood_eval_0 = whir_parse_commitment(fs_state);

        // In the future point / eval will come from the PIOP
        fs_state_2, pcs_point = fs_hint(fs_state_1, N_VARS);
        fs_state_3, pcs_eval = fs_hint(fs_state_2, 1);
        fs_state_4, combination_randomness_gen_0 = fs_sample_ef(fs_state_3);  // vectorized pointer of len 1

        claimed_sum_side = mul_extension_vec(combination_randomness_gen_0, pcs_eval);
        claimed_sum = add_extension_vec(ood_eval_0, claimed_sum_side);

        fs_states = malloc(FOLDING_FACTOR_0 + 1);
        fs_states[0] = fs_state_4;

        claimed_sums = malloc(FOLDING_FACTOR_0 + 1);
        claimed_sums[0] = claimed_sum;

        randomness = malloc(FOLDING_FACTOR_0); // in reverse order

        for sc_round in 0..FOLDING_FACTOR_0 {
            fs_state_5, poly = fs_receive(fs_states[sc_round], 3); // vectorized pointer of len 1
            sum_over_boolean_hypercube = degree_two_polynomial_sum_at_0_and_1(poly);
            consistent = eq_extension_vec(sum_over_boolean_hypercube, claimed_sums[sc_round]);
            if consistent == 0 {
                panic();
            }
            fs_state_6, rand = fs_sample_ef(fs_state_5);  // vectorized pointer of len 1
            fs_states[sc_round + 1] = fs_state_6;
            new_claimed_sum = degree_two_polynomial_eval(poly, rand);
            claimed_sums[sc_round + 1] = new_claimed_sum;
            randomness[FOLDING_FACTOR_0 - 1 - sc_round] = rand;
        }

        fs_state_7 = fs_states[FOLDING_FACTOR_0];

        fs_print_state(fs_state_7);

        return;
    }

    fn degree_two_polynomial_sum_at_0_and_1(coeffs) -> 1 {
        // coeffs is a vectorized pointer to 3 chunks of 8 field elements
        // return a vectorized pointer to 1 chunk of 8 field elements
        a = add_extension_vec(coeffs, coeffs);
        b = add_extension_vec(a, coeffs + 1);
        c = add_extension_vec(b, coeffs + 2);
        return c;
    }

    fn degree_two_polynomial_eval(coeffs, point) -> 1 {
        // everythiing is vectorized pointers
        point_squared = mul_extension_vec(point, point);
        a_xx = mul_extension_vec(coeffs + 2, point_squared);
        b_x = mul_extension_vec(coeffs + 1, point);
        c = coeffs;
        res_0 = add_extension_vec(a_xx, b_x);
        res_1 = add_extension_vec(res_0, c);
        return res_1;
    }

    fn whir_parse_commitment(fs_state) -> 4 {
        fs_state_1, root = fs_receive(fs_state, 1); // vectorized pointer of len 1
        fs_state_2, ood_point = fs_sample_ef(fs_state_1);  // vectorized pointer of len 1
        fs_state_3, ood_eval = fs_receive(fs_state_2, 1); // vectorized pointer of len 1
        return fs_state_3, root, ood_point, ood_eval;
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

    fn fs_sample_ef(fs_state) -> 2 {
        // return the updated fs_state, and a vectorized pointer to 1 chunk of 8 field elements
        res = malloc_vec(1);
        new_fs_state = fs_sample_helper(fs_state, 8, res * 8);
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
            res[i] = output_buffer_ptr[7 - i];
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
        l, r = poseidon16(fs_state[1], fs_state[2]);
        new_fs_state = malloc(4);
        new_fs_state[0] = fs_state[0];
        new_fs_state[1] = l;
        new_fs_state[2] = r;
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
        l, r = poseidon16(fs_state[0], fs_state[2]);
        new_fs_state = malloc(4);
        new_fs_state[0] = fs_state[0] + 1;
        new_fs_state[1] = l;
        new_fs_state[2] = r;
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

    fn print_chunck(vec_ptr) {
        ptr = vec_ptr * 8;
        for i in 0..8 {
            print(ptr[i]);
        }
        return;
    }

    fn mul_extension_vec(a, b) -> 1 {
        c = malloc_vec(1);
        a_ptr = a * 8;
        b_ptr = b * 8;
        c_ptr = c * 8;
        mul_extension(a_ptr, b_ptr, c_ptr);
        return c;
    }

    fn mul_extension(a, b, c) {
        // a, b and c are pointers
        // c = a * b

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

        c[0] = c0;
        c[1] = c1;
        c[2] = c2;
        c[3] = c3;
        c[4] = c4;
        c[5] = c5;
        c[6] = c6;
        c[7] = c7;
        return;
    }

    fn add_extension_vec(a, b) -> 1 {
        c = malloc_vec(1);
        a_ptr = a * 8;
        b_ptr = b * 8;
        c_ptr = c * 8;
        add_extension(a_ptr, b_ptr, c_ptr);
        return c;
    }

    fn add_extension(a, b, c) {
        // a, b and c are pointers
        // c = a + b
        c[0] = a[0] + b[0];
        c[1] = a[1] + b[1];
        c[2] = a[2] + b[2];
        c[3] = a[3] + b[3];
        c[4] = a[4] + b[4];
        c[5] = a[5] + b[5];
        c[6] = a[6] + b[6];
        c[7] = a[7] + b[7];
        return;
    }

    fn eq_extension_vec(a, b) -> 1 {
        // a and b are vectorized pointers
        // return 1 if a == b, 0 otherwise
        a_ptr = a * 8;
        b_ptr = b * 8;
        res = eq_extension(a_ptr, b_ptr);
        return res;
    }

    fn eq_extension(a, b) -> 1 {
        // a and b are pointers
        // return 1 if a == b, 0 otherwise
        if a[0] != b[0] { return 0; }
        if a[1] != b[1] { return 0; }
        if a[2] != b[2] { return 0; }
        if a[3] != b[3] { return 0; }
        if a[4] != b[4] { return 0; }
        if a[5] != b[5] { return 0; }
        if a[6] != b[6] { return 0; }
        if a[7] != b[7] { return 0; }
        return 1; // a == b
    }

    fn print_chunk(vec) {
        // vec is a vectorized pointer
        ptr = vec * 8;
        for i in 0..8 {
            print(ptr[i]);
        }
        return;
    }

   "#;

    let poseidon16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let poseidon24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(0));

    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    let whir_params = ProtocolParameters {
        initial_statement: true,
        security_level: 128,
        pow_bits: 0,
        folding_factor: FoldingFactor::ConstantFromSecondRound(7, 4),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        rs_domain_initial_reduction_factor: 5,
        univariate_skip: false,
    };

    let num_variables = 25;

    let mv_params = MultivariateParameters::<EF>::new(num_variables);

    let params =
        WhirConfig::<EF, F, MerkleHash, MerkleCompress, MyChallenger>::new(mv_params, whir_params);
    assert_eq!(params.committment_ood_samples, 1);

    let mut rng = StdRng::seed_from_u64(0);
    let polynomial =
        EvaluationsList::<F>::new((0..1 << num_variables).map(|_| rng.random()).collect());

    let point = MultilinearPoint::<EF>::rand(&mut rng, num_variables);

    let mut statement = Statement::<EF>::new(num_variables);
    let eval = polynomial.evaluate(&point);
    let weights = Weights::evaluation(point.clone());
    statement.add_constraint(weights, eval);

    let challenger = MyChallenger::new(poseidon16);
    let domainsep = DomainSeparator::new(vec![]);

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    // Commit to the polynomial and produce a witness
    let committer = CommitmentWriter::new(&params);

    let dft = EvalsDft::<F>::new(1 << params.max_fft_size());

    let witness = committer
        .commit(&dft, &mut prover_state, polynomial)
        .unwrap();

    let mut public_input = prover_state.proof_data().to_vec();
    let commitment_size = public_input.len();
    assert_eq!(commitment_size, 16);
    public_input.extend(
        point
            .iter()
            .flat_map(|x| <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(x).to_vec()),
    );
    public_input.extend(<EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(&eval).to_vec());

    let prover = Prover(&params);

    prover
        .prove(&dft, &mut prover_state, statement.clone(), witness)
        .unwrap();

    public_input.extend(prover_state.proof_data()[commitment_size..].to_vec());

    let commitment_reader = CommitmentReader::new(&params);

    // Create a verifier with matching parameters
    let verifier = Verifier::new(&params);

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state =
        domainsep.to_verifier_state(prover_state.proof_data().to_vec(), challenger);

    // Parse the commitment
    let parsed_commitment = commitment_reader
        .parse_commitment::<8>(&mut verifier_state)
        .unwrap();

    verifier
        .verify(&mut verifier_state, &parsed_commitment, &statement)
        .unwrap();

    compile_and_run(program, &public_input, &[]);
}
