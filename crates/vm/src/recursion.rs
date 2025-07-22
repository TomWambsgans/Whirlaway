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
    const FOLDING_FACTOR_3 = 4;

    const FINAL_VARS = N_VARS - (FOLDING_FACTOR_0 + FOLDING_FACTOR_1 + FOLDING_FACTOR_2 + FOLDING_FACTOR_3);
    const TWO_POW_FINAL_VARS = 64;

    const TWO_POW_FOLDING_FACTOR_0 = 128;
    const TWO_POW_FOLDING_FACTOR_1 = 16;
    const TWO_POW_FOLDING_FACTOR_2 = 16;
    const TWO_POW_FOLDING_FACTOR_3 = 16;

    const RS_REDUCTION_FACTOR_0 = 5;
    const RS_REDUCTION_FACTOR_1 = 1;
    const RS_REDUCTION_FACTOR_2 = 1;
    const RS_REDUCTION_FACTOR_3 = 1;

    const NUM_QUERIES_0 = 138;
    const NUM_QUERIES_1 = 44;
    const NUM_QUERIES_2 = 22;
    const NUM_QUERIES_3 = 15;

    const ROOT_19 = 339671193;
    const ROOT_18 = 1816824389;
    const ROOT_17 = 373019801;
    const ROOT_16 = 1848593786;
    const ROOT_15 = 1168510561;
    const ROOT_14 = 1047035213;
    const ROOT_13 = 809067698;
    const ROOT_12 = 1989134074;
    const ROOT_11 = 2000983452;
    const ROOT_10 = 860702919;
    const ROOT_9 = 665670555;
    const ROOT_8 = 392596362;
    const ROOT_7 = 699882112;
    const ROOT_6 = 1548376985;
    const ROOT_5 = 170455089;
    const ROOT_4 = 148625052;
    const ROOT_3 = 1748172362;
    const ROOT_2 = 2113994754;
    const ROOT_1 = 2130706432;
    const ROOT_0 = 1;

    fn main() {
        transcript_start = public_input_start / 8;
        fs_state = fs_new(transcript_start);

        fs_state_1, root_0, ood_point_0, ood_eval_0 = parse_commitment(fs_state);

        // In the future point / eval will come from the PIOP
        fs_state_2, pcs_point = fs_hint(fs_state_1, N_VARS);
        fs_state_3, pcs_eval = fs_hint(fs_state_2, 1);
        fs_state_4, combination_randomness_gen_0 = fs_sample_ef(fs_state_3);  // vectorized pointer of len 1

        claimed_sum_side = mul_extension_ret(combination_randomness_gen_0, pcs_eval);
        claimed_sum_0 = add_extension_ret(ood_eval_0, claimed_sum_side);

        domain_size_0 = N_VARS + LOG_INV_RATE;
        fs_state_5, folding_randomness_1, ood_point_1, root_1, circle_values_1, combination_randomness_powers_1, claimed_sum_1 = 
            whir_round(fs_state_4, root_0, FOLDING_FACTOR_0, TWO_POW_FOLDING_FACTOR_0, 1, NUM_QUERIES_0, domain_size_0, claimed_sum_0);

        domain_size_1 = domain_size_0 - RS_REDUCTION_FACTOR_0;
        fs_state_6, folding_randomness_2, ood_point_2, root_2, circle_values_2, combination_randomness_powers_2, claimed_sum_2 = 
            whir_round(fs_state_5, root_1, FOLDING_FACTOR_1, TWO_POW_FOLDING_FACTOR_1, 0, NUM_QUERIES_1, domain_size_1, claimed_sum_1);

        domain_size_2 = domain_size_1 - RS_REDUCTION_FACTOR_1;
        fs_state_7, folding_randomness_3, ood_point_3, root_3, circle_values_3, combination_randomness_powers_3, claimed_sum_3 = 
            whir_round(fs_state_6, root_2, FOLDING_FACTOR_2, TWO_POW_FOLDING_FACTOR_2, 0, NUM_QUERIES_2, domain_size_2, claimed_sum_2);

        domain_size_3 = domain_size_2 - RS_REDUCTION_FACTOR_2;
        fs_state_8, folding_randomness_4, final_claimed_sum = sumcheck(fs_state_7, FOLDING_FACTOR_3, claimed_sum_3);
        fs_state_9, final_coeffcients = fs_receive(fs_state_8, TWO_POW_FINAL_VARS);
        fs_state_10, final_circle_values, final_folds = sample_stir_indexes_and_fold(fs_state_9, NUM_QUERIES_3, 0, FOLDING_FACTOR_3, TWO_POW_FOLDING_FACTOR_3, domain_size_3, root_3, folding_randomness_4);

        // 68711 cycles
        for i in 0..NUM_QUERIES_3 {
            powers_of_2_rev = powers_of_two_rev_base(final_circle_values[i], FINAL_VARS);
            poly_eq = poly_eq_base(powers_of_2_rev, FINAL_VARS, TWO_POW_FINAL_VARS);
            final_pol_evaluated_on_circle = malloc_vec(1);
            dot_product_base_extension(poly_eq, final_coeffcients, final_pol_evaluated_on_circle, TWO_POW_FINAL_VARS);
            correct_eval = eq_extension(final_pol_evaluated_on_circle, final_folds + i);
            assert correct_eval == 1;
        }

        fs_state_11, folding_randomness_5, end_sum = sumcheck(fs_state_10, FINAL_VARS, final_claimed_sum);

        folding_randomness_global = malloc_vec(N_VARS);

        ffs = malloc(N_ROUNDS + 2);
        ffs[0] = FOLDING_FACTOR_0; ffs[1] = FOLDING_FACTOR_1; ffs[2] = FOLDING_FACTOR_2; ffs[3] = FOLDING_FACTOR_3; ffs[4] = FINAL_VARS;
        frs = malloc(N_ROUNDS + 2);
        frs[0] = folding_randomness_5; frs[1] = folding_randomness_4; frs[2] = folding_randomness_3; frs[3] = folding_randomness_2; frs[4] = folding_randomness_1;
        ffs_sums = malloc(N_ROUNDS + 2);
        ffs_sums[0] = FOLDING_FACTOR_0;
        for i in 0..N_ROUNDS + 1 {
            ffs_sums[i + 1] = ffs_sums[i] + ffs[i + 1];
        }
        for i in 0..N_ROUNDS + 2 {
            start = folding_randomness_global + N_VARS - ffs_sums[N_ROUNDS + 1 - i];
            for j in 0..ffs[N_ROUNDS + 1 - i] {
                copy_chunk_vec(frs[i] + j, start + j);
            }
        }

        ood_0_expanded_from_univariate = powers_of_two_rev_extension(ood_point_0, N_VARS);
        s0 = eq_mle_extension(ood_0_expanded_from_univariate, folding_randomness_global, N_VARS);
        s1 = eq_mle_extension(pcs_point, folding_randomness_global, N_VARS);
        s3 = mul_extension_ret(s1, combination_randomness_gen_0);
        s4 = add_extension_ret(s0, s3);

        weight_sums = malloc(N_ROUNDS + 1);
        weight_sums[0] = s4;

        ood_points = malloc(N_ROUNDS + 1); ood_points[0] = ood_point_0; ood_points[1] = ood_point_1; ood_points[2] = ood_point_2; ood_points[3] = ood_point_3;
        num_queries = malloc(N_ROUNDS + 1); num_queries[0] = NUM_QUERIES_0; num_queries[1] = NUM_QUERIES_1; num_queries[2] = NUM_QUERIES_2; num_queries[3] = NUM_QUERIES_3;
        circle_values = malloc(N_ROUNDS + 1); circle_values[0] = circle_values_1; circle_values[1] = circle_values_2; circle_values[2] = circle_values_3; circle_values[3] = final_circle_values;
        combination_randomness_powers = malloc(N_ROUNDS); combination_randomness_powers[0] = combination_randomness_powers_1; combination_randomness_powers[1] = combination_randomness_powers_2; combination_randomness_powers[2] = combination_randomness_powers_3;

        for i in 0..N_ROUNDS {
            ood_expanded_from_univariate = powers_of_two_rev_extension(ood_points[i + 1], N_VARS - ffs_sums[i]); // 456 cycles
            s5 = eq_mle_extension(ood_expanded_from_univariate, folding_randomness_global, N_VARS - ffs_sums[i]); // 2248 cycles
            s6s = malloc_vec(num_queries[i] + 1);
            copy_chunk_vec(s5, s6s);
            circle_value_i = circle_values[i];
            for j in 0..num_queries[i] {
                expanded_from_univariate = powers_of_two_rev_base(circle_value_i[j], N_VARS - ffs_sums[i]); // 302 cycles
                temp = eq_mle_extension_base(expanded_from_univariate, folding_randomness_global, N_VARS - ffs_sums[i]); // 1415 cycles
                copy_chunk_vec(temp, s6s + j + 1);
            }
            s7 = dot_product_extension(s6s, combination_randomness_powers[i], num_queries[i] + 1);  // 10720 cycles
            wsum = add_extension_ret(weight_sums[i], s7);
            weight_sums[i+1] = wsum;
        }

        evaluation_of_weights = weight_sums[N_ROUNDS];
        poly_eq_final = poly_eq_extension(folding_randomness_5, FINAL_VARS, TWO_POW_FINAL_VARS);
        final_value = dot_product_extension(poly_eq_final, final_coeffcients, TWO_POW_FINAL_VARS);
        evaluation_of_weights_times_final_value = mul_extension_ret(evaluation_of_weights, final_value);
        final_check = eq_extension(evaluation_of_weights_times_final_value, end_sum);
        assert final_check == 1;

        return;
    }

    fn eq_mle_extension(a, b, n) -> 1 {

        buff = malloc_vec(n);

        for i in 0..n {
            ai = a + i;
            bi = b + i;
            buffi = buff + i;
            ab = mul_extension_ret(ai, bi);
            a_ptr = ai * 8;
            b_ptr = bi * 8;
            ab_ptr = ab * 8;
            buff_ptr = buffi * 8;
            buff_ptr[0] = 1 + 2 * ab_ptr[0] - a_ptr[0] - b_ptr[0];
            for j in 1..8 unroll {
                buff_ptr[j] = 2 * ab_ptr[j] - a_ptr[j] - b_ptr[j];
            }
        }

        prods = malloc_vec(n);
        copy_chunk_vec(buff, prods);
        for i in 0..n - 1 {
            mul_extension(prods + i, buff + i + 1, prods + i + 1);
        }

        return prods + n - 1;
    }

    fn eq_mle_extension_base(a, b, n) -> 1 {
        // a: base
        // b: extension

        buff = malloc_vec(n);

        for i in 0..n {
            ai = a[i];
            bi = (b + i) * 8;
            buffi = (buff + i) * 8;
            ai_double = ai * 2;
            buffi[0] = 1 + ai_double * bi[0] - ai - bi[0];
            for j in 1..8 unroll {
                buffi[j] = ai_double * bi[j] - bi[j];
            }
        }


        prods = malloc_vec(n);
        copy_chunk_vec(buff, prods);
        for i in 0..n - 1 {
            mul_extension(prods + i, buff + i + 1, prods + i + 1);
        }
        return prods + n - 1;
    }

    fn powers_of_two_rev_base(alpha, n) -> 1 {
        // "expand_from_univariate"
        // alpha: F

        res = malloc(n);
        res[n - 1] = alpha;
        for i in 1..n {
            res[n - 1 - i] = res[n - i] * res[n - i];
        }
        return res;
    }

    fn powers_of_two_rev_extension(alpha, n) -> 1 {
        // "expand_from_univariate"

        res = malloc_vec(n);
        copy_chunk_vec(alpha, res + n - 1);
        for i in 1..n {
            mul_extension(res + (n - i), res + (n - i), res + (n - 1 - i));
        }
        return res;
    }

    fn sumcheck(fs_state, n_steps, claimed_sum) -> 3 {
        fs_states_a = malloc(n_steps + 1);
        fs_states_a[0] = fs_state;

        claimed_sums = malloc(n_steps + 1);
        claimed_sums[0] = claimed_sum;

        folding_randomness = malloc_vec(n_steps); // in reverse order.

        for sc_round in 0..n_steps {
            fs_state_5, poly = fs_receive(fs_states_a[sc_round], 3); // vectorized pointer of len 1
            sum_over_boolean_hypercube = degree_two_polynomial_sum_at_0_and_1(poly);
            consistent = eq_extension(sum_over_boolean_hypercube, claimed_sums[sc_round]);
            if consistent == 0 {
                panic();
            }
            fs_state_6, rand = fs_sample_ef(fs_state_5);  // vectorized pointer of len 1
            fs_states_a[sc_round + 1] = fs_state_6;
            new_claimed_sum = degree_two_polynomial_eval(poly, rand);
            claimed_sums[sc_round + 1] = new_claimed_sum;
            copy_chunk_vec(rand, folding_randomness +  n_steps - 1 - sc_round);
        }
        new_state = fs_states_a[n_steps];
        new_claimed_sum = claimed_sums[n_steps];

        return new_state, folding_randomness, new_claimed_sum;
    }

    fn sample_stir_indexes_and_fold(fs_state, num_queries, is_first_round, folding_factor, two_pow_folding_factor, domain_size, prev_root, folding_randomness) -> 3 {

        fs_state_9, stir_challenges_indexes = sample_bits(fs_state, num_queries);

        answers = malloc(num_queries); // a vector of vectorized pointers, each pointing to `two_pow_folding_factor` field elements (base if first rounds, extension otherwise)
        fs_states_b = malloc(num_queries + 1);
        fs_states_b[0] = fs_state_9;
        
        // the number of chunk of 8 field elements per merkle leaf opened
        if is_first_round == 1 {
            n_chuncks_per_answer = two_pow_folding_factor / 8; // "/ 8" because initial merkle leaves are in the basefield
        } else {
            n_chuncks_per_answer = two_pow_folding_factor;
        }

        for i in 0..num_queries {
            new_fs_state, answer = fs_hint(fs_states_b[i], n_chuncks_per_answer); 
            fs_states_b[i + 1] = new_fs_state;
            answers[i] = answer;
        }
        fs_state_10 = fs_states_b[num_queries];

        leaf_hashes = malloc(num_queries); // a vector of vectorized pointers, each pointing to 1 chunk of 8 field elements
        for i in 0..num_queries {
            answer = answers[i];
            internal_states = malloc(1 + (n_chuncks_per_answer / 2)); // "/ 2" because with poseidon24 we hash 2 chuncks of 8 field elements at each permutation
            internal_states[0] = pointer_to_zero_vector; // initial state
            for j in 0..n_chuncks_per_answer / 2 {
                new_state_0, _, new_state_2 = poseidon24(answer + (2*j), answer + (2*j) + 1, internal_states[j]);
                if j == (n_chuncks_per_answer / 2) - 1 {
                    // last step
                    internal_states[j + 1] = new_state_0;
                } else {
                    internal_states[j + 1] = new_state_2;
                }
            }
            leaf_hashes[i] = internal_states[n_chuncks_per_answer / 2];
        }

        folded_domain_size = domain_size - folding_factor;

        fs_states_c = malloc(num_queries + 1);
        fs_states_c[0] = fs_state_10;

        for i in 0..num_queries {
            fs_state_11, merkle_path = fs_hint(fs_states_c[i], folded_domain_size);
            fs_states_c[i + 1] = fs_state_11;

            stir_index_bits = stir_challenges_indexes[i]; // a pointer to 31 bits

            states = malloc(1 + folded_domain_size);
            states[0] = leaf_hashes[i];
            for j in 0..folded_domain_size {
                if stir_index_bits[j] == 1 {
                    left = merkle_path + j;
                    right = states[j];
                } else {
                    left = states[j];
                    right = merkle_path + j;
                }
                state_j_plus_1, _ = poseidon16(left, right);
                states[j + 1] = state_j_plus_1;
            }
            correct_root = eq_extension(states[folded_domain_size], prev_root);
            assert correct_root == 1;
        }

        fs_state_11 = fs_states_c[num_queries];

        poly_eq = poly_eq_extension(folding_randomness, folding_factor, two_pow_folding_factor);

        folds = malloc_vec(num_queries);
        if is_first_round == 1 {
            for i in 0..num_queries {
                dot_product_base_extension(answers[i] * 8, poly_eq, folds + i, TWO_POW_FOLDING_FACTOR_0);
            }
        } else {
            for i in 0..num_queries {
                temp = dot_product_extension(answers[i], poly_eq, two_pow_folding_factor);
                copy_chunk_vec(temp, (folds + i));
            }
        }

        circle_values = malloc(num_queries); // ROOT^each_stir_index
        for i in 0..num_queries {
            stir_index_bits = stir_challenges_indexes[i];
            circle_value = unit_root_pow(folded_domain_size, stir_index_bits);
            circle_values[i] = circle_value;
        }

        return fs_state_11, circle_values, folds;
    }


    fn whir_round(fs_state, prev_root, folding_factor, two_pow_folding_factor, is_first_round, num_queries, domain_size, claimed_sum) -> 7 {
        fs_state_7, folding_randomness, new_claimed_sum_a = sumcheck(fs_state, folding_factor, claimed_sum);

        fs_state_8, root, ood_point, ood_eval = parse_commitment(fs_state_7);
   
        fs_state_11, circle_values, folds = 
            sample_stir_indexes_and_fold(fs_state_8, num_queries, is_first_round, folding_factor, two_pow_folding_factor, domain_size, prev_root, folding_randomness);

        fs_state_12, combination_randomness_gen = fs_sample_ef(fs_state_11);

        combination_randomness_powers = powers(combination_randomness_gen, num_queries + 1); // "+ 1" because of one OOD sample

        claimed_sum_supplement_side = dot_product_extension(folds, combination_randomness_powers + 1, num_queries);
        claimed_sum_supplement = add_extension_ret(claimed_sum_supplement_side, ood_eval);
        new_claimed_sum_b = add_extension_ret(claimed_sum_supplement, new_claimed_sum_a);

        return fs_state_12, folding_randomness, ood_point, root, circle_values, combination_randomness_powers, new_claimed_sum_b;
    }

    fn copy_chunk(src, dst) {
        // src: pointer to 8 F
        // dst: pointer to 8 F
        for i in 0..8 unroll { dst[i] = src[i]; }
        return;
    }

    fn copy_chunk_vec(src, dst) {
        src_ptr = src * 8;
        dst_ptr = dst * 8;
        for i in 0..8 unroll { dst_ptr[i] = src_ptr[i]; }
        return;
    }

    fn powers(alpha, n) -> 1 {
        // alpha: EF
        // n: F

        res = malloc_vec(n);
        set_to_one(res);
        for i in 0..n - 1 {
            mul_extension(res + i, alpha, res + i + 1);
        }
        return res;
    }

    fn dot_product_extension(a, b, n) -> 1 {

        prods = malloc_vec(n);
        for i in 0..n {
            mul_extension(a + i, b + i, prods + i);
        }

        sums = malloc_vec(n);
        copy_chunk_vec(prods, sums);
        for i in 0..n - 1 {
            add_extension(sums + i, prods + i + 1, sums + i + 1);
        }

        return sums + n - 1;
    }

    fn unit_root_pow(domain_size, index_bits) -> 1 {
        // index_bits is a pointer to domain_size bits

        if domain_size == 19 {
            return ((index_bits[0] * ROOT_19) + (1 - index_bits[0])) * ((index_bits[1] * ROOT_18) + (1 - index_bits[1])) * ((index_bits[2] * ROOT_17) + (1 - index_bits[2])) * ((index_bits[3] * ROOT_16) + (1 - index_bits[3])) * ((index_bits[4] * ROOT_15) + (1 - index_bits[4])) * ((index_bits[5] * ROOT_14) + (1 - index_bits[5])) * ((index_bits[6] * ROOT_13) + (1 - index_bits[6])) * ((index_bits[7] * ROOT_12) + (1 - index_bits[7])) * ((index_bits[8] * ROOT_11) + (1 - index_bits[8])) * ((index_bits[9] * ROOT_10) + (1 - index_bits[9])) * ((index_bits[10] * ROOT_9) + (1 - index_bits[10])) * ((index_bits[11] * ROOT_8) + (1 - index_bits[11])) * ((index_bits[12] * ROOT_7) + (1 - index_bits[12])) * ((index_bits[13] * ROOT_6) + (1 - index_bits[13])) * ((index_bits[14] * ROOT_5) + (1 - index_bits[14])) * ((index_bits[15] * ROOT_4) + (1 - index_bits[15])) * ((index_bits[16] * ROOT_3) + (1 - index_bits[16])) * ((index_bits[17] * ROOT_2) + (1 - index_bits[17])) * ((index_bits[18] * ROOT_1) + (1 - index_bits[18]));
        }

        if domain_size == 17 {
            return ((index_bits[0] * ROOT_17) + (1 - index_bits[0])) * ((index_bits[1] * ROOT_16) + (1 - index_bits[1])) * ((index_bits[2] * ROOT_15) + (1 - index_bits[2])) * ((index_bits[3] * ROOT_14) + (1 - index_bits[3])) * ((index_bits[4] * ROOT_13) + (1 - index_bits[4])) * ((index_bits[5] * ROOT_12) + (1 - index_bits[5])) * ((index_bits[6] * ROOT_11) + (1 - index_bits[6])) * ((index_bits[7] * ROOT_10) + (1 - index_bits[7])) * ((index_bits[8] * ROOT_9) + (1 - index_bits[8])) * ((index_bits[9] * ROOT_8) + (1 - index_bits[9])) * ((index_bits[10] * ROOT_7) + (1 - index_bits[10])) * ((index_bits[11] * ROOT_6) + (1 - index_bits[11])) * ((index_bits[12] * ROOT_5) + (1 - index_bits[12])) * ((index_bits[13] * ROOT_4) + (1 - index_bits[13])) * ((index_bits[14] * ROOT_3) + (1 - index_bits[14])) * ((index_bits[15] * ROOT_2) + (1 - index_bits[15])) * ((index_bits[16] * ROOT_1) + (1 - index_bits[16]));
        }

        if domain_size == 16 {
            return ((index_bits[0] * ROOT_16) + (1 - index_bits[0])) * ((index_bits[1] * ROOT_15) + (1 - index_bits[1])) * ((index_bits[2] * ROOT_14) + (1 - index_bits[2])) * ((index_bits[3] * ROOT_13) + (1 - index_bits[3])) * ((index_bits[4] * ROOT_12) + (1 - index_bits[4])) * ((index_bits[5] * ROOT_11) + (1 - index_bits[5])) * ((index_bits[6] * ROOT_10) + (1 - index_bits[6])) * ((index_bits[7] * ROOT_9) + (1 - index_bits[7])) * ((index_bits[8] * ROOT_8) + (1 - index_bits[8])) * ((index_bits[9] * ROOT_7) + (1 - index_bits[9])) * ((index_bits[10] * ROOT_6) + (1 - index_bits[10])) * ((index_bits[11] * ROOT_5) + (1 - index_bits[11])) * ((index_bits[12] * ROOT_4) + (1 - index_bits[12])) * ((index_bits[13] * ROOT_3) + (1 - index_bits[13])) * ((index_bits[14] * ROOT_2) + (1 - index_bits[14])) * ((index_bits[15] * ROOT_1) + (1 - index_bits[15]));
        }

        if domain_size == 15 {
            return ((index_bits[0] * ROOT_15) + (1 - index_bits[0])) * ((index_bits[1] * ROOT_14) + (1 - index_bits[1])) * ((index_bits[2] * ROOT_13) + (1 - index_bits[2])) * ((index_bits[3] * ROOT_12) + (1 - index_bits[3])) * ((index_bits[4] * ROOT_11) + (1 - index_bits[4])) * ((index_bits[5] * ROOT_10) + (1 - index_bits[5])) * ((index_bits[6] * ROOT_9) + (1 - index_bits[6])) * ((index_bits[7] * ROOT_8) + (1 - index_bits[7])) * ((index_bits[8] * ROOT_7) + (1 - index_bits[8])) * ((index_bits[9] * ROOT_6) + (1 - index_bits[9])) * ((index_bits[10] * ROOT_5) + (1 - index_bits[10])) * ((index_bits[11] * ROOT_4) + (1 - index_bits[11])) * ((index_bits[12] * ROOT_3) + (1 - index_bits[12])) * ((index_bits[13] * ROOT_2) + (1 - index_bits[13])) * ((index_bits[14] * ROOT_1) + (1 - index_bits[14]));
        }

        UNIMPLEMENTED = 0;
        print(UNIMPLEMENTED, domain_size);
        panic();
    }

    fn dot_product_base_extension(a, b, res, n) {
        // a is a pointer to n base field elements
        // b is a pointer to n extension field elements

        b_ptr = b * 8;
        res_ptr = res * 8;

        
        // prods = malloc(n * 8);
        // for i in 0..n {
        //     for j in 0..8 {
        //         prods[i * 8 + j] = a[i] * b_ptr[i * 8 + j];
        //     }
        // }

        // for i in 0..8 {
        //     my_buff = malloc(n);
        //     my_buff[0] = prods[i];
        //     for j in 0..n - 1  {
        //         my_buff[j + 1] = my_buff[j] + prods[i + ((j + 1) * 8)];
        //     }
        //     res_ptr[i] = my_buff[n - 1];
        // }

        if n == TWO_POW_FOLDING_FACTOR_0 {
           
            // OPTIMIZED VERSION

            prods = malloc(TWO_POW_FOLDING_FACTOR_0 * 8);
            for i in 0..TWO_POW_FOLDING_FACTOR_0 unroll {
                for j in 0..8 unroll {
                    prods[i * 8 + j] = a[i] * b_ptr[i * 8 + j];
                }
            }

            res_ptr[0] = prods[0] + prods[8] + prods[16] + prods[24] + prods[32] + prods[40] + prods[48] + prods[56] + prods[64] + prods[72] + prods[80] + prods[88] + prods[96] + prods[104] + prods[112] + prods[120] + prods[128] + prods[136] + prods[144] + prods[152] + prods[160] + prods[168] + prods[176] + prods[184] + prods[192] + prods[200] + prods[208] + prods[216] + prods[224] + prods[232] + prods[240] + prods[248] + prods[256] + prods[264] + prods[272] + prods[280] + prods[288] + prods[296] + prods[304] + prods[312] + prods[320] + prods[328] + prods[336] + prods[344] + prods[352] + prods[360] + prods[368] + prods[376] + prods[384] + prods[392] + prods[400] + prods[408] + prods[416] + prods[424] + prods[432] + prods[440] + prods[448] + prods[456] + prods[464] + prods[472] + prods[480] + prods[488] + prods[496] + prods[504] + prods[512] + prods[520] + prods[528] + prods[536] + prods[544] + prods[552] + prods[560] + prods[568] + prods[576] + prods[584] + prods[592] + prods[600] + prods[608] + prods[616] + prods[624] + prods[632] + prods[640] + prods[648] + prods[656] + prods[664] + prods[672] + prods[680] + prods[688] + prods[696] + prods[704] + prods[712] + prods[720] + prods[728] + prods[736] + prods[744] + prods[752] + prods[760] + prods[768] + prods[776] + prods[784] + prods[792] + prods[800] + prods[808] + prods[816] + prods[824] + prods[832] + prods[840] + prods[848] + prods[856] + prods[864] + prods[872] + prods[880] + prods[888] + prods[896] + prods[904] + prods[912] + prods[920] + prods[928] + prods[936] + prods[944] + prods[952] + prods[960] + prods[968] + prods[976] + prods[984] + prods[992] + prods[1000] + prods[1008] + prods[1016];
            res_ptr[1] = prods[1] + prods[9] + prods[17] + prods[25] + prods[33] + prods[41] + prods[49] + prods[57] + prods[65] + prods[73] + prods[81] + prods[89] + prods[97] + prods[105] + prods[113] + prods[121] + prods[129] + prods[137] + prods[145] + prods[153] + prods[161] + prods[169] + prods[177] + prods[185] + prods[193] + prods[201] + prods[209] + prods[217] + prods[225] + prods[233] + prods[241] + prods[249] + prods[257] + prods[265] + prods[273] + prods[281] + prods[289] + prods[297] + prods[305] + prods[313] + prods[321] + prods[329] + prods[337] + prods[345] + prods[353] + prods[361] + prods[369] + prods[377] + prods[385] + prods[393] + prods[401] + prods[409] + prods[417] + prods[425] + prods[433] + prods[441] + prods[449] + prods[457] + prods[465] + prods[473] + prods[481] + prods[489] + prods[497] + prods[505] + prods[513] + prods[521] + prods[529] + prods[537] + prods[545] + prods[553] + prods[561] + prods[569] + prods[577] + prods[585] + prods[593] + prods[601] + prods[609] + prods[617] + prods[625] + prods[633] + prods[641] + prods[649] + prods[657] + prods[665] + prods[673] + prods[681] + prods[689] + prods[697] + prods[705] + prods[713] + prods[721] + prods[729] + prods[737] + prods[745] + prods[753] + prods[761] + prods[769] + prods[777] + prods[785] + prods[793] + prods[801] + prods[809] + prods[817] + prods[825] + prods[833] + prods[841] + prods[849] + prods[857] + prods[865] + prods[873] + prods[881] + prods[889] + prods[897] + prods[905] + prods[913] + prods[921] + prods[929] + prods[937] + prods[945] + prods[953] + prods[961] + prods[969] + prods[977] + prods[985] + prods[993] + prods[1001] + prods[1009] + prods[1017];
            res_ptr[2] = prods[2] + prods[10] + prods[18] + prods[26] + prods[34] + prods[42] + prods[50] + prods[58] + prods[66] + prods[74] + prods[82] + prods[90] + prods[98] + prods[106] + prods[114] + prods[122] + prods[130] + prods[138] + prods[146] + prods[154] + prods[162] + prods[170] + prods[178] + prods[186] + prods[194] + prods[202] + prods[210] + prods[218] + prods[226] + prods[234] + prods[242] + prods[250] + prods[258] + prods[266] + prods[274] + prods[282] + prods[290] + prods[298] + prods[306] + prods[314] + prods[322] + prods[330] + prods[338] + prods[346] + prods[354] + prods[362] + prods[370] + prods[378] + prods[386] + prods[394] + prods[402] + prods[410] + prods[418] + prods[426] + prods[434] + prods[442] + prods[450] + prods[458] + prods[466] + prods[474] + prods[482] + prods[490] + prods[498] + prods[506] + prods[514] + prods[522] + prods[530] + prods[538] + prods[546] + prods[554] + prods[562] + prods[570] + prods[578] + prods[586] + prods[594] + prods[602] + prods[610] + prods[618] + prods[626] + prods[634] + prods[642] + prods[650] + prods[658] + prods[666] + prods[674] + prods[682] + prods[690] + prods[698] + prods[706] + prods[714] + prods[722] + prods[730] + prods[738] + prods[746] + prods[754] + prods[762] + prods[770] + prods[778] + prods[786] + prods[794] + prods[802] + prods[810] + prods[818] + prods[826] + prods[834] + prods[842] + prods[850] + prods[858] + prods[866] + prods[874] + prods[882] + prods[890] + prods[898] + prods[906] + prods[914] + prods[922] + prods[930] + prods[938] + prods[946] + prods[954] + prods[962] + prods[970] + prods[978] + prods[986] + prods[994] + prods[1002] + prods[1010] + prods[1018];
            res_ptr[3] = prods[3] + prods[11] + prods[19] + prods[27] + prods[35] + prods[43] + prods[51] + prods[59] + prods[67] + prods[75] + prods[83] + prods[91] + prods[99] + prods[107] + prods[115] + prods[123] + prods[131] + prods[139] + prods[147] + prods[155] + prods[163] + prods[171] + prods[179] + prods[187] + prods[195] + prods[203] + prods[211] + prods[219] + prods[227] + prods[235] + prods[243] + prods[251] + prods[259] + prods[267] + prods[275] + prods[283] + prods[291] + prods[299] + prods[307] + prods[315] + prods[323] + prods[331] + prods[339] + prods[347] + prods[355] + prods[363] + prods[371] + prods[379] + prods[387] + prods[395] + prods[403] + prods[411] + prods[419] + prods[427] + prods[435] + prods[443] + prods[451] + prods[459] + prods[467] + prods[475] + prods[483] + prods[491] + prods[499] + prods[507] + prods[515] + prods[523] + prods[531] + prods[539] + prods[547] + prods[555] + prods[563] + prods[571] + prods[579] + prods[587] + prods[595] + prods[603] + prods[611] + prods[619] + prods[627] + prods[635] + prods[643] + prods[651] + prods[659] + prods[667] + prods[675] + prods[683] + prods[691] + prods[699] + prods[707] + prods[715] + prods[723] + prods[731] + prods[739] + prods[747] + prods[755] + prods[763] + prods[771] + prods[779] + prods[787] + prods[795] + prods[803] + prods[811] + prods[819] + prods[827] + prods[835] + prods[843] + prods[851] + prods[859] + prods[867] + prods[875] + prods[883] + prods[891] + prods[899] + prods[907] + prods[915] + prods[923] + prods[931] + prods[939] + prods[947] + prods[955] + prods[963] + prods[971] + prods[979] + prods[987] + prods[995] + prods[1003] + prods[1011] + prods[1019];
            res_ptr[4] = prods[4] + prods[12] + prods[20] + prods[28] + prods[36] + prods[44] + prods[52] + prods[60] + prods[68] + prods[76] + prods[84] + prods[92] + prods[100] + prods[108] + prods[116] + prods[124] + prods[132] + prods[140] + prods[148] + prods[156] + prods[164] + prods[172] + prods[180] + prods[188] + prods[196] + prods[204] + prods[212] + prods[220] + prods[228] + prods[236] + prods[244] + prods[252] + prods[260] + prods[268] + prods[276] + prods[284] + prods[292] + prods[300] + prods[308] + prods[316] + prods[324] + prods[332] + prods[340] + prods[348] + prods[356] + prods[364] + prods[372] + prods[380] + prods[388] + prods[396] + prods[404] + prods[412] + prods[420] + prods[428] + prods[436] + prods[444] + prods[452] + prods[460] + prods[468] + prods[476] + prods[484] + prods[492] + prods[500] + prods[508] + prods[516] + prods[524] + prods[532] + prods[540] + prods[548] + prods[556] + prods[564] + prods[572] + prods[580] + prods[588] + prods[596] + prods[604] + prods[612] + prods[620] + prods[628] + prods[636] + prods[644] + prods[652] + prods[660] + prods[668] + prods[676] + prods[684] + prods[692] + prods[700] + prods[708] + prods[716] + prods[724] + prods[732] + prods[740] + prods[748] + prods[756] + prods[764] + prods[772] + prods[780] + prods[788] + prods[796] + prods[804] + prods[812] + prods[820] + prods[828] + prods[836] + prods[844] + prods[852] + prods[860] + prods[868] + prods[876] + prods[884] + prods[892] + prods[900] + prods[908] + prods[916] + prods[924] + prods[932] + prods[940] + prods[948] + prods[956] + prods[964] + prods[972] + prods[980] + prods[988] + prods[996] + prods[1004] + prods[1012] + prods[1020];
            res_ptr[5] = prods[5] + prods[13] + prods[21] + prods[29] + prods[37] + prods[45] + prods[53] + prods[61] + prods[69] + prods[77] + prods[85] + prods[93] + prods[101] + prods[109] + prods[117] + prods[125] + prods[133] + prods[141] + prods[149] + prods[157] + prods[165] + prods[173] + prods[181] + prods[189] + prods[197] + prods[205] + prods[213] + prods[221] + prods[229] + prods[237] + prods[245] + prods[253] + prods[261] + prods[269] + prods[277] + prods[285] + prods[293] + prods[301] + prods[309] + prods[317] + prods[325] + prods[333] + prods[341] + prods[349] + prods[357] + prods[365] + prods[373] + prods[381] + prods[389] + prods[397] + prods[405] + prods[413] + prods[421] + prods[429] + prods[437] + prods[445] + prods[453] + prods[461] + prods[469] + prods[477] + prods[485] + prods[493] + prods[501] + prods[509] + prods[517] + prods[525] + prods[533] + prods[541] + prods[549] + prods[557] + prods[565] + prods[573] + prods[581] + prods[589] + prods[597] + prods[605] + prods[613] + prods[621] + prods[629] + prods[637] + prods[645] + prods[653] + prods[661] + prods[669] + prods[677] + prods[685] + prods[693] + prods[701] + prods[709] + prods[717] + prods[725] + prods[733] + prods[741] + prods[749] + prods[757] + prods[765] + prods[773] + prods[781] + prods[789] + prods[797] + prods[805] + prods[813] + prods[821] + prods[829] + prods[837] + prods[845] + prods[853] + prods[861] + prods[869] + prods[877] + prods[885] + prods[893] + prods[901] + prods[909] + prods[917] + prods[925] + prods[933] + prods[941] + prods[949] + prods[957] + prods[965] + prods[973] + prods[981] + prods[989] + prods[997] + prods[1005] + prods[1013] + prods[1021];
            res_ptr[6] = prods[6] + prods[14] + prods[22] + prods[30] + prods[38] + prods[46] + prods[54] + prods[62] + prods[70] + prods[78] + prods[86] + prods[94] + prods[102] + prods[110] + prods[118] + prods[126] + prods[134] + prods[142] + prods[150] + prods[158] + prods[166] + prods[174] + prods[182] + prods[190] + prods[198] + prods[206] + prods[214] + prods[222] + prods[230] + prods[238] + prods[246] + prods[254] + prods[262] + prods[270] + prods[278] + prods[286] + prods[294] + prods[302] + prods[310] + prods[318] + prods[326] + prods[334] + prods[342] + prods[350] + prods[358] + prods[366] + prods[374] + prods[382] + prods[390] + prods[398] + prods[406] + prods[414] + prods[422] + prods[430] + prods[438] + prods[446] + prods[454] + prods[462] + prods[470] + prods[478] + prods[486] + prods[494] + prods[502] + prods[510] + prods[518] + prods[526] + prods[534] + prods[542] + prods[550] + prods[558] + prods[566] + prods[574] + prods[582] + prods[590] + prods[598] + prods[606] + prods[614] + prods[622] + prods[630] + prods[638] + prods[646] + prods[654] + prods[662] + prods[670] + prods[678] + prods[686] + prods[694] + prods[702] + prods[710] + prods[718] + prods[726] + prods[734] + prods[742] + prods[750] + prods[758] + prods[766] + prods[774] + prods[782] + prods[790] + prods[798] + prods[806] + prods[814] + prods[822] + prods[830] + prods[838] + prods[846] + prods[854] + prods[862] + prods[870] + prods[878] + prods[886] + prods[894] + prods[902] + prods[910] + prods[918] + prods[926] + prods[934] + prods[942] + prods[950] + prods[958] + prods[966] + prods[974] + prods[982] + prods[990] + prods[998] + prods[1006] + prods[1014] + prods[1022];
            res_ptr[7] = prods[7] + prods[15] + prods[23] + prods[31] + prods[39] + prods[47] + prods[55] + prods[63] + prods[71] + prods[79] + prods[87] + prods[95] + prods[103] + prods[111] + prods[119] + prods[127] + prods[135] + prods[143] + prods[151] + prods[159] + prods[167] + prods[175] + prods[183] + prods[191] + prods[199] + prods[207] + prods[215] + prods[223] + prods[231] + prods[239] + prods[247] + prods[255] + prods[263] + prods[271] + prods[279] + prods[287] + prods[295] + prods[303] + prods[311] + prods[319] + prods[327] + prods[335] + prods[343] + prods[351] + prods[359] + prods[367] + prods[375] + prods[383] + prods[391] + prods[399] + prods[407] + prods[415] + prods[423] + prods[431] + prods[439] + prods[447] + prods[455] + prods[463] + prods[471] + prods[479] + prods[487] + prods[495] + prods[503] + prods[511] + prods[519] + prods[527] + prods[535] + prods[543] + prods[551] + prods[559] + prods[567] + prods[575] + prods[583] + prods[591] + prods[599] + prods[607] + prods[615] + prods[623] + prods[631] + prods[639] + prods[647] + prods[655] + prods[663] + prods[671] + prods[679] + prods[687] + prods[695] + prods[703] + prods[711] + prods[719] + prods[727] + prods[735] + prods[743] + prods[751] + prods[759] + prods[767] + prods[775] + prods[783] + prods[791] + prods[799] + prods[807] + prods[815] + prods[823] + prods[831] + prods[839] + prods[847] + prods[855] + prods[863] + prods[871] + prods[879] + prods[887] + prods[895] + prods[903] + prods[911] + prods[919] + prods[927] + prods[935] + prods[943] + prods[951] + prods[959] + prods[967] + prods[975] + prods[983] + prods[991] + prods[999] + prods[1007] + prods[1015] + prods[1023];

            return;

        }

        if n == TWO_POW_FINAL_VARS {
           
            // OPTIMIZED VERSION

            prods = malloc(TWO_POW_FINAL_VARS * 8);
            for i in 0..TWO_POW_FINAL_VARS unroll {
                for j in 0..8 unroll {
                    prods[i * 8 + j] = a[i] * b_ptr[i * 8 + j];
                }
            }

            res_ptr[0] = prods[0] + prods[8] + prods[16] + prods[24] + prods[32] + prods[40] + prods[48] + prods[56] + prods[64] + prods[72] + prods[80] + prods[88] + prods[96] + prods[104] + prods[112] + prods[120] + prods[128] + prods[136] + prods[144] + prods[152] + prods[160] + prods[168] + prods[176] + prods[184] + prods[192] + prods[200] + prods[208] + prods[216] + prods[224] + prods[232] + prods[240] + prods[248] + prods[256] + prods[264] + prods[272] + prods[280] + prods[288] + prods[296] + prods[304] + prods[312] + prods[320] + prods[328] + prods[336] + prods[344] + prods[352] + prods[360] + prods[368] + prods[376] + prods[384] + prods[392] + prods[400] + prods[408] + prods[416] + prods[424] + prods[432] + prods[440] + prods[448] + prods[456] + prods[464] + prods[472] + prods[480] + prods[488] + prods[496] + prods[504];
            res_ptr[1] = prods[1] + prods[9] + prods[17] + prods[25] + prods[33] + prods[41] + prods[49] + prods[57] + prods[65] + prods[73] + prods[81] + prods[89] + prods[97] + prods[105] + prods[113] + prods[121] + prods[129] + prods[137] + prods[145] + prods[153] + prods[161] + prods[169] + prods[177] + prods[185] + prods[193] + prods[201] + prods[209] + prods[217] + prods[225] + prods[233] + prods[241] + prods[249] + prods[257] + prods[265] + prods[273] + prods[281] + prods[289] + prods[297] + prods[305] + prods[313] + prods[321] + prods[329] + prods[337] + prods[345] + prods[353] + prods[361] + prods[369] + prods[377] + prods[385] + prods[393] + prods[401] + prods[409] + prods[417] + prods[425] + prods[433] + prods[441] + prods[449] + prods[457] + prods[465] + prods[473] + prods[481] + prods[489] + prods[497] + prods[505];
            res_ptr[2] = prods[2] + prods[10] + prods[18] + prods[26] + prods[34] + prods[42] + prods[50] + prods[58] + prods[66] + prods[74] + prods[82] + prods[90] + prods[98] + prods[106] + prods[114] + prods[122] + prods[130] + prods[138] + prods[146] + prods[154] + prods[162] + prods[170] + prods[178] + prods[186] + prods[194] + prods[202] + prods[210] + prods[218] + prods[226] + prods[234] + prods[242] + prods[250] + prods[258] + prods[266] + prods[274] + prods[282] + prods[290] + prods[298] + prods[306] + prods[314] + prods[322] + prods[330] + prods[338] + prods[346] + prods[354] + prods[362] + prods[370] + prods[378] + prods[386] + prods[394] + prods[402] + prods[410] + prods[418] + prods[426] + prods[434] + prods[442] + prods[450] + prods[458] + prods[466] + prods[474] + prods[482] + prods[490] + prods[498] + prods[506];
            res_ptr[3] = prods[3] + prods[11] + prods[19] + prods[27] + prods[35] + prods[43] + prods[51] + prods[59] + prods[67] + prods[75] + prods[83] + prods[91] + prods[99] + prods[107] + prods[115] + prods[123] + prods[131] + prods[139] + prods[147] + prods[155] + prods[163] + prods[171] + prods[179] + prods[187] + prods[195] + prods[203] + prods[211] + prods[219] + prods[227] + prods[235] + prods[243] + prods[251] + prods[259] + prods[267] + prods[275] + prods[283] + prods[291] + prods[299] + prods[307] + prods[315] + prods[323] + prods[331] + prods[339] + prods[347] + prods[355] + prods[363] + prods[371] + prods[379] + prods[387] + prods[395] + prods[403] + prods[411] + prods[419] + prods[427] + prods[435] + prods[443] + prods[451] + prods[459] + prods[467] + prods[475] + prods[483] + prods[491] + prods[499] + prods[507];
            res_ptr[4] = prods[4] + prods[12] + prods[20] + prods[28] + prods[36] + prods[44] + prods[52] + prods[60] + prods[68] + prods[76] + prods[84] + prods[92] + prods[100] + prods[108] + prods[116] + prods[124] + prods[132] + prods[140] + prods[148] + prods[156] + prods[164] + prods[172] + prods[180] + prods[188] + prods[196] + prods[204] + prods[212] + prods[220] + prods[228] + prods[236] + prods[244] + prods[252] + prods[260] + prods[268] + prods[276] + prods[284] + prods[292] + prods[300] + prods[308] + prods[316] + prods[324] + prods[332] + prods[340] + prods[348] + prods[356] + prods[364] + prods[372] + prods[380] + prods[388] + prods[396] + prods[404] + prods[412] + prods[420] + prods[428] + prods[436] + prods[444] + prods[452] + prods[460] + prods[468] + prods[476] + prods[484] + prods[492] + prods[500] + prods[508];
            res_ptr[5] = prods[5] + prods[13] + prods[21] + prods[29] + prods[37] + prods[45] + prods[53] + prods[61] + prods[69] + prods[77] + prods[85] + prods[93] + prods[101] + prods[109] + prods[117] + prods[125] + prods[133] + prods[141] + prods[149] + prods[157] + prods[165] + prods[173] + prods[181] + prods[189] + prods[197] + prods[205] + prods[213] + prods[221] + prods[229] + prods[237] + prods[245] + prods[253] + prods[261] + prods[269] + prods[277] + prods[285] + prods[293] + prods[301] + prods[309] + prods[317] + prods[325] + prods[333] + prods[341] + prods[349] + prods[357] + prods[365] + prods[373] + prods[381] + prods[389] + prods[397] + prods[405] + prods[413] + prods[421] + prods[429] + prods[437] + prods[445] + prods[453] + prods[461] + prods[469] + prods[477] + prods[485] + prods[493] + prods[501] + prods[509];
            res_ptr[6] = prods[6] + prods[14] + prods[22] + prods[30] + prods[38] + prods[46] + prods[54] + prods[62] + prods[70] + prods[78] + prods[86] + prods[94] + prods[102] + prods[110] + prods[118] + prods[126] + prods[134] + prods[142] + prods[150] + prods[158] + prods[166] + prods[174] + prods[182] + prods[190] + prods[198] + prods[206] + prods[214] + prods[222] + prods[230] + prods[238] + prods[246] + prods[254] + prods[262] + prods[270] + prods[278] + prods[286] + prods[294] + prods[302] + prods[310] + prods[318] + prods[326] + prods[334] + prods[342] + prods[350] + prods[358] + prods[366] + prods[374] + prods[382] + prods[390] + prods[398] + prods[406] + prods[414] + prods[422] + prods[430] + prods[438] + prods[446] + prods[454] + prods[462] + prods[470] + prods[478] + prods[486] + prods[494] + prods[502] + prods[510];
            res_ptr[7] = prods[7] + prods[15] + prods[23] + prods[31] + prods[39] + prods[47] + prods[55] + prods[63] + prods[71] + prods[79] + prods[87] + prods[95] + prods[103] + prods[111] + prods[119] + prods[127] + prods[135] + prods[143] + prods[151] + prods[159] + prods[167] + prods[175] + prods[183] + prods[191] + prods[199] + prods[207] + prods[215] + prods[223] + prods[231] + prods[239] + prods[247] + prods[255] + prods[263] + prods[271] + prods[279] + prods[287] + prods[295] + prods[303] + prods[311] + prods[319] + prods[327] + prods[335] + prods[343] + prods[351] + prods[359] + prods[367] + prods[375] + prods[383] + prods[391] + prods[399] + prods[407] + prods[415] + prods[423] + prods[431] + prods[439] + prods[447] + prods[455] + prods[463] + prods[471] + prods[479] + prods[487] + prods[495] + prods[503] + prods[511];

            return;

        }


        UNIMPLEMENTED = 1;
        print(UNIMPLEMENTED, n);

        return;
    }

    fn poly_eq_extension(point, n, two_pow_n) -> 1 {
        // Example: for n = 2: eq(x, y) = [(1 - x)(1 - y), (1 - x)y, x(1 - y), xy]

        if n == 0 {
            res = malloc_vec(1);
            set_to_one(res);
            return res;
        }

        res = malloc_vec(two_pow_n);

        inner_res = poly_eq_extension(point + 1, n - 1, two_pow_n / 2);

        two_pow_n_minus_1 = two_pow_n / 2;

        for i in 0..two_pow_n_minus_1 {
            mul_extension(point, inner_res + i, res + two_pow_n_minus_1 + i);
            sub_extension(inner_res + i, res + two_pow_n_minus_1 + i, res + i);
        }
        
        return res;
    }

    fn poly_eq_base(point, n, two_pow_n) -> 1 {
        // return a (normal) pointer to 2^n base field elements, corresponding to the "equality polynomial" at point
        // Example: for n = 2: eq(x, y) = [(1 - x)(1 - y), (1 - x)y, x(1 - y), xy]

        if n == 0 {
            // base case
            res = malloc(1);
            res[0] = 1;
            return res;
        }

        res = malloc(two_pow_n);

        inner_res = poly_eq_base(point + 1, n - 1, two_pow_n / 2);

        two_pow_n_minus_1 = two_pow_n / 2;

        for i in 0..two_pow_n_minus_1 {
            res[two_pow_n_minus_1 + i] = inner_res[i] * point[0];
            res[i] = inner_res[i] - res[two_pow_n_minus_1 + i];
        }
        
        return res;
    }


    fn pow(a, b) -> 1 {
        if b == 0 {
            return 1; // a^0 = 1
        } else {
            p = pow(a, b - 1);
            return a * p;
        }
    }

    fn sample_bits(fs_state, n) -> 2 {
        // return the updated fs_state, and a pointer to n pointers, each pointing to 31 (boolean) field elements
        samples = malloc(n);
        new_fs_state = fs_sample_helper(fs_state, n, samples);
        sampled_bits = malloc(n);
        for i in 0..n {
            bits = checked_decompose_bits(samples[i]);
            sampled_bits[i] = bits;
        }

        return new_fs_state, sampled_bits;
    }

    fn checked_decompose_bits(a) -> 1 {
        // return a pointer to bits of a
        bits = decompose_bits(a); // hint
        bit0 = bits[0]; bit1 = bits[1]; bit2 = bits[2]; bit3 = bits[3]; bit4 = bits[4]; bit5 = bits[5]; bit6 = bits[6]; bit7 = bits[7]; bit8 = bits[8]; bit9 = bits[9]; bit10 = bits[10]; bit11 = bits[11]; bit12 = bits[12]; bit13 = bits[13]; bit14 = bits[14]; bit15 = bits[15]; bit16 = bits[16]; bit17 = bits[17]; bit18 = bits[18]; bit19 = bits[19]; bit20 = bits[20]; bit21 = bits[21]; bit22 = bits[22]; bit23 = bits[23]; bit24 = bits[24]; bit25 = bits[25]; bit26 = bits[26]; bit27 = bits[27]; bit28 = bits[28]; bit29 = bits[29]; bit30 = bits[30];
        assert bit0 * (1 - bit0) == 0; assert bit1 * (1 - bit1) == 0; assert bit2 * (1 - bit2) == 0; assert bit3 * (1 - bit3) == 0; assert bit4 * (1 - bit4) == 0; assert bit5 * (1 - bit5) == 0; assert bit6 * (1 - bit6) == 0; assert bit7 * (1 - bit7) == 0; assert bit8 * (1 - bit8) == 0; assert bit9 * (1 - bit9) == 0; assert bit10 * (1 - bit10) == 0; assert bit11 * (1 - bit11) == 0; assert bit12 * (1 - bit12) == 0; assert bit13 * (1 - bit13) == 0; assert bit14 * (1 - bit14) == 0; assert bit15 * (1 - bit15) == 0; assert bit16 * (1 - bit16) == 0; assert bit17 * (1 - bit17) == 0; assert bit18 * (1 - bit18) == 0; assert bit19 * (1 - bit19) == 0; assert bit20 * (1 - bit20) == 0; assert bit21 * (1 - bit21) == 0; assert bit22 * (1 - bit22) == 0; assert bit23 * (1 - bit23) == 0; assert bit24 * (1 - bit24) == 0; assert bit25 * (1 - bit25) == 0; assert bit26 * (1 - bit26) == 0; assert bit27 * (1 - bit27) == 0; assert bit28 * (1 - bit28) == 0; assert bit29 * (1 - bit29) == 0; assert bit30 * (1 - bit30) == 0;
        assert a == bit0 + (2 * bit1) + (4 * bit2) + (8 * bit3) + (16 * bit4) + (32 * bit5) + (64 * bit6) + (128 * bit7) + (256 * bit8) + (512 * bit9) + (1024 * bit10) + (2048 * bit11) + (4096 * bit12) + (8192 * bit13) + (16384 * bit14) + (32768 * bit15) + (65536 * bit16) + (131072 * bit17) + (262144 * bit18) + (524288 * bit19) + (1048576 * bit20) + (2097152 * bit21) + (4194304 * bit22) + (8388608 * bit23) + (16777216 * bit24) + (33554432 * bit25) + (67108864 * bit26) + (134217728 * bit27) + (268435456 * bit28) + (536870912 * bit29) + (1073741824 * bit30);
        return bits; // a pointer to 31 bits
    }

    fn degree_two_polynomial_sum_at_0_and_1(coeffs) -> 1 {
        // coeffs is a vectorized pointer to 3 chunks of 8 field elements
        // return a vectorized pointer to 1 chunk of 8 field elements
        a = add_extension_ret(coeffs, coeffs);
        b = add_extension_ret(a, coeffs + 1);
        c = add_extension_ret(b, coeffs + 2);
        return c;
    }

    fn degree_two_polynomial_eval(coeffs, point) -> 1 {
        // coefs: vectorized
        // res: normal pointer to 8 field elements
        point_squared = mul_extension_ret(point, point);
        a_xx = mul_extension_ret(coeffs + 2, point_squared);
        b_x = mul_extension_ret(coeffs + 1, point);
        c = coeffs;
        res_0 = add_extension_ret(a_xx, b_x);
        res_1 = add_extension_ret(res_0, c);
        return res_1;
    }

    fn parse_commitment(fs_state) -> 4 {
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

    fn mul_extension_ret(a, b) -> 1 {
        c = malloc_vec(1);
        mul_extension(a, b, c);
        return c;
    }

    fn mul_extension(a, b, c) {
        // c = a * b

        ap = a * 8;
        bp = b * 8;
        cp = c * 8;
       
        cp[0] = (ap[0] * bp[0]) + W * ((ap[1] * bp[7]) + (ap[2] * bp[6]) + (ap[3] * bp[5]) + (ap[4] * bp[4]) + (ap[5] * bp[3]) + (ap[6] * bp[2]) + (ap[7] * bp[1]));
        cp[1] = (ap[1] * bp[0]) + (ap[0] * bp[1]) + W * ((ap[2] * bp[7]) + (ap[3] * bp[6]) + (ap[4] * bp[5]) + (ap[5] * bp[4]) + (ap[6] * bp[3]) + (ap[7] * bp[2]));
        cp[2] = (ap[2] * bp[0]) + (ap[1] * bp[1]) + (ap[0] * bp[2]) + W * ((ap[3] * bp[7]) + (ap[4] * bp[6]) + (ap[5] * bp[5]) + (ap[6] * bp[4]) + (ap[7] * bp[3]));
        cp[3] = (ap[3] * bp[0]) + (ap[2] * bp[1]) + (ap[1] * bp[2]) + (ap[0] * bp[3]) + W * ((ap[4] * bp[7]) + (ap[5] * bp[6]) + (ap[6] * bp[5]) + (ap[7] * bp[4]));
        cp[4] = (ap[4] * bp[0]) + (ap[3] * bp[1]) + (ap[2] * bp[2]) + (ap[1] * bp[3]) + (ap[0] * bp[4]) + W * ((ap[5] * bp[7]) + (ap[6] * bp[6]) + (ap[7] * bp[5]));
        cp[5] = (ap[5] * bp[0]) + (ap[4] * bp[1]) + (ap[3] * bp[2]) + (ap[2] * bp[3]) + (ap[1] * bp[4]) + (ap[0] * bp[5]) + W * ((ap[6] * bp[7]) + (ap[7] * bp[6]));
        cp[6] = (ap[6] * bp[0]) + (ap[5] * bp[1]) + (ap[4] * bp[2]) + (ap[3] * bp[3]) + (ap[2] * bp[4]) + (ap[1] * bp[5]) + (ap[0] * bp[6]) + W * (ap[7] * bp[7]);
        cp[7] = (ap[7] * bp[0]) + (ap[6] * bp[1]) + (ap[5] * bp[2]) + (ap[4] * bp[3]) + (ap[3] * bp[4]) + (ap[2] * bp[5]) + (ap[1] * bp[6]) + (ap[0] * bp[7]);

        return;
    }

    fn add_extension_ret(a, b) -> 1 {
        c = malloc_vec(1);
        add_extension(a, b, c);
        return c;
    }

      fn add_extension(a, b, c) {
        // c = a + b

        ap = a * 8;
        bp = b * 8;
        cp = c * 8;

        for i in 0..8 unroll {
            cp[i] = ap[i] + bp[i];
        }
        return;
    }

    fn sub_extension(a, b, c) {
        // c = a - b

        ap = a * 8;
        bp = b * 8;
        cp = c * 8;

        for i in 0..8 unroll {
            cp[i] = ap[i] - bp[i];
        }
        return;
    }

    fn eq_extension(a, b) -> 1 {
        // a and b are vectorized pointers
        // return 1 if a == b, 0 otherwise
        a_ptr = a * 8;
        b_ptr = b * 8;
        for i in 0..8 unroll {
            if a_ptr[i] != b_ptr[i] {
                return 0; // a != b
            }
        }
        return 1; // a == b
    }

    fn set_to_one(a) {
        a_ptr = 8 * a;
        a_ptr[0] = 1;
        for i in 1..8 unroll { a_ptr[i] = 0; }
        return;
    }

   fn print_chunk(a) {
        a_ptr = a * 8;
        for i in 0..8 {
            print(a_ptr[i]);
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
