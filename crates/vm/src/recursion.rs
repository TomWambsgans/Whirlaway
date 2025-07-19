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

    const TWO_POW_FOLDING_FACTOR_0 = 128;
    const TWO_POW_FOLDING_FACTOR_1 = 16;
    const TWO_POW_FOLDING_FACTOR_2 = 16;

    const RS_REDUCTION_FACTOR_0 = 5;
    const RS_REDUCTION_FACTOR_1 = 1;
    const RS_REDUCTION_FACTOR_2 = 1;

    const NUM_QUERIES_0 = 138;

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

        claimed_sum_side = mul_extension_vec(combination_randomness_gen_0, pcs_eval);
        claimed_sum = add_extension_vec(ood_eval_0, claimed_sum_side);

        fs_states_a = malloc(FOLDING_FACTOR_0 + 1);
        fs_states_a[0] = fs_state_4;

        claimed_sums = malloc(FOLDING_FACTOR_0 + 1);
        claimed_sums[0] = claimed_sum;

        folding_randomness = malloc(FOLDING_FACTOR_0); // in reverse order. A vector of vectorized pointers, each pointing to 1 chunk of 8 field elements

        for sc_round in 0..FOLDING_FACTOR_0 {
            fs_state_5, poly = fs_receive(fs_states_a[sc_round], 3); // vectorized pointer of len 1
            sum_over_boolean_hypercube = degree_two_polynomial_sum_at_0_and_1(poly);
            consistent = eq_extension_vec(sum_over_boolean_hypercube, claimed_sums[sc_round]);
            if consistent == 0 {
                panic();
            }
            fs_state_6, rand = fs_sample_ef(fs_state_5);  // vectorized pointer of len 1
            fs_states_a[sc_round + 1] = fs_state_6;
            new_claimed_sum = degree_two_polynomial_eval(poly, rand);
            claimed_sums[sc_round + 1] = new_claimed_sum;
            folding_randomness[FOLDING_FACTOR_0 - 1 - sc_round] = rand;
        }

        fs_state_7 = fs_states_a[FOLDING_FACTOR_0];

        fs_state_8, root_1, ood_point_1, ood_eval_1 = parse_commitment(fs_state_7);

        fs_state_9, stir_challenges_indexes = sample_bits(fs_state_8, NUM_QUERIES_0);

        // fs_print_state(fs_state_9); // 614216178 .. 310158447

        answers = malloc(NUM_QUERIES_0); // a vector of vectorized pointers, each pointing to TWO_POW_FOLDING_FACTOR_0 base field elements
        fs_states_b = malloc(NUM_QUERIES_0 + 1);
        fs_states_b[0] = fs_state_9;

        for i in 0..NUM_QUERIES_0 {
            new_fs_state, answer = fs_hint(fs_states_b[i], TWO_POW_FOLDING_FACTOR_0 / 8); // "/ 8" because initial merkle leaves are in the basefield
            fs_states_b[i + 1] = new_fs_state;
            answers[i] = answer;
        }
        fs_state_10 = fs_states_b[NUM_QUERIES_0];

        leaf_hashes = malloc(NUM_QUERIES_0); // a vector of vectorized pointers, each pointing to 1 chunk of 8 field elements
        for i in 0..NUM_QUERIES_0 {
            answer = answers[i];
            internal_states = malloc(1 + ((TWO_POW_FOLDING_FACTOR_0 / 8) / 2)); // "/ 2" because with poseidon24 we hash 2 chuncks of 8 field elements at each permutation
            internal_states[0] = pointer_to_zero_vector; // initial state
            for j in 0..(TWO_POW_FOLDING_FACTOR_0 / 8) / 2 {
                new_state_0, _, new_state_2 = poseidon24(answer + (2*j), answer + (2*j) + 1, internal_states[j]);
                if j == ((TWO_POW_FOLDING_FACTOR_0 / 8) / 2) - 1 {
                    // last step
                    internal_states[j + 1] = new_state_0;
                } else {
                    internal_states[j + 1] = new_state_2;
                }
            }
            leaf_hashes[i] = internal_states[(TWO_POW_FOLDING_FACTOR_0 / 8) / 2];
        }

        folded_domain_size = N_VARS + LOG_INV_RATE - FOLDING_FACTOR_0;

        fs_states_c = malloc(NUM_QUERIES_0 + 1);
        fs_states_c[0] = fs_state_10;

        for i in 0..NUM_QUERIES_0 {
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
            correct_root = eq_extension_vec(states[folded_domain_size], root_0);
            assert correct_root == 1;
        }

        fs_state_11 = fs_states_c[NUM_QUERIES_0];

        two_pow_FOLDING_FACTOR_0 = pow(2, FOLDING_FACTOR_0);

        poly_eq_0 = poly_eq(folding_randomness, FOLDING_FACTOR_0, two_pow_FOLDING_FACTOR_0);

        folds = malloc_vec(NUM_QUERIES_0);
        for i in 0..NUM_QUERIES_0 {
            dot_product_base_extension(answers[i] * 8, poly_eq_0, folds + i, two_pow_FOLDING_FACTOR_0);
        }

        circle_values = malloc(NUM_QUERIES_0); // ROOT^each_stir_index
        for i in 0..NUM_QUERIES_0 {
            stir_index_bits = stir_challenges_indexes[i];
            circle_value = unit_root_pow(folded_domain_size, stir_index_bits);
            circle_values[i] = circle_value;
            print(circle_value);
        }

        return;
    }

    fn unit_root_pow(domain_size, index_bits) -> 1 {
        // index_bits is a pointer to domain_size bits

        if domain_size == 19 {
            return ((index_bits[0] * ROOT_19) + (1 - index_bits[0])) * ((index_bits[1] * ROOT_18) + (1 - index_bits[1])) * ((index_bits[2] * ROOT_17) + (1 - index_bits[2])) * ((index_bits[3] * ROOT_16) + (1 - index_bits[3])) * ((index_bits[4] * ROOT_15) + (1 - index_bits[4])) * ((index_bits[5] * ROOT_14) + (1 - index_bits[5])) * ((index_bits[6] * ROOT_13) + (1 - index_bits[6])) * ((index_bits[7] * ROOT_12) + (1 - index_bits[7])) * ((index_bits[8] * ROOT_11) + (1 - index_bits[8])) * ((index_bits[9] * ROOT_10) + (1 - index_bits[9])) * ((index_bits[10] * ROOT_9) + (1 - index_bits[10])) * ((index_bits[11] * ROOT_8) + (1 - index_bits[11])) * ((index_bits[12] * ROOT_7) + (1 - index_bits[12])) * ((index_bits[13] * ROOT_6) + (1 - index_bits[13])) * ((index_bits[14] * ROOT_5) + (1 - index_bits[14])) * ((index_bits[15] * ROOT_4) + (1 - index_bits[15])) * ((index_bits[16] * ROOT_3) + (1 - index_bits[16])) * ((index_bits[17] * ROOT_2) + (1 - index_bits[17])) * ((index_bits[18] * ROOT_1) + (1 - index_bits[18]));
        }

        panic(); // unimplemented
    }

    fn dot_product_base_extension(a, b, res, n) {
        // a is a pointer to n base field elements
        // b is a vectorized pointer to n extension field elements
        // res is a vectorized pointer to 1 extension field element, will be set to the dot product of a and b
        // n is the number of elements in a and b

        prods = malloc(n * 8);
        b_ptr = b * 8;
        // TODO macro / loop unrolling
        a0 = a[0]; prods[0] = a0 * b_ptr[0]; prods[1] = a0 * b_ptr[1]; prods[2] = a0 * b_ptr[2]; prods[3] = a0 * b_ptr[3]; prods[4] = a0 * b_ptr[4]; prods[5] = a0 * b_ptr[5]; prods[6] = a0 * b_ptr[6]; prods[7] = a0 * b_ptr[7];
        a1 = a[1]; prods[8] = a1 * b_ptr[8]; prods[9] = a1 * b_ptr[9]; prods[10] = a1 * b_ptr[10]; prods[11] = a1 * b_ptr[11]; prods[12] = a1 * b_ptr[12]; prods[13] = a1 * b_ptr[13]; prods[14] = a1 * b_ptr[14]; prods[15] = a1 * b_ptr[15];
        a2 = a[2]; prods[16] = a2 * b_ptr[16]; prods[17] = a2 * b_ptr[17]; prods[18] = a2 * b_ptr[18]; prods[19] = a2 * b_ptr[19]; prods[20] = a2 * b_ptr[20]; prods[21] = a2 * b_ptr[21]; prods[22] = a2 * b_ptr[22]; prods[23] = a2 * b_ptr[23];
        a3 = a[3]; prods[24] = a3 * b_ptr[24]; prods[25] = a3 * b_ptr[25]; prods[26] = a3 * b_ptr[26]; prods[27] = a3 * b_ptr[27]; prods[28] = a3 * b_ptr[28]; prods[29] = a3 * b_ptr[29]; prods[30] = a3 * b_ptr[30]; prods[31] = a3 * b_ptr[31];
        a4 = a[4]; prods[32] = a4 * b_ptr[32]; prods[33] = a4 * b_ptr[33]; prods[34] = a4 * b_ptr[34]; prods[35] = a4 * b_ptr[35]; prods[36] = a4 * b_ptr[36]; prods[37] = a4 * b_ptr[37]; prods[38] = a4 * b_ptr[38]; prods[39] = a4 * b_ptr[39];
        a5 = a[5]; prods[40] = a5 * b_ptr[40]; prods[41] = a5 * b_ptr[41]; prods[42] = a5 * b_ptr[42]; prods[43] = a5 * b_ptr[43]; prods[44] = a5 * b_ptr[44]; prods[45] = a5 * b_ptr[45]; prods[46] = a5 * b_ptr[46]; prods[47] = a5 * b_ptr[47];
        a6 = a[6]; prods[48] = a6 * b_ptr[48]; prods[49] = a6 * b_ptr[49]; prods[50] = a6 * b_ptr[50]; prods[51] = a6 * b_ptr[51]; prods[52] = a6 * b_ptr[52]; prods[53] = a6 * b_ptr[53]; prods[54] = a6 * b_ptr[54]; prods[55] = a6 * b_ptr[55];
        a7 = a[7]; prods[56] = a7 * b_ptr[56]; prods[57] = a7 * b_ptr[57]; prods[58] = a7 * b_ptr[58]; prods[59] = a7 * b_ptr[59]; prods[60] = a7 * b_ptr[60]; prods[61] = a7 * b_ptr[61]; prods[62] = a7 * b_ptr[62]; prods[63] = a7 * b_ptr[63];
        a8 = a[8]; prods[64] = a8 * b_ptr[64]; prods[65] = a8 * b_ptr[65]; prods[66] = a8 * b_ptr[66]; prods[67] = a8 * b_ptr[67]; prods[68] = a8 * b_ptr[68]; prods[69] = a8 * b_ptr[69]; prods[70] = a8 * b_ptr[70]; prods[71] = a8 * b_ptr[71];
        a9 = a[9]; prods[72] = a9 * b_ptr[72]; prods[73] = a9 * b_ptr[73]; prods[74] = a9 * b_ptr[74]; prods[75] = a9 * b_ptr[75]; prods[76] = a9 * b_ptr[76]; prods[77] = a9 * b_ptr[77]; prods[78] = a9 * b_ptr[78]; prods[79] = a9 * b_ptr[79];
        a10 = a[10]; prods[80] = a10 * b_ptr[80]; prods[81] = a10 * b_ptr[81]; prods[82] = a10 * b_ptr[82]; prods[83] = a10 * b_ptr[83]; prods[84] = a10 * b_ptr[84]; prods[85] = a10 * b_ptr[85]; prods[86] = a10 * b_ptr[86]; prods[87] = a10 * b_ptr[87];
        a11 = a[11]; prods[88] = a11 * b_ptr[88]; prods[89] = a11 * b_ptr[89]; prods[90] = a11 * b_ptr[90]; prods[91] = a11 * b_ptr[91]; prods[92] = a11 * b_ptr[92]; prods[93] = a11 * b_ptr[93]; prods[94] = a11 * b_ptr[94]; prods[95] = a11 * b_ptr[95];
        a12 = a[12]; prods[96] = a12 * b_ptr[96]; prods[97] = a12 * b_ptr[97]; prods[98] = a12 * b_ptr[98]; prods[99] = a12 * b_ptr[99]; prods[100] = a12 * b_ptr[100]; prods[101] = a12 * b_ptr[101]; prods[102] = a12 * b_ptr[102]; prods[103] = a12 * b_ptr[103];
        a13 = a[13]; prods[104] = a13 * b_ptr[104]; prods[105] = a13 * b_ptr[105]; prods[106] = a13 * b_ptr[106]; prods[107] = a13 * b_ptr[107]; prods[108] = a13 * b_ptr[108]; prods[109] = a13 * b_ptr[109]; prods[110] = a13 * b_ptr[110]; prods[111] = a13 * b_ptr[111];
        a14 = a[14]; prods[112] = a14 * b_ptr[112]; prods[113] = a14 * b_ptr[113]; prods[114] = a14 * b_ptr[114]; prods[115] = a14 * b_ptr[115]; prods[116] = a14 * b_ptr[116]; prods[117] = a14 * b_ptr[117]; prods[118] = a14 * b_ptr[118]; prods[119] = a14 * b_ptr[119];
        a15 = a[15]; prods[120] = a15 * b_ptr[120]; prods[121] = a15 * b_ptr[121]; prods[122] = a15 * b_ptr[122]; prods[123] = a15 * b_ptr[123]; prods[124] = a15 * b_ptr[124]; prods[125] = a15 * b_ptr[125]; prods[126] = a15 * b_ptr[126]; prods[127] = a15 * b_ptr[127];
        a16 = a[16]; prods[128] = a16 * b_ptr[128]; prods[129] = a16 * b_ptr[129]; prods[130] = a16 * b_ptr[130]; prods[131] = a16 * b_ptr[131]; prods[132] = a16 * b_ptr[132]; prods[133] = a16 * b_ptr[133]; prods[134] = a16 * b_ptr[134]; prods[135] = a16 * b_ptr[135];
        a17 = a[17]; prods[136] = a17 * b_ptr[136]; prods[137] = a17 * b_ptr[137]; prods[138] = a17 * b_ptr[138]; prods[139] = a17 * b_ptr[139]; prods[140] = a17 * b_ptr[140]; prods[141] = a17 * b_ptr[141]; prods[142] = a17 * b_ptr[142]; prods[143] = a17 * b_ptr[143];
        a18 = a[18]; prods[144] = a18 * b_ptr[144]; prods[145] = a18 * b_ptr[145]; prods[146] = a18 * b_ptr[146]; prods[147] = a18 * b_ptr[147]; prods[148] = a18 * b_ptr[148]; prods[149] = a18 * b_ptr[149]; prods[150] = a18 * b_ptr[150]; prods[151] = a18 * b_ptr[151];
        a19 = a[19]; prods[152] = a19 * b_ptr[152]; prods[153] = a19 * b_ptr[153]; prods[154] = a19 * b_ptr[154]; prods[155] = a19 * b_ptr[155]; prods[156] = a19 * b_ptr[156]; prods[157] = a19 * b_ptr[157]; prods[158] = a19 * b_ptr[158]; prods[159] = a19 * b_ptr[159];
        a20 = a[20]; prods[160] = a20 * b_ptr[160]; prods[161] = a20 * b_ptr[161]; prods[162] = a20 * b_ptr[162]; prods[163] = a20 * b_ptr[163]; prods[164] = a20 * b_ptr[164]; prods[165] = a20 * b_ptr[165]; prods[166] = a20 * b_ptr[166]; prods[167] = a20 * b_ptr[167];
        a21 = a[21]; prods[168] = a21 * b_ptr[168]; prods[169] = a21 * b_ptr[169]; prods[170] = a21 * b_ptr[170]; prods[171] = a21 * b_ptr[171]; prods[172] = a21 * b_ptr[172]; prods[173] = a21 * b_ptr[173]; prods[174] = a21 * b_ptr[174]; prods[175] = a21 * b_ptr[175];
        a22 = a[22]; prods[176] = a22 * b_ptr[176]; prods[177] = a22 * b_ptr[177]; prods[178] = a22 * b_ptr[178]; prods[179] = a22 * b_ptr[179]; prods[180] = a22 * b_ptr[180]; prods[181] = a22 * b_ptr[181]; prods[182] = a22 * b_ptr[182]; prods[183] = a22 * b_ptr[183];
        a23 = a[23]; prods[184] = a23 * b_ptr[184]; prods[185] = a23 * b_ptr[185]; prods[186] = a23 * b_ptr[186]; prods[187] = a23 * b_ptr[187]; prods[188] = a23 * b_ptr[188]; prods[189] = a23 * b_ptr[189]; prods[190] = a23 * b_ptr[190]; prods[191] = a23 * b_ptr[191];
        a24 = a[24]; prods[192] = a24 * b_ptr[192]; prods[193] = a24 * b_ptr[193]; prods[194] = a24 * b_ptr[194]; prods[195] = a24 * b_ptr[195]; prods[196] = a24 * b_ptr[196]; prods[197] = a24 * b_ptr[197]; prods[198] = a24 * b_ptr[198]; prods[199] = a24 * b_ptr[199];
        a25 = a[25]; prods[200] = a25 * b_ptr[200]; prods[201] = a25 * b_ptr[201]; prods[202] = a25 * b_ptr[202]; prods[203] = a25 * b_ptr[203]; prods[204] = a25 * b_ptr[204]; prods[205] = a25 * b_ptr[205]; prods[206] = a25 * b_ptr[206]; prods[207] = a25 * b_ptr[207];
        a26 = a[26]; prods[208] = a26 * b_ptr[208]; prods[209] = a26 * b_ptr[209]; prods[210] = a26 * b_ptr[210]; prods[211] = a26 * b_ptr[211]; prods[212] = a26 * b_ptr[212]; prods[213] = a26 * b_ptr[213]; prods[214] = a26 * b_ptr[214]; prods[215] = a26 * b_ptr[215];
        a27 = a[27]; prods[216] = a27 * b_ptr[216]; prods[217] = a27 * b_ptr[217]; prods[218] = a27 * b_ptr[218]; prods[219] = a27 * b_ptr[219]; prods[220] = a27 * b_ptr[220]; prods[221] = a27 * b_ptr[221]; prods[222] = a27 * b_ptr[222]; prods[223] = a27 * b_ptr[223];
        a28 = a[28]; prods[224] = a28 * b_ptr[224]; prods[225] = a28 * b_ptr[225]; prods[226] = a28 * b_ptr[226]; prods[227] = a28 * b_ptr[227]; prods[228] = a28 * b_ptr[228]; prods[229] = a28 * b_ptr[229]; prods[230] = a28 * b_ptr[230]; prods[231] = a28 * b_ptr[231];
        a29 = a[29]; prods[232] = a29 * b_ptr[232]; prods[233] = a29 * b_ptr[233]; prods[234] = a29 * b_ptr[234]; prods[235] = a29 * b_ptr[235]; prods[236] = a29 * b_ptr[236]; prods[237] = a29 * b_ptr[237]; prods[238] = a29 * b_ptr[238]; prods[239] = a29 * b_ptr[239];
        a30 = a[30]; prods[240] = a30 * b_ptr[240]; prods[241] = a30 * b_ptr[241]; prods[242] = a30 * b_ptr[242]; prods[243] = a30 * b_ptr[243]; prods[244] = a30 * b_ptr[244]; prods[245] = a30 * b_ptr[245]; prods[246] = a30 * b_ptr[246]; prods[247] = a30 * b_ptr[247];
        a31 = a[31]; prods[248] = a31 * b_ptr[248]; prods[249] = a31 * b_ptr[249]; prods[250] = a31 * b_ptr[250]; prods[251] = a31 * b_ptr[251]; prods[252] = a31 * b_ptr[252]; prods[253] = a31 * b_ptr[253]; prods[254] = a31 * b_ptr[254]; prods[255] = a31 * b_ptr[255];
        a32 = a[32]; prods[256] = a32 * b_ptr[256]; prods[257] = a32 * b_ptr[257]; prods[258] = a32 * b_ptr[258]; prods[259] = a32 * b_ptr[259]; prods[260] = a32 * b_ptr[260]; prods[261] = a32 * b_ptr[261]; prods[262] = a32 * b_ptr[262]; prods[263] = a32 * b_ptr[263];
        a33 = a[33]; prods[264] = a33 * b_ptr[264]; prods[265] = a33 * b_ptr[265]; prods[266] = a33 * b_ptr[266]; prods[267] = a33 * b_ptr[267]; prods[268] = a33 * b_ptr[268]; prods[269] = a33 * b_ptr[269]; prods[270] = a33 * b_ptr[270]; prods[271] = a33 * b_ptr[271];
        a34 = a[34]; prods[272] = a34 * b_ptr[272]; prods[273] = a34 * b_ptr[273]; prods[274] = a34 * b_ptr[274]; prods[275] = a34 * b_ptr[275]; prods[276] = a34 * b_ptr[276]; prods[277] = a34 * b_ptr[277]; prods[278] = a34 * b_ptr[278]; prods[279] = a34 * b_ptr[279];
        a35 = a[35]; prods[280] = a35 * b_ptr[280]; prods[281] = a35 * b_ptr[281]; prods[282] = a35 * b_ptr[282]; prods[283] = a35 * b_ptr[283]; prods[284] = a35 * b_ptr[284]; prods[285] = a35 * b_ptr[285]; prods[286] = a35 * b_ptr[286]; prods[287] = a35 * b_ptr[287];
        a36 = a[36]; prods[288] = a36 * b_ptr[288]; prods[289] = a36 * b_ptr[289]; prods[290] = a36 * b_ptr[290]; prods[291] = a36 * b_ptr[291]; prods[292] = a36 * b_ptr[292]; prods[293] = a36 * b_ptr[293]; prods[294] = a36 * b_ptr[294]; prods[295] = a36 * b_ptr[295];
        a37 = a[37]; prods[296] = a37 * b_ptr[296]; prods[297] = a37 * b_ptr[297]; prods[298] = a37 * b_ptr[298]; prods[299] = a37 * b_ptr[299]; prods[300] = a37 * b_ptr[300]; prods[301] = a37 * b_ptr[301]; prods[302] = a37 * b_ptr[302]; prods[303] = a37 * b_ptr[303];
        a38 = a[38]; prods[304] = a38 * b_ptr[304]; prods[305] = a38 * b_ptr[305]; prods[306] = a38 * b_ptr[306]; prods[307] = a38 * b_ptr[307]; prods[308] = a38 * b_ptr[308]; prods[309] = a38 * b_ptr[309]; prods[310] = a38 * b_ptr[310]; prods[311] = a38 * b_ptr[311];
        a39 = a[39]; prods[312] = a39 * b_ptr[312]; prods[313] = a39 * b_ptr[313]; prods[314] = a39 * b_ptr[314]; prods[315] = a39 * b_ptr[315]; prods[316] = a39 * b_ptr[316]; prods[317] = a39 * b_ptr[317]; prods[318] = a39 * b_ptr[318]; prods[319] = a39 * b_ptr[319];
        a40 = a[40]; prods[320] = a40 * b_ptr[320]; prods[321] = a40 * b_ptr[321]; prods[322] = a40 * b_ptr[322]; prods[323] = a40 * b_ptr[323]; prods[324] = a40 * b_ptr[324]; prods[325] = a40 * b_ptr[325]; prods[326] = a40 * b_ptr[326]; prods[327] = a40 * b_ptr[327];
        a41 = a[41]; prods[328] = a41 * b_ptr[328]; prods[329] = a41 * b_ptr[329]; prods[330] = a41 * b_ptr[330]; prods[331] = a41 * b_ptr[331]; prods[332] = a41 * b_ptr[332]; prods[333] = a41 * b_ptr[333]; prods[334] = a41 * b_ptr[334]; prods[335] = a41 * b_ptr[335];
        a42 = a[42]; prods[336] = a42 * b_ptr[336]; prods[337] = a42 * b_ptr[337]; prods[338] = a42 * b_ptr[338]; prods[339] = a42 * b_ptr[339]; prods[340] = a42 * b_ptr[340]; prods[341] = a42 * b_ptr[341]; prods[342] = a42 * b_ptr[342]; prods[343] = a42 * b_ptr[343];
        a43 = a[43]; prods[344] = a43 * b_ptr[344]; prods[345] = a43 * b_ptr[345]; prods[346] = a43 * b_ptr[346]; prods[347] = a43 * b_ptr[347]; prods[348] = a43 * b_ptr[348]; prods[349] = a43 * b_ptr[349]; prods[350] = a43 * b_ptr[350]; prods[351] = a43 * b_ptr[351];
        a44 = a[44]; prods[352] = a44 * b_ptr[352]; prods[353] = a44 * b_ptr[353]; prods[354] = a44 * b_ptr[354]; prods[355] = a44 * b_ptr[355]; prods[356] = a44 * b_ptr[356]; prods[357] = a44 * b_ptr[357]; prods[358] = a44 * b_ptr[358]; prods[359] = a44 * b_ptr[359];
        a45 = a[45]; prods[360] = a45 * b_ptr[360]; prods[361] = a45 * b_ptr[361]; prods[362] = a45 * b_ptr[362]; prods[363] = a45 * b_ptr[363]; prods[364] = a45 * b_ptr[364]; prods[365] = a45 * b_ptr[365]; prods[366] = a45 * b_ptr[366]; prods[367] = a45 * b_ptr[367];
        a46 = a[46]; prods[368] = a46 * b_ptr[368]; prods[369] = a46 * b_ptr[369]; prods[370] = a46 * b_ptr[370]; prods[371] = a46 * b_ptr[371]; prods[372] = a46 * b_ptr[372]; prods[373] = a46 * b_ptr[373]; prods[374] = a46 * b_ptr[374]; prods[375] = a46 * b_ptr[375];
        a47 = a[47]; prods[376] = a47 * b_ptr[376]; prods[377] = a47 * b_ptr[377]; prods[378] = a47 * b_ptr[378]; prods[379] = a47 * b_ptr[379]; prods[380] = a47 * b_ptr[380]; prods[381] = a47 * b_ptr[381]; prods[382] = a47 * b_ptr[382]; prods[383] = a47 * b_ptr[383];
        a48 = a[48]; prods[384] = a48 * b_ptr[384]; prods[385] = a48 * b_ptr[385]; prods[386] = a48 * b_ptr[386]; prods[387] = a48 * b_ptr[387]; prods[388] = a48 * b_ptr[388]; prods[389] = a48 * b_ptr[389]; prods[390] = a48 * b_ptr[390]; prods[391] = a48 * b_ptr[391];
        a49 = a[49]; prods[392] = a49 * b_ptr[392]; prods[393] = a49 * b_ptr[393]; prods[394] = a49 * b_ptr[394]; prods[395] = a49 * b_ptr[395]; prods[396] = a49 * b_ptr[396]; prods[397] = a49 * b_ptr[397]; prods[398] = a49 * b_ptr[398]; prods[399] = a49 * b_ptr[399];
        a50 = a[50]; prods[400] = a50 * b_ptr[400]; prods[401] = a50 * b_ptr[401]; prods[402] = a50 * b_ptr[402]; prods[403] = a50 * b_ptr[403]; prods[404] = a50 * b_ptr[404]; prods[405] = a50 * b_ptr[405]; prods[406] = a50 * b_ptr[406]; prods[407] = a50 * b_ptr[407];
        a51 = a[51]; prods[408] = a51 * b_ptr[408]; prods[409] = a51 * b_ptr[409]; prods[410] = a51 * b_ptr[410]; prods[411] = a51 * b_ptr[411]; prods[412] = a51 * b_ptr[412]; prods[413] = a51 * b_ptr[413]; prods[414] = a51 * b_ptr[414]; prods[415] = a51 * b_ptr[415];
        a52 = a[52]; prods[416] = a52 * b_ptr[416]; prods[417] = a52 * b_ptr[417]; prods[418] = a52 * b_ptr[418]; prods[419] = a52 * b_ptr[419]; prods[420] = a52 * b_ptr[420]; prods[421] = a52 * b_ptr[421]; prods[422] = a52 * b_ptr[422]; prods[423] = a52 * b_ptr[423];
        a53 = a[53]; prods[424] = a53 * b_ptr[424]; prods[425] = a53 * b_ptr[425]; prods[426] = a53 * b_ptr[426]; prods[427] = a53 * b_ptr[427]; prods[428] = a53 * b_ptr[428]; prods[429] = a53 * b_ptr[429]; prods[430] = a53 * b_ptr[430]; prods[431] = a53 * b_ptr[431];
        a54 = a[54]; prods[432] = a54 * b_ptr[432]; prods[433] = a54 * b_ptr[433]; prods[434] = a54 * b_ptr[434]; prods[435] = a54 * b_ptr[435]; prods[436] = a54 * b_ptr[436]; prods[437] = a54 * b_ptr[437]; prods[438] = a54 * b_ptr[438]; prods[439] = a54 * b_ptr[439];
        a55 = a[55]; prods[440] = a55 * b_ptr[440]; prods[441] = a55 * b_ptr[441]; prods[442] = a55 * b_ptr[442]; prods[443] = a55 * b_ptr[443]; prods[444] = a55 * b_ptr[444]; prods[445] = a55 * b_ptr[445]; prods[446] = a55 * b_ptr[446]; prods[447] = a55 * b_ptr[447];
        a56 = a[56]; prods[448] = a56 * b_ptr[448]; prods[449] = a56 * b_ptr[449]; prods[450] = a56 * b_ptr[450]; prods[451] = a56 * b_ptr[451]; prods[452] = a56 * b_ptr[452]; prods[453] = a56 * b_ptr[453]; prods[454] = a56 * b_ptr[454]; prods[455] = a56 * b_ptr[455];
        a57 = a[57]; prods[456] = a57 * b_ptr[456]; prods[457] = a57 * b_ptr[457]; prods[458] = a57 * b_ptr[458]; prods[459] = a57 * b_ptr[459]; prods[460] = a57 * b_ptr[460]; prods[461] = a57 * b_ptr[461]; prods[462] = a57 * b_ptr[462]; prods[463] = a57 * b_ptr[463];
        a58 = a[58]; prods[464] = a58 * b_ptr[464]; prods[465] = a58 * b_ptr[465]; prods[466] = a58 * b_ptr[466]; prods[467] = a58 * b_ptr[467]; prods[468] = a58 * b_ptr[468]; prods[469] = a58 * b_ptr[469]; prods[470] = a58 * b_ptr[470]; prods[471] = a58 * b_ptr[471];
        a59 = a[59]; prods[472] = a59 * b_ptr[472]; prods[473] = a59 * b_ptr[473]; prods[474] = a59 * b_ptr[474]; prods[475] = a59 * b_ptr[475]; prods[476] = a59 * b_ptr[476]; prods[477] = a59 * b_ptr[477]; prods[478] = a59 * b_ptr[478]; prods[479] = a59 * b_ptr[479];
        a60 = a[60]; prods[480] = a60 * b_ptr[480]; prods[481] = a60 * b_ptr[481]; prods[482] = a60 * b_ptr[482]; prods[483] = a60 * b_ptr[483]; prods[484] = a60 * b_ptr[484]; prods[485] = a60 * b_ptr[485]; prods[486] = a60 * b_ptr[486]; prods[487] = a60 * b_ptr[487];
        a61 = a[61]; prods[488] = a61 * b_ptr[488]; prods[489] = a61 * b_ptr[489]; prods[490] = a61 * b_ptr[490]; prods[491] = a61 * b_ptr[491]; prods[492] = a61 * b_ptr[492]; prods[493] = a61 * b_ptr[493]; prods[494] = a61 * b_ptr[494]; prods[495] = a61 * b_ptr[495];
        a62 = a[62]; prods[496] = a62 * b_ptr[496]; prods[497] = a62 * b_ptr[497]; prods[498] = a62 * b_ptr[498]; prods[499] = a62 * b_ptr[499]; prods[500] = a62 * b_ptr[500]; prods[501] = a62 * b_ptr[501]; prods[502] = a62 * b_ptr[502]; prods[503] = a62 * b_ptr[503];
        a63 = a[63]; prods[504] = a63 * b_ptr[504]; prods[505] = a63 * b_ptr[505]; prods[506] = a63 * b_ptr[506]; prods[507] = a63 * b_ptr[507]; prods[508] = a63 * b_ptr[508]; prods[509] = a63 * b_ptr[509]; prods[510] = a63 * b_ptr[510]; prods[511] = a63 * b_ptr[511];
        a64 = a[64]; prods[512] = a64 * b_ptr[512]; prods[513] = a64 * b_ptr[513]; prods[514] = a64 * b_ptr[514]; prods[515] = a64 * b_ptr[515]; prods[516] = a64 * b_ptr[516]; prods[517] = a64 * b_ptr[517]; prods[518] = a64 * b_ptr[518]; prods[519] = a64 * b_ptr[519];
        a65 = a[65]; prods[520] = a65 * b_ptr[520]; prods[521] = a65 * b_ptr[521]; prods[522] = a65 * b_ptr[522]; prods[523] = a65 * b_ptr[523]; prods[524] = a65 * b_ptr[524]; prods[525] = a65 * b_ptr[525]; prods[526] = a65 * b_ptr[526]; prods[527] = a65 * b_ptr[527];
        a66 = a[66]; prods[528] = a66 * b_ptr[528]; prods[529] = a66 * b_ptr[529]; prods[530] = a66 * b_ptr[530]; prods[531] = a66 * b_ptr[531]; prods[532] = a66 * b_ptr[532]; prods[533] = a66 * b_ptr[533]; prods[534] = a66 * b_ptr[534]; prods[535] = a66 * b_ptr[535];
        a67 = a[67]; prods[536] = a67 * b_ptr[536]; prods[537] = a67 * b_ptr[537]; prods[538] = a67 * b_ptr[538]; prods[539] = a67 * b_ptr[539]; prods[540] = a67 * b_ptr[540]; prods[541] = a67 * b_ptr[541]; prods[542] = a67 * b_ptr[542]; prods[543] = a67 * b_ptr[543];
        a68 = a[68]; prods[544] = a68 * b_ptr[544]; prods[545] = a68 * b_ptr[545]; prods[546] = a68 * b_ptr[546]; prods[547] = a68 * b_ptr[547]; prods[548] = a68 * b_ptr[548]; prods[549] = a68 * b_ptr[549]; prods[550] = a68 * b_ptr[550]; prods[551] = a68 * b_ptr[551];
        a69 = a[69]; prods[552] = a69 * b_ptr[552]; prods[553] = a69 * b_ptr[553]; prods[554] = a69 * b_ptr[554]; prods[555] = a69 * b_ptr[555]; prods[556] = a69 * b_ptr[556]; prods[557] = a69 * b_ptr[557]; prods[558] = a69 * b_ptr[558]; prods[559] = a69 * b_ptr[559];
        a70 = a[70]; prods[560] = a70 * b_ptr[560]; prods[561] = a70 * b_ptr[561]; prods[562] = a70 * b_ptr[562]; prods[563] = a70 * b_ptr[563]; prods[564] = a70 * b_ptr[564]; prods[565] = a70 * b_ptr[565]; prods[566] = a70 * b_ptr[566]; prods[567] = a70 * b_ptr[567];
        a71 = a[71]; prods[568] = a71 * b_ptr[568]; prods[569] = a71 * b_ptr[569]; prods[570] = a71 * b_ptr[570]; prods[571] = a71 * b_ptr[571]; prods[572] = a71 * b_ptr[572]; prods[573] = a71 * b_ptr[573]; prods[574] = a71 * b_ptr[574]; prods[575] = a71 * b_ptr[575];
        a72 = a[72]; prods[576] = a72 * b_ptr[576]; prods[577] = a72 * b_ptr[577]; prods[578] = a72 * b_ptr[578]; prods[579] = a72 * b_ptr[579]; prods[580] = a72 * b_ptr[580]; prods[581] = a72 * b_ptr[581]; prods[582] = a72 * b_ptr[582]; prods[583] = a72 * b_ptr[583];
        a73 = a[73]; prods[584] = a73 * b_ptr[584]; prods[585] = a73 * b_ptr[585]; prods[586] = a73 * b_ptr[586]; prods[587] = a73 * b_ptr[587]; prods[588] = a73 * b_ptr[588]; prods[589] = a73 * b_ptr[589]; prods[590] = a73 * b_ptr[590]; prods[591] = a73 * b_ptr[591];
        a74 = a[74]; prods[592] = a74 * b_ptr[592]; prods[593] = a74 * b_ptr[593]; prods[594] = a74 * b_ptr[594]; prods[595] = a74 * b_ptr[595]; prods[596] = a74 * b_ptr[596]; prods[597] = a74 * b_ptr[597]; prods[598] = a74 * b_ptr[598]; prods[599] = a74 * b_ptr[599];
        a75 = a[75]; prods[600] = a75 * b_ptr[600]; prods[601] = a75 * b_ptr[601]; prods[602] = a75 * b_ptr[602]; prods[603] = a75 * b_ptr[603]; prods[604] = a75 * b_ptr[604]; prods[605] = a75 * b_ptr[605]; prods[606] = a75 * b_ptr[606]; prods[607] = a75 * b_ptr[607];
        a76 = a[76]; prods[608] = a76 * b_ptr[608]; prods[609] = a76 * b_ptr[609]; prods[610] = a76 * b_ptr[610]; prods[611] = a76 * b_ptr[611]; prods[612] = a76 * b_ptr[612]; prods[613] = a76 * b_ptr[613]; prods[614] = a76 * b_ptr[614]; prods[615] = a76 * b_ptr[615];
        a77 = a[77]; prods[616] = a77 * b_ptr[616]; prods[617] = a77 * b_ptr[617]; prods[618] = a77 * b_ptr[618]; prods[619] = a77 * b_ptr[619]; prods[620] = a77 * b_ptr[620]; prods[621] = a77 * b_ptr[621]; prods[622] = a77 * b_ptr[622]; prods[623] = a77 * b_ptr[623];
        a78 = a[78]; prods[624] = a78 * b_ptr[624]; prods[625] = a78 * b_ptr[625]; prods[626] = a78 * b_ptr[626]; prods[627] = a78 * b_ptr[627]; prods[628] = a78 * b_ptr[628]; prods[629] = a78 * b_ptr[629]; prods[630] = a78 * b_ptr[630]; prods[631] = a78 * b_ptr[631];
        a79 = a[79]; prods[632] = a79 * b_ptr[632]; prods[633] = a79 * b_ptr[633]; prods[634] = a79 * b_ptr[634]; prods[635] = a79 * b_ptr[635]; prods[636] = a79 * b_ptr[636]; prods[637] = a79 * b_ptr[637]; prods[638] = a79 * b_ptr[638]; prods[639] = a79 * b_ptr[639];
        a80 = a[80]; prods[640] = a80 * b_ptr[640]; prods[641] = a80 * b_ptr[641]; prods[642] = a80 * b_ptr[642]; prods[643] = a80 * b_ptr[643]; prods[644] = a80 * b_ptr[644]; prods[645] = a80 * b_ptr[645]; prods[646] = a80 * b_ptr[646]; prods[647] = a80 * b_ptr[647];
        a81 = a[81]; prods[648] = a81 * b_ptr[648]; prods[649] = a81 * b_ptr[649]; prods[650] = a81 * b_ptr[650]; prods[651] = a81 * b_ptr[651]; prods[652] = a81 * b_ptr[652]; prods[653] = a81 * b_ptr[653]; prods[654] = a81 * b_ptr[654]; prods[655] = a81 * b_ptr[655];
        a82 = a[82]; prods[656] = a82 * b_ptr[656]; prods[657] = a82 * b_ptr[657]; prods[658] = a82 * b_ptr[658]; prods[659] = a82 * b_ptr[659]; prods[660] = a82 * b_ptr[660]; prods[661] = a82 * b_ptr[661]; prods[662] = a82 * b_ptr[662]; prods[663] = a82 * b_ptr[663];
        a83 = a[83]; prods[664] = a83 * b_ptr[664]; prods[665] = a83 * b_ptr[665]; prods[666] = a83 * b_ptr[666]; prods[667] = a83 * b_ptr[667]; prods[668] = a83 * b_ptr[668]; prods[669] = a83 * b_ptr[669]; prods[670] = a83 * b_ptr[670]; prods[671] = a83 * b_ptr[671];
        a84 = a[84]; prods[672] = a84 * b_ptr[672]; prods[673] = a84 * b_ptr[673]; prods[674] = a84 * b_ptr[674]; prods[675] = a84 * b_ptr[675]; prods[676] = a84 * b_ptr[676]; prods[677] = a84 * b_ptr[677]; prods[678] = a84 * b_ptr[678]; prods[679] = a84 * b_ptr[679];
        a85 = a[85]; prods[680] = a85 * b_ptr[680]; prods[681] = a85 * b_ptr[681]; prods[682] = a85 * b_ptr[682]; prods[683] = a85 * b_ptr[683]; prods[684] = a85 * b_ptr[684]; prods[685] = a85 * b_ptr[685]; prods[686] = a85 * b_ptr[686]; prods[687] = a85 * b_ptr[687];
        a86 = a[86]; prods[688] = a86 * b_ptr[688]; prods[689] = a86 * b_ptr[689]; prods[690] = a86 * b_ptr[690]; prods[691] = a86 * b_ptr[691]; prods[692] = a86 * b_ptr[692]; prods[693] = a86 * b_ptr[693]; prods[694] = a86 * b_ptr[694]; prods[695] = a86 * b_ptr[695];
        a87 = a[87]; prods[696] = a87 * b_ptr[696]; prods[697] = a87 * b_ptr[697]; prods[698] = a87 * b_ptr[698]; prods[699] = a87 * b_ptr[699]; prods[700] = a87 * b_ptr[700]; prods[701] = a87 * b_ptr[701]; prods[702] = a87 * b_ptr[702]; prods[703] = a87 * b_ptr[703];
        a88 = a[88]; prods[704] = a88 * b_ptr[704]; prods[705] = a88 * b_ptr[705]; prods[706] = a88 * b_ptr[706]; prods[707] = a88 * b_ptr[707]; prods[708] = a88 * b_ptr[708]; prods[709] = a88 * b_ptr[709]; prods[710] = a88 * b_ptr[710]; prods[711] = a88 * b_ptr[711];
        a89 = a[89]; prods[712] = a89 * b_ptr[712]; prods[713] = a89 * b_ptr[713]; prods[714] = a89 * b_ptr[714]; prods[715] = a89 * b_ptr[715]; prods[716] = a89 * b_ptr[716]; prods[717] = a89 * b_ptr[717]; prods[718] = a89 * b_ptr[718]; prods[719] = a89 * b_ptr[719];
        a90 = a[90]; prods[720] = a90 * b_ptr[720]; prods[721] = a90 * b_ptr[721]; prods[722] = a90 * b_ptr[722]; prods[723] = a90 * b_ptr[723]; prods[724] = a90 * b_ptr[724]; prods[725] = a90 * b_ptr[725]; prods[726] = a90 * b_ptr[726]; prods[727] = a90 * b_ptr[727];
        a91 = a[91]; prods[728] = a91 * b_ptr[728]; prods[729] = a91 * b_ptr[729]; prods[730] = a91 * b_ptr[730]; prods[731] = a91 * b_ptr[731]; prods[732] = a91 * b_ptr[732]; prods[733] = a91 * b_ptr[733]; prods[734] = a91 * b_ptr[734]; prods[735] = a91 * b_ptr[735];
        a92 = a[92]; prods[736] = a92 * b_ptr[736]; prods[737] = a92 * b_ptr[737]; prods[738] = a92 * b_ptr[738]; prods[739] = a92 * b_ptr[739]; prods[740] = a92 * b_ptr[740]; prods[741] = a92 * b_ptr[741]; prods[742] = a92 * b_ptr[742]; prods[743] = a92 * b_ptr[743];
        a93 = a[93]; prods[744] = a93 * b_ptr[744]; prods[745] = a93 * b_ptr[745]; prods[746] = a93 * b_ptr[746]; prods[747] = a93 * b_ptr[747]; prods[748] = a93 * b_ptr[748]; prods[749] = a93 * b_ptr[749]; prods[750] = a93 * b_ptr[750]; prods[751] = a93 * b_ptr[751];
        a94 = a[94]; prods[752] = a94 * b_ptr[752]; prods[753] = a94 * b_ptr[753]; prods[754] = a94 * b_ptr[754]; prods[755] = a94 * b_ptr[755]; prods[756] = a94 * b_ptr[756]; prods[757] = a94 * b_ptr[757]; prods[758] = a94 * b_ptr[758]; prods[759] = a94 * b_ptr[759];
        a95 = a[95]; prods[760] = a95 * b_ptr[760]; prods[761] = a95 * b_ptr[761]; prods[762] = a95 * b_ptr[762]; prods[763] = a95 * b_ptr[763]; prods[764] = a95 * b_ptr[764]; prods[765] = a95 * b_ptr[765]; prods[766] = a95 * b_ptr[766]; prods[767] = a95 * b_ptr[767];
        a96 = a[96]; prods[768] = a96 * b_ptr[768]; prods[769] = a96 * b_ptr[769]; prods[770] = a96 * b_ptr[770]; prods[771] = a96 * b_ptr[771]; prods[772] = a96 * b_ptr[772]; prods[773] = a96 * b_ptr[773]; prods[774] = a96 * b_ptr[774]; prods[775] = a96 * b_ptr[775];
        a97 = a[97]; prods[776] = a97 * b_ptr[776]; prods[777] = a97 * b_ptr[777]; prods[778] = a97 * b_ptr[778]; prods[779] = a97 * b_ptr[779]; prods[780] = a97 * b_ptr[780]; prods[781] = a97 * b_ptr[781]; prods[782] = a97 * b_ptr[782]; prods[783] = a97 * b_ptr[783];
        a98 = a[98]; prods[784] = a98 * b_ptr[784]; prods[785] = a98 * b_ptr[785]; prods[786] = a98 * b_ptr[786]; prods[787] = a98 * b_ptr[787]; prods[788] = a98 * b_ptr[788]; prods[789] = a98 * b_ptr[789]; prods[790] = a98 * b_ptr[790]; prods[791] = a98 * b_ptr[791];
        a99 = a[99]; prods[792] = a99 * b_ptr[792]; prods[793] = a99 * b_ptr[793]; prods[794] = a99 * b_ptr[794]; prods[795] = a99 * b_ptr[795]; prods[796] = a99 * b_ptr[796]; prods[797] = a99 * b_ptr[797]; prods[798] = a99 * b_ptr[798]; prods[799] = a99 * b_ptr[799];
        a100 = a[100]; prods[800] = a100 * b_ptr[800]; prods[801] = a100 * b_ptr[801]; prods[802] = a100 * b_ptr[802]; prods[803] = a100 * b_ptr[803]; prods[804] = a100 * b_ptr[804]; prods[805] = a100 * b_ptr[805]; prods[806] = a100 * b_ptr[806]; prods[807] = a100 * b_ptr[807];
        a101 = a[101]; prods[808] = a101 * b_ptr[808]; prods[809] = a101 * b_ptr[809]; prods[810] = a101 * b_ptr[810]; prods[811] = a101 * b_ptr[811]; prods[812] = a101 * b_ptr[812]; prods[813] = a101 * b_ptr[813]; prods[814] = a101 * b_ptr[814]; prods[815] = a101 * b_ptr[815];
        a102 = a[102]; prods[816] = a102 * b_ptr[816]; prods[817] = a102 * b_ptr[817]; prods[818] = a102 * b_ptr[818]; prods[819] = a102 * b_ptr[819]; prods[820] = a102 * b_ptr[820]; prods[821] = a102 * b_ptr[821]; prods[822] = a102 * b_ptr[822]; prods[823] = a102 * b_ptr[823];
        a103 = a[103]; prods[824] = a103 * b_ptr[824]; prods[825] = a103 * b_ptr[825]; prods[826] = a103 * b_ptr[826]; prods[827] = a103 * b_ptr[827]; prods[828] = a103 * b_ptr[828]; prods[829] = a103 * b_ptr[829]; prods[830] = a103 * b_ptr[830]; prods[831] = a103 * b_ptr[831];
        a104 = a[104]; prods[832] = a104 * b_ptr[832]; prods[833] = a104 * b_ptr[833]; prods[834] = a104 * b_ptr[834]; prods[835] = a104 * b_ptr[835]; prods[836] = a104 * b_ptr[836]; prods[837] = a104 * b_ptr[837]; prods[838] = a104 * b_ptr[838]; prods[839] = a104 * b_ptr[839];
        a105 = a[105]; prods[840] = a105 * b_ptr[840]; prods[841] = a105 * b_ptr[841]; prods[842] = a105 * b_ptr[842]; prods[843] = a105 * b_ptr[843]; prods[844] = a105 * b_ptr[844]; prods[845] = a105 * b_ptr[845]; prods[846] = a105 * b_ptr[846]; prods[847] = a105 * b_ptr[847];
        a106 = a[106]; prods[848] = a106 * b_ptr[848]; prods[849] = a106 * b_ptr[849]; prods[850] = a106 * b_ptr[850]; prods[851] = a106 * b_ptr[851]; prods[852] = a106 * b_ptr[852]; prods[853] = a106 * b_ptr[853]; prods[854] = a106 * b_ptr[854]; prods[855] = a106 * b_ptr[855];
        a107 = a[107]; prods[856] = a107 * b_ptr[856]; prods[857] = a107 * b_ptr[857]; prods[858] = a107 * b_ptr[858]; prods[859] = a107 * b_ptr[859]; prods[860] = a107 * b_ptr[860]; prods[861] = a107 * b_ptr[861]; prods[862] = a107 * b_ptr[862]; prods[863] = a107 * b_ptr[863];
        a108 = a[108]; prods[864] = a108 * b_ptr[864]; prods[865] = a108 * b_ptr[865]; prods[866] = a108 * b_ptr[866]; prods[867] = a108 * b_ptr[867]; prods[868] = a108 * b_ptr[868]; prods[869] = a108 * b_ptr[869]; prods[870] = a108 * b_ptr[870]; prods[871] = a108 * b_ptr[871];
        a109 = a[109]; prods[872] = a109 * b_ptr[872]; prods[873] = a109 * b_ptr[873]; prods[874] = a109 * b_ptr[874]; prods[875] = a109 * b_ptr[875]; prods[876] = a109 * b_ptr[876]; prods[877] = a109 * b_ptr[877]; prods[878] = a109 * b_ptr[878]; prods[879] = a109 * b_ptr[879];
        a110 = a[110]; prods[880] = a110 * b_ptr[880]; prods[881] = a110 * b_ptr[881]; prods[882] = a110 * b_ptr[882]; prods[883] = a110 * b_ptr[883]; prods[884] = a110 * b_ptr[884]; prods[885] = a110 * b_ptr[885]; prods[886] = a110 * b_ptr[886]; prods[887] = a110 * b_ptr[887];
        a111 = a[111]; prods[888] = a111 * b_ptr[888]; prods[889] = a111 * b_ptr[889]; prods[890] = a111 * b_ptr[890]; prods[891] = a111 * b_ptr[891]; prods[892] = a111 * b_ptr[892]; prods[893] = a111 * b_ptr[893]; prods[894] = a111 * b_ptr[894]; prods[895] = a111 * b_ptr[895];
        a112 = a[112]; prods[896] = a112 * b_ptr[896]; prods[897] = a112 * b_ptr[897]; prods[898] = a112 * b_ptr[898]; prods[899] = a112 * b_ptr[899]; prods[900] = a112 * b_ptr[900]; prods[901] = a112 * b_ptr[901]; prods[902] = a112 * b_ptr[902]; prods[903] = a112 * b_ptr[903];
        a113 = a[113]; prods[904] = a113 * b_ptr[904]; prods[905] = a113 * b_ptr[905]; prods[906] = a113 * b_ptr[906]; prods[907] = a113 * b_ptr[907]; prods[908] = a113 * b_ptr[908]; prods[909] = a113 * b_ptr[909]; prods[910] = a113 * b_ptr[910]; prods[911] = a113 * b_ptr[911];
        a114 = a[114]; prods[912] = a114 * b_ptr[912]; prods[913] = a114 * b_ptr[913]; prods[914] = a114 * b_ptr[914]; prods[915] = a114 * b_ptr[915]; prods[916] = a114 * b_ptr[916]; prods[917] = a114 * b_ptr[917]; prods[918] = a114 * b_ptr[918]; prods[919] = a114 * b_ptr[919];
        a115 = a[115]; prods[920] = a115 * b_ptr[920]; prods[921] = a115 * b_ptr[921]; prods[922] = a115 * b_ptr[922]; prods[923] = a115 * b_ptr[923]; prods[924] = a115 * b_ptr[924]; prods[925] = a115 * b_ptr[925]; prods[926] = a115 * b_ptr[926]; prods[927] = a115 * b_ptr[927];
        a116 = a[116]; prods[928] = a116 * b_ptr[928]; prods[929] = a116 * b_ptr[929]; prods[930] = a116 * b_ptr[930]; prods[931] = a116 * b_ptr[931]; prods[932] = a116 * b_ptr[932]; prods[933] = a116 * b_ptr[933]; prods[934] = a116 * b_ptr[934]; prods[935] = a116 * b_ptr[935];
        a117 = a[117]; prods[936] = a117 * b_ptr[936]; prods[937] = a117 * b_ptr[937]; prods[938] = a117 * b_ptr[938]; prods[939] = a117 * b_ptr[939]; prods[940] = a117 * b_ptr[940]; prods[941] = a117 * b_ptr[941]; prods[942] = a117 * b_ptr[942]; prods[943] = a117 * b_ptr[943];
        a118 = a[118]; prods[944] = a118 * b_ptr[944]; prods[945] = a118 * b_ptr[945]; prods[946] = a118 * b_ptr[946]; prods[947] = a118 * b_ptr[947]; prods[948] = a118 * b_ptr[948]; prods[949] = a118 * b_ptr[949]; prods[950] = a118 * b_ptr[950]; prods[951] = a118 * b_ptr[951];
        a119 = a[119]; prods[952] = a119 * b_ptr[952]; prods[953] = a119 * b_ptr[953]; prods[954] = a119 * b_ptr[954]; prods[955] = a119 * b_ptr[955]; prods[956] = a119 * b_ptr[956]; prods[957] = a119 * b_ptr[957]; prods[958] = a119 * b_ptr[958]; prods[959] = a119 * b_ptr[959];
        a120 = a[120]; prods[960] = a120 * b_ptr[960]; prods[961] = a120 * b_ptr[961]; prods[962] = a120 * b_ptr[962]; prods[963] = a120 * b_ptr[963]; prods[964] = a120 * b_ptr[964]; prods[965] = a120 * b_ptr[965]; prods[966] = a120 * b_ptr[966]; prods[967] = a120 * b_ptr[967];
        a121 = a[121]; prods[968] = a121 * b_ptr[968]; prods[969] = a121 * b_ptr[969]; prods[970] = a121 * b_ptr[970]; prods[971] = a121 * b_ptr[971]; prods[972] = a121 * b_ptr[972]; prods[973] = a121 * b_ptr[973]; prods[974] = a121 * b_ptr[974]; prods[975] = a121 * b_ptr[975];
        a122 = a[122]; prods[976] = a122 * b_ptr[976]; prods[977] = a122 * b_ptr[977]; prods[978] = a122 * b_ptr[978]; prods[979] = a122 * b_ptr[979]; prods[980] = a122 * b_ptr[980]; prods[981] = a122 * b_ptr[981]; prods[982] = a122 * b_ptr[982]; prods[983] = a122 * b_ptr[983];
        a123 = a[123]; prods[984] = a123 * b_ptr[984]; prods[985] = a123 * b_ptr[985]; prods[986] = a123 * b_ptr[986]; prods[987] = a123 * b_ptr[987]; prods[988] = a123 * b_ptr[988]; prods[989] = a123 * b_ptr[989]; prods[990] = a123 * b_ptr[990]; prods[991] = a123 * b_ptr[991];
        a124 = a[124]; prods[992] = a124 * b_ptr[992]; prods[993] = a124 * b_ptr[993]; prods[994] = a124 * b_ptr[994]; prods[995] = a124 * b_ptr[995]; prods[996] = a124 * b_ptr[996]; prods[997] = a124 * b_ptr[997]; prods[998] = a124 * b_ptr[998]; prods[999] = a124 * b_ptr[999];
        a125 = a[125]; prods[1000] = a125 * b_ptr[1000]; prods[1001] = a125 * b_ptr[1001]; prods[1002] = a125 * b_ptr[1002]; prods[1003] = a125 * b_ptr[1003]; prods[1004] = a125 * b_ptr[1004]; prods[1005] = a125 * b_ptr[1005]; prods[1006] = a125 * b_ptr[1006]; prods[1007] = a125 * b_ptr[1007];
        a126 = a[126]; prods[1008] = a126 * b_ptr[1008]; prods[1009] = a126 * b_ptr[1009]; prods[1010] = a126 * b_ptr[1010]; prods[1011] = a126 * b_ptr[1011]; prods[1012] = a126 * b_ptr[1012]; prods[1013] = a126 * b_ptr[1013]; prods[1014] = a126 * b_ptr[1014]; prods[1015] = a126 * b_ptr[1015];
        a127 = a[127]; prods[1016] = a127 * b_ptr[1016]; prods[1017] = a127 * b_ptr[1017]; prods[1018] = a127 * b_ptr[1018]; prods[1019] = a127 * b_ptr[1019]; prods[1020] = a127 * b_ptr[1020]; prods[1021] = a127 * b_ptr[1021]; prods[1022] = a127 * b_ptr[1022]; prods[1023] = a127 * b_ptr[1023];

        res_ptr = res * 8;
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

    fn poly_eq(point, n, two_pow_n) -> 1 {
        // point is a pointer to n vectorized pointers, each pointing to 1 chunk of 8 field elements
        // return a vectorized pointer to 2^n extension field elements, corresponding to the "equality polynomial" at point
        // Example: for n = 2: eq(x, y) = [(1 - x)(1 - y), (1 - x)y, x(1 - y), xy]

        if n == 0 {
            // base case
            res = malloc_vec(1);
            res_ptr = res * 8;
            res_ptr[0] = 1;
            res_ptr[1] = 0; res_ptr[2] = 0; res_ptr[3] = 0; res_ptr[4] = 0; res_ptr[5] = 0; res_ptr[6] = 0; res_ptr[7] = 0;
            return res;
        }

        res = malloc_vec(two_pow_n);

        inner_res = poly_eq(point + 1, n - 1, two_pow_n / 2);

        two_pow_n_minus_1 = two_pow_n / 2;

        point_ptr = point[0] * 8;

        for i in 0..two_pow_n_minus_1 {
            inner_ptr = (inner_res + i) * 8;
            left_ptr = (res + i) * 8; 
            right_ptr = (res + two_pow_n_minus_1 + i) * 8; 
            mul_extension(point_ptr, inner_ptr, right_ptr);
            sub_extension(inner_ptr, right_ptr, left_ptr);
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

    fn mul_base_by_extension_vec(a_in_base, b_in_extension) -> 1 {
        c = malloc_vec(1);
        b_ptr = b_in_extension * 8;
        c_ptr = c * 8;
        c_ptr[0] = a_in_base * b_ptr[0]; c_ptr[1] = a_in_base * b_ptr[1]; c_ptr[2] = a_in_base * b_ptr[2]; c_ptr[3] = a_in_base * b_ptr[3]; c_ptr[4] = a_in_base * b_ptr[4]; c_ptr[5] = a_in_base * b_ptr[5]; c_ptr[6] = a_in_base * b_ptr[6]; c_ptr[7] = a_in_base * b_ptr[7];
        return c;
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

    fn sub_extension(a, b, c) {
        // a, b and c are pointers
        // c = a - b
        c[0] = a[0] - b[0];
        c[1] = a[1] - b[1];
        c[2] = a[2] - b[2];
        c[3] = a[3] - b[3];
        c[4] = a[4] - b[4];
        c[5] = a[5] - b[5];
        c[6] = a[6] - b[6];
        c[7] = a[7] - b[7];
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
