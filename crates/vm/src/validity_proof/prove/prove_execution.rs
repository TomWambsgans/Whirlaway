use crate::dot_product_air::DOT_PRODUCT_AIR_COLUMN_GROUPS;
use crate::dot_product_air::DotProductAir;
use crate::dot_product_air::build_dot_product_columns;
use crate::prove::all_poseidon_16_indexes;
use crate::prove::all_poseidon_24_indexes;
use crate::validity_proof::common::fold_bytecode;
use crate::validity_proof::common::intitial_and_final_pc_conditions;
use crate::validity_proof::common::poseidon_16_column_groups;
use crate::validity_proof::common::poseidon_24_column_groups;
use crate::validity_proof::common::poseidon_lookup_index_statements;
use ::air::prove_many_air_2;
use ::air::{table::AirTable, witness::AirWitness};
use lookup::prove_gkr_product;
use lookup::{compute_pushforward, prove_logup_star};
use p3_air::BaseAir;
use p3_field::BasedVectorSpace;
use p3_field::PrimeCharacteristicRing;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use pcs::num_packed_vars_for_pols;
use pcs::{BatchPCS, packed_pcs_commit, packed_pcs_global_statements};
use rayon::prelude::*;
use tracing::info_span;
use utils::ToUsize;
use utils::assert_eq_many;
use utils::dot_product_with_base;
use utils::field_slice_as_base;
use utils::fold_multilinear_in_large_field;
use utils::pack_extension;
use utils::{
    Evaluation, PF, build_poseidon_16_air, build_poseidon_24_air, build_prover_state,
    padd_with_zero_to_next_power_of_two,
};
use whir_p3::dft::EvalsDft;
use whir_p3::poly::evals::{eval_eq, fold_multilinear};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};
use whir_p3::utils::compute_eval_eq;

use crate::prove::build_poseidon_columns;
use crate::validity_proof::common::poseidon_lookup_value;
use crate::{
    air::VMAir,
    bytecode::bytecode::Bytecode,
    prove::{ExecutionTrace, get_execution_trace},
    runner::execute_bytecode,
    *,
};

pub fn prove_execution(
    bytecode: &Bytecode,
    public_input: &[F],
    private_input: &[F],
    pcs: &impl BatchPCS<PF<EF>, EF, EF>,
) -> Vec<PF<EF>> {
    let ExecutionTrace {
        full_trace,
        n_poseidons_16,
        n_poseidons_24,
        poseidons_16, // padded with empty poseidons
        poseidons_24, // padded with empty poseidons
        dot_products,
        vm_multilinear_evals,
        public_memory_size,
        memory,
    } = info_span!("Witness generation").in_scope(|| {
        let execution_result = execute_bytecode(&bytecode, &public_input, private_input);
        get_execution_trace(&bytecode, &execution_result)
    });

    let public_memory = &memory[..public_memory_size];
    let private_memory = &memory[public_memory_size..];
    let log_memory = log2_ceil_usize(memory.len());
    let log_public_memory = log2_strict_usize(public_memory.len());

    let n_cycles = full_trace[0].len();
    let log_n_cycles = log2_strict_usize(n_cycles);
    assert!(full_trace.iter().all(|col| col.len() == 1 << log_n_cycles));

    let dft = EvalsDft::default();

    let mut exec_columns = full_trace[..N_INSTRUCTION_COLUMNS_IN_AIR]
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    exec_columns.extend(
        full_trace[N_INSTRUCTION_COLUMNS..]
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>(),
    );
    let exec_witness = AirWitness::<PF<EF>>::new(&exec_columns, &COLUMN_GROUPS_EXEC);
    let exec_table = AirTable::<EF, _>::new(VMAir);

    #[cfg(test)]
    exec_table.check_trace_validity(&exec_witness).unwrap();

    let _validity_proof_span = info_span!("Validity proof generation").entered();

    let p16_air = build_poseidon_16_air();
    let p24_air = build_poseidon_24_air();
    let p16_table = AirTable::<EF, _>::new(p16_air.clone());
    let p24_table = AirTable::<EF, _>::new(p24_air.clone());

    let dot_product_table = AirTable::<EF, _>::new(DotProductAir);

    let (p16_columns, p24_columns) = build_poseidon_columns(&poseidons_16, &poseidons_24);
    let p16_witness = AirWitness::new(&p16_columns, &poseidon_16_column_groups(&p16_air));
    let p24_witness = AirWitness::new(&p24_columns, &poseidon_24_column_groups(&p24_air));

    let (dot_product_columns, dot_product_padding_len) = build_dot_product_columns(&dot_products);
    let dot_product_witness = AirWitness::new(&dot_product_columns, &DOT_PRODUCT_AIR_COLUMN_GROUPS);
    #[cfg(test)]
    dot_product_table
        .check_trace_validity(&dot_product_witness)
        .unwrap();

    let p16_commited =
        padd_with_zero_to_next_power_of_two(&p16_columns[16..p16_air.width() - 16].concat());
    let p24_commited =
        padd_with_zero_to_next_power_of_two(&p24_columns[24..p24_air.width() - 24].concat());

    let dot_product_flags: Vec<PF<EF>> = field_slice_as_base(&dot_product_columns[0]).unwrap();
    let dot_product_lengths: Vec<PF<EF>> = field_slice_as_base(&dot_product_columns[1]).unwrap();
    let dot_product_indexes: Vec<PF<EF>> = padd_with_zero_to_next_power_of_two(
        &field_slice_as_base(
            &[
                dot_product_columns[2].clone(),
                dot_product_columns[3].clone(),
                dot_product_columns[4].clone(),
            ]
            .concat(),
        )
        .unwrap(),
    );
    let dot_product_computations: &[EF] = &dot_product_columns[8];
    let dot_product_computations_base = dot_product_computations
        .par_iter()
        .flat_map(|ef| <EF as BasedVectorSpace<PF<EF>>>::as_basis_coefficients_slice(ef).to_vec())
        .collect::<Vec<_>>();

    let exec_memory_addresses = padd_with_zero_to_next_power_of_two(
        &full_trace[COL_INDEX_MEM_ADDRESS_A..=COL_INDEX_MEM_ADDRESS_C].concat(),
    );

    assert!(private_memory.len() % public_memory.len() == 0);
    let n_private_memory_chunks = private_memory.len() / public_memory.len();
    let private_memory_commited_chunks = (0..n_private_memory_chunks)
        .map(|i| &private_memory[i * public_memory.len()..(i + 1) * public_memory.len()])
        .collect::<Vec<_>>();

    let log_n_rows_dot_product_table = log2_strict_usize(dot_product_columns[0].len());

    let mut prover_state = build_prover_state::<EF>();
    prover_state.add_base_scalars(
        &[
            log_n_cycles,
            n_poseidons_16,
            n_poseidons_24,
            dot_products.len(),
            log_n_rows_dot_product_table,
            dot_product_padding_len,
            private_memory.len(),
        ]
        .into_iter()
        .map(F::from_usize)
        .collect::<Vec<_>>(),
    );

    // 1st Commitment
    let packed_pcs_witness_base = packed_pcs_commit(
        pcs.pcs_a(),
        &[
            vec![
                full_trace[COL_INDEX_PC].as_slice(),
                full_trace[COL_INDEX_FP].as_slice(),
                exec_memory_addresses.as_slice(),
                all_poseidon_16_indexes(&poseidons_16).as_slice(),
                all_poseidon_24_indexes(&poseidons_24).as_slice(),
                p16_commited.as_slice(),
                p24_commited.as_slice(),
                dot_product_flags.as_slice(),
                dot_product_lengths.as_slice(),
                dot_product_indexes.as_slice(),
                dot_product_computations_base.as_slice(),
            ],
            private_memory_commited_chunks.clone(),
        ]
        .concat(),
        &dft,
        &mut prover_state,
    );

    // Grand Product for consistency with precompiles
    let grand_product_challenge_global = prover_state.sample();
    let grand_product_challenge_p16 = prover_state.sample().powers().collect_n(5);
    let grand_product_challenge_p24 = prover_state.sample().powers().collect_n(5);
    let grand_product_dot_product_challenge = prover_state.sample().powers().collect_n(6);
    let mut exec_column_for_grand_product = vec![grand_product_challenge_global; n_cycles];
    for pos16 in &poseidons_16 {
        let Some(cycle) = pos16.cycle else {
            break;
        };
        exec_column_for_grand_product[cycle] = grand_product_challenge_global
            + grand_product_challenge_p16[1]
            + grand_product_challenge_p16[2] * F::from_usize(pos16.addr_input_a)
            + grand_product_challenge_p16[3] * F::from_usize(pos16.addr_input_b)
            + grand_product_challenge_p16[4] * F::from_usize(pos16.addr_output);
    }
    for pos24 in &poseidons_24 {
        let Some(cycle) = pos24.cycle else {
            break;
        };
        exec_column_for_grand_product[cycle] = grand_product_challenge_global
            + grand_product_challenge_p24[1]
            + grand_product_challenge_p24[2] * F::from_usize(pos24.addr_input_a)
            + grand_product_challenge_p24[3] * F::from_usize(pos24.addr_input_b)
            + grand_product_challenge_p24[4] * F::from_usize(pos24.addr_output);
    }
    for dot_product in &dot_products {
        exec_column_for_grand_product[dot_product.cycle] = grand_product_challenge_global
            + grand_product_dot_product_challenge[1]
            + grand_product_dot_product_challenge[2] * F::from_usize(dot_product.addr_0)
            + grand_product_dot_product_challenge[3] * F::from_usize(dot_product.addr_1)
            + grand_product_dot_product_challenge[4] * F::from_usize(dot_product.addr_res)
            + grand_product_dot_product_challenge[5] * F::from_usize(dot_product.len);
    }

    let (grand_product_exec_res, grand_product_exec_statement) = prove_gkr_product(
        &mut prover_state,
        pack_extension(&exec_column_for_grand_product),
    );

    let p16_column_for_grand_product = poseidons_16
        .par_iter()
        .map(|pos_16| {
            grand_product_challenge_global
                + grand_product_challenge_p16[1]
                + grand_product_challenge_p16[2] * F::from_usize(pos_16.addr_input_a)
                + grand_product_challenge_p16[3] * F::from_usize(pos_16.addr_input_b)
                + grand_product_challenge_p16[4] * F::from_usize(pos_16.addr_output)
        })
        .collect::<Vec<_>>();

    let (grand_product_p16_res, grand_product_p16_statement) = prove_gkr_product(
        &mut prover_state,
        pack_extension(&p16_column_for_grand_product),
    );

    let p24_column_for_grand_product = poseidons_24
        .par_iter()
        .map(|pos_24| {
            grand_product_challenge_global
                + grand_product_challenge_p24[1]
                + grand_product_challenge_p24[2] * F::from_usize(pos_24.addr_input_a)
                + grand_product_challenge_p24[3] * F::from_usize(pos_24.addr_input_b)
                + grand_product_challenge_p24[4] * F::from_usize(pos_24.addr_output)
        })
        .collect::<Vec<_>>();

    let (grand_product_p24_res, grand_product_p24_statement) = prove_gkr_product(
        &mut prover_state,
        pack_extension(&p24_column_for_grand_product),
    );

    let dot_product_column_for_grand_product = (0..1 << log_n_rows_dot_product_table)
        .into_par_iter()
        .map(|i| {
            grand_product_challenge_global
                + grand_product_dot_product_challenge[1]
                + (grand_product_dot_product_challenge[2] * dot_product_columns[2][i]
                    + grand_product_dot_product_challenge[3] * dot_product_columns[3][i]
                    + grand_product_dot_product_challenge[4] * dot_product_columns[4][i]
                    + grand_product_dot_product_challenge[5] * dot_product_columns[1][i])
                    * dot_product_columns[0][i]
        })
        .collect::<Vec<_>>();

    let (grand_product_dot_product_res, grand_product_dot_product_statement) = prove_gkr_product(
        &mut prover_state,
        pack_extension(&dot_product_column_for_grand_product),
    );

    let corrected_prod_exec = grand_product_exec_res
        / grand_product_challenge_global
            .exp_u64((n_cycles - n_poseidons_16 - n_poseidons_24 - dot_products.len()) as u64);
    let corrected_prod_p16 = grand_product_p16_res
        / (grand_product_challenge_global
            + grand_product_challenge_p16[1]
            + grand_product_challenge_p16[4] * F::from_usize(POSEIDON_16_NULL_HASH_PTR))
        .exp_u64((n_poseidons_16.next_power_of_two() - n_poseidons_16) as u64);

    let corrected_prod_p24 = grand_product_p24_res
        / (grand_product_challenge_global
            + grand_product_challenge_p24[1]
            + grand_product_challenge_p24[4] * F::from_usize(POSEIDON_24_NULL_HASH_PTR))
        .exp_u64((n_poseidons_24.next_power_of_two() - n_poseidons_24) as u64);

    let corrected_dot_product = grand_product_dot_product_res
        / ((grand_product_challenge_global
            + grand_product_dot_product_challenge[1]
            + grand_product_dot_product_challenge[5])
            .exp_u64(dot_product_padding_len as u64)
            * (grand_product_challenge_global + grand_product_dot_product_challenge[1]).exp_u64(
                ((1 << log_n_rows_dot_product_table) - dot_product_padding_len - dot_products.len())
                    as u64,
            ));

    // Grand product statements
    let grand_product_fp_eval =
        full_trace[COL_INDEX_FP].evaluate(&grand_product_exec_statement.point);
    prover_state.add_extension_scalar(grand_product_fp_eval);
    let grand_product_fp_statement = Evaluation {
        point: grand_product_exec_statement.point.clone(),
        value: grand_product_fp_eval,
    };

    let grand_product_mem_value_a_eval =
        full_trace[COL_INDEX_MEM_VALUE_A].evaluate(&grand_product_exec_statement.point);
    let grand_product_mem_value_b_eval =
        full_trace[COL_INDEX_MEM_VALUE_B].evaluate(&grand_product_exec_statement.point);
    let grand_product_mem_value_c_eval =
        full_trace[COL_INDEX_MEM_VALUE_C].evaluate(&grand_product_exec_statement.point);
    prover_state.add_extension_scalars(&[
        grand_product_mem_value_a_eval,
        grand_product_mem_value_b_eval,
        grand_product_mem_value_c_eval,
    ]);
    let grand_product_mem_values_mixing_challenges = MultilinearPoint(prover_state.sample_vec(2));
    let grand_product_mem_values_statement = Evaluation {
        point: MultilinearPoint(
            [
                grand_product_mem_values_mixing_challenges.0.clone(),
                grand_product_exec_statement.point.0.clone(),
            ]
            .concat(),
        ),
        value: [
            grand_product_mem_value_a_eval,
            grand_product_mem_value_b_eval,
            grand_product_mem_value_c_eval,
            EF::ZERO,
        ]
        .evaluate(&grand_product_mem_values_mixing_challenges),
    };

    assert_eq!(
        corrected_prod_exec,
        corrected_prod_p16 * corrected_prod_p24 * corrected_dot_product
    );

    let exec_evals_to_prove =
        exec_table.prove_base(&mut prover_state, UNIVARIATE_SKIPS, exec_witness);

    let poseidon_evals_to_prove = prove_many_air_2(
        &mut prover_state,
        UNIVARIATE_SKIPS,
        &[&p16_table],
        &[&p24_table],
        &[p16_witness],
        &[p24_witness],
    );
    let p16_evals_to_prove = &poseidon_evals_to_prove[0];
    let p24_evals_to_prove = &poseidon_evals_to_prove[1];

    let dot_product_evals_to_prove =
        dot_product_table.prove_extension(&mut prover_state, 1, dot_product_witness);

    // Main memory lookup
    let exec_memory_indexes = padd_with_zero_to_next_power_of_two(
        &full_trace[COL_INDEX_MEM_ADDRESS_A..=COL_INDEX_MEM_ADDRESS_C].concat(),
    );
    let mut memory_poly_eq_point = eval_eq(&exec_evals_to_prove[1].point);
    let memory_poly_eq_point_alpha = prover_state.sample();
    compute_eval_eq::<PF<EF>, EF, true>(
        &grand_product_mem_values_statement.point,
        &mut memory_poly_eq_point,
        memory_poly_eq_point_alpha,
    );
    let padded_memory = padd_with_zero_to_next_power_of_two(&memory); // TODO avoid this padding
    let exec_pushforward = compute_pushforward(
        &exec_memory_indexes,
        padded_memory.len(),
        &memory_poly_eq_point,
    );

    let max_n_poseidons = poseidons_16.len().max(poseidons_24.len());
    let min_n_poseidons = poseidons_16.len().min(poseidons_24.len());
    let (
        all_poseidon_indexes,
        folded_memory,
        poseidon_pushforward,
        poseidon_lookup_challenge,
        poseidon_poly_eq_point,
        memory_folding_challenges,
    ) = {
        // Poseidons 16/24 memory addresses lookup

        assert_eq_many!(
            &p16_evals_to_prove[0].point,
            &p16_evals_to_prove[1].point,
            &p16_evals_to_prove[3].point,
            &p16_evals_to_prove[4].point,
        );
        assert_eq_many!(
            &p24_evals_to_prove[0].point,
            &p24_evals_to_prove[1].point,
            &p24_evals_to_prove[2].point,
            &p24_evals_to_prove[5].point,
        );
        assert_eq!(
            &p16_evals_to_prove[0].point[..3 + log2_ceil_usize(min_n_poseidons)],
            &p24_evals_to_prove[0].point[..3 + log2_ceil_usize(min_n_poseidons)]
        );

        let memory_folding_challenges = MultilinearPoint(p16_evals_to_prove[0].point[..3].to_vec());
        let poseidon_lookup_batching_chalenges = MultilinearPoint(prover_state.sample_vec(3));
        let mut poseidon_lookup_point = poseidon_lookup_batching_chalenges.0.clone();
        poseidon_lookup_point.extend_from_slice({
            if poseidons_16.len() > poseidons_24.len() {
                &p16_evals_to_prove[0].point[3..]
            } else {
                &p24_evals_to_prove[0].point[3..]
            }
        });
        let poseidon_lookup_value = poseidon_lookup_value(
            n_poseidons_16,
            n_poseidons_24,
            p16_evals_to_prove,
            p24_evals_to_prove,
            &poseidon_lookup_batching_chalenges,
        );
        let poseidon_lookup_challenge = Evaluation {
            point: MultilinearPoint(poseidon_lookup_point),
            value: poseidon_lookup_value,
        };

        let poseidon16_steps =
            1 << (log2_strict_usize(max_n_poseidons) - log2_strict_usize(poseidons_16.len()));
        let poseidon24_steps =
            1 << (log2_strict_usize(max_n_poseidons) - log2_strict_usize(poseidons_24.len()));
        let mut all_poseidon_indexes = F::zero_vec(8 * max_n_poseidons);

        #[rustfmt::skip]
        let chunks = [
            (poseidons_16.par_iter().map(|p| p.addr_input_a).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| p.addr_input_b).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| p.addr_output).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| p.addr_output + 1).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_24.par_iter().map(|p| p.addr_input_a).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| p.addr_input_a + 1).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| p.addr_input_b).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| p.addr_output).collect::<Vec<_>>(), poseidon24_steps),
        ];
        for (chunk_idx, (addrs, step)) in chunks.into_iter().enumerate() {
            let offset = chunk_idx * max_n_poseidons;
            all_poseidon_indexes[offset..]
                .par_iter_mut()
                .step_by(step)
                .zip(addrs)
                .for_each(|(slot, addr)| {
                    *slot = F::from_usize(addr);
                });
        }

        let mut all_poseidon_values = EF::zero_vec(8 * max_n_poseidons);
        #[rustfmt::skip]
        let chunks = [
            (poseidons_16.par_iter().map(|p| (&p.input[0..8]).evaluate(&memory_folding_challenges)).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| (&p.input[8..16]).evaluate(&memory_folding_challenges)).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| (&p.output[0..8]).evaluate(&memory_folding_challenges)).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| (&p.output[8..16]).evaluate(&memory_folding_challenges)).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_24.par_iter().map(|p| (&p.input[0..8]).evaluate(&memory_folding_challenges)).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| (&p.input[8..16]).evaluate(&memory_folding_challenges)).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| (&p.input[16..24]).evaluate(&memory_folding_challenges)).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| (&p.output).evaluate(&memory_folding_challenges)).collect::<Vec<_>>(), poseidon24_steps),
        ];
        for (chunk_idx, (values, step)) in chunks.into_iter().enumerate() {
            let offset = chunk_idx * max_n_poseidons;
            all_poseidon_values[offset..]
                .par_iter_mut()
                .step_by(step)
                .zip(values)
                .for_each(|(slot, value)| {
                    *slot = value;
                });
        }

        let poseidon_folded_memory = fold_multilinear(&padded_memory, &memory_folding_challenges);

        assert_eq!(all_poseidon_indexes.len(), all_poseidon_values.len(),);

        // TODO remove these checks
        {
            for (index, value) in all_poseidon_indexes.iter().zip(&all_poseidon_values) {
                assert_eq!(value, &poseidon_folded_memory[index.to_usize()]);
            }
            assert_eq!(
                all_poseidon_values.evaluate(&poseidon_lookup_challenge.point),
                poseidon_lookup_challenge.value
            );
        }

        let poseidon_poly_eq_point = eval_eq(&poseidon_lookup_challenge.point);

        let poseidon_pushforward = compute_pushforward(
            &all_poseidon_indexes,
            poseidon_folded_memory.len(),
            &poseidon_poly_eq_point,
        );

        (
            all_poseidon_indexes,
            poseidon_folded_memory,
            poseidon_pushforward,
            poseidon_lookup_challenge,
            poseidon_poly_eq_point,
            memory_folding_challenges,
        )
    };

    // As long as we dont have the gkr grand product, for consistency accross all the tables, that will require opening
    // on the instruction precompiles columns (btw we should as always do the sumchecks "in parallel" to get common challenges),
    // we perform the bytecode lookup only on the first N_INSTRUCTION_COLUMNS_IN_AIR.
    // But it will on the full N_INSTRUCTION_COLUMNS eventually.

    let bytecode_compression_challenges =
        MultilinearPoint(exec_evals_to_prove[0].point[..LOG_N_INSTRUCTION_COLUMNS_IN_AIR].to_vec());

    let compressed_exec_instructions = fold_multilinear_in_large_field(
        &padd_with_zero_to_next_power_of_two(&full_trace[..N_INSTRUCTION_COLUMNS_IN_AIR].concat()),
        &eval_eq(&bytecode_compression_challenges.0),
    );
    let folded_bytecode = fold_bytecode(bytecode, &bytecode_compression_challenges);
    let pc_column = &full_trace[COL_INDEX_PC];
    // TODO remove this sanity check
    for (i, pc) in pc_column.iter().enumerate() {
        assert_eq!(
            folded_bytecode[pc.to_usize()],
            compressed_exec_instructions[i]
        );
    }
    let bytecode_lookup_point =
        MultilinearPoint(exec_evals_to_prove[0].point[LOG_N_INSTRUCTION_COLUMNS_IN_AIR..].to_vec());
    let bytecode_lookup_claim = Evaluation {
        point: bytecode_lookup_point.clone(),
        value: exec_evals_to_prove[0].value,
    };
    let bytecode_poly_eq_point = eval_eq(&bytecode_lookup_point);
    let bytecode_pushforward =
        compute_pushforward(&pc_column, folded_bytecode.len(), &bytecode_poly_eq_point);

    let dot_product_poly_eq_point = eval_eq(&dot_product_evals_to_prove[3].point.clone());
    let dot_product_pushforward = compute_pushforward(
        &dot_product_indexes,
        padded_memory.len() / DIMENSION,
        &dot_product_poly_eq_point,
    );

    // 2nd Commitment
    let commited_extension = [
        exec_pushforward.as_slice(),
        poseidon_pushforward.as_slice(),
        bytecode_pushforward.as_slice(),
        dot_product_pushforward.as_slice(),
    ];
    let packed_pcs_witness_extension = packed_pcs_commit(
        &pcs.pcs_b(
            log2_strict_usize(packed_pcs_witness_base.packed_polynomial.len()),
            num_packed_vars_for_pols(&commited_extension),
        ),
        &commited_extension,
        &dft,
        &mut prover_state,
    );

    let exec_logup_star_statements = prove_logup_star(
        &mut prover_state,
        &padded_memory,
        &exec_memory_indexes,
        exec_evals_to_prove[1].value
            + memory_poly_eq_point_alpha * grand_product_mem_values_statement.value,
        &memory_poly_eq_point,
        &exec_pushforward,
    );

    let poseidon_logup_star_statements = prove_logup_star(
        &mut prover_state,
        &folded_memory,
        &all_poseidon_indexes,
        poseidon_lookup_challenge.value,
        &poseidon_poly_eq_point,
        &poseidon_pushforward,
    );

    let bytecode_logup_star_statements = prove_logup_star(
        &mut prover_state,
        &folded_bytecode,
        &pc_column,
        bytecode_lookup_claim.value,
        &bytecode_poly_eq_point,
        &bytecode_pushforward,
    );

    let memory_in_extension_field = padded_memory
        .par_chunks_exact(DIMENSION)
        .map(|chunk| EF::from_basis_coefficients_slice(chunk).unwrap())
        .collect::<Vec<_>>();
    let dot_product_logup_star_statements = prove_logup_star(
        &mut prover_state,
        &memory_in_extension_field,
        &dot_product_indexes,
        dot_product_evals_to_prove[3].value,
        &dot_product_poly_eq_point,
        &dot_product_pushforward,
    );

    let poseidon_lookup_memory_point = MultilinearPoint(
        [
            poseidon_logup_star_statements.on_table.point.0.clone(),
            memory_folding_challenges.0.clone(),
        ]
        .concat(),
    );

    let dot_product_folded_memory_evals = fold_multilinear_in_large_field(
        &padded_memory,
        &eval_eq(&dot_product_logup_star_statements.on_table.point),
    );
    assert_eq!(
        dot_product_with_base(&dot_product_folded_memory_evals),
        dot_product_logup_star_statements.on_table.value
    );
    prover_state.add_extension_scalars(&dot_product_folded_memory_evals);
    let dot_product_memory_mixing_challenges = prover_state.sample_vec(3);
    let dot_product_memory_challenge = Evaluation {
        point: MultilinearPoint(
            [
                dot_product_logup_star_statements.on_table.point.0.clone(),
                dot_product_memory_mixing_challenges.clone(),
            ]
            .concat(),
        ),
        value: dot_product_folded_memory_evals
            .evaluate(&MultilinearPoint(dot_product_memory_mixing_challenges)),
    };

    // open memory
    let exec_lookup_chunk_point = MultilinearPoint(
        exec_logup_star_statements.on_table.point[log_memory - log_public_memory..].to_vec(),
    );
    let poseidon_lookup_chunk_point =
        MultilinearPoint(poseidon_lookup_memory_point[log_memory - log_public_memory..].to_vec());
    let dot_product_lookup_chunk_point = MultilinearPoint(
        dot_product_memory_challenge.point.0[log_memory - log_public_memory..].to_vec(),
    );
    let mut private_memory_statements = vec![];
    for private_memory_chunk in &private_memory_commited_chunks {
        let chunk_eval_exec_lookup = private_memory_chunk.evaluate(&exec_lookup_chunk_point);
        let chunk_eval_poseidon_lookup =
            private_memory_chunk.evaluate(&poseidon_lookup_chunk_point);
        let chunk_eval_dot_product_lookup =
            private_memory_chunk.evaluate(&dot_product_lookup_chunk_point);
        prover_state.add_extension_scalar(chunk_eval_exec_lookup);
        prover_state.add_extension_scalar(chunk_eval_poseidon_lookup);
        prover_state.add_extension_scalar(chunk_eval_dot_product_lookup);
        private_memory_statements.push(vec![
            Evaluation {
                point: exec_lookup_chunk_point.clone(),
                value: chunk_eval_exec_lookup,
            },
            Evaluation {
                point: poseidon_lookup_chunk_point.clone(),
                value: chunk_eval_poseidon_lookup,
            },
            Evaluation {
                point: dot_product_lookup_chunk_point.clone(),
                value: chunk_eval_dot_product_lookup,
            },
        ]);
    }

    // index opening for poseidon lookup
    let poseidon_index_evals = fold_multilinear(
        &all_poseidon_indexes,
        &MultilinearPoint(poseidon_logup_star_statements.on_indexes.point[3..].to_vec()),
    );
    prover_state.add_extension_scalars(&poseidon_index_evals);

    let (p16_indexes_statements, p24_indexes_statements) = poseidon_lookup_index_statements(
        &poseidon_index_evals,
        n_poseidons_16,
        n_poseidons_24,
        &poseidon_logup_star_statements.on_indexes.point,
    )
    .unwrap();

    let (initial_pc_statement, final_pc_statement) =
        intitial_and_final_pc_conditions(bytecode, log_n_cycles);

    let dot_product_computation_column_evals = fold_multilinear_in_large_field(
        &dot_product_computations_base,
        &eval_eq(&dot_product_evals_to_prove[4].point),
    );
    assert_eq!(
        dot_product_with_base(&dot_product_computation_column_evals),
        dot_product_evals_to_prove[4].value
    );
    prover_state.add_extension_scalars(&dot_product_computation_column_evals);
    let dot_product_computation_mixing_challenges = prover_state.sample_vec(3);
    let dot_product_computation_column_challenge = Evaluation {
        point: MultilinearPoint(
            [
                dot_product_evals_to_prove[4].point.0.clone(),
                dot_product_computation_mixing_challenges.clone(),
            ]
            .concat(),
        ),
        value: dot_product_computation_column_evals
            .evaluate(&MultilinearPoint(dot_product_computation_mixing_challenges)),
    };

    // First Opening
    let global_statements_base = packed_pcs_global_statements(
        &packed_pcs_witness_base.tree,
        &[
            vec![
                vec![
                    exec_evals_to_prove[2].clone(),
                    bytecode_logup_star_statements.on_indexes.clone(),
                    initial_pc_statement,
                    final_pc_statement,
                ], // pc
                vec![exec_evals_to_prove[3].clone(), grand_product_fp_statement], // fp
                vec![
                    exec_evals_to_prove[4].clone(),
                    exec_logup_star_statements.on_indexes,
                ], // memory addresses
                p16_indexes_statements,
                p24_indexes_statements,
                vec![p16_evals_to_prove[2].clone()],
                vec![p24_evals_to_prove[3].clone()],
                vec![dot_product_evals_to_prove[0].clone()], // dot product: (start) flag
                vec![dot_product_evals_to_prove[1].clone()], // dot product: length
                vec![
                    dot_product_evals_to_prove[2].clone(),
                    dot_product_logup_star_statements.on_indexes,
                ], // dot product: indexes
                vec![dot_product_computation_column_challenge],
            ],
            private_memory_statements,
        ]
        .concat(),
    );

    // Second Opening
    let global_statements_extension = packed_pcs_global_statements(
        &packed_pcs_witness_extension.tree,
        &vec![
            exec_logup_star_statements.on_pushforward,
            poseidon_logup_star_statements.on_pushforward,
            bytecode_logup_star_statements.on_pushforward,
            dot_product_logup_star_statements.on_pushforward,
        ],
    );

    pcs.batch_open(
        &dft,
        &mut prover_state,
        &global_statements_base,
        packed_pcs_witness_base.inner_witness,
        &packed_pcs_witness_base.packed_polynomial,
        &global_statements_extension,
        packed_pcs_witness_extension.inner_witness,
        &packed_pcs_witness_extension.packed_polynomial,
    );

    prover_state.proof_data().to_vec()
}
