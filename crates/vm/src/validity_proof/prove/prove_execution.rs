use crate::prove::all_poseidon_16_indexes;
use crate::prove::all_poseidon_24_indexes;
use crate::validity_proof::common::fold_bytecode;
use crate::validity_proof::common::intitial_and_final_pc_conditions;
use crate::validity_proof::common::poseidon_16_column_groups;
use crate::validity_proof::common::poseidon_24_column_groups;
use crate::validity_proof::common::poseidon_lookup_index_statements;
use ::air::prove_many_air;
use ::air::{table::AirTable, witness::AirWitness};
use lookup::{compute_pushforward, prove_logup_star};
use p3_air::BaseAir;
use p3_field::PrimeCharacteristicRing;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use pcs::num_packed_vars_for_pols;
use pcs::{BatchPCS, packed_pcs_commit, packed_pcs_global_statements};
use rayon::prelude::*;
use tracing::info_span;
use utils::ToUsize;
use utils::assert_eq_many;
use utils::fold_multilinear_in_large_field;
use utils::{
    Evaluation, PF, build_poseidon_16_air, build_poseidon_24_air, build_prover_state,
    padd_with_zero_to_next_power_of_two,
};
use whir_p3::dft::EvalsDft;
use whir_p3::poly::evals::{eval_eq, fold_multilinear};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

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
        dot_products_ee,
        dot_products_be,
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
    let mut prover_state = build_prover_state::<EF>();
    prover_state.add_base_scalars(
        &[
            log_n_cycles,
            n_poseidons_16,
            n_poseidons_24,
            dot_products_ee.len(),
            dot_products_be.len(),
            private_memory.len(),
        ]
        .into_iter()
        .map(F::from_usize)
        .collect::<Vec<_>>(),
    );

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

    let (p16_columns, p24_columns) = build_poseidon_columns(&poseidons_16, &poseidons_24);
    let p16_witness = AirWitness::new(&p16_columns, &poseidon_16_column_groups(&p16_air));
    let p24_witness = AirWitness::new(&p24_columns, &poseidon_24_column_groups(&p24_air));

    let p16_commited =
        padd_with_zero_to_next_power_of_two(&p16_columns[16..p16_air.width() - 16].concat());
    let p24_commited =
        padd_with_zero_to_next_power_of_two(&p24_columns[24..p24_air.width() - 24].concat());

    let commited_pc_fp = [
        full_trace[COL_INDEX_PC].clone(),
        full_trace[COL_INDEX_FP].clone(),
    ]
    .concat();

    let exec_memory_addresses = padd_with_zero_to_next_power_of_two(
        &full_trace[COL_INDEX_MEM_ADDRESS_A..=COL_INDEX_MEM_ADDRESS_C].concat(),
    );

    assert!(private_memory.len() % public_memory.len() == 0);
    let n_private_memory_chunks = private_memory.len() / public_memory.len();
    let private_memory_commited_chunks = (0..n_private_memory_chunks)
        .map(|i| &private_memory[i * public_memory.len()..(i + 1) * public_memory.len()])
        .collect::<Vec<_>>();

    // 1st Commitment
    let packed_pcs_witness_base = packed_pcs_commit(
        pcs.pcs_a(),
        &[
            vec![
                commited_pc_fp.as_slice(),
                exec_memory_addresses.as_slice(),
                all_poseidon_16_indexes(&poseidons_16).as_slice(),
                all_poseidon_24_indexes(&poseidons_24).as_slice(),
                p16_commited.as_slice(),
                p24_commited.as_slice(),
            ],
            private_memory_commited_chunks.clone(),
        ]
        .concat(),
        &dft,
        &mut prover_state,
    );

    // PIOP
    let exec_evals_to_prove = exec_table.prove(&mut prover_state, UNIVARIATE_SKIPS, exec_witness);

    let poseidon_evals_to_prove = prove_many_air(
        &mut prover_state,
        UNIVARIATE_SKIPS,
        &[&p16_table],
        &[&p24_table],
        &[p16_witness],
        &[p24_witness]
    );
    let p16_evals_to_prove = &poseidon_evals_to_prove[0];
    let p24_evals_to_prove = &poseidon_evals_to_prove[1];

    // Main memory lookup
    let exec_memory_indexes = padd_with_zero_to_next_power_of_two(
        &full_trace[COL_INDEX_MEM_ADDRESS_A..=COL_INDEX_MEM_ADDRESS_C].concat(),
    );
    let memory_poly_eq_point = eval_eq(&exec_evals_to_prove[1].point);
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

        let folded_memory = fold_multilinear(&padded_memory, &memory_folding_challenges);

        assert_eq!(all_poseidon_indexes.len(), all_poseidon_values.len(),);

        // TODO remove these checks
        {
            for (index, value) in all_poseidon_indexes.iter().zip(&all_poseidon_values) {
                assert_eq!(value, &folded_memory[index.to_usize()]);
            }
            assert_eq!(
                all_poseidon_values.evaluate(&poseidon_lookup_challenge.point),
                poseidon_lookup_challenge.value
            );
        }

        let poseidon_poly_eq_point = eval_eq(&poseidon_lookup_challenge.point);

        let poseidon_pushforward = compute_pushforward(
            &all_poseidon_indexes,
            folded_memory.len(),
            &poseidon_poly_eq_point,
        );

        (
            all_poseidon_indexes,
            folded_memory,
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

    // 2nd Commitment
    let commited_extension = [
        exec_pushforward.as_slice(),
        poseidon_pushforward.as_slice(),
        bytecode_pushforward.as_slice(),
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
        &exec_evals_to_prove[1],
        &memory_poly_eq_point,
        &exec_pushforward,
    );

    let poseidon_logup_star_statements = prove_logup_star(
        &mut prover_state,
        &folded_memory,
        &all_poseidon_indexes,
        &poseidon_lookup_challenge,
        &poseidon_poly_eq_point,
        &poseidon_pushforward,
    );

    let bytecode_logup_star_statements = prove_logup_star(
        &mut prover_state,
        &folded_bytecode,
        &pc_column,
        &bytecode_lookup_claim,
        &bytecode_poly_eq_point,
        &bytecode_pushforward,
    );
    let mut bytecode_lookup_index_statement = bytecode_logup_star_statements.on_indexes.clone();
    bytecode_lookup_index_statement.point.0.insert(0, EF::ZERO); // because we commit both pc and fp together

    let poseidon_lookup_memory_point = MultilinearPoint(
        [
            poseidon_logup_star_statements.on_table.point.0.clone(),
            memory_folding_challenges.0.clone(),
        ]
        .concat(),
    );
    // open memory at point logup_star_statements.on_table.point
    let exec_lookup_chunk_point = MultilinearPoint(
        exec_logup_star_statements.on_table.point[log_memory - log_public_memory..].to_vec(),
    );
    let poseidon_lookup_chunk_point =
        MultilinearPoint(poseidon_lookup_memory_point[log_memory - log_public_memory..].to_vec());
    let mut private_memory_statements = vec![];
    for private_memory_chunk in &private_memory_commited_chunks {
        let chunk_eval_exec_lookup = private_memory_chunk.evaluate(&exec_lookup_chunk_point);
        let chunk_eval_poseidon_lookup =
            private_memory_chunk.evaluate(&poseidon_lookup_chunk_point);
        prover_state.add_extension_scalar(chunk_eval_exec_lookup);
        prover_state.add_extension_scalar(chunk_eval_poseidon_lookup);
        private_memory_statements.push(vec![
            Evaluation {
                point: exec_lookup_chunk_point.clone(),
                value: chunk_eval_exec_lookup,
            },
            Evaluation {
                point: poseidon_lookup_chunk_point.clone(),
                value: chunk_eval_poseidon_lookup,
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

    // First Opening
    let global_statements_base = packed_pcs_global_statements(
        &packed_pcs_witness_base.tree,
        &[
            vec![
                vec![
                    exec_evals_to_prove[2].clone(),
                    bytecode_lookup_index_statement,
                    initial_pc_statement,
                    final_pc_statement,
                ], // pc, fp
                vec![
                    exec_evals_to_prove[3].clone(),
                    exec_logup_star_statements.on_indexes,
                ], // memory addresses
                p16_indexes_statements,
                p24_indexes_statements,
                vec![p16_evals_to_prove[2].clone()],
                vec![p24_evals_to_prove[3].clone()],
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
