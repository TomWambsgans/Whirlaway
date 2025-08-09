use ::air::prove_many_air;
use ::air::{table::AirTable, witness::AirWitness};
use lookup::{compute_pushforward, prove_logup_star};
use p3_air::BaseAir;
use p3_field::PrimeCharacteristicRing;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use pcs::{BatchPCS, packed_pcs_commit, packed_pcs_global_statements};
use rayon::prelude::*;
use tracing::info_span;
use utils::ToUsize;
use utils::assert_eq_many;
use utils::{
    Evaluation, PF, build_poseidon_16_air, build_poseidon_24_air, build_prover_state,
    generate_trace_poseidon_16, generate_trace_poseidon_24, padd_with_zero_to_next_power_of_two,
};
use whir_p3::dft::EvalsDft;
use whir_p3::poly::evals::{eval_eq, fold_multilinear};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

use crate::validity_proof::common::poseidon_lookup_value;
use crate::{
    air::VMAir,
    bytecode::bytecode::Bytecode,
    runner::execute_bytecode,
    tracer::{ExecutionTrace, get_execution_trace},
    *,
};

pub fn prove_execution(
    bytecode: &Bytecode,
    public_input: &[F],
    private_input: &[F],
    pcs: &impl BatchPCS<PF<EF>, EF, EF>,
) -> Vec<PF<EF>> {
    let ExecutionTrace {
        main_trace,
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

    let log_n_rows = log2_strict_usize(main_trace[0].len());
    assert!(main_trace.iter().all(|col| col.len() == (1 << log_n_rows)));
    let mut prover_state = build_prover_state::<EF>();
    prover_state.add_base_scalars(
        &[
            log_n_rows,
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

    let main_witness = AirWitness::<PF<EF>>::new(&main_trace, &COLUMN_GROUPS_EXEC);
    let main_table = AirTable::<EF, _>::new(VMAir);

    #[cfg(test)]
    main_table.check_trace_validity(&main_witness).unwrap();

    let _validity_proof_span = info_span!("Validity proof generation").entered();

    let poseidon_16_air = build_poseidon_16_air();
    let poseidon_24_air = build_poseidon_24_air();
    let table_poseidon_16 = AirTable::<EF, _>::new(poseidon_16_air.clone());
    let table_poseidon_24 = AirTable::<EF, _>::new(poseidon_24_air.clone());

    let poseidon_16_data = poseidons_16.iter().map(|w| w.input).collect::<Vec<_>>();
    let poseidon_24_data = poseidons_24.iter().map(|w| w.input).collect::<Vec<_>>();
    let witness_matrix_poseidon_16 = generate_trace_poseidon_16(poseidon_16_data);
    let witness_matrix_poseidon_24 = generate_trace_poseidon_24(poseidon_24_data);

    let witness_matrix_poseidon_16_transposed = witness_matrix_poseidon_16.transpose();
    let witness_matrix_poseidon_24_transposed = witness_matrix_poseidon_24.transpose();

    assert_eq!(
        witness_matrix_poseidon_16_transposed.width,
        poseidons_16.len()
    );
    let witness_columns_poseidon_16 = (0..poseidon_16_air.width())
        .map(|col| {
            witness_matrix_poseidon_16_transposed.values[col
                * witness_matrix_poseidon_16_transposed.width
                ..(col + 1) * witness_matrix_poseidon_16_transposed.width]
                .to_vec()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        witness_matrix_poseidon_24_transposed.width,
        poseidons_24.len()
    );
    let witness_columns_poseidon_24 = (0..poseidon_24_air.width())
        .map(|col| {
            witness_matrix_poseidon_24_transposed.values[col
                * witness_matrix_poseidon_24_transposed.width
                ..(col + 1) * witness_matrix_poseidon_24_transposed.width]
                .to_vec()
        })
        .collect::<Vec<_>>();

    let witness_poseidon_16 = AirWitness::new(
        &witness_columns_poseidon_16,
        &[
            0..8,
            8..16,
            16..poseidon_16_air.width() - 16,
            poseidon_16_air.width() - 16..poseidon_16_air.width() - 8,
            poseidon_16_air.width() - 8..poseidon_16_air.width(),
        ],
    );
    let witness_poseidon_24 = AirWitness::new(
        &witness_columns_poseidon_24,
        &[
            0..8,
            8..16,
            16..24,
            24..poseidon_24_air.width() - 24,
            poseidon_24_air.width() - 24..poseidon_24_air.width() - 8, // TODO should we commit to this part ? Probably not, but careful here, we will not check evaluations for this part
            poseidon_24_air.width() - 8..poseidon_24_air.width(),
        ],
    );

    let commited_poseidon_16_table = padd_with_zero_to_next_power_of_two(
        &witness_columns_poseidon_16[16..poseidon_16_air.width() - 16].concat(),
    );
    let commited_poseidon_24_table = padd_with_zero_to_next_power_of_two(
        &witness_columns_poseidon_24[24..poseidon_24_air.width() - 24].concat(),
    );

    let all_poseidon_16_indexes = [
        padd_with_zero_to_next_power_of_two(
            &poseidons_16
                .iter()
                .map(|p| F::from_usize(p.addr_input_a))
                .collect::<Vec<_>>(),
        ),
        padd_with_zero_to_next_power_of_two(
            &poseidons_16
                .iter()
                .map(|p| F::from_usize(p.addr_input_b))
                .collect::<Vec<_>>(),
        ),
        padd_with_zero_to_next_power_of_two(
            &poseidons_16
                .iter()
                .map(|p| F::from_usize(p.addr_output))
                .collect::<Vec<_>>(),
        ),
    ]
    .concat();
    let all_poseidon_16_indexes_padded =
        padd_with_zero_to_next_power_of_two(&all_poseidon_16_indexes);

    let all_poseidon_24_indexes = [
        padd_with_zero_to_next_power_of_two(
            &poseidons_24
                .iter()
                .map(|p| F::from_usize(p.addr_input_a))
                .collect::<Vec<_>>(),
        ),
        padd_with_zero_to_next_power_of_two(
            &poseidons_24
                .iter()
                .map(|p| F::from_usize(p.addr_input_b))
                .collect::<Vec<_>>(),
        ),
        padd_with_zero_to_next_power_of_two(
            &poseidons_24
                .iter()
                .map(|p| F::from_usize(p.addr_output))
                .collect::<Vec<_>>(),
        ),
    ]
    .concat();
    let all_poseidon_24_indexes_padded =
        padd_with_zero_to_next_power_of_two(&all_poseidon_24_indexes);

    // Commit A
    let commited_pc_fp = main_trace[N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS
        ..N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS + 2]
        .concat();
    let commited_memory_addreses = padd_with_zero_to_next_power_of_two(
        &main_trace[N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS + 2
            ..N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS + N_COMMITTED_EXEC_COLUMNS]
            .concat(),
    );

    assert!(private_memory.len() % public_memory.len() == 0);
    let n_private_memory_chunks = private_memory.len() / public_memory.len();
    let private_memory_commited_chunks = (0..n_private_memory_chunks)
        .map(|i| &private_memory[i * public_memory.len()..(i + 1) * public_memory.len()])
        .collect::<Vec<_>>();

    // Commit A
    let packed_pcs_witness_base = packed_pcs_commit(
        pcs.pcs_a(),
        &[
            vec![
                commited_pc_fp.as_slice(),
                commited_memory_addreses.as_slice(),
                all_poseidon_16_indexes_padded.as_slice(),
                all_poseidon_24_indexes_padded.as_slice(),
                commited_poseidon_16_table.as_slice(),
                commited_poseidon_24_table.as_slice(),
            ],
            private_memory_commited_chunks.clone(),
        ]
        .concat(),
        &dft,
        &mut prover_state,
    );

    // PIOP
    let main_table_evals_to_prove =
        main_table.prove(&mut prover_state, UNIVARIATE_SKIPS, main_witness);

    let poseidon_evals_to_prove = prove_many_air(
        &mut prover_state,
        UNIVARIATE_SKIPS,
        &[&table_poseidon_16],
        &[&table_poseidon_24],
        &[witness_poseidon_16, witness_poseidon_24],
    );
    let poseidon16_evals_to_prove = &poseidon_evals_to_prove[0];
    let poseidon24_evals_to_prove = &poseidon_evals_to_prove[1];

    // Main memory lookup
    let exec_memory_indexes = padd_with_zero_to_next_power_of_two(
        &main_trace[COL_INDEX_MEM_ADDRESS_A..=COL_INDEX_MEM_ADDRESS_C].concat(),
    );
    let memory_poly_eq_point = eval_eq(&main_table_evals_to_prove[1].point);
    // TODO avoid this padding
    let padded_memory = padd_with_zero_to_next_power_of_two(&memory);
    let memory_pushforward = compute_pushforward(
        &exec_memory_indexes,
        padded_memory.len(),
        &memory_poly_eq_point,
    );
    let log_padded_memory = log2_strict_usize(padded_memory.len());
    let log_public_memory = log2_strict_usize(public_memory_size);

    let (
        all_poseidon_indexes,
        folded_memory,
        poseidon_pushforward,
        poseidon_lookup_challenge,
        poseidon_poly_eq_point,
    ) = {
        // Poseidons 16/24 memory addresses lookup

        let max_n_poseidons = poseidons_16.len().max(poseidons_24.len());
        let min_n_poseidons = poseidons_16.len().min(poseidons_24.len());

        assert_eq_many!(
            &poseidon16_evals_to_prove[0].point,
            &poseidon16_evals_to_prove[1].point,
            &poseidon16_evals_to_prove[3].point,
            &poseidon16_evals_to_prove[4].point,
        );
        assert_eq_many!(
            &poseidon24_evals_to_prove[0].point,
            &poseidon24_evals_to_prove[1].point,
            &poseidon24_evals_to_prove[2].point,
            &poseidon24_evals_to_prove[5].point,
        );
        assert_eq!(
            &poseidon16_evals_to_prove[0].point[..3 + log2_ceil_usize(min_n_poseidons)],
            &poseidon24_evals_to_prove[0].point[..3 + log2_ceil_usize(min_n_poseidons)]
        );

        let mixing_challenges = MultilinearPoint(poseidon16_evals_to_prove[0].point[..3].to_vec());
        let poseidon_lookup_batching_chalenges = MultilinearPoint(prover_state.sample_vec(3));
        let mut poseidon_lookup_point = poseidon_lookup_batching_chalenges.0.clone();
        poseidon_lookup_point.extend_from_slice({
            if poseidons_16.len() > poseidons_24.len() {
                &poseidon16_evals_to_prove[0].point[3..]
            } else {
                &poseidon24_evals_to_prove[0].point[3..]
            }
        });
        let poseidon_lookup_value = poseidon_lookup_value(
            n_poseidons_16,
            n_poseidons_24,
            poseidon16_evals_to_prove,
            poseidon24_evals_to_prove,
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
            (poseidons_16.par_iter().map(|p| (&p.input[0..8]).evaluate(&mixing_challenges)).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| (&p.input[8..16]).evaluate(&mixing_challenges)).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| (&p.output[0..8]).evaluate(&mixing_challenges)).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_16.par_iter().map(|p| (&p.output[8..16]).evaluate(&mixing_challenges)).collect::<Vec<_>>(), poseidon16_steps),
            (poseidons_24.par_iter().map(|p| (&p.input[0..8]).evaluate(&mixing_challenges)).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| (&p.input[8..16]).evaluate(&mixing_challenges)).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| (&p.input[16..24]).evaluate(&mixing_challenges)).collect::<Vec<_>>(), poseidon24_steps),
            (poseidons_24.par_iter().map(|p| (&p.output).evaluate(&mixing_challenges)).collect::<Vec<_>>(), poseidon24_steps),
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

        let folded_memory = fold_multilinear(&padded_memory, &mixing_challenges);

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
        )
    };

    // Commit B
    let packed_pcs_witness_extension = packed_pcs_commit(
        &pcs.pcs_b(
            log2_strict_usize(packed_pcs_witness_base.packed_polynomial.len()),
            1 + log2_ceil_usize(private_memory.len())
                .max(log2_strict_usize(poseidon_pushforward.len())),
        ),
        &[
            memory_pushforward.as_slice(),
            poseidon_pushforward.as_slice(),
        ],
        &dft,
        &mut prover_state,
    );

    let main_trace_logup_star_statements = prove_logup_star(
        &mut prover_state,
        &padded_memory,
        &exec_memory_indexes,
        &main_table_evals_to_prove[1],
        &memory_poly_eq_point,
        &memory_pushforward,
    );

    let poseidon_logup_star_statements = prove_logup_star(
        &mut prover_state,
        &folded_memory,
        &all_poseidon_indexes,
        &poseidon_lookup_challenge,
        &poseidon_poly_eq_point,
        &poseidon_pushforward,
    );

    // open memory at point logup_star_statements.on_table.point
    let private_memory_chunk_point = MultilinearPoint(
        main_trace_logup_star_statements.on_table.point[log_padded_memory - log_public_memory..]
            .to_vec(),
    );
    let mut private_memory_statements = vec![];
    for private_memory_chunk in &private_memory_commited_chunks {
        let chunk_eval = private_memory_chunk.evaluate(&private_memory_chunk_point);
        private_memory_statements.push(chunk_eval);
    }
    prover_state.add_extension_scalars(&private_memory_statements);
    let private_memory_statements = private_memory_statements
        .into_iter()
        .map(|value| {
            vec![Evaluation {
                point: private_memory_chunk_point.clone(),
                value,
            }]
        })
        .collect::<Vec<_>>();

    // TODO open remaining logup_star_statements statements

    // Open A
    let global_statements_base_polynomial = packed_pcs_global_statements(
        &packed_pcs_witness_base.tree,
        &[
            vec![
                vec![main_table_evals_to_prove[2].clone()], // pc, fp
                vec![
                    main_table_evals_to_prove[3].clone(),
                    main_trace_logup_star_statements.on_indexes,
                ], // memory addresses
                vec![],
                vec![],
                vec![poseidon16_evals_to_prove[2].clone()],
                vec![poseidon24_evals_to_prove[3].clone()],
            ],
            private_memory_statements,
        ]
        .concat(),
    );

    // Open B
    let global_statements_extension_polynomial = packed_pcs_global_statements(
        &packed_pcs_witness_extension.tree,
        &vec![main_trace_logup_star_statements.on_pushforward, vec![]],
    );

    pcs.batch_open(
        &dft,
        &mut prover_state,
        &global_statements_base_polynomial,
        packed_pcs_witness_base.inner_witness,
        &packed_pcs_witness_base.packed_polynomial,
        &global_statements_extension_polynomial,
        packed_pcs_witness_extension.inner_witness,
        &packed_pcs_witness_extension.packed_polynomial,
    );

    prover_state.proof_data().to_vec()
}
