use ::air::{table::AirTable, witness::AirWitness};
use lookup::{compute_pushforward, prove_logup_star};
use p3_air::BaseAir;
use p3_field::PrimeCharacteristicRing;
use p3_util::log2_strict_usize;
use pcs::{BatchPCS, PCS, packed_pcs_commit, packed_pcs_global_statements};
use tracing::info_span;
use utils::{
    Evaluation, PF, build_poseidon_16_air, build_poseidon_24_air, build_prover_state,
    generate_trace_poseidon_16, generate_trace_poseidon_24, padd_with_zero_to_next_power_of_two,
};
use whir_p3::dft::EvalsDft;
use whir_p3::poly::evals::eval_eq;
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

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
        poseidons_16,
        poseidons_24,
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
            poseidons_16.len(),
            poseidons_24.len(),
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
    let main_table = AirTable::<EF, _>::new(VMAir, UNIVARIATE_SKIPS);

    #[cfg(test)]
    main_table.check_trace_validity(&main_witness).unwrap();

    let _validity_proof_span = info_span!("Validity proof generation").entered();

    let poseidon_16_air = build_poseidon_16_air();
    let poseidon_24_air = build_poseidon_24_air();
    let table_poseidon_16 = AirTable::<EF, _>::new(poseidon_16_air.clone(), UNIVARIATE_SKIPS);
    let table_poseidon_24 = AirTable::<EF, _>::new(poseidon_24_air.clone(), UNIVARIATE_SKIPS);

    let mut poseidon_16_data_padded = poseidons_16
        .iter()
        .map(|w| w.hashed_data)
        .collect::<Vec<_>>();
    poseidon_16_data_padded.resize(poseidons_16.len().next_power_of_two(), [F::ZERO; 16]);
    let mut poseidon_24_data_padded = poseidons_24
        .iter()
        .map(|w| w.hashed_data)
        .collect::<Vec<_>>();
    poseidon_24_data_padded.resize(poseidons_24.len().next_power_of_two(), [F::ZERO; 24]);
    let witness_matrix_poseidon_16 = generate_trace_poseidon_16(poseidon_16_data_padded);
    let witness_matrix_poseidon_24 = generate_trace_poseidon_24(poseidon_24_data_padded);

    let witness_matrix_poseidon_16_transposed = witness_matrix_poseidon_16.transpose();
    let witness_matrix_poseidon_24_transposed = witness_matrix_poseidon_24.transpose();

    assert_eq!(
        witness_matrix_poseidon_16_transposed.width,
        poseidons_16.len().next_power_of_two()
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
        poseidons_24.len().next_power_of_two()
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
            0..16,
            16..poseidon_16_air.width() - 16,
            poseidon_16_air.width() - 16..poseidon_16_air.width(),
        ],
    );
    let witness_poseidon_24 = AirWitness::new(
        &witness_columns_poseidon_24,
        &[
            0..24,
            24..poseidon_24_air.width() - 24,
            poseidon_24_air.width() - 24..poseidon_24_air.width(),
        ],
    );

    let commited_poseidon_16 = padd_with_zero_to_next_power_of_two(
        &witness_columns_poseidon_16[16..poseidon_16_air.width() - 16].concat(),
    );
    let commited_poseidon_24 = padd_with_zero_to_next_power_of_two(
        &witness_columns_poseidon_24[24..poseidon_24_air.width() - 24].concat(),
    );

    // 1) Commit A
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

    let packed_pcs_witness_base = packed_pcs_commit(
        pcs.pcs_a(),
        &[
            vec![
                commited_pc_fp.as_slice(),
                commited_memory_addreses.as_slice(),
                commited_poseidon_16.as_slice(),
                commited_poseidon_24.as_slice(),
            ],
            private_memory_commited_chunks.clone(),
        ]
        .concat(),
        &dft,
        &mut prover_state,
    );

    // 2) PIOP
    let main_table_evals_to_prove = main_table.prove(&mut prover_state, main_witness);
    let poseidon16_evals_to_prove = table_poseidon_16.prove(&mut prover_state, witness_poseidon_16);
    let poseidon24_evals_to_prove = table_poseidon_24.prove(&mut prover_state, witness_poseidon_24);

    // Main memory lookup
    let exec_memory_indexes = padd_with_zero_to_next_power_of_two(
        &main_trace[COL_INDEX_MEM_ADDRESS_A..=COL_INDEX_MEM_ADDRESS_C].concat(),
    );
    let exec_memory_values = padd_with_zero_to_next_power_of_two(
        &main_trace[COL_INDEX_MEM_VALUE_A..=COL_INDEX_MEM_VALUE_C].concat(),
    );
    let memory_poly_eq_point =
        info_span!("eval_eq for logup*").in_scope(|| eval_eq(&main_table_evals_to_prove[1].point));
    // TODO avoid this padding
    let padded_memory = padd_with_zero_to_next_power_of_two(&memory);
    let memory_pushforward = compute_pushforward(
        &exec_memory_indexes,
        padded_memory.len(),
        &memory_poly_eq_point,
    );
    let log_padded_memory = log2_strict_usize(padded_memory.len());
    let log_public_memory = log2_strict_usize(public_memory_size);

    let packed_pcs_witness_extension = packed_pcs_commit(
        pcs.pcs_b(),
        &[memory_pushforward.as_slice()],
        &dft,
        &mut prover_state,
    );

    let logup_star_statements = prove_logup_star(
        &mut prover_state,
        &padded_memory,
        &exec_memory_indexes,
        &exec_memory_values,
        &main_table_evals_to_prove[1],
        &memory_poly_eq_point,
        &memory_pushforward,
    );

    // open memory at point logup_star_statements.on_table.point
    let private_memory_chunk_point = MultilinearPoint(
        logup_star_statements.on_table.point[log_padded_memory - log_public_memory..].to_vec(),
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

    // 3) Open A
    let global_statements_base_polynomial = packed_pcs_global_statements(
        &packed_pcs_witness_base.tree,
        &[
            vec![
                vec![main_table_evals_to_prove[2].clone()], // pc, fp
                vec![
                    main_table_evals_to_prove[3].clone(),
                    logup_star_statements.on_indexes,
                ], // memory addresses
                vec![poseidon16_evals_to_prove[1].clone()],
                vec![poseidon24_evals_to_prove[1].clone()],
            ],
            private_memory_statements,
        ]
        .concat(),
    );
    pcs.pcs_a().open(
        &dft,
        &mut prover_state,
        &global_statements_base_polynomial,
        packed_pcs_witness_base.inner_witness,
        &packed_pcs_witness_base.packed_polynomial,
    );

    prover_state.proof_data().to_vec()
}
