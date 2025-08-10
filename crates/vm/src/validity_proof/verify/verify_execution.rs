use crate::validity_proof::common::fold_bytecode;
use ::air::table::AirTable;
use ::air::verify_many_air;
use lookup::verify_logup_star;
use p3_air::BaseAir;
use p3_field::PrimeCharacteristicRing;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use pcs::num_packed_vars_for_vars;
use pcs::packed_pcs_global_statements;
use pcs::{BatchPCS, NumVariables as _, packed_pcs_parse_commitment};
use utils::{Evaluation, PF, build_challenger, padd_with_zero_to_next_power_of_two};
use utils::{ToUsize, build_poseidon_16_air, build_poseidon_24_air};
use whir_p3::fiat_shamir::{errors::ProofError, verifier::VerifierState};
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::poly::multilinear::MultilinearPoint;

use crate::runner::build_public_memory;
use crate::validity_proof::common::{
    poseidon_16_column_groups, poseidon_24_column_groups, poseidon_lookup_index_statements,
    poseidon_lookup_value,
};
use crate::{air::VMAir, bytecode::bytecode::Bytecode, *};

pub fn verify_execution(
    bytecode: &Bytecode,
    public_input: &[F],
    proof_data: Vec<PF<EF>>,
    pcs: &impl BatchPCS<PF<EF>, EF, EF>,
) -> Result<(), ProofError> {
    let mut verifier_state = VerifierState::new(proof_data, build_challenger());

    let exec_table = AirTable::<EF, _>::new(VMAir);
    let p16_air = build_poseidon_16_air();
    let p24_air = build_poseidon_24_air();
    let p16_table = AirTable::<EF, _>::new(p16_air.clone());
    let p24_table = AirTable::<EF, _>::new(p24_air.clone());

    let [
        log_n_cycles,
        n_poseidons_16,
        n_poseidons_24,
        n_dot_products_ee,
        n_dot_products_be,
        private_memory_len,
    ] = verifier_state
        .next_base_scalars_const::<6>()?
        .into_iter()
        .map(|x| x.to_usize())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    if log_n_cycles > 32
        || n_poseidons_16 > 1 << 32
        || n_poseidons_24 > 1 << 32
        || n_dot_products_ee > 1 << 32
        || n_dot_products_be > 1 << 32
        || private_memory_len > 1 << 32
    {
        // To avoid "DOS" attack
        return Err(ProofError::InvalidProof);
    }

    let public_memory = build_public_memory(public_input);
    let public_memory_len = public_memory.len();
    if private_memory_len % public_memory_len != 0 {
        return Err(ProofError::InvalidProof);
    }
    let n_private_memory_chunks = private_memory_len / public_memory_len;
    let log_public_memory = log2_strict_usize(public_memory_len);
    let log_memory = log2_ceil_usize(public_memory_len + private_memory_len);
    let log_n_p16 = log2_ceil_usize(n_poseidons_16);
    let log_n_p24 = log2_ceil_usize(n_poseidons_24);

    let vars_pc_fp = log_n_cycles + 1;
    let vars_exec_memory_addresses = log_n_cycles + 2; // 3 memory addresses, rounded to 2^2
    let vars_p16_indexes = log_n_p16 + 2;
    let vars_p24_indexes = log_n_p24 + 2;

    let vars_p16_table = log_n_p16 + log2_ceil_usize(p16_air.width() - 16 * 2);
    let vars_p24_table = log_n_p24 + log2_ceil_usize(p24_air.width() - 24 * 2);

    let vars_private_memory = vec![log_public_memory; n_private_memory_chunks];

    let vars_pcs_base = [
        vec![
            vars_pc_fp,
            vars_exec_memory_addresses,
            vars_p16_indexes,
            vars_p24_indexes,
            vars_p16_table,
            vars_p24_table,
        ],
        vars_private_memory,
    ]
    .concat();

    let parsed_commitment_base =
        packed_pcs_parse_commitment(pcs.pcs_a(), &mut verifier_state, vars_pcs_base)?;

    let exec_evals_to_verify = exec_table.verify(
        &mut verifier_state,
        UNIVARIATE_SKIPS,
        log_n_cycles,
        &COLUMN_GROUPS_EXEC,
    )?;

    let poseidon_evals_to_verify = verify_many_air(
        &mut verifier_state,
        &[&p16_table],
        &[&p24_table],
        UNIVARIATE_SKIPS,
        &[log_n_p16, log_n_p24],
        &[
            poseidon_16_column_groups(&p16_air),
            poseidon_24_column_groups(&p24_air),
        ],
    )?;
    let p16_evals_to_verify = &poseidon_evals_to_verify[0];
    let p24_evals_to_verify = &poseidon_evals_to_verify[1];
    let memory_folding_challenges = MultilinearPoint(p16_evals_to_verify[0].point[..3].to_vec());

    // Poseidons 16/24 memory addresses lookup
    let poseidon_lookup_batching_chalenges = MultilinearPoint(verifier_state.sample_vec(3));

    let poseidon_lookup_log_length = 3 + log_n_p16.max(log_n_p24);

    let log_bytecode_len = log2_ceil_usize(bytecode.instructions.len());
    let vars_pcs_extension = vec![log_memory, log_memory - 3, log_bytecode_len];
    let packed_parsed_commitment_extension = packed_pcs_parse_commitment(
        &pcs.pcs_b(
            parsed_commitment_base
                .inner_parsed_commitment
                .num_variables(),
            num_packed_vars_for_vars(&vars_pcs_extension),
        ),
        &mut verifier_state,
        vars_pcs_extension,
    )
    .unwrap();

    let exec_logup_star_statements = verify_logup_star(
        &mut verifier_state,
        log_memory,
        log_n_cycles + 2, // 3 memory columns, rounded to 2^2
        &exec_evals_to_verify[1],
    )
    .unwrap();

    let mut poseidon_lookup_point = poseidon_lookup_batching_chalenges.0.clone();
    poseidon_lookup_point.extend_from_slice({
        if n_poseidons_16 > n_poseidons_24 {
            &p16_evals_to_verify[0].point[3..]
        } else {
            &p24_evals_to_verify[0].point[3..]
        }
    });
    let poseidon_lookup_value = poseidon_lookup_value(
        n_poseidons_16,
        n_poseidons_24,
        &p16_evals_to_verify,
        &p24_evals_to_verify,
        &poseidon_lookup_batching_chalenges,
    );
    let poseidon_lookup_challenge = Evaluation {
        point: MultilinearPoint(poseidon_lookup_point),
        value: poseidon_lookup_value,
    };
    let poseidon_logup_star_statements = verify_logup_star(
        &mut verifier_state,
        log_memory - 3, // "-3" because it's folded memory
        poseidon_lookup_log_length,
        &poseidon_lookup_challenge,
    )
    .unwrap();

    let bytecode_compression_challenges = MultilinearPoint(
        exec_evals_to_verify[0].point[..LOG_N_INSTRUCTION_COLUMNS_IN_AIR].to_vec(),
    );
    let bytecode_lookup_claim = Evaluation {
        point: MultilinearPoint(
            exec_evals_to_verify[0].point[LOG_N_INSTRUCTION_COLUMNS_IN_AIR..].to_vec(),
        ),
        value: exec_evals_to_verify[0].value,
    };
    let bytecode_logup_star_statements = verify_logup_star(
        &mut verifier_state,
        log_bytecode_len,
        log_n_cycles,
        &bytecode_lookup_claim,
    )
    .unwrap();
    let folded_bytecode = fold_bytecode(bytecode, &bytecode_compression_challenges);
    if folded_bytecode.evaluate(&bytecode_logup_star_statements.on_table.point)
        != bytecode_logup_star_statements.on_table.value
    {
        return Err(ProofError::InvalidProof);
    }

    let mut bytecode_lookup_index_statement = bytecode_logup_star_statements.on_indexes.clone();
    bytecode_lookup_index_statement.point.0.insert(0, EF::ZERO); // because we commit both pc and fp together

    let poseidon_lookup_memory_point = MultilinearPoint(
        [
            poseidon_logup_star_statements.on_table.point.0.clone(),
            memory_folding_challenges.0.clone(),
        ]
        .concat(),
    );

    let exec_lookup_chunk_point = MultilinearPoint(
        exec_logup_star_statements.on_table.point[log_memory - log_public_memory..].to_vec(),
    );
    let poseidon_lookup_chunk_point =
        MultilinearPoint(poseidon_lookup_memory_point[log_memory - log_public_memory..].to_vec());

    let mut chunk_evals_exec_lookup = vec![public_memory.evaluate(&exec_lookup_chunk_point)];
    let mut chunk_evals_poseidon_lookup =
        vec![public_memory.evaluate(&poseidon_lookup_chunk_point)];

    let mut private_memory_statements = vec![];
    for _ in 0..n_private_memory_chunks {
        let chunk_eval_exec_lookup = verifier_state.next_extension_scalar()?;
        let chunk_eval_poseidon_lookup = verifier_state.next_extension_scalar()?;
        chunk_evals_exec_lookup.push(chunk_eval_exec_lookup);
        chunk_evals_poseidon_lookup.push(chunk_eval_poseidon_lookup);
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

    if exec_logup_star_statements.on_table.value
        != padd_with_zero_to_next_power_of_two(&chunk_evals_exec_lookup).evaluate(
            &MultilinearPoint(
                exec_logup_star_statements.on_table.point[..log_memory - log_public_memory]
                    .to_vec(),
            ),
        )
    {
        return Err(ProofError::InvalidProof);
    }
    if poseidon_logup_star_statements.on_table.value
        != padd_with_zero_to_next_power_of_two(&chunk_evals_poseidon_lookup).evaluate(
            &MultilinearPoint(
                poseidon_logup_star_statements.on_table.point[..log_memory - log_public_memory]
                    .to_vec(),
            ),
        )
    {
        return Err(ProofError::InvalidProof);
    }

    // index opening for poseidon lookup
    let poseidon_index_evals = verifier_state.next_extension_scalars_vec(8)?;
    if poseidon_index_evals.evaluate(&MultilinearPoint(
        poseidon_logup_star_statements.on_indexes.point[..3].to_vec(),
    )) != poseidon_logup_star_statements.on_indexes.value
    {
        return Err(ProofError::InvalidProof);
    }

    let (p16_indexes_statements, p24_indexes_statements) = poseidon_lookup_index_statements(
        &poseidon_index_evals,
        n_poseidons_16,
        n_poseidons_24,
        &poseidon_logup_star_statements.on_indexes.point,
    )?;

    let global_statements_base = packed_pcs_global_statements(
        &parsed_commitment_base.tree,
        &[
            vec![
                vec![
                    exec_evals_to_verify[2].clone(),
                    bytecode_lookup_index_statement,
                ], // pc, fp
                vec![
                    exec_evals_to_verify[3].clone(),
                    exec_logup_star_statements.on_indexes,
                ], // memory addresses
                p16_indexes_statements,
                p24_indexes_statements,
                vec![p16_evals_to_verify[2].clone()],
                vec![p24_evals_to_verify[3].clone()],
            ],
            private_memory_statements,
        ]
        .concat(),
    );

    let global_statements_extension = packed_pcs_global_statements(
        &packed_parsed_commitment_extension.tree,
        &vec![
            exec_logup_star_statements.on_pushforward,
            poseidon_logup_star_statements.on_pushforward,
            bytecode_logup_star_statements.on_pushforward,
        ],
    );

    pcs.batch_verify(
        &mut verifier_state,
        &parsed_commitment_base.inner_parsed_commitment,
        &global_statements_base,
        &packed_parsed_commitment_extension.inner_parsed_commitment,
        &global_statements_extension,
    )?;

    Ok(())
}
