use ::air::table::AirTable;
use ::air::verify_many_air;
use lookup::verify_logup_star;
use p3_air::BaseAir;
use p3_field::PrimeCharacteristicRing;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use pcs::packed_pcs_global_statements;
use pcs::{BatchPCS, NumVariables as _, packed_pcs_parse_commitment};
use utils::from_end;
use utils::{Evaluation, PF, build_challenger, padd_with_zero_to_next_power_of_two};
use utils::{ToUsize, build_poseidon_16_air, build_poseidon_24_air};
use whir_p3::fiat_shamir::{errors::ProofError, verifier::VerifierState};
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::poly::multilinear::MultilinearPoint;

use crate::validity_proof::common::poseidon_lookup_value;
use crate::{air::VMAir, bytecode::bytecode::Bytecode, *};

pub fn verify_execution(
    _bytecode: &Bytecode,
    public_input: &[F],
    proof_data: Vec<PF<EF>>,
    pcs: &impl BatchPCS<PF<EF>, EF, EF>,
) -> Result<(), ProofError> {
    let table = AirTable::<EF, _>::new(VMAir);

    let mut verifier_state = VerifierState::new(proof_data, build_challenger());

    let poseidon_16_air = build_poseidon_16_air();
    let poseidon_24_air = build_poseidon_24_air();
    let table_poseidon_16 = AirTable::<EF, _>::new(poseidon_16_air.clone());
    let table_poseidon_24 = AirTable::<EF, _>::new(poseidon_24_air.clone());

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
        .map(|x| {
            if x.to_usize() > 1 << 32 {
                Err(ProofError::InvalidProof) // To avoid DDOS
            } else {
                Ok(x.to_usize())
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .unwrap();

    if log_n_cycles > 32 {
        return Err(ProofError::InvalidProof);
    }

    let public_memory_len = (PUBLIC_INPUT_START + public_input.len()).next_power_of_two();
    if private_memory_len % public_memory_len != 0 {
        return Err(ProofError::InvalidProof);
    }
    let n_private_memory_chunks = private_memory_len / public_memory_len;
    let padded_memory_len = (public_memory_len + private_memory_len).next_power_of_two();
    let log_padded_memory = log2_strict_usize(padded_memory_len);
    let log_public_memory = log2_strict_usize(public_memory_len);

    let vars_pc_fp = log_n_cycles + 1;
    let vars_memory_addresses = log_n_cycles + 2; // 3 memory addresses, rounded to 2^2
    let vars_poseidon_16_indexes = log2_ceil_usize(n_poseidons_16) + 2;
    let vars_poseidon_24_indexes = log2_ceil_usize(n_poseidons_24) + 2;

    let vars_poseidon_16_table =
        log2_ceil_usize(n_poseidons_16) + log2_ceil_usize(poseidon_16_air.width() - 16 * 2);
    let vars_poseidon_24_table =
        log2_ceil_usize(n_poseidons_24) + log2_ceil_usize(poseidon_24_air.width() - 24 * 2);

    let vars_for_private_memory =
        vec![log2_strict_usize(public_memory_len); n_private_memory_chunks];

    let vars_per_polynomial_base = [
        vec![
            vars_pc_fp,
            vars_memory_addresses,
            vars_poseidon_16_indexes,
            vars_poseidon_24_indexes,
            vars_poseidon_16_table,
            vars_poseidon_24_table,
        ],
        vars_for_private_memory,
    ]
    .concat();

    let packed_parsed_commitment_base =
        packed_pcs_parse_commitment(pcs.pcs_a(), &mut verifier_state, vars_per_polynomial_base)?;

    let main_table_evals_to_verify = table.verify(
        &mut verifier_state,
        UNIVARIATE_SKIPS,
        log_n_cycles,
        &COLUMN_GROUPS_EXEC,
    )?;

    let poseidon_evals_to_verify = verify_many_air(
        &mut verifier_state,
        &[&table_poseidon_16],
        &[&table_poseidon_24],
        UNIVARIATE_SKIPS,
        &[
            log2_ceil_usize(n_poseidons_16),
            log2_ceil_usize(n_poseidons_24),
        ],
        &[
            vec![
                0..8,
                8..16,
                16..poseidon_16_air.width() - 16,
                poseidon_16_air.width() - 16..poseidon_16_air.width() - 8,
                poseidon_16_air.width() - 8..poseidon_16_air.width(),
            ],
            vec![
                0..8,
                8..16,
                16..24,
                24..poseidon_24_air.width() - 24,
                poseidon_24_air.width() - 24..poseidon_24_air.width() - 8, // TODO should we commit to this part ? Probably not, but careful here, we will not check evaluations for this part
                poseidon_24_air.width() - 8..poseidon_24_air.width(),
            ],
        ],
    )?;
    let poseidon16_evals_to_verify = &poseidon_evals_to_verify[0];
    let poseidon24_evals_to_verify = &poseidon_evals_to_verify[1];

    // Poseidons 16/24 memory addresses lookup
    let poseidon_lookup_batching_chalenges = MultilinearPoint(verifier_state.sample_vec(3));

    let poseidon_lookup_table_log_length = 3 + log2_ceil_usize(n_poseidons_16.max(n_poseidons_24));

    let padded_memory_len = (public_memory_len + private_memory_len).next_power_of_two();
    let main_pushforward_variables = log2_strict_usize(padded_memory_len);
    let vars_per_polynomial_extension =
        vec![main_pushforward_variables, poseidon_lookup_table_log_length];
    let packed_parsed_commitment_extension = packed_pcs_parse_commitment(
        &pcs.pcs_b(
            packed_parsed_commitment_base
                .inner_parsed_commitment
                .num_variables(),
            1 + log2_ceil_usize(private_memory_len).max(poseidon_lookup_table_log_length),
        ),
        &mut verifier_state,
        vars_per_polynomial_extension,
    )
    .unwrap();

    let main_logup_star_statements = verify_logup_star(
        &mut verifier_state,
        log_padded_memory,
        log_n_cycles + 2, // 3 memory columns, rounded to 2^2
        &main_table_evals_to_verify[1],
    )
    .unwrap();

    let mut poseidon_lookup_point = poseidon_lookup_batching_chalenges.0.clone();
    poseidon_lookup_point.extend_from_slice({
        if n_poseidons_16 > n_poseidons_24 {
            &poseidon16_evals_to_verify[0].point[3..]
        } else {
            &poseidon24_evals_to_verify[0].point[3..]
        }
    });
    let poseidon_lookup_value = poseidon_lookup_value(
        n_poseidons_16,
        n_poseidons_24,
        &poseidon16_evals_to_verify,
        &poseidon24_evals_to_verify,
        &poseidon_lookup_batching_chalenges,
    );
    let poseidon_lookup_challenge = Evaluation {
        point: MultilinearPoint(poseidon_lookup_point),
        value: poseidon_lookup_value,
    };
    let poseidon_logup_star_statements = verify_logup_star(
        &mut verifier_state,
        log2_strict_usize(padded_memory_len) - 3, // "-3" because it's folded memory
        poseidon_lookup_table_log_length,
        &poseidon_lookup_challenge,
    )
    .unwrap();

    let private_memory_statements =
        verifier_state.next_extension_scalars_vec(n_private_memory_chunks)?;
    if main_logup_star_statements.on_table.value
        != padd_with_zero_to_next_power_of_two(&private_memory_statements).evaluate(
            &MultilinearPoint(
                main_logup_star_statements.on_table.point[..log_padded_memory - log_public_memory]
                    .to_vec(),
            ),
        )
    {}
    let private_memory_chunk_point = MultilinearPoint(
        main_logup_star_statements.on_table.point[log_padded_memory - log_public_memory..].to_vec(),
    );
    let private_memory_statements = private_memory_statements
        .into_iter()
        .map(|value| {
            vec![Evaluation {
                point: private_memory_chunk_point.clone(),
                value,
            }]
        })
        .collect::<Vec<_>>();

    let global_statements_base_polynomial = packed_pcs_global_statements(
        &packed_parsed_commitment_base.tree,
        &[
            vec![
                vec![main_table_evals_to_verify[2].clone()], // pc, fp
                vec![
                    main_table_evals_to_verify[3].clone(),
                    main_logup_star_statements.on_indexes,
                ], // memory addresses
                vec![],
                vec![],
                vec![poseidon16_evals_to_verify[2].clone()],
                vec![poseidon24_evals_to_verify[3].clone()],
            ],
            private_memory_statements,
        ]
        .concat(),
    );

    // Open B
    let global_statements_extension_polynomial = packed_pcs_global_statements(
        &packed_parsed_commitment_extension.tree,
        &vec![main_logup_star_statements.on_pushforward, vec![]],
    );

    pcs.batch_verify(
        &mut verifier_state,
        &packed_parsed_commitment_base.inner_parsed_commitment,
        &global_statements_base_polynomial,
        &packed_parsed_commitment_extension.inner_parsed_commitment,
        &global_statements_extension_polynomial,
    )?;

    Ok(())
}
