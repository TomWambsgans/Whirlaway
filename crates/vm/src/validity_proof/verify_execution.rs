use ::air::table::AirTable;
use lookup::verify_logup_star;
use p3_air::BaseAir;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use pcs::{BatchPCS, packed_pcs_parse_commitment};
use pcs::{PCS, packed_pcs_global_statements};
use utils::{Evaluation, PF, build_challenger, padd_with_zero_to_next_power_of_two};
use utils::{ToUsize, build_poseidon_16_air, build_poseidon_24_air};
use whir_p3::fiat_shamir::{errors::ProofError, verifier::VerifierState};
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::poly::multilinear::MultilinearPoint;

use crate::{air::VMAir, bytecode::bytecode::Bytecode, *};

pub fn verify_execution(
    _bytecode: &Bytecode,
    public_input: &[F],
    proof_data: Vec<PF<EF>>,
    pcs: &impl BatchPCS<PF<EF>, EF, EF>,
) -> Result<(), ProofError> {
    let table = AirTable::<EF, _>::new(VMAir, UNIVARIATE_SKIPS);

    let mut verifier_state = VerifierState::new(proof_data, build_challenger());

    let poseidon_16_air = build_poseidon_16_air();
    let poseidon_24_air = build_poseidon_24_air();
    let table_poseidon_16 = AirTable::<EF, _>::new(poseidon_16_air.clone(), UNIVARIATE_SKIPS);
    let table_poseidon_24 = AirTable::<EF, _>::new(poseidon_24_air.clone(), UNIVARIATE_SKIPS);

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
    let vars_poseidon_16 =
        log2_ceil_usize(n_poseidons_16) + log2_ceil_usize(poseidon_16_air.width() - 16 * 2);
    let vars_poseidon_24 =
        log2_ceil_usize(n_poseidons_24) + log2_ceil_usize(poseidon_24_air.width() - 24 * 2);

    let vars_for_private_memory =
        vec![log2_strict_usize(public_memory_len); n_private_memory_chunks];

    let vars_per_polynomial_base = [
        vec![
            vars_pc_fp,
            vars_memory_addresses,
            vars_poseidon_16,
            vars_poseidon_24,
        ],
        vars_for_private_memory,
    ]
    .concat();

    let packed_parsed_commitment_base =
        packed_pcs_parse_commitment(pcs.pcs_a(), &mut verifier_state, vars_per_polynomial_base)?;

    let main_table_evals_to_verify =
        table.verify(&mut verifier_state, log_n_cycles, &COLUMN_GROUPS_EXEC)?;
    let poseidon16_evals_to_verify = table_poseidon_16.verify(
        &mut verifier_state,
        log2_ceil_usize(n_poseidons_16),
        &[
            0..16,
            16..poseidon_16_air.width() - 16,
            poseidon_16_air.width() - 16..poseidon_16_air.width(),
        ],
    )?;
    let poseidon24_evals_to_verify = table_poseidon_24.verify(
        &mut verifier_state,
        log2_ceil_usize(n_poseidons_24),
        &[
            0..24,
            24..poseidon_24_air.width() - 24,
            poseidon_24_air.width() - 24..poseidon_24_air.width(),
        ],
    )?;

    let padded_memory_len = (public_memory_len + private_memory_len).next_power_of_two();
    let main_pushforward_variables = log2_strict_usize(padded_memory_len);
    let vars_per_polynomial_extension = vec![main_pushforward_variables];
    let packed_parsed_commitment_extension = packed_pcs_parse_commitment(
        pcs.pcs_b(),
        &mut verifier_state,
        vars_per_polynomial_extension,
    )
    .unwrap();

    let logup_star_statements = verify_logup_star(
        &mut verifier_state,
        log_padded_memory,
        log_n_cycles + 2, // 3 memory columns, rounded to 2^2
        &main_table_evals_to_verify[1],
    )
    .unwrap();

    let private_memory_statements =
        verifier_state.next_extension_scalars_vec(n_private_memory_chunks)?;
    if logup_star_statements.on_table.value
        != padd_with_zero_to_next_power_of_two(&private_memory_statements).evaluate(
            &MultilinearPoint(
                logup_star_statements.on_table.point[..log_padded_memory - log_public_memory]
                    .to_vec(),
            ),
        )
    {}
    let private_memory_chunk_point = MultilinearPoint(
        logup_star_statements.on_table.point[log_padded_memory - log_public_memory..].to_vec(),
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
                    logup_star_statements.on_indexes,
                ], // memory addresses
                vec![poseidon16_evals_to_verify[1].clone()],
                vec![poseidon24_evals_to_verify[1].clone()],
            ],
            private_memory_statements,
        ]
        .concat(),
    );
    pcs.pcs_a().verify(
        &mut verifier_state,
        &packed_parsed_commitment_base.inner_parsed_commitment,
        &global_statements_base_polynomial,
    )?;

    Ok(())
}
