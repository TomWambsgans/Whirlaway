use ::air::table::AirTable;
use p3_air::BaseAir;
use p3_util::log2_ceil_usize;
use pcs::{PCS, parse_multi_commitment, verify_multi_commitment};
use utils::{PF, build_challenger};
use utils::{ToUsize, build_poseidon_16_air, build_poseidon_24_air};
use whir_p3::fiat_shamir::{errors::ProofError, verifier::VerifierState};

use crate::{air::VMAir, bytecode::bytecode::Bytecode, *};

pub fn verify_execution(
    _bytecode: &Bytecode,
    _public_input: &[F],
    proof_data: Vec<PF<EF>>,
    base_pcs: &impl PCS<PF<EF>, EF>,
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
    ] = verifier_state
        .next_base_scalars_const::<5>()?
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

    let vars_main_table = log_n_cycles + log2_ceil_usize(N_COMMITTED_EXEC_COLUMNS);
    let vars_poseidon_16 =
        log2_ceil_usize(n_poseidons_16) + log2_ceil_usize(poseidon_16_air.width() - 16 * 2);
    let vars_poseidon_24 =
        log2_ceil_usize(n_poseidons_24) + log2_ceil_usize(poseidon_24_air.width() - 24 * 2);

    let vars_per_polynomial = vec![vars_main_table, vars_poseidon_16, vars_poseidon_24];

    let parsed_commitment =
        parse_multi_commitment(base_pcs, &mut verifier_state, vars_per_polynomial)?;

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

    verify_multi_commitment(
        base_pcs,
        &mut verifier_state,
        &parsed_commitment,
        &[
            vec![main_table_evals_to_verify[2].clone()],
            vec![poseidon16_evals_to_verify[1].clone()],
            vec![poseidon24_evals_to_verify[1].clone()],
        ],
    )?;

    Ok(())
}
