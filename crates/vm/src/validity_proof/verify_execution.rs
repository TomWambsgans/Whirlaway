use ::air::table::AirTable;
use p3_field::PrimeField64;
use p3_util::log2_ceil_usize;
use pcs::PCS;
use utils::{PF, build_challenger};
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

    let log_n_cycles = verifier_state.next_base_scalars_const::<1>()?[0];
    let log_n_cycles = log_n_cycles.as_canonical_u64() as usize;
    if log_n_cycles <= UNIVARIATE_SKIPS || log_n_cycles > 32 {
        return Err(ProofError::InvalidProof); // To avoid DDOS
    }

    let parsed_commitment = base_pcs.parse_commitment(
        &mut verifier_state,
        log_n_cycles + log2_ceil_usize(N_COMMITTED_COLUMNS),
    )?;
    let evaluations_remaining_to_verify =
        table.verify(&mut verifier_state, log_n_cycles, &COLUMN_GROUPS)?;

    // assert_eq!(
    //     padd_with_zero_to_next_power_of_two(&trace[..N_INSTRUCTION_FIELDS_IN_AIR].concat())
    //         .evaluate(&evaluations_remaining_to_verify[0].point),
    //     evaluations_remaining_to_verify[0].value
    // );
    // assert_eq!(
    //     padd_with_zero_to_next_power_of_two(
    //         &trace[N_INSTRUCTION_FIELDS_IN_AIR..N_INSTRUCTION_FIELDS_IN_AIR + 3].concat()
    //     )
    //     .evaluate(&evaluations_remaining_to_verify[1].point),
    //     evaluations_remaining_to_verify[1].value
    // );
    base_pcs.verify(
        &mut verifier_state,
        &parsed_commitment,
        &[evaluations_remaining_to_verify[2].clone()],
    )?;
    Ok(())
}
