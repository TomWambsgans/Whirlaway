use ::air::{table::AirTable, witness::AirWitness};
use p3_field::PrimeCharacteristicRing;
use p3_util::log2_strict_usize;
use pcs::PCS;
use tracing::info_span;
use utils::{build_poseidon_16_air, build_poseidon_24_air, build_prover_state, padd_with_zero_to_next_power_of_two, PF};
use whir_p3::dft::EvalsDft;

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
    base_pcs: &impl PCS<PF<EF>, EF>,
) -> Vec<PF<EF>> {
    let ExecutionTrace {
        main_trace,
        poseidons_16,
        poseidons_24,
        dot_products_ee,
        dot_products_be,
    } = info_span!("Witness generation").in_scope(|| {
        let execution_result = execute_bytecode(&bytecode, &public_input, private_input);
        get_execution_trace(&bytecode, &execution_result)
    });

    let log_n_rows = log2_strict_usize(main_trace[0].len());
    assert!(main_trace.iter().all(|col| col.len() == (1 << log_n_rows)));
    let mut prover_state = build_prover_state::<EF>();
    prover_state.add_base_scalars(&[F::from_usize(log_n_rows)]);

    let dft = EvalsDft::default();

    let witness = AirWitness::<PF<EF>>::new(&main_trace, &COLUMN_GROUPS);
    let table = AirTable::<EF, _>::new(VMAir, UNIVARIATE_SKIPS);

    #[cfg(test)]
    table.check_trace_validity(&witness).unwrap();

    let _validity_proof_span = info_span!("Validity proof generation").entered();

    let poseidon_16_air = build_poseidon_16_air();
    let poseidon_24_air = build_poseidon_24_air();

    // 1) Commit
    let commited_trace_polynomial = padd_with_zero_to_next_power_of_two(
        &main_trace[N_INSTRUCTION_FIELDS_IN_AIR + N_MEMORY_VALUE_COLUMNS..].concat(),
    );
    let pcs_witness = base_pcs.commit(&dft, &mut prover_state, &commited_trace_polynomial);

    // 2) PIOP
    let evaluations_remaining_to_prove = table.prove(&mut prover_state, witness);

    // 3) Open
    base_pcs.open(
        &dft,
        &mut prover_state,
        &[evaluations_remaining_to_prove[2].clone()],
        pcs_witness,
        &commited_trace_polynomial,
    );

    prover_state.proof_data().to_vec()
}
