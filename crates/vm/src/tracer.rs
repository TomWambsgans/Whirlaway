use crate::{
    F, N_AIR_COLUMNS, N_INSTRUCTION_FIELDS_IN_AIR,
    bytecode::bytecode::{Bytecode, Instruction},
    runner::ExecutionResult,
};
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField64;

pub fn get_execution_trace(bytecode: &Bytecode, execution_result: &ExecutionResult) -> Vec<Vec<F>> {
    assert_eq!(execution_result.pcs.len(), execution_result.fps.len());
    let log_n_rows = execution_result.pcs.len().next_power_of_two().ilog2() as usize;
    let mut trace = (0..N_AIR_COLUMNS)
        .map(|_| F::zero_vec(1 << log_n_rows))
        .collect::<Vec<Vec<F>>>();

    for (i, (&pc, &fp)) in execution_result
        .pcs
        .iter()
        .zip(&execution_result.fps)
        .enumerate()
    {
        let instruction = &bytecode.instructions[pc];
        let field_repr = instruction.field_representation();

        for (j, field) in field_repr
            .iter()
            .enumerate()
            .take(N_INSTRUCTION_FIELDS_IN_AIR)
        {
            trace[j][i] = *field;
        }

        let mut addr_a = F::ZERO;
        if field_repr[3].is_zero() {
            // flag_a == 0
            addr_a = F::from_usize(fp) + field_repr[0]; // fp + operand_a
        }
        let value_a = execution_result.memory.0[addr_a.as_canonical_u64() as usize].unwrap();
        let mut addr_b = F::ZERO;
        if field_repr[4].is_zero() {
            // flag_b == 0
            addr_b = F::from_usize(fp) + field_repr[1]; // fp + operand_b
        }
        let value_b = execution_result.memory.0[addr_b.as_canonical_u64() as usize].unwrap();

        let mut addr_c = F::ZERO;
        if field_repr[5].is_zero() {
            // flag_c == 0
            addr_c = F::from_usize(fp) + field_repr[2]; // fp + operand_c
        } else if let Instruction::Deref { shift_1, .. } = instruction {
            let operand_c = F::from_usize(*shift_1);
            assert_eq!(field_repr[2], operand_c); // debug purpose
            addr_c = value_a + operand_c;
        }
        let value_c = execution_result.memory.0[addr_c.as_canonical_u64() as usize].unwrap();

        trace[11][i] = value_a;
        trace[12][i] = value_b;
        trace[13][i] = value_c;
        trace[14][i] = F::from_usize(pc);
        trace[15][i] = F::from_usize(fp);
        trace[16][i] = addr_a;
        trace[17][i] = addr_b;
        trace[18][i] = addr_c;
    }

    trace
}
