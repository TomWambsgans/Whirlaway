use crate::{
    EF, F, N_AIR_COLUMNS, N_INSTRUCTION_FIELDS_IN_AIR,
    bytecode::bytecode::{Bytecode, Instruction},
    runner::ExecutionResult,
};
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use utils::ToUsize;

pub struct WitnessDotProductEE {
    pub addr_0: usize,   // vectorized pointer
    pub addr_1: usize,   // vectorized pointer
    pub addr_res: usize, // vectorized pointer
    pub len: usize,
    pub slice_0: Vec<EF>,
    pub slice_1: Vec<EF>,
}

pub struct WitnessDotProductBE {
    pub addr_base: usize, // normal pointer
    pub addr_ext: usize,  // vectorized pointer
    pub addr_res: usize,  // vectorized pointer
    pub len: usize,
    pub slice_base: Vec<F>,
    pub slice_ext: Vec<EF>,
}

pub struct WitnessPoseidon16 {
    pub addr_input_a: usize, // vectorized pointer (of size 1)
    pub addr_input_b: usize, // vectorized pointer (of size 1)
    pub addr_output: usize,  // vectorized pointer (of size 2)
    pub hashed_data: [F; 16],
}

pub struct WitnessPoseidon24 {
    pub addr_input_a: usize, // vectorized pointer (of size 2)
    pub addr_input_b: usize, // vectorized pointer (of size 1)
    pub addr_output: usize,  // vectorized pointer (of size 1)
    pub hashed_data: [F; 24],
}

pub struct ExecutionTrace {
    pub main_trace: Vec<Vec<F>>,
    pub poseidons_16: Vec<WitnessPoseidon16>,
    pub poseidons_24: Vec<WitnessPoseidon24>,
    pub dot_products_ee: Vec<WitnessDotProductEE>,
    pub dot_products_be: Vec<WitnessDotProductBE>,
}

pub fn get_execution_trace(
    bytecode: &Bytecode,
    execution_result: &ExecutionResult,
) -> ExecutionTrace {
    assert_eq!(execution_result.pcs.len(), execution_result.fps.len());
    let n_cycles = execution_result.pcs.len();
    let memory = &execution_result.memory;
    let log_n_cycles_rounded_up = n_cycles.next_power_of_two().ilog2() as usize;
    let mut trace = (0..N_AIR_COLUMNS)
        .map(|_| F::zero_vec(1 << log_n_cycles_rounded_up))
        .collect::<Vec<Vec<F>>>();
    let mut poseidons_16 = Vec::new();
    let mut poseidons_24 = Vec::new();
    let mut dot_products_ee = Vec::new();
    let mut dot_products_be = Vec::new();

    for (i, (&pc, &fp)) in execution_result
        .pcs
        .iter()
        .zip(&execution_result.fps)
        .enumerate()
    {
        let instruction = &bytecode.instructions[pc];
        let field_repr = instruction.field_representation();

        // println!(
        //     "Cycle {}: PC = {}, FP = {}, Instruction = {}",
        //     i, pc, fp, instruction.to_string()
        // );

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
        let value_a = memory.0[addr_a.to_usize()].unwrap();
        let mut addr_b = F::ZERO;
        if field_repr[4].is_zero() {
            // flag_b == 0
            addr_b = F::from_usize(fp) + field_repr[1]; // fp + operand_b
        }
        let value_b = memory.0[addr_b.to_usize()].unwrap();

        let mut addr_c = F::ZERO;
        if field_repr[5].is_zero() {
            // flag_c == 0
            addr_c = F::from_usize(fp) + field_repr[2]; // fp + operand_c
        } else if let Instruction::Deref { shift_1, .. } = instruction {
            let operand_c = F::from_usize(*shift_1);
            assert_eq!(field_repr[2], operand_c); // debug purpose
            addr_c = value_a + operand_c;
        }
        let value_c = memory.0[addr_c.to_usize()].unwrap();

        trace[11][i] = value_a;
        trace[12][i] = value_b;
        trace[13][i] = value_c;
        trace[14][i] = F::from_usize(pc);
        trace[15][i] = F::from_usize(fp);
        trace[16][i] = addr_a;
        trace[17][i] = addr_b;
        trace[18][i] = addr_c;

        match instruction {
            Instruction::Poseidon2_16 { arg_a, arg_b, res } => {
                let addr_input_a = arg_a.read_value(&memory, fp).unwrap().to_usize();
                let addr_input_b = arg_b.read_value(&memory, fp).unwrap().to_usize();
                let addr_output = res.read_value(&memory, fp).unwrap().to_usize();
                let value_a = memory.get_vector(addr_input_a).unwrap();
                let value_b = memory.get_vector(addr_input_b).unwrap();
                poseidons_16.push(WitnessPoseidon16 {
                    addr_input_a,
                    addr_input_b,
                    addr_output,
                    hashed_data: [value_a, value_b].concat().try_into().unwrap(),
                });
            }
            Instruction::Poseidon2_24 { arg_a, arg_b, res } => {
                let addr_input_a = arg_a.read_value(&memory, fp).unwrap().to_usize();
                let addr_input_b = arg_b.read_value(&memory, fp).unwrap().to_usize();
                let addr_output = res.read_value(&memory, fp).unwrap().to_usize();
                let value_a = memory.get_vectorized_slice(addr_input_a, 2).unwrap();
                let value_b = memory.get_vector(addr_input_b).unwrap().to_vec();
                poseidons_24.push(WitnessPoseidon24 {
                    addr_input_a,
                    addr_input_b,
                    addr_output,
                    hashed_data: [value_a, value_b].concat().try_into().unwrap(),
                });
            }
            Instruction::DotProductExtensionExtension {
                arg0,
                arg1,
                res,
                size,
            } => {
                let addr_0 = arg0.read_value(&memory, fp).unwrap().to_usize();
                let addr_1 = arg1.read_value(&memory, fp).unwrap().to_usize();
                let addr_res = res.read_value(&memory, fp).unwrap().to_usize();
                let slice_0 = memory
                    .get_vectorized_slice_extension(addr_0, *size)
                    .unwrap();
                let slice_1 = memory
                    .get_vectorized_slice_extension(addr_1, *size)
                    .unwrap();
                dot_products_ee.push(WitnessDotProductEE {
                    addr_0,
                    addr_1,
                    addr_res,
                    len: *size,
                    slice_0,
                    slice_1,
                });
            }
            Instruction::DotProductBaseExtension {
                arg_base,
                arg_ext,
                res,
                size,
            } => {
                let addr_base = arg_base.read_value(&memory, fp).unwrap().to_usize();
                let addr_ext = arg_ext.read_value(&memory, fp).unwrap().to_usize();
                let addr_res = res.read_value(&memory, fp).unwrap().to_usize();
                let slice_base = (0..*size)
                    .map(|i| memory.get(addr_base + i).unwrap())
                    .collect::<Vec<F>>();
                let slice_ext = memory
                    .get_vectorized_slice_extension(addr_ext, *size)
                    .unwrap();
                dot_products_be.push(WitnessDotProductBE {
                    addr_base,
                    addr_ext,
                    addr_res,
                    len: *size,
                    slice_base,
                    slice_ext,
                });
            }
            _ => {}
        }
    }

    // repeat the last row to get to a power of two
    for j in 0..N_AIR_COLUMNS {
        let last_value = trace[j][n_cycles - 1];
        for i in n_cycles..(1 << log_n_cycles_rounded_up) {
            trace[j][i] = last_value;
        }
    }

    ExecutionTrace {
        main_trace: trace,
        poseidons_16,
        poseidons_24,
        dot_products_ee,
        dot_products_be,
    }
}
