use p3_field::BasedVectorSpace;
use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::Poseidon2KoalaBear;

use crate::AIR_COLUMNS_PER_OPCODE;
use crate::bytecode::final_bytecode::Instruction;
use crate::bytecode::final_bytecode::Value;
use crate::{DIMENSION, EF, F, bytecode::final_bytecode::Bytecode};
use p3_field::PrimeField64;
use p3_symmetric::Permutation;

#[derive(Debug, Clone, Default)]
struct Memory {
    data: Vec<Option<F>>,
    _extension: std::marker::PhantomData<EF>,
}

impl Memory {
    fn is_set(&self, index: usize) -> bool {
        self.data.get(index).map_or(false, |opt| opt.is_some())
    }

    fn try_get(&self, index: usize) -> Option<F> {
        self.data.get(index).and_then(|opt| *opt)
    }

    fn get(&self, index: usize) -> F {
        self.data.get(index).and_then(|opt| *opt).unwrap()
    }

    fn read_value(&self, value: Value, fp: usize) -> F {
        match value {
            Value::Constant(c) => F::from_usize(c),
            Value::Fp => F::from_usize(fp),
            Value::MemoryAfterFp { shift } => self.get(fp + shift),
            Value::DirectMemory { shift } => self.get(shift),
        }
    }

    fn set(&mut self, index: usize, value: F) {
        if index >= self.data.len() {
            self.data.resize(index + 1, None);
        }
        if let Some(existing) = &mut self.data[index] {
            assert_eq!(
                existing, &value,
                "Memory already has a value at index {}: expected {}, found {}",
                index, value, existing
            );
        } else {
            self.data[index] = Some(value);
        }
    }

    fn get_vector(&self, index: usize) -> [F; DIMENSION] {
        let mut vector = [F::default(); DIMENSION];
        for i in 0..DIMENSION {
            vector[i] = self.get(index * DIMENSION + i);
        }
        vector
    }

    fn set_vector(&mut self, index: usize, value: [F; DIMENSION]) {
        for (i, v) in value.iter().enumerate() {
            let idx = DIMENSION * index + i;
            self.set(idx, *v);
        }
    }
}

impl Value {
    fn is_in_memory(&self) -> bool {
        matches!(
            self,
            Value::MemoryAfterFp { .. } | Value::DirectMemory { .. }
        )
    }

    fn memory_address(&self, fp: usize) -> Option<usize> {
        match self {
            Value::Constant(_) | Value::Fp => None,
            Value::MemoryAfterFp { shift } => Some(fp + *shift),
            Value::DirectMemory { shift } => Some(*shift),
        }
    }
}

pub fn execute_bytecode(
    bytecode: &Bytecode,
    public_input: &[F],
    posiedon_16: Poseidon2KoalaBear<16>,
    poseidon_24: Poseidon2KoalaBear<24>,
) {
    let mut memory = Memory::default();

    // TODO place the bytecode into memory
    // For now we will it with zeros
    for _ in 0..bytecode.instructions.len() * AIR_COLUMNS_PER_OPCODE {
        memory.data.push(Some(F::ZERO));
    }

    for (i, value) in public_input.iter().enumerate() {
        memory.set(bytecode.public_input_start + i, *value);
    }

    let mut pc = 0;
    let mut fp = bytecode.public_input_start + public_input.len();
    let mut ap = bytecode.public_input_start + public_input.len() + bytecode.starting_frame_memory;

    let constant_value = |value: Value, fp: usize| -> Option<F> {
        match value {
            Value::Constant(c) => Some(F::from_usize(c)),
            Value::Fp => Some(F::from_usize(fp)),
            Value::MemoryAfterFp { .. } | Value::DirectMemory { .. } => None,
        }
    };

    while pc != bytecode.ending_pc {
        if pc >= bytecode.instructions.len() {
            panic!(
                "Program counter out of bounds: {} >= {}",
                pc,
                bytecode.instructions.len()
            );
        }
        let instruction = &bytecode.instructions[pc];
        match instruction {
            Instruction::Computation {
                operation,
                arg_a,
                arg_b,
                res,
            } => {
                if let Some(memory_address_res) = res.memory_address(fp) {
                    let a_value = constant_value(*arg_a, fp).unwrap();
                    let b_value = constant_value(*arg_b, fp).unwrap();
                    let res_value = operation.compute(a_value, b_value);
                    memory.set(memory_address_res, res_value);
                } else if let Some(memory_address_a) = arg_a.memory_address(fp) {
                    let res_value = constant_value(*res, fp).unwrap();
                    let b_value = constant_value(*arg_b, fp).unwrap();
                    let a_value = operation.inverse_compute(res_value, b_value);
                    memory.set(memory_address_a, a_value);
                } else if let Some(memory_address_b) = arg_b.memory_address(fp) {
                    let res_value = constant_value(*res, fp).unwrap();
                    let a_value = constant_value(*arg_a, fp).unwrap();
                    let b_value = operation.inverse_compute(res_value, a_value);
                    memory.set(memory_address_b, b_value);
                } else {
                    let a_value = constant_value(*arg_a, fp).unwrap();
                    let b_value = constant_value(*arg_b, fp).unwrap();
                    let res_value = operation.compute(a_value, b_value);
                    assert_eq!(res_value, operation.compute(a_value, b_value));
                }
                pc += 1;
            }
            Instruction::MemoryPointerEq {
                shift_0,
                shift_1,
                res,
            } => {
                if let Some(memory_address_res) = res.memory_address(fp) {
                    let ptr = memory.get(fp + shift_0);
                    let value = memory.get(ptr.to_unique_u64() as usize + shift_1);
                    memory.set(memory_address_res, value);
                } else {
                    let value = constant_value(*res, fp).unwrap();
                    let ptr = memory.get(fp + shift_0);
                    memory.set(ptr.to_unique_u64() as usize + shift_1, value);
                }
                pc += 1;
            }
            Instruction::JumpIfNotZero {
                condition,
                dest,
                updated_fp,
            } => {
                if memory.read_value(*condition, fp) != F::ZERO {
                    pc = memory.read_value(*dest, fp).to_unique_u64() as usize;
                    fp = memory.read_value(*updated_fp, fp).to_unique_u64() as usize;
                } else {
                    pc += 1;
                }
            }
            Instruction::RequestMemory { shift, size } => {
                memory.set(fp + shift, F::from_usize(ap));
                ap += size;
                pc += 1;
            }
            Instruction::Poseidon2_16 { shift } => {
                let ptr_arg_0 = memory.get(fp + shift);
                let ptr_arg_1 = memory.get(fp + shift + 1);
                let ptr_res_0 = memory.get(fp + shift + 2);
                let ptr_res_1 = memory.get(fp + shift + 3);

                let arg0 = memory.get_vector(ptr_arg_0.to_unique_u64() as usize);
                let arg1 = memory.get_vector(ptr_arg_1.to_unique_u64() as usize);

                let mut input = [F::ZERO; DIMENSION * 2];
                input[..DIMENSION].copy_from_slice(&arg0);
                input[DIMENSION..].copy_from_slice(&arg1);

                posiedon_16.permute_mut(&mut input);

                let res0: [F; DIMENSION] = input[..DIMENSION].try_into().unwrap();
                let res1: [F; DIMENSION] = input[DIMENSION..].try_into().unwrap();

                memory.set_vector(ptr_res_0.to_unique_u64() as usize, res0);
                memory.set_vector(ptr_res_1.to_unique_u64() as usize, res1);

                pc += 1;
            }
            Instruction::Poseidon2_24 { shift } => {
                let ptr_arg_0 = memory.get(fp + shift);
                let ptr_arg_1 = memory.get(fp + shift + 1);
                let ptr_arg_2 = memory.get(fp + shift + 2);
                let ptr_res_0 = memory.get(fp + shift + 3);
                let ptr_res_1 = memory.get(fp + shift + 4);
                let ptr_res_2 = memory.get(fp + shift + 5);

                let arg0 = memory.get_vector(ptr_arg_0.to_unique_u64() as usize);
                let arg1 = memory.get_vector(ptr_arg_1.to_unique_u64() as usize);
                let arg2 = memory.get_vector(ptr_arg_2.to_unique_u64() as usize);

                let mut input = [F::ZERO; DIMENSION * 3];
                input[..DIMENSION].copy_from_slice(&arg0);
                input[DIMENSION..2 * DIMENSION].copy_from_slice(&arg1);
                input[2 * DIMENSION..].copy_from_slice(&arg2);

                poseidon_24.permute_mut(&mut input);

                let res0: [F; DIMENSION] = input[..DIMENSION].try_into().unwrap();
                let res1: [F; DIMENSION] = input[DIMENSION..2 * DIMENSION].try_into().unwrap();
                let res2: [F; DIMENSION] = input[2 * DIMENSION..].try_into().unwrap();

                memory.set_vector(ptr_res_0.to_unique_u64() as usize, res0);
                memory.set_vector(ptr_res_1.to_unique_u64() as usize, res1);
                memory.set_vector(ptr_res_2.to_unique_u64() as usize, res2);

                pc += 1;
            }
            Instruction::ExtComputation {
                operation,
                arg_a,
                arg_b,
                res,
            } => {
                // TODO

            
            }
        }
    }

    // TODO fill the bytecode into memory
}
