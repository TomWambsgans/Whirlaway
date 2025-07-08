use p3_field::BasedVectorSpace;
use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::Poseidon2KoalaBear;

use crate::AIR_COLUMNS_PER_OPCODE;
use crate::bytecode::final_bytecode::Hint;
use crate::bytecode::final_bytecode::Instruction;
use crate::bytecode::final_bytecode::Operation;
use crate::bytecode::final_bytecode::Value;
use crate::{DIMENSION, EF, F, bytecode::final_bytecode::Bytecode};
use p3_field::PrimeField64;
use p3_symmetric::Permutation;

const MAX_MEMORY_SIZE: usize = 1 << 20;

#[derive(Debug, Clone, Default)]
struct Memory {
    data: Vec<Option<F>>,
    _extension: std::marker::PhantomData<EF>,
}

impl Memory {
    // fn is_set(&self, index: usize) -> bool {
    //     self.data.get(index).map_or(false, |opt| opt.is_some())
    // }

    fn try_get(&self, index: usize) -> Option<F> {
        self.data.get(index).and_then(|opt| *opt)
    }

    fn get(&self, index: usize) -> F {
        self.data.get(index).and_then(|opt| *opt).unwrap()
    }

    fn read_value(&self, value: Value, fp: usize) -> F {
        self.try_read_value(value, fp).unwrap()
    }

    fn try_read_value(&self, value: Value, fp: usize) -> Option<F> {
        match value {
            Value::Constant(c) => Some(F::from_usize(c)),
            Value::Fp => Some(F::from_usize(fp)),
            Value::MemoryAfterFp { shift } => self.try_get(fp + shift),
            Value::DirectMemory { shift } => self.try_get(shift),
        }
    }

    fn is_value_unknown(&self, value: Value, fp: usize) -> bool {
        self.try_read_value(value, fp).is_none()
    }

    fn set(&mut self, index: usize, value: F) {
        if index >= self.data.len() {
            assert!(
                index < MAX_MEMORY_SIZE,
                "Memory index out of bounds: {} >= {}",
                index,
                MAX_MEMORY_SIZE
            );
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
    // fn is_in_memory(&self) -> bool {
    //     matches!(
    //         self,
    //         Value::MemoryAfterFp { .. } | Value::DirectMemory { .. }
    //     )
    // }

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
    poseidon_16: Poseidon2KoalaBear<16>,
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

    while pc != bytecode.ending_pc {
        if pc >= bytecode.instructions.len() {
            panic!(
                "Program counter out of bounds: {} >= {}",
                pc,
                bytecode.instructions.len()
            );
        }

        //  dbg!(pc, fp);

        for hint in bytecode.hints.get(&pc).unwrap_or(&vec![]) {
            match hint {
                Hint::RequestMemory { shift, size } => {
                    memory.set(fp + shift, F::from_usize(ap));
                    ap += size;
                    // does not increase PC
                }
                Hint::Print { line_info, content } => {
                    let values = content
                        .iter()
                        .map(|value| memory.read_value(*value, fp).to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let line_info = line_info.replace(";", "");
                    println!("\"{}\" -> {}", line_info, values);
                    // does not increase PC
                }
            }
        }

        let instruction = &bytecode.instructions[pc];
        match instruction {
            Instruction::Computation {
                operation,
                arg_a,
                arg_b,
                res,
            } => {
                if memory.is_value_unknown(*res, fp) {
                    let memory_address_res = res.memory_address(fp).unwrap();
                    let a_value = memory.read_value(*arg_a, fp);
                    let b_value = memory.read_value(*arg_b, fp);
                    let res_value = operation.compute(a_value, b_value);
                    memory.set(memory_address_res, res_value);
                } else if memory.is_value_unknown(*arg_a, fp) {
                    let memory_address_a = arg_a.memory_address(fp).unwrap();
                    let res_value = memory.read_value(*res, fp);
                    let b_value = memory.read_value(*arg_b, fp);
                    let a_value = operation.inverse_compute(res_value, b_value);
                    memory.set(memory_address_a, a_value);
                } else if memory.is_value_unknown(*arg_b, fp) {
                    let memory_address_b = arg_b.memory_address(fp).unwrap();
                    let res_value = memory.read_value(*res, fp);
                    let a_value = memory.read_value(*arg_a, fp);
                    let b_value = operation.inverse_compute(res_value, a_value);
                    memory.set(memory_address_b, b_value);
                } else {
                    let a_value = memory.read_value(*arg_a, fp);
                    let b_value = memory.read_value(*arg_b, fp);
                    let res_value = memory.read_value(*res, fp);
                    assert_eq!(res_value, operation.compute(a_value, b_value));
                }

                pc += 1;
            }
            Instruction::MemoryPointerEq {
                shift_0,
                shift_1,
                res,
            } => {
                if memory.is_value_unknown(*res, fp) {
                    let memory_address_res = res.memory_address(fp).unwrap();
                    let ptr = memory.get(fp + shift_0);
                    let value = memory.get(ptr.as_canonical_u64() as usize + shift_1);
                    memory.set(memory_address_res, value);
                } else {
                    let value = memory.read_value(*res, fp);
                    let ptr = memory.get(fp + shift_0);
                    memory.set(ptr.as_canonical_u64() as usize + shift_1, value);
                }
                pc += 1;
            }
            Instruction::JumpIfNotZero {
                condition,
                dest,
                updated_fp,
            } => {
                if memory.read_value(*condition, fp) != F::ZERO {
                    pc = memory.read_value(*dest, fp).as_canonical_u64() as usize;
                    fp = memory.read_value(*updated_fp, fp).as_canonical_u64() as usize;
                } else {
                    pc += 1;
                }
            }
            Instruction::Poseidon2_16 { shift } => {
                let ptr_arg_0 = memory.get(fp + shift);
                let ptr_arg_1 = memory.get(fp + shift + 1);
                let ptr_res_0 = memory.get(fp + shift + 2);
                let ptr_res_1 = memory.get(fp + shift + 3);

                let arg0 = memory.get_vector(ptr_arg_0.as_canonical_u64() as usize);
                let arg1 = memory.get_vector(ptr_arg_1.as_canonical_u64() as usize);

                let mut input = [F::ZERO; DIMENSION * 2];
                input[..DIMENSION].copy_from_slice(&arg0);
                input[DIMENSION..].copy_from_slice(&arg1);

                poseidon_16.permute_mut(&mut input);

                let res0: [F; DIMENSION] = input[..DIMENSION].try_into().unwrap();
                let res1: [F; DIMENSION] = input[DIMENSION..].try_into().unwrap();

                memory.set_vector(ptr_res_0.as_canonical_u64() as usize, res0);
                memory.set_vector(ptr_res_1.as_canonical_u64() as usize, res1);

                pc += 1;
            }
            Instruction::Poseidon2_24 { shift } => {
                let ptr_arg_0 = memory.get(fp + shift);
                let ptr_arg_1 = memory.get(fp + shift + 1);
                let ptr_arg_2 = memory.get(fp + shift + 2);
                let ptr_res_0 = memory.get(fp + shift + 3);
                let ptr_res_1 = memory.get(fp + shift + 4);
                let ptr_res_2 = memory.get(fp + shift + 5);

                let arg0 = memory.get_vector(ptr_arg_0.as_canonical_u64() as usize);
                let arg1 = memory.get_vector(ptr_arg_1.as_canonical_u64() as usize);
                let arg2 = memory.get_vector(ptr_arg_2.as_canonical_u64() as usize);

                let mut input = [F::ZERO; DIMENSION * 3];
                input[..DIMENSION].copy_from_slice(&arg0);
                input[DIMENSION..2 * DIMENSION].copy_from_slice(&arg1);
                input[2 * DIMENSION..].copy_from_slice(&arg2);

                poseidon_24.permute_mut(&mut input);

                let res0: [F; DIMENSION] = input[..DIMENSION].try_into().unwrap();
                let res1: [F; DIMENSION] = input[DIMENSION..2 * DIMENSION].try_into().unwrap();
                let res2: [F; DIMENSION] = input[2 * DIMENSION..].try_into().unwrap();

                memory.set_vector(ptr_res_0.as_canonical_u64() as usize, res0);
                memory.set_vector(ptr_res_1.as_canonical_u64() as usize, res1);
                memory.set_vector(ptr_res_2.as_canonical_u64() as usize, res2);

                pc += 1;
            }
            Instruction::ExtComputation {
                operation,
                arg_a,
                arg_b,
                res,
            } => {
                // TODO if res and a (resp b) are known, but not b (resp a)

                let a_address = memory.read_value(*arg_a, fp);
                let b_address = memory.read_value(*arg_b, fp);
                let res_address = memory.read_value(*res, fp);

                let a = EF::from_basis_coefficients_slice(
                    memory
                        .get_vector(a_address.as_canonical_u64() as usize)
                        .as_slice(),
                )
                .unwrap();
                let b = EF::from_basis_coefficients_slice(
                    memory
                        .get_vector(b_address.as_canonical_u64() as usize)
                        .as_slice(),
                )
                .unwrap();

                let res = match operation {
                    Operation::Add => a + b,
                    Operation::Mul => a * b,
                };

                memory.set_vector(
                    res_address.as_canonical_u64() as usize,
                    res.as_basis_coefficients_slice().try_into().unwrap(),
                );
            }
        }
    }

    // TODO fill the bytecode into memory
}
