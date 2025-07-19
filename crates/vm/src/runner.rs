use p3_field::PrimeCharacteristicRing;

use crate::FIELD_ELEMENTS_PER_OPCODE;
use crate::Poseidon16;
use crate::Poseidon24;
use crate::bytecode::bytecode::Hint;
use crate::bytecode::bytecode::Instruction;
use crate::bytecode::bytecode::MemOrConstant;
use crate::bytecode::bytecode::MemOrFp;
use crate::bytecode::bytecode::MemOrFpOrConstant;
use crate::{DIMENSION, EF, F, bytecode::bytecode::Bytecode};
use p3_field::Field;
use p3_field::PrimeField64;
use p3_symmetric::Permutation;

const MAX_MEMORY_SIZE: usize = 1 << 20;

#[derive(Debug, Clone, Default)]
struct Memory {
    data: Vec<Option<F>>,
    _extension: std::marker::PhantomData<EF>,
}

impl MemOrConstant {
    fn try_read_value(&self, memory: &Memory, fp: usize) -> Option<F> {
        match self {
            MemOrConstant::Constant(c) => Some(*c),
            MemOrConstant::MemoryAfterFp { shift } => memory.try_get(fp + *shift),
        }
    }

    fn read_value(&self, memory: &Memory, fp: usize) -> F {
        self.try_read_value(memory, fp).expect(&format!(
            "Memory access error, value: {:?}, fp: {}",
            self, fp
        ))
    }

    fn is_value_unknown(&self, memory: &Memory, fp: usize) -> bool {
        self.try_read_value(memory, fp).is_none()
    }

    fn memory_address(&self, fp: usize) -> Option<usize> {
        match self {
            MemOrConstant::Constant(_) => None,
            MemOrConstant::MemoryAfterFp { shift } => Some(fp + *shift),
        }
    }
}

impl MemOrFp {
    fn try_read_value(&self, memory: &Memory, fp: usize) -> Option<F> {
        match self {
            MemOrFp::MemoryAfterFp { shift } => memory.try_get(fp + *shift),
            MemOrFp::Fp => Some(F::from_usize(fp)),
        }
    }

    fn read_value(&self, memory: &Memory, fp: usize) -> F {
        self.try_read_value(memory, fp).expect(&format!(
            "Memory access error, value: {:?}, fp: {}",
            self, fp
        ))
    }

    fn is_value_unknown(&self, memory: &Memory, fp: usize) -> bool {
        self.try_read_value(memory, fp).is_none()
    }

    fn memory_address(&self, fp: usize) -> Option<usize> {
        match self {
            MemOrFp::MemoryAfterFp { shift } => Some(fp + *shift),
            MemOrFp::Fp => None,
        }
    }
}

impl MemOrFpOrConstant {
    fn try_read_value(&self, memory: &Memory, fp: usize) -> Option<F> {
        match self {
            MemOrFpOrConstant::MemoryAfterFp { shift } => memory.try_get(fp + *shift),
            MemOrFpOrConstant::Fp => Some(F::from_usize(fp)),
            MemOrFpOrConstant::Constant(c) => Some(*c),
        }
    }

    fn read_value(&self, memory: &Memory, fp: usize) -> F {
        self.try_read_value(memory, fp).expect(&format!(
            "Memory access error, value: {:?}, fp: {}",
            self, fp
        ))
    }

    fn is_value_unknown(&self, memory: &Memory, fp: usize) -> bool {
        self.try_read_value(memory, fp).is_none()
    }

    fn memory_address(&self, fp: usize) -> Option<usize> {
        match self {
            MemOrFpOrConstant::MemoryAfterFp { shift } => Some(fp + *shift),
            MemOrFpOrConstant::Fp => None,
            MemOrFpOrConstant::Constant(_) => None,
        }
    }
}

impl Memory {
    // fn is_set(&self, index: usize) -> bool {
    //     self.data.get(index).map_or(false, |opt| opt.is_some())
    // }

    fn try_get(&self, index: usize) -> Option<F> {
        self.data.get(index).and_then(|opt| *opt)
    }

    fn get(&self, index: usize) -> F {
        self.data
            .get(index)
            .unwrap_or_else(|| {
                panic!(
                    "Memory access error: index {} out of bounds (max: {})",
                    index,
                    self.data.len()
                )
            })
            .unwrap_or_else(|| panic!("Memory access error: index {} is None", index,))
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

pub fn execute_bytecode(
    bytecode: &Bytecode,
    public_input: &[F],
    private_input: &[F],
    poseidon_16: Poseidon16,
    poseidon_24: Poseidon24,
) {
    let mut memory = Memory::default();

    // TODO place the bytecode into memory
    // For now we will it with zeros
    for _ in 0..bytecode.instructions.len() * FIELD_ELEMENTS_PER_OPCODE {
        memory.data.push(Some(F::ZERO));
    }

    for _ in bytecode.instructions.len() * FIELD_ELEMENTS_PER_OPCODE..bytecode.public_input_start {
        memory.data.push(Some(F::ZERO)); // For "pointer_to_zero_vector"
    }

    for (i, value) in public_input.iter().enumerate() {
        memory.set(bytecode.public_input_start + i, *value);
    }

    let mut fp = bytecode.public_input_start + public_input.len();
    if fp % 8 != 0 {
        fp += 8 - (fp % 8); // Align to 8 field elements
    }

    for (i, value) in private_input.iter().enumerate() {
        memory.set(fp + i, *value);
    }
    fp += private_input.len();
    if fp % 8 != 0 {
        fp += 8 - (fp % 8); // Align to 8 field elements
    }

    let mut pc = 0;
    let mut ap = fp + bytecode.starting_frame_memory;

    let mut poseidon16_calls = 0;
    let mut poseidon24_calls = 0;
    let mut instructions_run = 0;

    while pc != bytecode.ending_pc {
        if pc >= bytecode.instructions.len() {
            panic!(
                "Program counter out of bounds: {} >= {}",
                pc,
                bytecode.instructions.len()
            );
        }

        // dbg!(pc, fp);

        instructions_run += 1;

        for hint in bytecode.hints.get(&pc).unwrap_or(&vec![]) {
            match hint {
                Hint::RequestMemory {
                    shift,
                    size,
                    vectorized,
                } => {
                    malloc(&mut memory, &mut ap, fp, *shift, *size, *vectorized);
                    // does not increase PC
                }
                Hint::DecomposeBits {
                    res_offset: result_offset,
                    to_decompose,
                } => {
                    let size = MemOrConstant::Constant(F::from_usize(F::bits())); // 31 for KoalaBear
                    malloc(&mut memory, &mut ap, fp, *result_offset, size, false);
                    let start = memory.get(fp + *result_offset).as_canonical_u64() as usize;
                    let to_decompose_value =
                        to_decompose.read_value(&memory, fp).as_canonical_u64();
                    for i in 0..F::bits() {
                        let bit = if to_decompose_value & (1 << i) != 0 {
                            F::ONE
                        } else {
                            F::ZERO
                        };
                        memory.set(start + i, bit);
                    }
                }
                Hint::Print { line_info, content } => {
                    let values = content
                        .iter()
                        .map(|value| value.read_value(&memory, fp).to_string())
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
                if res.is_value_unknown(&memory, fp) {
                    let memory_address_res = res.memory_address(fp).unwrap();
                    let a_value = arg_a.read_value(&memory, fp);
                    let b_value = arg_b.read_value(&memory, fp);
                    let res_value = operation.compute(a_value, b_value);
                    memory.set(memory_address_res, res_value);
                } else if arg_a.is_value_unknown(&memory, fp) {
                    let memory_address_a = arg_a.memory_address(fp).unwrap();
                    let res_value = res.read_value(&memory, fp);
                    let b_value = arg_b.read_value(&memory, fp);
                    let a_value = operation.inverse_compute(res_value, b_value);
                    memory.set(memory_address_a, a_value);
                } else if arg_b.is_value_unknown(&memory, fp) {
                    let memory_address_b = arg_b.memory_address(fp).unwrap();
                    let res_value = res.read_value(&memory, fp);
                    let a_value = arg_a.read_value(&memory, fp);
                    let b_value = operation.inverse_compute(res_value, a_value);
                    memory.set(memory_address_b, b_value);
                } else {
                    let a_value = arg_a.read_value(&memory, fp);
                    let b_value = arg_b.read_value(&memory, fp);
                    let res_value = res.read_value(&memory, fp);
                    assert_eq!(res_value, operation.compute(a_value, b_value));
                }

                pc += 1;
            }
            Instruction::Deref {
                shift_0,
                shift_1,
                res,
            } => {
                if res.is_value_unknown(&memory, fp) {
                    let memory_address_res = res.memory_address(fp).unwrap();
                    let ptr = memory.get(fp + shift_0);
                    let value = memory.get(ptr.as_canonical_u64() as usize + shift_1);
                    memory.set(memory_address_res, value);
                } else {
                    let value = res.read_value(&memory, fp);
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
                if condition.read_value(&memory, fp) != F::ZERO {
                    pc = dest.read_value(&memory, fp).as_canonical_u64() as usize;
                    fp = updated_fp.read_value(&memory, fp).as_canonical_u64() as usize;
                } else {
                    pc += 1;
                }
            }
            Instruction::Poseidon2_16 { shift } => {
                poseidon16_calls += 1;

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
                poseidon24_calls += 1;

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
        }
    }

    if poseidon16_calls + poseidon24_calls > 0 {
        println!(
            "\nExecuted {} instructions, Poseidon2_16 calls: {}, Poseidon2_24 calls: {} (1 poseidon per {} instructions)",
            instructions_run,
            poseidon16_calls,
            poseidon24_calls,
            instructions_run / (poseidon16_calls + poseidon24_calls)
        );
        println!(
            "Final memory size: {} ({} cells per poseidon)",
            memory.data.len(),
            memory.data.len() / (poseidon16_calls + poseidon24_calls)
        );
    }
    // TODO fill the bytecode into memory
}

fn malloc(
    memory: &mut Memory,
    ap: &mut usize,
    fp: usize,
    res_offset: usize,
    size: MemOrConstant,
    vectorized: bool,
) {
    // TODO avoid memory fragmentation (easy perf boost in perspective)

    let size = size.read_value(&memory, fp).as_canonical_u64() as usize;

    if vectorized {
        // find the next multiple of 8
        let ap_next_multiple_of_8 = (*ap + 7) / 8 * 8;
        memory.set(fp + res_offset, F::from_usize(ap_next_multiple_of_8 / 8));
        *ap = ap_next_multiple_of_8 + size * 8;
    } else {
        memory.set(fp + res_offset, F::from_usize(*ap));
        *ap += size;
    }
}
