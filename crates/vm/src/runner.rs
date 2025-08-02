use p3_field::BasedVectorSpace;
use p3_field::PrimeCharacteristicRing;
use utils::pretty_integer;

use crate::Poseidon16;
use crate::Poseidon24;
use crate::bytecode::bytecode::Hint;
use crate::bytecode::bytecode::Instruction;
use crate::bytecode::bytecode::MemOrConstant;
use crate::bytecode::bytecode::MemOrFp;
use crate::bytecode::bytecode::MemOrFpOrConstant;
use crate::precompiles::PRECOMPILES;
use crate::precompiles::PrecompileName;
use crate::{DIMENSION, EF, F, bytecode::bytecode::Bytecode};
use p3_field::Field;
use p3_field::PrimeField64;
use p3_symmetric::Permutation;

const MAX_MEMORY_SIZE: usize = 1 << 26;

#[derive(Debug, Clone)]
enum RunnerError {
    OutOfMemory,
    MemoryAlreadySet,
    NotAPointer,
    DivByZero,
    NotEqual(F, F),
    UndefinedMemory,
    PCOutOfBounds,
}

impl ToString for RunnerError {
    fn to_string(&self) -> String {
        match self {
            RunnerError::OutOfMemory => "Out of memory".to_string(),
            RunnerError::MemoryAlreadySet => "Memory already set".to_string(),
            RunnerError::NotAPointer => "Not a pointer".to_string(),
            RunnerError::DivByZero => "Division by zero".to_string(),
            RunnerError::NotEqual(expected, actual) => {
                format!("Computation Invalid: {} != {}", expected, actual)
            }
            RunnerError::UndefinedMemory => "Undefined memory access".to_string(),
            RunnerError::PCOutOfBounds => "Program counter out of bounds".to_string(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Memory {
    data: Vec<Option<F>>,
    _extension: std::marker::PhantomData<EF>,
}

impl MemOrConstant {
    fn read_value(&self, memory: &Memory, fp: usize) -> Result<F, RunnerError> {
        match self {
            MemOrConstant::Constant(c) => Ok(*c),
            MemOrConstant::MemoryAfterFp { shift } => memory.get(fp + *shift),
        }
    }

    fn is_value_unknown(&self, memory: &Memory, fp: usize) -> bool {
        self.read_value(memory, fp).is_err()
    }

    fn memory_address(&self, fp: usize) -> Result<usize, RunnerError> {
        match self {
            MemOrConstant::Constant(_) => Err(RunnerError::NotAPointer),
            MemOrConstant::MemoryAfterFp { shift } => Ok(fp + *shift),
        }
    }
}

impl MemOrFp {
    fn read_value(&self, memory: &Memory, fp: usize) -> Result<F, RunnerError> {
        match self {
            MemOrFp::MemoryAfterFp { shift } => memory.get(fp + *shift),
            MemOrFp::Fp => Ok(F::from_usize(fp)),
        }
    }

    fn is_value_unknown(&self, memory: &Memory, fp: usize) -> bool {
        self.read_value(memory, fp).is_err()
    }

    fn memory_address(&self, fp: usize) -> Result<usize, RunnerError> {
        match self {
            MemOrFp::MemoryAfterFp { shift } => Ok(fp + *shift),
            MemOrFp::Fp => Err(RunnerError::NotAPointer),
        }
    }
}

impl MemOrFpOrConstant {
    fn read_value(&self, memory: &Memory, fp: usize) -> Result<F, RunnerError> {
        match self {
            MemOrFpOrConstant::MemoryAfterFp { shift } => memory.get(fp + *shift),
            MemOrFpOrConstant::Fp => Ok(F::from_usize(fp)),
            MemOrFpOrConstant::Constant(c) => Ok(*c),
        }
    }

    fn is_value_unknown(&self, memory: &Memory, fp: usize) -> bool {
        self.read_value(memory, fp).is_err()
    }

    fn memory_address(&self, fp: usize) -> Result<usize, RunnerError> {
        match self {
            MemOrFpOrConstant::MemoryAfterFp { shift } => Ok(fp + *shift),
            MemOrFpOrConstant::Fp => Err(RunnerError::NotAPointer),
            MemOrFpOrConstant::Constant(_) => Err(RunnerError::NotAPointer),
        }
    }
}

impl Memory {
    // fn is_set(&self, index: usize) -> bool {
    //     self.data.get(index).map_or(false, |opt| opt.is_some())
    // }

    fn get(&self, index: usize) -> Result<F, RunnerError> {
        self.data
            .get(index)
            .and_then(|opt| *opt)
            .ok_or(RunnerError::UndefinedMemory)
    }

    fn set(&mut self, index: usize, value: F) -> Result<(), RunnerError> {
        if index >= self.data.len() {
            if index >= MAX_MEMORY_SIZE {
                return Err(RunnerError::OutOfMemory);
            }
            self.data.resize(index + 1, None);
        }
        if let Some(existing) = &mut self.data[index] {
            if *existing != value {
                return Err(RunnerError::MemoryAlreadySet);
            }
        } else {
            self.data[index] = Some(value);
        }
        Ok(())
    }

    fn get_vector(&self, index: usize) -> Result<[F; DIMENSION], RunnerError> {
        let mut vector = [F::default(); DIMENSION];
        for i in 0..DIMENSION {
            vector[i] = self.get(index * DIMENSION + i)?;
        }
        Ok(vector)
    }

    fn set_vector(&mut self, index: usize, value: [F; DIMENSION]) -> Result<(), RunnerError> {
        for (i, v) in value.iter().enumerate() {
            let idx = DIMENSION * index + i;
            self.set(idx, *v)?;
        }
        Ok(())
    }
}

pub fn execute_bytecode(
    bytecode: &Bytecode,
    public_input: &[F],
    private_input: &[F],
    poseidon_16: &Poseidon16,
    poseidon_24: &Poseidon24,
) {
    let mut std_out = String::new();
    let no_vec_runtime_memory = match execute_bytecode_helper(
        bytecode,
        public_input,
        private_input,
        poseidon_16,
        poseidon_24,
        MAX_MEMORY_SIZE / 2,
        false,
        &mut std_out,
    ) {
        Ok(no_vec_runtime_memory) => no_vec_runtime_memory,
        Err(err) => {
            if !std_out.is_empty() {
                print!("{}", std_out);
            }
            panic!("Error during bytecode execution: {}", err.to_string());
        }
    };
    execute_bytecode_helper(
        bytecode,
        public_input,
        private_input,
        poseidon_16,
        poseidon_24,
        no_vec_runtime_memory,
        true,
        &mut String::new(),
    )
    .unwrap();
}

fn execute_bytecode_helper(
    bytecode: &Bytecode,
    public_input: &[F],
    private_input: &[F],
    poseidon_16: &Poseidon16,
    poseidon_24: &Poseidon24,
    no_vec_runtime_memory: usize,
    final_execution: bool,
    std_out: &mut String,
) -> Result<usize, RunnerError> {
    let mut memory = Memory::default();

    for _ in 0..8 {
        memory.data.push(Some(F::ZERO)); // For "pointer_to_zero_vector"
    }

    for (i, value) in public_input.iter().enumerate() {
        memory.set(bytecode.public_input_start + i, *value)?;
    }

    let mut fp = bytecode.public_input_start + public_input.len();
    if fp % 8 != 0 {
        fp += 8 - (fp % 8); // Align to 8 field elements
    }

    for (i, value) in private_input.iter().enumerate() {
        memory.set(fp + i, *value)?;
    }
    fp += private_input.len();
    fp = fp.next_multiple_of(DIMENSION);

    let initial_ap = fp + bytecode.starting_frame_memory;
    let initial_ap_vec =
        (initial_ap + no_vec_runtime_memory).next_multiple_of(DIMENSION) / DIMENSION;

    let mut pc = 0;
    let mut ap = initial_ap;
    let mut ap_vec = initial_ap_vec;

    let mut poseidon16_calls = 0;
    let mut poseidon24_calls = 0;
    let mut extension_mul_calls = 0;
    let mut extension_add_calls = 0;
    let mut cpu_cycles = 0;

    let mut last_checkpoint_cpu_cycles = 0;
    let mut checkpoint_ap = initial_ap;
    let mut checkpoint_ap_vec = ap_vec;

    while pc != bytecode.ending_pc {
        if pc >= bytecode.instructions.len() {
            return Err(RunnerError::PCOutOfBounds);
        }

        cpu_cycles += 1;

        for hint in bytecode.hints.get(&pc).unwrap_or(&vec![]) {
            match hint {
                Hint::RequestMemory {
                    shift,
                    size,
                    vectorized,
                } => {
                    let size = size.read_value(&memory, fp)?.as_canonical_u64() as usize;

                    if *vectorized {
                        // find the next multiple of 8
                        memory.set(fp + *shift, F::from_usize(ap_vec))?;
                        ap_vec += size;
                    } else {
                        memory.set(fp + *shift, F::from_usize(ap))?;
                        ap += size;
                    }
                    // does not increase PC
                }
                Hint::DecomposeBits {
                    res_offset,
                    to_decompose,
                } => {
                    let to_decompose_value =
                        to_decompose.read_value(&memory, fp)?.as_canonical_u64();
                    for i in 0..F::bits() {
                        let bit = if to_decompose_value & (1 << i) != 0 {
                            F::ONE
                        } else {
                            F::ZERO
                        };
                        memory.set(fp + *res_offset + i, bit)?;
                    }
                }
                Hint::Print { line_info, content } => {
                    let values = content
                        .iter()
                        .map(|value| Ok(value.read_value(&memory, fp)?.to_string()))
                        .collect::<Result<Vec<_>, _>>()?;
                    // Logs for performance analysis:
                    if values[0] == "123456789" {
                        if values.len() == 1 {
                            *std_out += &format!("[CHECKPOINT]\n");
                        } else {
                            assert_eq!(values.len(), 2);
                            let new_no_vec_memory = ap - checkpoint_ap;
                            let new_vec_memory = (ap_vec - checkpoint_ap_vec) * DIMENSION;
                            *std_out += &format!(
                                "[CHECKPOINT {}] new CPU cycles: {}, new runtime memory: {} ({:.1}% vec)\n",
                                values[1],
                                pretty_integer(cpu_cycles - last_checkpoint_cpu_cycles),
                                pretty_integer(new_no_vec_memory + new_vec_memory),
                                new_vec_memory as f64 / (new_no_vec_memory + new_vec_memory) as f64
                                    * 100.0
                            );
                        }

                        last_checkpoint_cpu_cycles = cpu_cycles;
                        checkpoint_ap = ap;
                        checkpoint_ap_vec = ap_vec;
                        continue;
                    }

                    let line_info = line_info.replace(";", "");
                    *std_out += &format!("\"{}\" -> {}\n", line_info, values.join(", "));
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
                    let memory_address_res = res.memory_address(fp)?;
                    let a_value = arg_a.read_value(&memory, fp)?;
                    let b_value = arg_b.read_value(&memory, fp)?;
                    let res_value = operation.compute(a_value, b_value);
                    memory.set(memory_address_res, res_value)?;
                } else if arg_a.is_value_unknown(&memory, fp) {
                    let memory_address_a = arg_a.memory_address(fp)?;
                    let res_value = res.read_value(&memory, fp)?;
                    let b_value = arg_b.read_value(&memory, fp)?;
                    let a_value = operation
                        .inverse_compute(res_value, b_value)
                        .ok_or(RunnerError::DivByZero)?;
                    memory.set(memory_address_a, a_value)?;
                } else if arg_b.is_value_unknown(&memory, fp) {
                    let memory_address_b = arg_b.memory_address(fp)?;
                    let res_value = res.read_value(&memory, fp)?;
                    let a_value = arg_a.read_value(&memory, fp)?;
                    let b_value = operation
                        .inverse_compute(res_value, a_value)
                        .ok_or(RunnerError::DivByZero)?;
                    memory.set(memory_address_b, b_value)?;
                } else {
                    let a_value = arg_a.read_value(&memory, fp)?;
                    let b_value = arg_b.read_value(&memory, fp)?;
                    let res_value = res.read_value(&memory, fp)?;
                    let computed_value = operation.compute(a_value, b_value);
                    if res_value != computed_value {
                        return Err(RunnerError::NotEqual(computed_value, res_value));
                    }
                }

                pc += 1;
            }
            Instruction::Deref {
                shift_0,
                shift_1,
                res,
            } => {
                if res.is_value_unknown(&memory, fp) {
                    let memory_address_res = res.memory_address(fp)?;
                    let ptr = memory.get(fp + shift_0)?;
                    let value = memory.get(ptr.as_canonical_u64() as usize + shift_1)?;
                    memory.set(memory_address_res, value)?;
                } else {
                    let value = res.read_value(&memory, fp)?;
                    let ptr = memory.get(fp + shift_0)?;
                    memory.set(ptr.as_canonical_u64() as usize + shift_1, value)?;
                }
                pc += 1;
            }
            Instruction::JumpIfNotZero {
                condition,
                dest,
                updated_fp,
            } => {
                if condition.read_value(&memory, fp)? != F::ZERO {
                    pc = dest.read_value(&memory, fp)?.as_canonical_u64() as usize;
                    fp = updated_fp.read_value(&memory, fp)?.as_canonical_u64() as usize;
                } else {
                    pc += 1;
                }
            }
            Instruction::Poseidon2_16 { arg_a, arg_b, res } => {
                poseidon16_calls += 1;

                let a_value = arg_a.read_value(&memory, fp)?;
                let b_value = arg_b.read_value(&memory, fp)?;
                let res_value = res.read_value(&memory, fp)?;

                let arg0 = memory.get_vector(a_value.as_canonical_u64() as usize)?;
                let arg1 = memory.get_vector(b_value.as_canonical_u64() as usize)?;

                let mut input = [F::ZERO; DIMENSION * 2];
                input[..DIMENSION].copy_from_slice(&arg0);
                input[DIMENSION..].copy_from_slice(&arg1);

                poseidon_16.permute_mut(&mut input);

                let res0: [F; DIMENSION] = input[..DIMENSION].try_into().unwrap();
                let res1: [F; DIMENSION] = input[DIMENSION..].try_into().unwrap();

                memory.set_vector(res_value.as_canonical_u64() as usize, res0)?;
                memory.set_vector(1 + res_value.as_canonical_u64() as usize, res1)?;

                pc += 1;
            }
            Instruction::Poseidon2_24 { arg_a, arg_b, res } => {
                poseidon24_calls += 1;

                let a_value = arg_a.read_value(&memory, fp)?;
                let b_value = arg_b.read_value(&memory, fp)?;
                let res_value = res.read_value(&memory, fp)?;

                let arg0 = memory.get_vector(a_value.as_canonical_u64() as usize)?;
                let arg1 = memory.get_vector(1 + a_value.as_canonical_u64() as usize)?;
                let arg2 = memory.get_vector(b_value.as_canonical_u64() as usize)?;

                let mut input = [F::ZERO; DIMENSION * 3];
                input[..DIMENSION].copy_from_slice(&arg0);
                input[DIMENSION..2 * DIMENSION].copy_from_slice(&arg1);
                input[2 * DIMENSION..].copy_from_slice(&arg2);

                poseidon_24.permute_mut(&mut input);

                let res: [F; DIMENSION] = input[2 * DIMENSION..].try_into().unwrap();

                memory.set_vector(res_value.as_canonical_u64() as usize, res)?;

                pc += 1;
            }
            Instruction::ExtensionMul { args } => {
                assert!(
                    PRECOMPILES
                        .iter()
                        .any(|p| p.name == PrecompileName::MulExtension)
                );

                extension_mul_calls += 1;

                let ptr_arg_0 = memory.get(fp + args[0])?.as_canonical_u64() as usize;
                let ptr_arg_1 = memory.get(fp + args[1])?.as_canonical_u64() as usize;
                let ptr_arg_2 = memory.get(fp + args[2])?.as_canonical_u64() as usize;

                let a = EF::from_basis_coefficients_slice(&memory.get_vector(ptr_arg_0)?).unwrap();
                let b = EF::from_basis_coefficients_slice(&memory.get_vector(ptr_arg_1)?).unwrap();
                let prod = (a * b).as_basis_coefficients_slice().try_into().unwrap();
                memory.set_vector(ptr_arg_2, prod)?;

                pc += 1;
            }
            Instruction::ExtensionAdd { args } => {
                assert!(
                    PRECOMPILES
                        .iter()
                        .any(|p| p.name == PrecompileName::AddExtension)
                );

                extension_add_calls += 1;

                let ptr_arg_0 = memory.get(fp + args[0])?.as_canonical_u64() as usize;
                let ptr_arg_1 = memory.get(fp + args[1])?.as_canonical_u64() as usize;
                let ptr_arg_2 = memory.get(fp + args[2])?.as_canonical_u64() as usize;

                let a = EF::from_basis_coefficients_slice(&memory.get_vector(ptr_arg_0)?).unwrap();
                let b = EF::from_basis_coefficients_slice(&memory.get_vector(ptr_arg_1)?).unwrap();
                let sum = (a + b).as_basis_coefficients_slice().try_into().unwrap();
                memory.set_vector(ptr_arg_2, sum)?;

                pc += 1;
            }
        }
    }

    if final_execution {
        if !std_out.is_empty() {
            print!("{}", std_out);
        }
        let runtime_memory_size =
            memory.data.len() - (bytecode.public_input_start + public_input.len());
        println!(
            "\nBytecode size: {}",
            pretty_integer(bytecode.instructions.len())
        );
        println!("Public input size: {}", pretty_integer(public_input.len()));
        println!(
            "Private input size: {}",
            pretty_integer(private_input.len())
        );
        println!("Executed {} instructions", pretty_integer(cpu_cycles));
        println!(
            "Runtime memory: {} ({:.2}% vec)",
            pretty_integer(runtime_memory_size),
            (DIMENSION * (ap_vec - initial_ap_vec)) as f64 / runtime_memory_size as f64 * 100.0
        );
        if poseidon16_calls + poseidon24_calls > 0 {
            println!(
                "Poseidon2_16 calls: {}, Poseidon2_24 calls: {} (1 poseidon per {} instructions)",
                pretty_integer(poseidon16_calls),
                pretty_integer(poseidon24_calls),
                cpu_cycles / (poseidon16_calls + poseidon24_calls)
            );
        }
        if extension_mul_calls > 0 {
            println!(
                "ExtensionMul calls: {}",
                pretty_integer(extension_mul_calls)
            );
        }
        if extension_add_calls > 0 {
            println!(
                "ExtensionAdd calls: {}",
                pretty_integer(extension_add_calls)
            );
        }
        let used_memory_cells = memory
            .data
            .iter()
            .skip(bytecode.public_input_start + public_input.len())
            .filter(|&&x| x.is_some())
            .count();
        println!(
            "Memory usage: {:.1}%",
            used_memory_cells as f64 / runtime_memory_size as f64 * 100.0
        );
    }

    Ok(ap - initial_ap)
}
