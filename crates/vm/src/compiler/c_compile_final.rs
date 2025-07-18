use p3_field::PrimeField64;
use std::collections::BTreeMap;

use crate::{
    F, FIELD_ELEMENTS_PER_OPCODE,
    bytecode::{
        bytecode::{
            Bytecode, Hint, Instruction, Label, MemOrConstant, MemOrFp, MemOrFpOrConstant,
            Operation,
        },
        intermediate_bytecode::{
            HighLevelOperation, IntermediaryMemOrFpOrConstant, IntermediateBytecode,
            IntermediateInstruction, IntermediateValue,
        },
    },
    lang::*,
};

impl IntermediateInstruction {
    fn is_meta(&self) -> bool {
        matches!(self, Self::RequestMemory { .. } | Self::Print { .. })
    }
}

struct Compiler {
    public_input_start: usize,
    pointer_to_zero_vector: usize,
    memory_size_per_function: BTreeMap<String, usize>,
    label_to_pc: BTreeMap<Label, usize>,
}

pub fn compile_to_low_level_bytecode(
    mut intermediate_bytecode: IntermediateBytecode,
) -> Result<Bytecode, String> {
    intermediate_bytecode.bytecode.insert(
        "@end_program".to_string(),
        vec![IntermediateInstruction::Jump {
            dest: IntermediateValue::label("@end_program".to_string()),
            updated_fp: None,
        }],
    );

    let starting_frame_memory = *intermediate_bytecode
        .memory_size_per_function
        .get("main")
        .unwrap();

    let bytecode_size = intermediate_bytecode
        .bytecode
        .iter()
        .map(|(_, instructions)| instructions.len())
        .sum::<usize>();
    let mut pointer_to_zero_vector = bytecode_size * FIELD_ELEMENTS_PER_OPCODE;
    // make it 8-aligned
    if pointer_to_zero_vector % 8 != 0 {
        pointer_to_zero_vector += 8 - (pointer_to_zero_vector % 8);
    }
    let public_input_start = pointer_to_zero_vector + 8;
    pointer_to_zero_vector /= 8;
    // ADD the zeros into the bytecode?

    let mut label_to_pc = BTreeMap::new();
    label_to_pc.insert("@function_main".to_string(), 0);
    let entrypoint = intermediate_bytecode
        .bytecode
        .remove("@function_main")
        .ok_or("No main function found in the compiled program")?;

    let mut code_chunks = vec![(0, entrypoint.clone())];
    let mut pc = entrypoint.iter().filter(|i| !i.is_meta()).count();
    for (label, instructions) in &intermediate_bytecode.bytecode {
        label_to_pc.insert(label.clone(), pc);
        code_chunks.push((pc, instructions.clone()));
        pc += instructions.iter().filter(|i| !i.is_meta()).count();
    }

    let ending_pc = label_to_pc.get("@end_program").cloned().unwrap();

    let mut low_level_bytecode = Vec::new();
    let mut hints = BTreeMap::new();

    let compiler = Compiler {
        public_input_start,
        pointer_to_zero_vector,
        memory_size_per_function: intermediate_bytecode.memory_size_per_function,
        label_to_pc,
    };

    let try_as_mem_or_constant = |value: &IntermediateValue| {
        if let Some(cst) = try_as_constant(value, &compiler) {
            return Some(MemOrConstant::Constant(cst));
        }
        if let IntermediateValue::MemoryAfterFp { shift } = value {
            return Some(MemOrConstant::MemoryAfterFp { shift: *shift });
        }
        return None;
    };

    let try_as_mem_or_fp = |value: &IntermediateValue| match value {
        IntermediateValue::MemoryAfterFp { shift } => {
            Some(MemOrFp::MemoryAfterFp { shift: *shift })
        }
        IntermediateValue::Fp => Some(MemOrFp::Fp),
        _ => None,
    };

    for (pc_start, chunk) in code_chunks {
        let mut pc = pc_start;
        for instruction in chunk {
            match instruction.clone() {
                IntermediateInstruction::Computation {
                    operation,
                    mut arg_a,
                    mut arg_b,
                    res,
                } => {
                    let operation: Operation = operation.try_into().unwrap();

                    if let Some(arg_a_cst) = try_as_constant(&arg_a, &compiler) {
                        if let Some(arg_b_cst) = try_as_constant(&arg_b, &compiler) {
                            // res = constant +/x constant

                            let op_res = operation
                                .compute(F::new(arg_a_cst as u32), F::new(arg_b_cst as u32));

                            let res: MemOrFp = res.try_into().unwrap();

                            low_level_bytecode.push(Instruction::Computation {
                                operation: Operation::Add,
                                arg_a: MemOrConstant::Constant(0),
                                arg_b: res,
                                res: MemOrConstant::Constant(op_res.as_canonical_u64() as usize),
                            });
                            pc += 1;
                            continue;
                        }
                    }

                    if arg_b.is_constant() {
                        std::mem::swap(&mut arg_a, &mut arg_b);
                    }

                    low_level_bytecode.push(Instruction::Computation {
                        operation,
                        arg_a: try_as_mem_or_constant(&arg_a).unwrap(),
                        arg_b: try_as_mem_or_fp(&arg_b).unwrap(),
                        res: try_as_mem_or_constant(&res).unwrap(),
                    });
                }
                IntermediateInstruction::Panic => {
                    low_level_bytecode.push(Instruction::Computation {
                        // fp x 0 = 1 is impossible, so we can use it to panic
                        operation: Operation::Mul,
                        arg_a: MemOrConstant::Constant(0),
                        arg_b: MemOrFp::Fp,
                        res: MemOrConstant::Constant(1),
                    });
                }
                IntermediateInstruction::Deref {
                    shift_0,
                    shift_1,
                    res,
                } => {
                    low_level_bytecode.push(Instruction::Deref {
                        shift_0,
                        shift_1,
                        res: match res {
                            IntermediaryMemOrFpOrConstant::MemoryAfterFp { shift } => {
                                MemOrFpOrConstant::MemoryAfterFp { shift }
                            }
                            IntermediaryMemOrFpOrConstant::Fp => MemOrFpOrConstant::Fp,
                            IntermediaryMemOrFpOrConstant::Constant(c) => {
                                MemOrFpOrConstant::Constant(convert_constant_expression(
                                    &c, &compiler,
                                ))
                            }
                        },
                    });
                }
                IntermediateInstruction::JumpIfNotZero {
                    condition,
                    dest,
                    updated_fp,
                } => {
                    let updated_fp = updated_fp
                        .map(|fp| try_as_mem_or_fp(&fp).unwrap())
                        .unwrap_or(MemOrFp::Fp);
                    low_level_bytecode.push(Instruction::JumpIfNotZero {
                        condition: try_as_mem_or_constant(&condition).unwrap(),
                        dest: try_as_mem_or_constant(&dest).unwrap(),
                        updated_fp,
                    });
                }
                IntermediateInstruction::Jump { dest, updated_fp } => {
                    low_level_bytecode.push(Instruction::JumpIfNotZero {
                        condition: MemOrConstant::Constant(1),
                        dest: try_as_mem_or_constant(&dest).unwrap(),
                        updated_fp: updated_fp
                            .map(|fp| try_as_mem_or_fp(&fp).unwrap())
                            .unwrap_or(MemOrFp::Fp),
                    });
                }
                IntermediateInstruction::Poseidon2_16 { shift } => {
                    low_level_bytecode.push(Instruction::Poseidon2_16 { shift });
                }
                IntermediateInstruction::Poseidon2_24 { shift } => {
                    low_level_bytecode.push(Instruction::Poseidon2_24 { shift });
                }
                IntermediateInstruction::RequestMemory {
                    shift,
                    size,
                    vectorized,
                } => {
                    let size = match size {
                        IntermediateValue::Constant(c) => {
                            MemOrConstant::Constant(convert_constant_expression(&c, &compiler))
                        }
                        IntermediateValue::MemoryAfterFp { shift } => {
                            MemOrConstant::MemoryAfterFp { shift }
                        }
                        IntermediateValue::Fp => unreachable!(),
                    };
                    let hint = Hint::RequestMemory {
                        shift,
                        vectorized,
                        size,
                    };
                    hints.entry(pc).or_insert_with(Vec::new).push(hint);
                }
                IntermediateInstruction::Print { line_info, content } => {
                    let hint = Hint::Print {
                        line_info: line_info.clone(),
                        content: content
                            .into_iter()
                            .map(|c| try_as_mem_or_constant(&c).unwrap())
                            .collect(),
                    };
                    hints.entry(pc).or_insert_with(Vec::new).push(hint);
                }
            }

            if !instruction.is_meta() {
                pc += 1;
            }
        }
    }

    return Ok(Bytecode {
        instructions: low_level_bytecode,
        hints,
        public_input_start,
        starting_frame_memory,
        ending_pc,
    });
}

fn convert_constant_value(constant: &ConstantValue, compiler: &Compiler) -> usize {
    match constant {
        ConstantValue::Scalar(scalar) => *scalar,
        ConstantValue::PublicInputStart => compiler.public_input_start,
        ConstantValue::PointerToZeroVector => compiler.pointer_to_zero_vector,
        ConstantValue::FunctionSize { function_name } => *compiler
            .memory_size_per_function
            .get(function_name)
            .expect(&format!(
                "Function {} not found in memory size map",
                function_name
            )),
        ConstantValue::Label(label) => compiler.label_to_pc.get(label).cloned().unwrap(),
    }
}

fn convert_constant_expression(constant: &ConstExpression, compiler: &Compiler) -> usize {
    match constant {
        ConstExpression::Value(value) => convert_constant_value(value, compiler),
        ConstExpression::Binary {
            left,
            operator,
            right,
        } => {
            let left = convert_constant_expression(left, compiler);
            let right = convert_constant_expression(right, compiler);
            match operator {
                HighLevelOperation::Add => left + right,
                HighLevelOperation::Sub => left - right,
                HighLevelOperation::Mul => left * right,
                HighLevelOperation::Div => {
                    assert!(right != 0, "Division by zero in constant expression");
                    left / right
                }
            }
        }
    }
}

fn try_as_constant(value: &IntermediateValue, compiler: &Compiler) -> Option<usize> {
    if let IntermediateValue::Constant(c) = value {
        Some(convert_constant_expression(c, compiler))
    } else {
        None
    }
}
