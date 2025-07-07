use std::collections::BTreeMap;

use crate::{
    AIR_COLUMNS_PER_OPCODE, PROGRAM_ENDING_ZEROS,
    bytecode::{
        final_bytecode::{Bytecode, Hint, Instruction, Operation, Value},
        intermediate_bytecode::*,
    },
    compiler::phase_2::compile_to_hight_level_bytecode,
    lang::*,
};

impl HighLevelInstruction {
    fn is_meta(&self) -> bool {
        matches!(
            self,
            HighLevelInstruction::RequestMemory { .. } | HighLevelInstruction::Print { .. }
        )
    }
}

pub fn compile_to_low_level_bytecode(program: Program) -> Result<Bytecode, String> {
    let mut high_level_bytecode = compile_to_hight_level_bytecode(program)?;
    clean(&mut high_level_bytecode);
    // println!("\nHigh level bytecode:\n\n{}\n", high_level_bytecode.to_string());

    high_level_bytecode.bytecode.insert(
        "@end_program".to_string(),
        vec![HighLevelInstruction::Jump {
            dest: HighLevelValue::Label("@end_program".to_string()),
            updated_fp: None,
        }],
    );

    let bytecode_size = high_level_bytecode
        .bytecode
        .iter()
        .map(|(_, instructions)| instructions.len())
        .sum::<usize>();
    let mut pointer_to_zero_vector = bytecode_size * AIR_COLUMNS_PER_OPCODE;
    // make it 8-aligned
    if pointer_to_zero_vector % 8 != 0 {
        pointer_to_zero_vector += 8 - (pointer_to_zero_vector % 8);
    }
    let public_input_start = pointer_to_zero_vector + PROGRAM_ENDING_ZEROS;

    let mut label_to_pc = BTreeMap::new();
    label_to_pc.insert("@function_main".to_string(), 0);
    let entrypoint = high_level_bytecode
        .bytecode
        .remove("@function_main")
        .ok_or("No main function found in the compiled program")?;

    let mut code_chunks = vec![(0, entrypoint.clone())];
    let mut pc = entrypoint.iter().filter(|i| !i.is_meta()).count();
    for (label, instructions) in &high_level_bytecode.bytecode {
        label_to_pc.insert(label.clone(), pc);
        code_chunks.push((pc, instructions.clone()));
        pc += instructions.iter().filter(|i| !i.is_meta()).count();
    }

    let ending_pc = label_to_pc.get("@end_program").cloned().unwrap();

    let mut low_level_bytecode = Vec::new();
    let mut hints = BTreeMap::new();

    let convert_value = |value: HighLevelValue| match value {
        HighLevelValue::Constant(c) => Ok(Value::Constant(c)),
        HighLevelValue::Fp => Ok(Value::Fp),
        HighLevelValue::MemoryAfterFp { shift } => Ok(Value::MemoryAfterFp { shift }),
        HighLevelValue::DirectMemory { shift } => Ok(Value::DirectMemory { shift }),
        HighLevelValue::ShiftedMemoryPointer { .. } | HighLevelValue::MemoryPointer { .. } => {
            Err("Memory Pointer should only be used in AssertEq".to_string())
        }
        HighLevelValue::PointerToZeroVector => Ok(Value::MemoryAfterFp {
            shift: pointer_to_zero_vector / 8,
        }),
        HighLevelValue::Label(label) => Ok(Value::Constant(
            label_to_pc
                .get(&label)
                .cloned()
                .expect("Label not found in the bytecode"),
        )),
        HighLevelValue::PublicInputStart => Ok(Value::Constant(public_input_start)),
    };

    for (pc_start, chunk) in code_chunks {
        let mut pc = pc_start;
        for instruction in chunk {
            match instruction.clone() {
                HighLevelInstruction::Computation {
                    operation,
                    arg_a,
                    arg_b,
                    res,
                } => {
                    low_level_bytecode.push(Instruction::Computation {
                        operation: match operation {
                            HighLevelOperation::Add => {
                                crate::bytecode::final_bytecode::Operation::Add
                            }
                            HighLevelOperation::Mul => {
                                crate::bytecode::final_bytecode::Operation::Mul
                            }
                            _ => {
                                unreachable!("Handled earlier in \"clean\"")
                            }
                        },
                        arg_a: convert_value(arg_a).unwrap(),
                        arg_b: convert_value(arg_b).unwrap(),
                        res: convert_value(res).unwrap(),
                    });
                }
                HighLevelInstruction::Eq { left, right } => match right {
                    HighLevelValue::ShiftedMemoryPointer { shift_0, shift_1 } => {
                        low_level_bytecode.push(Instruction::MemoryPointerEq {
                            shift_0,
                            shift_1,
                            res: convert_value(left.clone()).unwrap(),
                        });
                    }
                    HighLevelValue::MemoryPointer { shift } => {
                        low_level_bytecode.push(Instruction::MemoryPointerEq {
                            shift_0: shift,
                            shift_1: 0,
                            res: convert_value(left.clone()).unwrap(),
                        });
                    }
                    _ => {
                        let left = convert_value(left).unwrap();
                        let right = convert_value(right).unwrap();
                        low_level_bytecode.push(Instruction::Computation {
                            operation: Operation::Add,
                            arg_a: left,
                            arg_b: Value::Constant(0),
                            res: right,
                        });
                    }
                },
                HighLevelInstruction::JumpIfNotZero { condition, dest, updated_fp } => {
                    let updated_fp = updated_fp
                        .map(|fp| convert_value(fp).unwrap())
                        .unwrap_or(Value::Fp);
                    low_level_bytecode.push(Instruction::JumpIfNotZero {
                        condition: convert_value(condition).unwrap(),
                        dest: convert_value(dest).unwrap(),
                        updated_fp,
                    });
                }
                HighLevelInstruction::Jump { dest, updated_fp } => {
                    low_level_bytecode.push(Instruction::JumpIfNotZero {
                        condition: Value::Constant(1),
                        dest: convert_value(dest).unwrap(),
                        updated_fp: updated_fp
                            .map(|fp| convert_value(fp).unwrap())
                            .unwrap_or(Value::Fp),
                    });
                }
                HighLevelInstruction::Poseidon2_16 { shift } => {
                    low_level_bytecode.push(Instruction::Poseidon2_16 { shift });
                }
                HighLevelInstruction::Poseidon2_24 { shift } => {
                    low_level_bytecode.push(Instruction::Poseidon2_24 { shift });
                }
                HighLevelInstruction::ExtComputation {
                    operation,
                    arg_a,
                    arg_b,
                    res,
                } => {
                    low_level_bytecode.push(Instruction::ExtComputation {
                        operation: match operation {
                            HighLevelOperation::Add => Operation::Add,
                            HighLevelOperation::Mul => Operation::Mul,
                            _ => {
                                return Err("Only Add and Mul operations are supported".to_string());
                            }
                        },
                        arg_a: convert_value(arg_a).unwrap(),
                        arg_b: convert_value(arg_b).unwrap(),
                        res: convert_value(res).unwrap(),
                    });
                }
                HighLevelInstruction::RequestMemory { shift, size } => {
                    let hint = Hint::RequestMemory {
                        shift,
                        size: match size {
                            HighLevelMetaValue::Constant(len) => len,
                            HighLevelMetaValue::FunctionSize { function_name } => {
                                *high_level_bytecode
                                    .memory_size_per_function
                                    .get(&function_name)
                                    .unwrap()
                            }
                        },
                    };
                    hints.entry(pc).or_insert_with(Vec::new).push(hint);
                }
                HighLevelInstruction::Print { line_info, content } => {
                    let hint = Hint::Print {
                        line_info: line_info.clone(),
                        content: content
                            .into_iter()
                            .map(|c| convert_value(c).unwrap())
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

    let starting_frame_memory = *high_level_bytecode
        .memory_size_per_function
        .get("main")
        .unwrap();

    return Ok(Bytecode {
        instructions: low_level_bytecode,
        hints,
        public_input_start,
        starting_frame_memory,
        ending_pc,
    });
}

fn clean(bytecode: &mut HighLevelBytecode) {
    for (_, instructions) in &mut bytecode.bytecode {
        for instruction in instructions {
            match instruction {
                HighLevelInstruction::Computation {
                    operation,
                    arg_a,
                    arg_b,
                    res,
                } => {
                    if *operation == HighLevelOperation::Div {
                        (*operation, *res, *arg_a, *arg_b) = (
                            HighLevelOperation::Mul,
                            arg_a.clone(),
                            res.clone(),
                            arg_b.clone(),
                        );
                    }
                    if *operation == HighLevelOperation::Sub {
                        (*operation, *res, *arg_a, *arg_b) = (
                            HighLevelOperation::Add,
                            arg_a.clone(),
                            res.clone(),
                            arg_b.clone(),
                        );
                    }
                }
                HighLevelInstruction::Eq { left, right } => {
                    if let HighLevelValue::ShiftedMemoryPointer { .. }
                    | HighLevelValue::MemoryPointer { .. } = left
                    {
                        assert!(
                            !matches!(right, HighLevelValue::ShiftedMemoryPointer { .. }),
                            "Cannot compare two memory pointers directly"
                        );
                        assert!(
                            !matches!(right, HighLevelValue::MemoryPointer { .. }),
                            "Cannot compare memory pointer with shifted memory pointer"
                        );
                        std::mem::swap(left, right);
                    }
                }
                _ => {}
            }
        }
    }
}
