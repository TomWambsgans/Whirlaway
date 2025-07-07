use std::collections::BTreeMap;

use crate::{
    bytecode::{
        high_level::*,
        low_level::{Bytecode, Instruction, Operation, Value},
    },
    compiler::phase_2::compile_to_hight_level_bytecode,
    lang::*,
};

const AIR_COLUMNS_PER_OPCODE: usize = 10; // TODO
const PROGRAM_ENDING_ZEROS: usize = 8; // Every program ends with at least 8 zeros, useful for creating an "empty" pointer

pub fn compile_to_low_level_bytecode(program: Program) -> Result<Bytecode, String> {
    let mut high_level_bytecode = compile_to_hight_level_bytecode(program)?;
    clean(&mut high_level_bytecode);

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
    let mut pc = entrypoint.len();
    let mut code_chunks = vec![entrypoint.clone()];
    for (label, instructions) in &high_level_bytecode.bytecode {
        label_to_pc.insert(label.clone(), pc);
        code_chunks.push(instructions.clone());
        pc += instructions.len();
    }

    let mut low_level_bytecode = Vec::new();

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

    for chunk in code_chunks {
        for instruction in chunk {
            match instruction {
                HighLevelInstruction::Computation {
                    operation,
                    arg_a,
                    arg_b,
                    res,
                } => {
                    low_level_bytecode.push(Instruction::Computation {
                        operation: match operation {
                            HighLevelOperation::Add => crate::bytecode::low_level::Operation::Add,
                            HighLevelOperation::Mul => crate::bytecode::low_level::Operation::Mul,
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
                HighLevelInstruction::JumpIfNotZero { condition, dest } => {
                    low_level_bytecode.push(Instruction::JumpIfNotZero {
                        condition: convert_value(condition).unwrap(),
                        dest: convert_value(dest).unwrap(),
                        updated_fp: Value::Fp,
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
                HighLevelInstruction::FpAssign { value } => {
                    low_level_bytecode.push(Instruction::Computation {
                        operation: Operation::Add,
                        arg_a: convert_value(value).unwrap(),
                        arg_b: Value::Constant(0),
                        res: Value::Fp,
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
                    low_level_bytecode.push(Instruction::RequestMemory {
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
                    });
                }
            }
        }
    }

    return Ok(Bytecode(low_level_bytecode));
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
