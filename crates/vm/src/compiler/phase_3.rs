use p3_field::PrimeField64;
use std::collections::BTreeMap;

use crate::{
    AIR_COLUMNS_PER_OPCODE, F,
    bytecode::{
        final_bytecode::{
            Bytecode, Hint, Instruction, MemOrConstant, MemOrFp, MemOrFpOrConstant, Operation,
        },
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
    // println!(
    //     "\nHigh level bytecode:\n\n{}\n",
    //     high_level_bytecode.to_string()
    // );

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
    let public_input_start = pointer_to_zero_vector + 8;
    pointer_to_zero_vector /= 8;
    // ADD the zeros into the bytecode?

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

    let convert_constant = |constant: ConstantValue| match constant {
        ConstantValue::Scalar(scalar) => scalar,
        ConstantValue::PublicInputStart => public_input_start,
        ConstantValue::PointerToZeroVector => pointer_to_zero_vector,
    };

    let try_as_constant = |value: &HighLevelValue| match value {
        HighLevelValue::Constant(c) => Some(convert_constant(*c)),
        HighLevelValue::Label(label) => Some(label_to_pc.get(label).cloned().unwrap()),
        _ => None,
    };

    let try_as_mem_or_constant = |value: &HighLevelValue| {
        if let Some(cst) = try_as_constant(value) {
            return Some(MemOrConstant::Constant(cst));
        }
        if let HighLevelValue::MemoryAfterFp { shift } = value {
            return Some(MemOrConstant::MemoryAfterFp { shift: *shift });
        }
        return None;
    };

    let try_as_mem_or_fp = |value: &HighLevelValue| match value {
        HighLevelValue::MemoryAfterFp { shift } => Some(MemOrFp::MemoryAfterFp { shift: *shift }),
        HighLevelValue::Fp => Some(MemOrFp::Fp),
        _ => None,
    };

    let try_as_mem_or_fp_or_constant = |value: &HighLevelValue| {
        if let Some(cst) = try_as_constant(value) {
            return Some(MemOrFpOrConstant::Constant(cst));
        }
        if let HighLevelValue::MemoryAfterFp { shift } = value {
            return Some(MemOrFpOrConstant::MemoryAfterFp { shift: *shift });
        }
        if let HighLevelValue::Fp = value {
            return Some(MemOrFpOrConstant::Fp);
        }
        None
    };

    for (pc_start, chunk) in code_chunks {
        let mut pc = pc_start;
        for instruction in chunk {
            match instruction.clone() {
                HighLevelInstruction::Computation {
                    operation,
                    mut arg_a,
                    mut arg_b,
                    res,
                } => {
                    let operation: Operation = operation.try_into().unwrap();

                    if let Some(arg_a_cst) = try_as_constant(&arg_a) {
                        if let Some(arg_b_cst) = try_as_constant(&arg_b) {
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
                            continue;
                        }
                    }

                    if arg_b.compiles_to_constant() {
                        std::mem::swap(&mut arg_a, &mut arg_b);
                    }

                    low_level_bytecode.push(Instruction::Computation {
                        operation,
                        arg_a: try_as_mem_or_constant(&arg_a).unwrap(),
                        arg_b: try_as_mem_or_fp(&arg_b).unwrap(),
                        res: try_as_mem_or_constant(&res).unwrap(),
                    });
                }
                HighLevelInstruction::Panic => {
                    low_level_bytecode.push(Instruction::Computation {
                        // fp x 0 = 1 is impossible, so we can use it to panic
                        operation: Operation::Mul,
                        arg_a: MemOrConstant::Constant(0),
                        arg_b: MemOrFp::Fp,
                        res: MemOrConstant::Constant(1),
                    });
                }
                HighLevelInstruction::Eq {
                    mut left,
                    mut right,
                } => {
                    // ShiftedMemoryPointer
                    assert!(!matches!(left, HighLevelValue::ShiftedMemoryPointer { .. }));
                    if let HighLevelValue::ShiftedMemoryPointer { shift_0, shift_1 } = right {
                        low_level_bytecode.push(Instruction::MemoryPointerEq {
                            shift_0,
                            shift_1,
                            res: try_as_mem_or_fp_or_constant(&dbg!(left)).unwrap(),
                        });
                        continue;
                    }

                    // Fp
                    if right == HighLevelValue::Fp {
                        assert!(left != HighLevelValue::Fp);
                        std::mem::swap(&mut left, &mut right);
                    }
                    if left == HighLevelValue::Fp {
                        low_level_bytecode.push(Instruction::Computation {
                            operation: Operation::Add,
                            arg_a: MemOrConstant::Constant(0),
                            arg_b: MemOrFp::Fp,
                            res: try_as_mem_or_constant(&right).unwrap(),
                        });
                        continue;
                    }

                    if left.compiles_to_constant() && right.compiles_to_constant() {
                        panic!("Weird ?")
                    }

                    if matches!(right, HighLevelValue::MemoryAfterFp { .. }) {
                        std::mem::swap(&mut left, &mut right);
                    }

                    assert!(matches!(left, HighLevelValue::MemoryAfterFp { .. }));

                    low_level_bytecode.push(Instruction::Computation {
                        operation: Operation::Add,
                        arg_a: MemOrConstant::Constant(0),
                        arg_b: try_as_mem_or_fp(&left).unwrap(),
                        res: try_as_mem_or_constant(&right).unwrap(),
                    });
                }
                HighLevelInstruction::MemoryPointerEq { shift_0, shift_1, res } => {
                    low_level_bytecode.push(Instruction::MemoryPointerEq {
                        shift_0,
                        shift_1,
                        res: MemOrFpOrConstant::MemoryAfterFp { shift: res },
                    });
                }   
                HighLevelInstruction::JumpIfNotZero {
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
                HighLevelInstruction::Jump { dest, updated_fp } => {
                    low_level_bytecode.push(Instruction::JumpIfNotZero {
                        condition: MemOrConstant::Constant(1),
                        dest: try_as_mem_or_constant(&dest).unwrap(),
                        updated_fp: updated_fp
                            .map(|fp| try_as_mem_or_fp(&fp).unwrap())
                            .unwrap_or(MemOrFp::Fp),
                    });
                }
                HighLevelInstruction::Poseidon2_16 { shift } => {
                    low_level_bytecode.push(Instruction::Poseidon2_16 { shift });
                }
                HighLevelInstruction::Poseidon2_24 { shift } => {
                    low_level_bytecode.push(Instruction::Poseidon2_24 { shift });
                }
                HighLevelInstruction::RequestMemory {
                    shift,
                    size,
                    vectorized,
                } => {
                    let hint = Hint::RequestMemory {
                        shift,
                        vectorized,
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
                    if let HighLevelValue::ShiftedMemoryPointer { .. } = left {
                        assert!(
                            !matches!(right, HighLevelValue::ShiftedMemoryPointer { .. }),
                            "Cannot compare two memory pointers directly"
                        );

                        std::mem::swap(left, right);
                    }
                }
                _ => {}
            }
        }
    }
}
