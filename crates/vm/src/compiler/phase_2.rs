use crate::{
    bytecode::intermediate_bytecode::*,
    compiler::{
        phase_0::{
            remove_memory_raw_accesses_with_constants, replace_array_access, replace_assert_not_eq,
            replace_if_eq,
        },
        phase_1::{find_variable_usage, replace_loops_with_recursion},
    },
    lang::*,
};
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
};

struct Compiler {
    bytecode: BTreeMap<Label, Vec<HighLevelInstruction>>,
    if_counter: usize,
    call_counter: usize,
    func_name: String,
    var_positions: BTreeMap<Var, usize>, // var -> memory offset from fp
    args_count: usize,
    stack_size: usize,
}

impl Compiler {
    fn new() -> Self {
        Self {
            var_positions: BTreeMap::new(),
            stack_size: 0,
            bytecode: BTreeMap::new(),
            func_name: String::new(),
            args_count: 0,
            if_counter: 0,
            call_counter: 0,
        }
    }

    fn get_var_offset(&self, var: &Var) -> usize {
        self.var_positions
            .get(var)
            .unwrap_or_else(|| panic!("Variable {} not in scope", var.name))
            .clone()
    }
}

impl HighLevelValue {
    fn from_var_or_const(var_or_const: &VarOrConstant, compiler: &Compiler) -> Self {
        match var_or_const {
            VarOrConstant::Var(var) => Self::MemoryAfterFp {
                shift: compiler.get_var_offset(var),
            },
            VarOrConstant::Constant(c) => Self::Constant(*c),
        }
    }

    fn from_var(var: &Var, compiler: &Compiler) -> Self {
        Self::MemoryAfterFp {
            shift: compiler.get_var_offset(var),
        }
    }
}

pub fn compile_to_hight_level_bytecode(mut program: Program) -> Result<HighLevelBytecode, String> {
    replace_assert_not_eq(&mut program);
    replace_array_access(&mut program);
    replace_loops_with_recursion(&mut program);
    replace_if_eq(&mut program);
    remove_memory_raw_accesses_with_constants(&mut program);

    let mut compiler = Compiler::new();
    let mut memory_sizes = BTreeMap::new();

    for function in program.functions.values() {
        let instructions = compile_function(function, &mut compiler)?;
        compiler
            .bytecode
            .insert(format!("@function_{}", function.name), instructions);
        memory_sizes.insert(function.name.clone(), compiler.stack_size);
    }

    Ok(HighLevelBytecode {
        bytecode: compiler.bytecode,
        memory_size_per_function: memory_sizes,
    })
}

fn compile_function(
    function: &Function,
    compiler: &mut Compiler,
) -> Result<Vec<HighLevelInstruction>, String> {
    let (mut internal_vars, external_vars) = find_variable_usage(&function.instructions);

    internal_vars.retain(|var| !function.arguments.contains(var));

    for var in &external_vars {
        if !function.arguments.contains(var) {
            return Err(format!(
                "Variable {} not in function {} arguments",
                var.name, function.name
            ));
        }
    }

    // memory layout: pc, fp, args, return_vars, internal_vars
    let mut stack_pos = 2; // Reserve space for pc and fp
    let mut var_positions = BTreeMap::new();

    for (i, var) in function.arguments.iter().enumerate() {
        var_positions.insert(var.clone(), stack_pos + i);
    }
    stack_pos += function.arguments.len();

    stack_pos += function.n_returned_vars;

    for (i, var) in internal_vars.iter().enumerate() {
        var_positions.insert(var.clone(), stack_pos + i);
    }
    stack_pos += internal_vars.len();

    compiler.func_name = function.name.clone();
    compiler.var_positions = var_positions;
    compiler.stack_size = stack_pos;
    compiler.args_count = function.arguments.len();

    let mut declared_vars: BTreeSet<Var> = function.arguments.iter().cloned().collect();
    compile_lines(&function.instructions, compiler, None, &mut declared_vars)
}

fn compile_lines(
    lines: &[Line],
    compiler: &mut Compiler,
    final_jump: Option<Label>,
    declared_vars: &mut BTreeSet<Var>,
) -> Result<Vec<HighLevelInstruction>, String> {
    let mut instructions = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        match line {
            Line::Assignment {
                var,
                operation,
                arg0,
                arg1,
            } => {
                instructions.push(HighLevelInstruction::Computation {
                    operation: *operation,
                    arg_a: HighLevelValue::from_var_or_const(arg0, compiler),
                    arg_b: HighLevelValue::from_var_or_const(arg1, compiler),
                    res: HighLevelValue::from_var(var, compiler),
                });
                mark_vars_as_declared(&[arg0, arg1], declared_vars);
                declared_vars.insert(var.clone());
            }

            Line::Assert(Boolean::Equal { left, right }) => {
                instructions.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::from_var_or_const(left, compiler),
                    right: HighLevelValue::from_var_or_const(right, compiler),
                });
                mark_vars_as_declared(&[left, right], declared_vars);
            }

            Line::IfCondition {
                condition: Boolean::Different { left, right },
                then_branch,
                else_branch,
            } => {
                validate_vars_declared(&[left, right], declared_vars)?;

                let temp_pos = compiler.stack_size;
                compiler.stack_size += 1;

                let if_id = compiler.if_counter;
                compiler.if_counter += 1;

                let (if_label, else_label, end_label) = (
                    format!("@if_{}", if_id),
                    format!("@else_{}", if_id),
                    format!("@if_else_end_{}", if_id),
                );

                instructions.push(HighLevelInstruction::Computation {
                    operation: HighLevelOperation::Sub,
                    arg_a: HighLevelValue::from_var_or_const(left, compiler),
                    arg_b: HighLevelValue::from_var_or_const(right, compiler),
                    res: HighLevelValue::MemoryAfterFp { shift: temp_pos },
                });
                instructions.push(HighLevelInstruction::JumpIfNotZero {
                    condition: HighLevelValue::MemoryAfterFp { shift: temp_pos },
                    dest: HighLevelValue::Label(if_label.clone()),
                    updated_fp: None,
                });
                instructions.push(HighLevelInstruction::Jump {
                    dest: HighLevelValue::Label(else_label.clone()),
                    updated_fp: None,
                });

                let original_stack = compiler.stack_size;
                let if_instructions =
                    compile_branch(then_branch, compiler, &end_label, declared_vars)?;
                let if_stack = compiler.stack_size;

                compiler.stack_size = original_stack;
                let else_instructions =
                    compile_branch(else_branch, compiler, &end_label, declared_vars)?;
                let else_stack = compiler.stack_size;

                compiler.stack_size = if_stack.max(else_stack);

                compiler.bytecode.insert(if_label, if_instructions);
                compiler.bytecode.insert(else_label, else_instructions);

                let remaining =
                    compile_lines(&lines[i + 1..], compiler, final_jump, declared_vars)?;
                compiler.bytecode.insert(end_label, remaining);

                return Ok(instructions);
            }

            Line::RawAccess { var, index } => {
                let VarOrConstant::Var(index_var) = index else {
                    unreachable!("Constants should be handled earlier");
                };
                instructions.push(HighLevelInstruction::MemoryPointerEq {
                    shift_0: compiler.get_var_offset(index_var),
                    shift_1: 0,
                    res: compiler.get_var_offset(var),
                });
                declared_vars.insert(index_var.clone());
                declared_vars.insert(var.clone());
            }

            Line::FunctionCall {
                function_name,
                args,
                return_data,
            } => {
                let call_id = compiler.call_counter;
                compiler.call_counter += 1;
                let return_label = format!("@return_from_call_{}", call_id);

                let new_fp_pos = compiler.stack_size;
                compiler.stack_size += 1;

                instructions.extend(setup_function_call(
                    function_name,
                    args,
                    new_fp_pos,
                    &return_label,
                    compiler,
                )?);

                mark_vars_as_declared(args, declared_vars);
                declared_vars.extend(return_data.iter().cloned());

                let after_call = compile_function_return(
                    &lines[i + 1..],
                    return_data,
                    args.len(),
                    new_fp_pos,
                    compiler,
                    final_jump,
                    declared_vars,
                )?;
                compiler.bytecode.insert(return_label, after_call);

                return Ok(instructions);
            }

            Line::Poseidon16 {
                arg0,
                arg1,
                res0,
                res1,
            } => {
                compile_poseidon(
                    &mut instructions,
                    &[arg0, arg1],
                    &[res0, res1],
                    4,
                    compiler,
                    declared_vars,
                )?;
                instructions.push(HighLevelInstruction::Poseidon2_16 {
                    shift: compiler.stack_size - 4,
                });
            }

            Line::Poseidon24 {
                arg0,
                arg1,
                arg2,
                res0,
                res1,
                res2,
            } => {
                compile_poseidon(
                    &mut instructions,
                    &[arg0, arg1, arg2],
                    &[res0, res1, res2],
                    6,
                    compiler,
                    declared_vars,
                )?;
                instructions.push(HighLevelInstruction::Poseidon2_24 {
                    shift: compiler.stack_size - 6,
                });
            }

            Line::MAlloc { var, size } => {
                let ConstantValue::Scalar(size_val) = size else {
                    return Err("Invalid memory allocation size".to_string());
                };
                instructions.push(HighLevelInstruction::RequestMemory {
                    shift: compiler.get_var_offset(var),
                    size: HighLevelMetaValue::Constant(*size_val),
                    vectorized: false,
                });
            }

            Line::FunctionRet { return_data } => {
                if compiler.func_name == "main" {
                    instructions.push(HighLevelInstruction::Jump {
                        dest: HighLevelValue::Label("@end_program".to_string()),
                        updated_fp: None,
                    });
                } else {
                    compile_function_ret(&mut instructions, return_data, compiler);
                }
            }

            Line::Panic => instructions.push(HighLevelInstruction::Panic),

            Line::Print { line_info, content } => {
                instructions.push(HighLevelInstruction::Print {
                    line_info: line_info.clone(),
                    content: content
                        .iter()
                        .map(|c| HighLevelValue::from_var_or_const(c, compiler))
                        .collect(),
                });
            }

            Line::Assert(Boolean::Different { .. }) => {
                unreachable!("Assert not equal should be handled earlier")
            }
            Line::IfCondition {
                condition: Boolean::Equal { .. },
                ..
            } => unreachable!("If condition with equality should be handled earlier"),
            Line::ForLoop { .. } | Line::ArrayAccess { .. } => {
                unreachable!("Should be replaced earlier: {:?}", line)
            }
        }
    }

    if let Some(jump_label) = final_jump {
        instructions.push(HighLevelInstruction::Jump {
            dest: HighLevelValue::Label(jump_label),
            updated_fp: None,
        });
    }

    Ok(instructions)
}

// Helper functions
fn mark_vars_as_declared<VoC: Borrow<VarOrConstant>>(vars: &[VoC], declared: &mut BTreeSet<Var>) {
    for var in vars {
        if let VarOrConstant::Var(v) = var.borrow() {
            declared.insert(v.clone());
        }
    }
}

fn validate_vars_declared(vars: &[&VarOrConstant], declared: &BTreeSet<Var>) -> Result<(), String> {
    for var in vars {
        if let VarOrConstant::Var(v) = var {
            if !declared.contains(v) {
                return Err(format!("Variable {} not declared", v.name));
            }
        }
    }
    Ok(())
}

fn compile_branch(
    lines: &[Line],
    compiler: &mut Compiler,
    end_label: &str,
    declared_vars: &mut BTreeSet<Var>,
) -> Result<Vec<HighLevelInstruction>, String> {
    let mut branch_vars = declared_vars.clone();
    let instructions = compile_lines(
        lines,
        compiler,
        Some(end_label.to_string()),
        &mut branch_vars,
    )?;
    *declared_vars = declared_vars.intersection(&branch_vars).cloned().collect();
    Ok(instructions)
}

fn setup_function_call(
    func_name: &str,
    args: &[VarOrConstant],
    new_fp_pos: usize,
    return_label: &str,
    compiler: &Compiler,
) -> Result<Vec<HighLevelInstruction>, String> {
    let mut instructions = vec![
        HighLevelInstruction::RequestMemory {
            shift: new_fp_pos,
            size: HighLevelMetaValue::FunctionSize {
                function_name: func_name.to_string(),
            },
            vectorized: false,
        },
        HighLevelInstruction::Eq {
            left: HighLevelValue::Label(return_label.to_string()),
            right: HighLevelValue::ShiftedMemoryPointer {
                shift_0: new_fp_pos,
                shift_1: 0,
            },
        },
        HighLevelInstruction::Eq {
            left: HighLevelValue::Fp,
            right: HighLevelValue::ShiftedMemoryPointer {
                shift_0: new_fp_pos,
                shift_1: 1,
            },
        },
    ];

    // Copy arguments
    for (i, arg) in args.iter().enumerate() {
        instructions.push(HighLevelInstruction::Eq {
            left: HighLevelValue::from_var_or_const(arg, compiler),
            right: HighLevelValue::ShiftedMemoryPointer {
                shift_0: new_fp_pos,
                shift_1: 2 + i,
            },
        });
    }

    instructions.push(HighLevelInstruction::Jump {
        dest: HighLevelValue::Label(format!("@function_{}", func_name)),
        updated_fp: Some(HighLevelValue::MemoryAfterFp { shift: new_fp_pos }),
    });

    Ok(instructions)
}

fn compile_function_return(
    remaining_lines: &[Line],
    return_data: &[Var],
    args_len: usize,
    new_fp_pos: usize,
    compiler: &mut Compiler,
    final_jump: Option<Label>,
    declared_vars: &mut BTreeSet<Var>,
) -> Result<Vec<HighLevelInstruction>, String> {
    let mut instructions = Vec::new();

    // Copy return values
    for (i, ret_var) in return_data.iter().enumerate() {
        instructions.push(HighLevelInstruction::Eq {
            left: HighLevelValue::from_var(ret_var, compiler),
            right: HighLevelValue::ShiftedMemoryPointer {
                shift_0: new_fp_pos,
                shift_1: 2 + args_len + i,
            },
        });
    }

    instructions.extend(compile_lines(
        remaining_lines,
        compiler,
        final_jump,
        declared_vars,
    )?);
    Ok(instructions)
}

fn compile_poseidon(
    instructions: &mut Vec<HighLevelInstruction>,
    args: &[&VarOrConstant],
    results: &[&Var],
    total_size: usize,
    compiler: &mut Compiler,
    declared_vars: &mut BTreeSet<Var>,
) -> Result<(), String> {
    // Allocate memory for new result variables
    for res_var in results {
        if declared_vars.insert((*res_var).clone()) {
            instructions.push(HighLevelInstruction::RequestMemory {
                shift: compiler.get_var_offset(res_var),
                size: HighLevelMetaValue::Constant(1),
                vectorized: true,
            });
        }
    }

    let start_pos = compiler.stack_size;

    for (i, arg) in args.iter().enumerate() {
        instructions.push(HighLevelInstruction::Eq {
            left: HighLevelValue::MemoryAfterFp {
                shift: start_pos + i,
            },
            right: HighLevelValue::from_var_or_const(arg, compiler),
        });
    }

    for (i, res) in results.iter().enumerate() {
        instructions.push(HighLevelInstruction::Eq {
            left: HighLevelValue::MemoryAfterFp {
                shift: start_pos + args.len() + i,
            },
            right: HighLevelValue::from_var(res, compiler),
        });
    }

    compiler.stack_size += total_size;
    Ok(())
}

fn compile_function_ret(
    instructions: &mut Vec<HighLevelInstruction>,
    return_data: &[VarOrConstant],
    compiler: &Compiler,
) {
    for (i, ret_var) in return_data.iter().enumerate() {
        instructions.push(HighLevelInstruction::Eq {
            left: HighLevelValue::MemoryAfterFp {
                shift: 2 + compiler.args_count + i,
            },
            right: HighLevelValue::from_var_or_const(ret_var, compiler),
        });
    }
    instructions.push(HighLevelInstruction::Jump {
        dest: HighLevelValue::MemoryAfterFp { shift: 0 },
        updated_fp: Some(HighLevelValue::MemoryAfterFp { shift: 1 }),
    });
}
