use crate::{
    bytecode::intermediate_bytecode::*,
    compiler::{
        SimpleProgram,
        a_simplify_lang::{SimpleFunction, SimpleLine},
    },
    lang::*,
};
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
};

struct Compiler {
    bytecode: BTreeMap<Label, Vec<IntermediateInstruction>>,
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
            .unwrap_or_else(|| panic!("Variable {} not in scope", var))
            .clone()
    }
}

impl IntermediateValue {
    fn from_var_or_const(var_or_const: &VarOrConstant, compiler: &Compiler) -> Self {
        match var_or_const {
            VarOrConstant::Var(var) => Self::MemoryAfterFp {
                shift: compiler.get_var_offset(var),
            },
            VarOrConstant::Constant(c) => Self::Constant(c.clone()),
        }
    }

    fn from_var(var: &Var, compiler: &Compiler) -> Self {
        Self::MemoryAfterFp {
            shift: compiler.get_var_offset(var),
        }
    }
}

pub fn compile_to_intermediate_bytecode(
    simple_program: SimpleProgram,
) -> Result<IntermediateBytecode, String> {
    let mut compiler = Compiler::new();
    let mut memory_sizes = BTreeMap::new();

    for function in simple_program.functions.values() {
        let instructions = compile_function(function, &mut compiler)?;
        compiler
            .bytecode
            .insert(format!("@function_{}", function.name), instructions);
        memory_sizes.insert(function.name.clone(), compiler.stack_size);
    }

    Ok(IntermediateBytecode {
        bytecode: compiler.bytecode,
        memory_size_per_function: memory_sizes,
    })
}

fn compile_function(
    function: &SimpleFunction,
    compiler: &mut Compiler,
) -> Result<Vec<IntermediateInstruction>, String> {
    let mut internal_vars = find_internal_vars(&function.instructions);

    internal_vars.retain(|var| !function.arguments.contains(var));

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
    lines: &[SimpleLine],
    compiler: &mut Compiler,
    final_jump: Option<Label>,
    declared_vars: &mut BTreeSet<Var>,
) -> Result<Vec<IntermediateInstruction>, String> {
    let mut instructions = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        match line {
            SimpleLine::Assignment {
                var,
                operation,
                arg0,
                arg1,
            } => {
                instructions.push(IntermediateInstruction::computation(
                    *operation,
                    IntermediateValue::from_var_or_const(arg0, compiler),
                    IntermediateValue::from_var_or_const(arg1, compiler),
                    IntermediateValue::from_var(var, compiler),
                ));

                mark_vars_as_declared(&[arg0, arg1], declared_vars);
                declared_vars.insert(var.clone());
            }

            SimpleLine::IfNotZero {
                condition,
                then_branch,
                else_branch,
            } => {
                validate_vars_declared(&[condition], declared_vars)?;

                let if_id = compiler.if_counter;
                compiler.if_counter += 1;

                let (if_label, else_label, end_label) = (
                    format!("@if_{}", if_id),
                    format!("@else_{}", if_id),
                    format!("@if_else_end_{}", if_id),
                );

                instructions.push(IntermediateInstruction::JumpIfNotZero {
                    condition: IntermediateValue::from_var_or_const(condition, compiler),
                    dest: IntermediateValue::label(if_label.clone()),
                    updated_fp: None,
                });
                instructions.push(IntermediateInstruction::Jump {
                    dest: IntermediateValue::label(else_label.clone()),
                    updated_fp: None,
                });

                let original_stack = compiler.stack_size;

                let mut then_declared_vars = declared_vars.clone();
                let then_instructions = compile_lines(
                    &then_branch,
                    compiler,
                    Some(end_label.to_string()),
                    &mut then_declared_vars,
                )?;
                let then_stack = compiler.stack_size;

                compiler.stack_size = original_stack;
                let mut else_declared_vars = declared_vars.clone();
                let else_instructions = compile_lines(
                    &else_branch,
                    compiler,
                    Some(end_label.to_string()),
                    &mut else_declared_vars,
                )?;
                let else_stack = compiler.stack_size;

                compiler.stack_size = then_stack.max(else_stack);
                *declared_vars = then_declared_vars
                    .intersection(&else_declared_vars)
                    .cloned()
                    .collect();

                compiler.bytecode.insert(if_label, then_instructions);
                compiler.bytecode.insert(else_label, else_instructions);

                let remaining =
                    compile_lines(&lines[i + 1..], compiler, final_jump, declared_vars)?;
                compiler.bytecode.insert(end_label, remaining);

                return Ok(instructions);
            }

            SimpleLine::RawAccess { res, index, shift } => {
                validate_vars_declared(&[VarOrConstant::Var(index.clone())], declared_vars)?;
                if let VarOrConstant::Var(var) = res {
                    declared_vars.insert(var.clone());
                }
                instructions.push(IntermediateInstruction::Deref {
                    shift_0: compiler.get_var_offset(index).into(),
                    shift_1: shift.clone(),
                    res: match res {
                        VarOrConstant::Var(var) => IntermediaryMemOrFpOrConstant::MemoryAfterFp {
                            shift: compiler.get_var_offset(var),
                        },
                        VarOrConstant::Constant(c) => {
                            IntermediaryMemOrFpOrConstant::Constant(c.clone())
                        }
                    },
                });
            }

            SimpleLine::FunctionCall {
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

                validate_vars_declared(args, declared_vars)?;
                declared_vars.extend(return_data.iter().cloned());

                let after_call = {
                    let mut instructions = Vec::new();

                    // Copy return values
                    for (i, ret_var) in return_data.iter().enumerate() {
                        instructions.push(IntermediateInstruction::Deref {
                            shift_0: new_fp_pos.into(),
                            shift_1: (2 + args.len() + i).into(),
                            res: IntermediaryMemOrFpOrConstant::MemoryAfterFp {
                                shift: compiler.get_var_offset(ret_var),
                            },
                        });
                    }

                    instructions.extend(compile_lines(
                        &lines[i + 1..],
                        compiler,
                        final_jump,
                        declared_vars,
                    )?);

                    instructions
                };

                compiler.bytecode.insert(return_label, after_call);

                return Ok(instructions);
            }

            SimpleLine::Poseidon16 { args, res } => {
                compile_poseidon(&mut instructions, args, res, compiler, declared_vars)?;
                instructions.push(IntermediateInstruction::Poseidon2_16 {
                    shift: compiler.stack_size - 4,
                });
            }

            SimpleLine::Poseidon24 { args, res } => {
                compile_poseidon(&mut instructions, args, res, compiler, declared_vars)?;
                instructions.push(IntermediateInstruction::Poseidon2_24 {
                    shift: compiler.stack_size - 6,
                });
            }
            SimpleLine::FunctionRet { return_data } => {
                if compiler.func_name == "main" {
                    instructions.push(IntermediateInstruction::Jump {
                        dest: IntermediateValue::label("@end_program".to_string()),
                        updated_fp: None,
                    });
                } else {
                    compile_function_ret(&mut instructions, return_data, compiler);
                }
            }
            SimpleLine::Panic => instructions.push(IntermediateInstruction::Panic),
            SimpleLine::MAlloc {
                var,
                size,
                vectorized,
            } => {
                declared_vars.insert(var.clone());
                instructions.push(IntermediateInstruction::RequestMemory {
                    shift: compiler.get_var_offset(var),
                    size: IntermediateValue::from_var_or_const(size, compiler),
                    vectorized: *vectorized,
                });
            }
            SimpleLine::DecomposeBits { var, to_decompose } => {
                declared_vars.insert(var.clone());
                instructions.push(IntermediateInstruction::DecomposeBits {
                    res_offset: compiler.get_var_offset(var),
                    to_decompose: IntermediateValue::from_var_or_const(to_decompose, compiler),
                });
            }
            SimpleLine::Print { line_info, content } => {
                instructions.push(IntermediateInstruction::Print {
                    line_info: line_info.clone(),
                    content: content
                        .iter()
                        .map(|c| IntermediateValue::from_var_or_const(c, compiler))
                        .collect(),
                });
            }
        }
    }

    if let Some(jump_label) = final_jump {
        instructions.push(IntermediateInstruction::Jump {
            dest: IntermediateValue::label(jump_label),
            updated_fp: None,
        });
    }

    Ok(instructions)
}

// Helper functions
fn mark_vars_as_declared<VoC: Borrow<VarOrConstant>>(vocs: &[VoC], declared: &mut BTreeSet<Var>) {
    for voc in vocs {
        if let VarOrConstant::Var(v) = voc.borrow() {
            declared.insert(v.clone());
        }
    }
}

fn validate_vars_declared<VoC: Borrow<VarOrConstant>>(
    vocs: &[VoC],
    declared: &BTreeSet<Var>,
) -> Result<(), String> {
    for voc in vocs {
        if let VarOrConstant::Var(v) = voc.borrow() {
            if !declared.contains(v) {
                return Err(format!("Variable {} not declared", v));
            }
        }
    }
    Ok(())
}

fn setup_function_call(
    func_name: &str,
    args: &[VarOrConstant],
    new_fp_pos: usize,
    return_label: &str,
    compiler: &Compiler,
) -> Result<Vec<IntermediateInstruction>, String> {
    let mut instructions = vec![
        IntermediateInstruction::RequestMemory {
            shift: new_fp_pos,
            size: ConstExpression::function_size(func_name.to_string()).into(),
            vectorized: false,
        },
        IntermediateInstruction::Deref {
            shift_0: new_fp_pos.into(),
            shift_1: ConstExpression::zero(),
            res: IntermediaryMemOrFpOrConstant::Constant(ConstExpression::label(
                return_label.to_string(),
            )),
        },
        IntermediateInstruction::Deref {
            shift_0: new_fp_pos.into(),
            shift_1: ConstExpression::one(),
            res: IntermediaryMemOrFpOrConstant::Fp,
        },
    ];

    // Copy arguments
    for (i, arg) in args.iter().enumerate() {
        instructions.push(IntermediateInstruction::Deref {
            shift_0: new_fp_pos.into(),
            shift_1: (2 + i).into(),
            res: match arg {
                VarOrConstant::Var(var) => IntermediaryMemOrFpOrConstant::MemoryAfterFp {
                    shift: compiler.get_var_offset(var),
                },
                VarOrConstant::Constant(c) => IntermediaryMemOrFpOrConstant::Constant(c.clone()),
            },
        });
    }

    instructions.push(IntermediateInstruction::Jump {
        dest: IntermediateValue::label(format!("@function_{}", func_name)),
        updated_fp: Some(IntermediateValue::MemoryAfterFp { shift: new_fp_pos }),
    });

    Ok(instructions)
}

fn compile_poseidon(
    instructions: &mut Vec<IntermediateInstruction>,
    args: &[VarOrConstant],
    results: &[Var],
    compiler: &mut Compiler,
    declared_vars: &mut BTreeSet<Var>,
) -> Result<(), String> {
    // Allocate memory for new result variables
    for res_var in results {
        if declared_vars.insert((*res_var).clone()) {
            instructions.push(IntermediateInstruction::RequestMemory {
                shift: compiler.get_var_offset(res_var),
                size: ConstExpression::one().into(),
                vectorized: true,
            });
        }
    }

    for (i, arg) in args.iter().enumerate() {
        instructions.push(IntermediateInstruction::equality(
            IntermediateValue::MemoryAfterFp {
                shift: compiler.stack_size + i,
            },
            IntermediateValue::from_var_or_const(arg, compiler),
        ));
    }

    for (i, res) in results.iter().enumerate() {
        instructions.push(IntermediateInstruction::equality(
            IntermediateValue::from_var(res, compiler),
            IntermediateValue::MemoryAfterFp {
                shift: compiler.stack_size + args.len() + i,
            },
        ));
    }

    compiler.stack_size += args.len() + results.len();
    Ok(())
}

fn compile_function_ret(
    instructions: &mut Vec<IntermediateInstruction>,
    return_data: &[VarOrConstant],
    compiler: &Compiler,
) {
    for (i, ret_var) in return_data.iter().enumerate() {
        instructions.push(IntermediateInstruction::equality(
            IntermediateValue::MemoryAfterFp {
                shift: 2 + compiler.args_count + i,
            },
            IntermediateValue::from_var_or_const(ret_var, compiler),
        ));
    }
    instructions.push(IntermediateInstruction::Jump {
        dest: IntermediateValue::MemoryAfterFp { shift: 0 },
        updated_fp: Some(IntermediateValue::MemoryAfterFp { shift: 1 }),
    });
}

fn find_internal_vars(lines: &[SimpleLine]) -> BTreeSet<Var> {
    let mut internal_vars = BTreeSet::new();
    for line in lines {
        match line {
            SimpleLine::Assignment { var, .. }
            | SimpleLine::MAlloc { var, .. }
            | SimpleLine::DecomposeBits { var, .. } => {
                internal_vars.insert(var.clone());
            }
            SimpleLine::RawAccess { res, .. } => {
                if let VarOrConstant::Var(var) = res {
                    internal_vars.insert(var.clone());
                }
            }
            SimpleLine::FunctionCall { return_data, .. } => {
                internal_vars.extend(return_data.iter().cloned());
            }
            SimpleLine::Poseidon16 { res, .. } => {
                internal_vars.extend(res.iter().cloned());
            }
            SimpleLine::Poseidon24 { res, .. } => {
                internal_vars.extend(res.iter().cloned());
            }
            SimpleLine::IfNotZero {
                then_branch,
                else_branch,
                ..
            } => {
                internal_vars.extend(find_internal_vars(then_branch));
                internal_vars.extend(find_internal_vars(else_branch));
            }
            SimpleLine::Panic | SimpleLine::Print { .. } | SimpleLine::FunctionRet { .. } => {}
        }
    }
    internal_vars
}
