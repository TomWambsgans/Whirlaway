use std::collections::{BTreeMap, BTreeSet};

use crate::{
    bytecode::intermediate_bytecode::*,
    compiler::{
        phase_0::{replace_array_access, replace_assert_not_eq, replace_if_eq},
        phase_1::{get_internal_and_external_variables, replace_loops_with_recursion},
    },
    lang::*,
};

struct Compiler {
    bytecode: BTreeMap<Label, Vec<HighLevelInstruction>>,
    if_else_counter: usize,
    function_call_counter: usize,

    // relative to current Function's
    func_name: String, // name of the current function being compiled
    vars_in_scope: BTreeMap<Var, usize>, // var = m[fp + index]
    args_count: usize, // number of arguments in the current function
    current_stack_size: usize,
}

impl HighLevelValue {
    fn from_var_or_constant(var_or_const: &VarOrConstant, compiler: &Compiler) -> Self {
        match var_or_const {
            VarOrConstant::Var(var) => HighLevelValue::from_var(var, compiler),
            VarOrConstant::Constant(constant) => HighLevelValue::from_constant(constant),
        }
    }

    fn from_var(var: &Var, compiler: &Compiler) -> Self {
        HighLevelValue::MemoryAfterFp {
            shift: compiler.get_shift(var),
        }
    }

    fn from_constant(constant: &ConstantValue) -> Self {
        match constant {
            ConstantValue::Scalar(scalar) => Self::Constant(*scalar),
            ConstantValue::PublicInputStart => Self::PublicInputStart,
            ConstantValue::PointerToZeroVector => Self::PointerToZeroVector,
        }
    }
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            vars_in_scope: BTreeMap::new(),
            current_stack_size: 0,
            bytecode: BTreeMap::new(),
            func_name: String::new(),
            args_count: 0,
            if_else_counter: 0,
            function_call_counter: 0,
        }
    }

    pub fn get_shift(&self, var: &Var) -> usize {
        self.vars_in_scope
            .get(var)
            .cloned()
            .expect(&format!("Variable {} not found in current scope", var.name))
    }
}

pub fn compile_to_hight_level_bytecode(mut program: Program) -> Result<HighLevelBytecode, String> {
    replace_assert_not_eq(&mut program);
    replace_array_access(&mut program);
    replace_loops_with_recursion(&mut program);
    replace_if_eq(&mut program);
    // println!("Program after phase 2: \n{}", program.to_string());
    let mut compiler = Compiler::new();
    let mut memory_size_per_function = BTreeMap::new();
    for function in program.functions.values() {
        let instructions = compile_function(function, &mut compiler)?;
        compiler
            .bytecode
            .insert(format!("@function_{}", function.name), instructions);
        memory_size_per_function.insert(function.name.clone(), compiler.current_stack_size);
    }
    Ok(HighLevelBytecode {
        bytecode: compiler.bytecode,
        memory_size_per_function,
    })
}

fn compile_function(
    function: &Function,
    compiler: &mut Compiler,
) -> Result<Vec<HighLevelInstruction>, String> {
    let (mut internal_vars, external_vars) =
        get_internal_and_external_variables(&function.instructions);

    // In case we reassign a variable which  is already a function argument:
    internal_vars.retain(|var| !function.arguments.contains(var));

    for external_var in &external_vars {
        if !function.arguments.contains(external_var) {
            return Err(format!(
                "Variable {} not found in function arguments of function {}",
                external_var.name, function.name
            ));
        }
    }

    let mut current_stack_size = 2; // for pc and fp (when returning)

    // associate to each variable a shift (in memory, relative to fp)
    let mut vars_in_scope = BTreeMap::new();
    for (i, var) in function.arguments.iter().enumerate() {
        vars_in_scope.insert(var.clone(), i + current_stack_size);
    }

    current_stack_size += function.arguments.len();

    current_stack_size += function.n_returned_vars; // reserve space for returned vars

    for (i, var) in internal_vars.iter().enumerate() {
        vars_in_scope.insert(var.clone(), i + current_stack_size);
    }

    current_stack_size += internal_vars.len();

    compiler.func_name = function.name.clone();
    compiler.vars_in_scope = vars_in_scope;
    compiler.current_stack_size = current_stack_size;
    compiler.args_count = function.arguments.len();

    let mut variables_already_declared: BTreeSet<Var> =
        function.arguments.iter().cloned().collect();

    compile_lines(
        &function.instructions,
        compiler,
        None,
        &mut variables_already_declared,
    )
}

fn compile_lines(
    lines: &[Line],
    compiler: &mut Compiler,
    final_jump: Option<Label>,
    variables_already_declared: &mut BTreeSet<Var>,
) -> Result<Vec<HighLevelInstruction>, String> {
    let mut res = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        match line {
            Line::Assignment {
                var,
                operation,
                arg0,
                arg1,
            } => {
                let value_a = HighLevelValue::from_var_or_constant(arg0, compiler);
                let value_b = HighLevelValue::from_var_or_constant(arg1, compiler);
                let instruction = HighLevelInstruction::Computation {
                    operation: *operation,
                    arg_a: value_a,
                    arg_b: value_b,
                    res: HighLevelValue::from_var(var, compiler),
                };
                res.push(instruction);

                if let VarOrConstant::Var(var) = arg0 {
                    variables_already_declared.insert(var.clone());
                }
                if let VarOrConstant::Var(var) = arg1 {
                    variables_already_declared.insert(var.clone());
                }
                variables_already_declared.insert(var.clone());
            }
            Line::RawAccess { var, index } => {
                let instruction = HighLevelInstruction::Eq {
                    left: HighLevelValue::from_var(var, compiler),
                    right: match index {
                        VarOrConstant::Constant(ConstantValue::Scalar(shift)) => {
                            HighLevelValue::DirectMemory {
                                shift: ConstantValue::Scalar(*shift),
                            }
                        }
                        VarOrConstant::Constant(ConstantValue::PublicInputStart) => {
                            HighLevelValue::DirectMemory {
                                shift: ConstantValue::PublicInputStart,
                            }
                        }
                        VarOrConstant::Constant(ConstantValue::PointerToZeroVector) => {
                            HighLevelValue::DirectMemory {
                                shift: ConstantValue::PointerToZeroVector,
                            }
                        }
                        VarOrConstant::Var(index_var) => HighLevelValue::MemoryPointer {
                            shift: compiler.get_shift(index_var),
                        },
                    },
                };
                res.push(instruction);

                if let VarOrConstant::Var(var) = index {
                    variables_already_declared.insert(var.clone());
                }
                variables_already_declared.insert(var.clone());
            }
            Line::Assert(condition) => match condition {
                Boolean::Different { .. } => {
                    unreachable!("Assert not equal should have been handled earlier")
                }
                Boolean::Equal { left, right } => {
                    let left_value = HighLevelValue::from_var_or_constant(left, compiler);
                    let right_value = HighLevelValue::from_var_or_constant(right, compiler);
                    res.push(HighLevelInstruction::Eq {
                        left: left_value,
                        right: right_value,
                    });

                    if let VarOrConstant::Var(var) = left {
                        variables_already_declared.insert(var.clone());
                    }
                    if let VarOrConstant::Var(var) = right {
                        variables_already_declared.insert(var.clone());
                    }
                }
            },
            Line::IfCondition {
                condition,
                then_branch,
                else_branch,
            } => match condition {
                Boolean::Equal { .. } => {
                    unreachable!("If condition with equality should have been handled earlier")
                }
                Boolean::Different { left, right } => {
                    let left_value = HighLevelValue::from_var_or_constant(left, compiler);
                    let right_value = HighLevelValue::from_var_or_constant(right, compiler);

                    if let VarOrConstant::Var(var) = left {
                        assert!(
                            variables_already_declared.contains(var),
                            "{} not declared in scope",
                            var.name
                        );
                    }
                    if let VarOrConstant::Var(var) = right {
                        assert!(
                            variables_already_declared.contains(var),
                            "{} not declared in scope",
                            var.name
                        );
                    }

                    let difference = HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size,
                    };
                    compiler.current_stack_size += 1; // reserve space for the result of the comparison

                    let label_after = format!("@if_else_end_{}", compiler.if_else_counter);
                    let label_if = format!("@if_{}", compiler.if_else_counter);
                    let label_else = format!("@else_{}", compiler.if_else_counter);
                    compiler.if_else_counter += 1;

                    let current_stack_size = compiler.current_stack_size;
                    let mut variables_declared_in_if = variables_already_declared.clone();
                    let instructions_if = compile_lines(
                        then_branch,
                        compiler,
                        Some(label_after.clone()),
                        &mut variables_declared_in_if,
                    )?;
                    let if_stack_size = compiler.current_stack_size;
                    compiler.current_stack_size = current_stack_size;
                    let mut variables_declared_in_else = variables_already_declared.clone();
                    let instructions_else = compile_lines(
                        else_branch,
                        compiler,
                        Some(label_after.clone()),
                        &mut variables_declared_in_else,
                    )?;
                    let else_stack_size = compiler.current_stack_size;

                    compiler.current_stack_size = else_stack_size.max(if_stack_size);

                    res.push(HighLevelInstruction::Computation {
                        operation: HighLevelOperation::Sub,
                        arg_a: left_value,
                        arg_b: right_value,
                        res: difference.clone(),
                    });
                    res.push(HighLevelInstruction::JumpIfNotZero {
                        condition: difference,
                        dest: HighLevelValue::Label(label_if.clone()),
                        updated_fp: None,
                    });
                    res.push(HighLevelInstruction::Jump {
                        dest: HighLevelValue::Label(label_else.clone()),
                        updated_fp: None,
                    });

                    compiler.bytecode.insert(label_if, instructions_if);
                    compiler.bytecode.insert(label_else, instructions_else);

                    let mut variables_declared_after_if_else = variables_declared_in_if
                        .intersection(&variables_declared_in_else)
                        .cloned()
                        .collect::<BTreeSet<_>>();

                    let instructions_after_if_else = compile_lines(
                        &lines[i + 1..],
                        compiler,
                        final_jump,
                        &mut variables_declared_after_if_else,
                    )?;
                    compiler
                        .bytecode
                        .insert(label_after, instructions_after_if_else);

                    return Ok(res);
                }
            },
            Line::ForLoop { .. } | Line::ArrayAccess { .. } => {
                unreachable!("Replaced earlier.")
            }

            Line::FunctionCall {
                function_name,
                args,
                return_data,
            } => {
                // Memory layout in the calling function: Pointer to new FP
                // Memory layout in the called function: PC, FP, args, return_data

                for ret_var in return_data {
                    variables_already_declared.insert(ret_var.clone());
                }

                let return_label = format!("@return_from_call_{}", compiler.function_call_counter);
                compiler.function_call_counter += 1;

                res.push(HighLevelInstruction::RequestMemory {
                    shift: compiler.current_stack_size,
                    size: HighLevelMetaValue::FunctionSize {
                        function_name: function_name.clone(),
                    },
                    vectorized: false,
                });
                let shift_of_new_fp = compiler.current_stack_size;
                compiler.current_stack_size += 1;

                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::ShiftedMemoryPointer {
                        shift_0: shift_of_new_fp,
                        shift_1: 0,
                    },
                    right: HighLevelValue::Label(return_label.clone()),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::Fp,
                    right: HighLevelValue::ShiftedMemoryPointer {
                        shift_0: shift_of_new_fp,
                        shift_1: 1,
                    },
                });

                for (arg_index, arg) in args.iter().enumerate() {
                    res.push(HighLevelInstruction::Eq {
                        left: HighLevelValue::ShiftedMemoryPointer {
                            shift_0: shift_of_new_fp,
                            shift_1: 2 + arg_index,
                        },
                        right: HighLevelValue::from_var_or_constant(arg, compiler),
                    });

                    if let VarOrConstant::Var(var) = arg {
                        variables_already_declared.insert(var.clone());
                    }
                }

                let mut instructions_after_function_call = vec![];
                for (ret_index, ret_var) in return_data.iter().enumerate() {
                    instructions_after_function_call.push(HighLevelInstruction::Eq {
                        left: HighLevelValue::ShiftedMemoryPointer {
                            shift_0: shift_of_new_fp,
                            shift_1: 2 + args.len() + ret_index,
                        },
                        right: HighLevelValue::from_var(ret_var, compiler),
                    });
                }

                let dest = HighLevelValue::Label(format!("@function_{}", function_name));
                res.push(HighLevelInstruction::Jump {
                    dest,
                    updated_fp: Some(HighLevelValue::MemoryAfterFp {
                        shift: shift_of_new_fp,
                    }),
                });

                instructions_after_function_call.extend(compile_lines(
                    &lines[i + 1..],
                    compiler,
                    final_jump,
                    variables_already_declared,
                )?);

                compiler
                    .bytecode
                    .insert(return_label, instructions_after_function_call);

                return Ok(res);
            }

            Line::Poseidon16 {
                arg0,
                arg1,
                res0,
                res1,
            } => {
                for res_var in [res0, res1] {
                    if variables_already_declared.insert(res_var.clone()) {
                        // value is new, we need to allocate memory
                        res.push(HighLevelInstruction::RequestMemory {
                            shift: compiler.get_shift(res_var),
                            size: HighLevelMetaValue::Constant(1),
                            vectorized: true,
                        });
                    }
                }

                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size,
                    },
                    right: HighLevelValue::from_var_or_constant(arg0, compiler),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size + 1,
                    },
                    right: HighLevelValue::from_var_or_constant(arg1, compiler),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size + 2,
                    },
                    right: HighLevelValue::from_var(res0, compiler),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size + 3,
                    },
                    right: HighLevelValue::from_var(res1, compiler),
                });
                res.push(HighLevelInstruction::Poseidon2_16 {
                    shift: compiler.current_stack_size,
                });
                compiler.current_stack_size += 4;
            }
            Line::Poseidon24 {
                arg0,
                arg1,
                arg2,
                res0,
                res1,
                res2,
            } => {
                for res_var in [res0, res1, res2] {
                    if variables_already_declared.insert(res_var.clone()) {
                        // value is new, we need to allocate memory
                        res.push(HighLevelInstruction::RequestMemory {
                            shift: compiler.get_shift(res_var),
                            size: HighLevelMetaValue::Constant(1),
                            vectorized: true,
                        });
                    }
                }

                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size,
                    },
                    right: HighLevelValue::from_var_or_constant(arg0, compiler),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size + 1,
                    },
                    right: HighLevelValue::from_var_or_constant(arg1, compiler),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size + 2,
                    },
                    right: HighLevelValue::from_var_or_constant(arg2, compiler),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size + 3,
                    },
                    right: HighLevelValue::from_var(res0, compiler),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size + 4,
                    },
                    right: HighLevelValue::from_var(res1, compiler),
                });
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::MemoryAfterFp {
                        shift: compiler.current_stack_size + 5,
                    },
                    right: HighLevelValue::from_var(res2, compiler),
                });
                res.push(HighLevelInstruction::Poseidon2_24 {
                    shift: compiler.current_stack_size,
                });
                compiler.current_stack_size += 6;
            }
            Line::MAlloc { var, size } => {
                let size = match size {
                    ConstantValue::Scalar(scalar) => *scalar,
                    ConstantValue::PointerToZeroVector => {
                        return Err("Cannot allocate memory with pointer to zero vector".to_string());
                    }
                    ConstantValue::PublicInputStart => {
                        return Err("Cannot allocate memory with public input start".to_string());
                    }
                };
                res.push(HighLevelInstruction::RequestMemory {
                    shift: compiler.get_shift(var),
                    size: HighLevelMetaValue::Constant(size),
                    vectorized: false,
                });
            }
            Line::FunctionRet { return_data } => {
                if compiler.func_name == "main" {
                    res.push(HighLevelInstruction::Jump {
                        dest: HighLevelValue::Label("@end_program".to_string()),
                        updated_fp: None,
                    });
                } else {
                    for (ret_index, return_var) in return_data.iter().enumerate() {
                        res.push(HighLevelInstruction::Eq {
                            left: HighLevelValue::MemoryAfterFp {
                                shift: 2 + compiler.args_count + ret_index,
                            },
                            right: HighLevelValue::from_var_or_constant(return_var, compiler),
                        });
                    }
                    res.push(HighLevelInstruction::Jump {
                        dest: HighLevelValue::MemoryAfterFp { shift: 0 },
                        updated_fp: Some(HighLevelValue::MemoryAfterFp { shift: 1 }),
                    });
                }
            }
            Line::Panic => {
                res.push(HighLevelInstruction::Eq {
                    left: HighLevelValue::Constant(0),
                    right: HighLevelValue::Constant(1),
                });
            }

            Line::Print { line_info, content } => {
                res.push(HighLevelInstruction::Print {
                    line_info: line_info.clone(),
                    content: content
                        .into_iter()
                        .map(|c| HighLevelValue::from_var_or_constant(c, compiler))
                        .collect(),
                });
            }
        }
    }
    if let Some(final_jump_label) = final_jump {
        res.push(HighLevelInstruction::Jump {
            dest: HighLevelValue::Label(final_jump_label),
            updated_fp: None,
        });
    }
    Ok(res)
}
