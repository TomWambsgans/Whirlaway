use std::collections::HashMap;

use crate::{
    bytecode::*,
    compiler::{
        phase_0::{replace_assert_not_eq, replace_if_eq},
        phase_1::{get_internal_and_external_variables, replace_loops_with_recursion},
    },
    lang::*,
};

struct Compiler {
    bytecode: HashMap<Label, Vec<Instruction>>,
    if_else_counter: usize,
    function_call_counter: usize,

    // relative to current Function's
    vars_in_scope: HashMap<Var, usize>, // var = m[fp + index]
    args_count: usize,                  // number of arguments in the current function
    current_stack_size: usize,
}

impl Value {
    fn from_var_or_constant(var_or_const: &VarOrConstant, compiler: &Compiler) -> Self {
        match var_or_const {
            VarOrConstant::Var(var) => Value::from_var(var, compiler),
            VarOrConstant::Constant(constant) => Value::from_constant(constant),
        }
    }

    fn from_var(var: &Var, compiler: &Compiler) -> Self {
        Value::MemoryAfterFp {
            shift: compiler.get_shift(var),
        }
    }

    fn from_constant(constant: &ConstantValue) -> Self {
        match constant {
            ConstantValue::Scalar(scalar) => Self::Constant(*scalar),
            ConstantValue::PublicInputStart => Self::PublicInputStart,
        }
    }
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            vars_in_scope: HashMap::new(),
            current_stack_size: 0,
            bytecode: HashMap::new(),
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

pub fn first_compile_pass(mut program: Program) -> Result<CompiledProgram, String> {
    replace_assert_not_eq(&mut program);
    replace_loops_with_recursion(&mut program);
    replace_if_eq(&mut program);
    let mut compiler = Compiler::new();
    for function in program.functions.values() {
        let instructions = compile_function(function, &mut compiler)?;
        compiler
            .bytecode
            .insert(format!("@function_{}", function.name), instructions);
    }
    Ok(CompiledProgram(compiler.bytecode))
}

fn compile_function(
    function: &Function,
    compiler: &mut Compiler,
) -> Result<Vec<Instruction>, String> {
    let (internal_vars, external_vars) =
        get_internal_and_external_variables(&function.instructions);

    for external_var in &external_vars {
        if !function.arguments.contains(external_var) {
            return Err(format!(
                "Variable {} not found in function arguments of function {}",
                external_var.name, function.name
            ));
        }
    }

    let mut current_stack_size = 1; // for pc (when returning)

    // associate to each variable a shift (in memory, relative to fp)
    let mut vars_in_scope = HashMap::new();
    for (i, var) in function.arguments.iter().enumerate() {
        vars_in_scope.insert(var.clone(), i + current_stack_size);
    }
    current_stack_size += function.arguments.len();

    current_stack_size += function.n_returned_vars; // reserve space for returned vars

    for (i, var) in internal_vars.iter().enumerate() {
        vars_in_scope.insert(var.clone(), i + current_stack_size);
    }

    current_stack_size += internal_vars.len();

    compiler.vars_in_scope = vars_in_scope;
    compiler.current_stack_size = current_stack_size;
    compiler.args_count = function.arguments.len();

    compile_lines(&function.instructions, compiler)
}

fn compile_lines(lines: &[Line], compiler: &mut Compiler) -> Result<Vec<Instruction>, String> {
    let mut res = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        match line {
            Line::Assignment {
                var,
                operation,
                arg0,
                arg1,
            } => {
                let value_a = Value::from_var_or_constant(arg0, compiler);
                let value_b = Value::from_var_or_constant(arg1, compiler);
                let instruction = Instruction::Computation {
                    operation: *operation,
                    arg_a: value_a,
                    arg_b: value_b,
                    res: Value::from_var(var, compiler),
                };
                res.push(instruction);
            }
            Line::RawAccess { var, index } => {
                let instruction = Instruction::Eq {
                    left: Value::from_var(var, compiler),
                    right: Value::from_var_or_constant(index, compiler),
                };
                res.push(instruction);
            }
            Line::Assert(condition) => match condition {
                Boolean::Different { .. } => {
                    unreachable!("Assert not equal should have been handled earlier")
                }
                Boolean::Equal { left, right } => {
                    let left_value = Value::from_var_or_constant(left, compiler);
                    let right_value = Value::from_var_or_constant(right, compiler);
                    res.push(Instruction::Eq {
                        left: left_value,
                        right: right_value,
                    });
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
                    let left_value = Value::from_var_or_constant(left, compiler);
                    let right_value = Value::from_var_or_constant(right, compiler);
                    let difference = Value::MemoryAfterFp {
                        shift: compiler.current_stack_size,
                    };
                    compiler.current_stack_size += 1; // reserve space for the result of the comparison

                    let label_after = format!("@if_else_end_{}", compiler.if_else_counter);
                    let label_if = format!("@if_{}", compiler.if_else_counter);
                    let label_else = format!("@else_{}", compiler.if_else_counter);

                    let current_stack_size = compiler.current_stack_size;
                    let mut instructions_if = compile_lines(then_branch, compiler)?;
                    let if_stack_size = compiler.current_stack_size;
                    compiler.current_stack_size = current_stack_size;
                    let mut instructions_else = compile_lines(else_branch, compiler)?;
                    let else_stack_size = compiler.current_stack_size;

                    compiler.current_stack_size = else_stack_size.max(if_stack_size);

                    instructions_if.push(Instruction::Jump {
                        dest: Value::Label(label_after.clone()),
                        updated_fp: None,
                    });
                    instructions_else.push(Instruction::Jump {
                        dest: Value::Label(label_after.clone()),
                        updated_fp: None,
                    });

                    res.push(Instruction::Computation {
                        operation: Operation::Sub,
                        arg_a: left_value,
                        arg_b: right_value,
                        res: difference.clone(),
                    });
                    res.push(Instruction::JumpIfNotZero {
                        condition: difference,
                        dest: Value::Label(label_if.clone()),
                    });
                    res.push(Instruction::Jump {
                        dest: Value::Label(label_else.clone()),
                        updated_fp: None,
                    });

                    compiler.bytecode.insert(label_if, instructions_if);
                    compiler.bytecode.insert(label_else, instructions_else);

                    let instructions_after_if_else = compile_lines(&lines[i + 1..], compiler)?;
                    compiler
                        .bytecode
                        .insert(label_after, instructions_after_if_else);

                    return Ok(res);
                }
            },
            Line::ForLoop { .. } => {
                unreachable!("For loops should have been replaced with recursion earlier")
            }

            Line::FunctionCall {
                function_name,
                args,
                return_data,
            } => {
                // Memory layout in the calling function: Pointer to new FP
                // Memory layout in the called function: PC, FP, args, return_data

                let return_label = format!("@return_from_call_{}", compiler.function_call_counter);
                compiler.function_call_counter += 1;

                res.push(Instruction::RequestMemory {
                    shift: compiler.current_stack_size,
                    size: MetaValue::FunctionSize {
                        function_name: function_name.clone(),
                    },
                });
                let shift_of_new_fp = compiler.current_stack_size;
                compiler.current_stack_size += 1;

                res.push(Instruction::Eq {
                    left: Value::ShiftedMemoryPointer {
                        shift_0: shift_of_new_fp,
                        shift_1: 0,
                    },
                    right: Value::Label(return_label.clone()),
                });
                res.push(Instruction::Eq {
                    left: Value::Fp,
                    right: Value::ShiftedMemoryPointer {
                        shift_0: shift_of_new_fp,
                        shift_1: 1,
                    },
                });

                for (arg_index, arg) in args.iter().enumerate() {
                    res.push(Instruction::Eq {
                        left: Value::ShiftedMemoryPointer {
                            shift_0: shift_of_new_fp,
                            shift_1: 2 + arg_index,
                        },
                        right: Value::from_var_or_constant(arg, compiler),
                    });
                }
                for (ret_index, ret_var) in return_data.iter().enumerate() {
                    res.push(Instruction::Eq {
                        left: Value::ShiftedMemoryPointer {
                            shift_0: shift_of_new_fp,
                            shift_1: 2 + args.len() + ret_index,
                        },
                        right: Value::from_var(ret_var, compiler),
                    });
                }

                let dest = Value::Label(format!("@function_{}", function_name));
                res.push(Instruction::FpAssign {
                    value: Value::MemoryAfterFp {
                        shift: shift_of_new_fp,
                    },
                });
                res.push(Instruction::Jump {
                    dest,
                    updated_fp: None,
                });

                let instructions_after_function_call = compile_lines(&lines[i + 1..], compiler)?;
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
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size,
                    },
                    right: Value::from_var_or_constant(arg0, compiler),
                });
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size + 1,
                    },
                    right: Value::from_var_or_constant(arg1, compiler),
                });
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size + 2,
                    },
                    right: Value::from_var(res0, compiler),
                });
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp { shift: 3 },
                    right: Value::from_var(res1, compiler),
                });
                res.push(Instruction::Poseidon2_16 {
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
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size,
                    },
                    right: Value::from_var_or_constant(arg0, compiler),
                });
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size + 1,
                    },
                    right: Value::from_var_or_constant(arg1, compiler),
                });
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size + 2,
                    },
                    right: Value::from_var_or_constant(arg2, compiler),
                });
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size + 3,
                    },
                    right: Value::from_var(res0, compiler),
                });
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size + 4,
                    },
                    right: Value::from_var(res1, compiler),
                });
                res.push(Instruction::Eq {
                    left: Value::MemoryAfterFp {
                        shift: compiler.current_stack_size + 5,
                    },
                    right: Value::from_var(res2, compiler),
                });
                res.push(Instruction::Poseidon2_24 {
                    shift: compiler.current_stack_size,
                });
                compiler.current_stack_size += 6;
            }
            Line::MAlloc { var, size } => {
                let size = match size {
                    ConstantValue::Scalar(scalar) => *scalar,
                    ConstantValue::PublicInputStart => {
                        return Err("Cannot allocate memory with public input start".to_string());
                    }
                };
                res.push(Instruction::RequestMemory {
                    shift: compiler.get_shift(var),
                    size: MetaValue::Constant(size),
                });
            }
            Line::AssertEqExt { left, right } => {
                res.push(Instruction::ExtComputation {
                    operation: Operation::Add,
                    arg_a: Value::from_var_or_constant(left, compiler),
                    arg_b: Value::PointerToZeroVector,
                    res: Value::from_var_or_constant(right, compiler),
                });
            }
            Line::FunctionRet { return_data } => {
                for (ret_index, return_var) in return_data.iter().enumerate() {
                    res.push(Instruction::Eq {
                        left: Value::MemoryAfterFp {
                            shift: 2 + compiler.args_count + ret_index,
                        },
                        right: Value::from_var_or_constant(return_var, compiler),
                    });
                }
                res.push(Instruction::Jump {
                    dest: Value::MemoryAfterFp { shift: 0 },
                    updated_fp: Some(Value::MemoryAfterFp { shift: 1 }),
                });
            }
            Line::Panic => {
                res.push(Instruction::Eq {
                    left: Value::Constant(0),
                    right: Value::Constant(1),
                });
            }
        }
    }
    Ok(res)
}
