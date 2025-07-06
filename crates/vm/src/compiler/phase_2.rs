use std::collections::HashMap;

use crate::{bytecode::*, lang::*};

struct Compiler {
    var_count: usize,
    vars_in_scope: HashMap<Var, usize>, // var = m[fp + index]
    current_shift: usize,
    bytecode: HashMap<Label, Vec<Instruction>>,
    condition_labels: usize,
}

impl Value {
    pub fn from_var_or_constant(var_or_const: &VarOrConstant, compiler: &Compiler) -> Option<Self> {
        match var_or_const {
            VarOrConstant::Var(var) => Some(Value::MemoryAfterFp {
                shift: compiler.vars_in_scope.get(var).cloned()?,
            }),
            VarOrConstant::Constant(constant) => Some(Value::from_constant(constant)),
        }
    }

    pub fn from_constant(constant: &ConstantValue) -> Self {
        match constant {
            ConstantValue::Scalar(scalar) => Self::Constant(*scalar),
            ConstantValue::PublicInputStart => Self::PublicInputStart,
        }
    }
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            var_count: 0,
            vars_in_scope: HashMap::new(),
            current_shift: 0,
            bytecode: HashMap::new(),
            condition_labels: 0,
        }
    }

    fn add_var(&mut self, name: String) -> (Var, usize) {
        let var = Var { name };
        self.vars_in_scope.insert(var.clone(), self.current_shift);
        self.var_count += 1;
        self.current_shift += 1;
        (var, self.current_shift - 1)
    }

    pub fn get_shift(&self, var: &Var) -> Result<usize, String> {
        self.vars_in_scope
            .get(var)
            .cloned()
            .ok_or_else(|| format!("Variable {} not found", var.name))
    }
}

pub fn compile_function(program: Program) -> Result<Vec<Instruction>, String> {
    todo!()
}
fn compile_program(
    mut program: Program,
    compiler: &mut Compiler,
) -> Result<Vec<Instruction>, String> {
    handle_assert_not_eq(&mut program);
    todo!()
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
                let value_a = Value::from_var_or_constant(arg0, compiler).unwrap();
                let value_b = Value::from_var_or_constant(arg1, compiler).unwrap();
                let (new_var, new_shift) = compiler.add_var(var.name.clone());
                let instruction = Instruction::Computation {
                    operation: *operation,
                    arg_a: value_a,
                    arg_b: value_b,
                    res: Value::MemoryAfterFp { shift: new_shift },
                };
                res.push(instruction);
            }
            Line::RawAccess { var, index } => {
                let (new_var, new_shift) = compiler.add_var(var.name.clone());
                let right = Value::from_var_or_constant(index, compiler).unwrap();
                let instruction = Instruction::Eq {
                    left: Value::MemoryAfterFp { shift: new_shift },
                    right,
                };
                res.push(instruction);
            }
            Line::Assert(condition) => match condition {
                Boolean::Different { .. } => {
                    unreachable!("Assert not equal should have been handled earlier")
                }
                Boolean::Equal { left, right } => {
                    let left_value = Value::from_var_or_constant(left, compiler).unwrap();
                    let right_value = Value::from_var_or_constant(right, compiler).unwrap();
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
                Condition::NotZero { var } => {
                    let label_if = format!("if_not_zero_{}", compiler.condition_labels);
                    let label_else = format!("else_{}", compiler.condition_labels);
                    let label_end = format!("end_{}", compiler.condition_labels);
                    compiler.condition_labels += 1;

                    let mut if_compiled = compile_lines(then_branch, compiler)?;
                    if_compiled.push(Instruction::Jump {
                        dest: Value::Label(label_end.clone()),
                    });
                    compiler.bytecode.insert(label_if.clone(), if_compiled);

                    let mut else_compiled = compile_lines(else_branch, compiler)?;
                    else_compiled.push(Instruction::Jump {
                        dest: Value::Label(label_end.clone()),
                    });
                    compiler.bytecode.insert(label_else.clone(), else_compiled);

                    let end_lines = compile_lines(&lines[i + 1..], compiler)?;
                    compiler.bytecode.insert(label_end.clone(), end_lines);

                    let shift = compiler
                        .vars_in_scope
                        .get(var)
                        .ok_or_else(|| format!("Variable {} not found", var.name))?;
                    res.push(Instruction::JumpIfNotZero {
                        condition: Value::MemoryAfterFp { shift: *shift },
                        dest: Value::Label(label_if),
                    });
                    res.push(Instruction::Jump {
                        dest: Value::Label(label_else),
                    });
                    return Ok(res);
                }
            },
            Line::ForLoop {
                iterator,
                start,
                end,
                body,
            } => {}
            Line::FunctionCall {
                function_name,
                args,
                return_data,
            } => {}
            Line::Poseidon16 {
                arg0,
                arg1,
                res0,
                res1,
            } => {}
            _ => todo!(),
        }
    }
    Ok(res)
}
