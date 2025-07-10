use crate::{bytecode::intermediate_bytecode::HighLevelOperation, lang::*};

/// Replace all assert_not_eq with if condition eq then panic
pub fn replace_assert_not_eq(program: &mut Program) {
    for function in program.functions.values_mut() {
        for line in &mut function.instructions {
            if let Line::Assert(condition) = line {
                if let Boolean::Different { left, right } = condition {
                    let eq_condition = Boolean::Equal {
                        left: left.clone(),
                        right: right.clone(),
                    };
                    *line = Line::IfCondition {
                        condition: eq_condition,
                        then_branch: vec![Line::Panic],
                        else_branch: vec![],
                    };
                }
            }
        }
    }
}

pub fn replace_array_access(program: &mut Program) {
    let mut index_var_counter = 0;
    for function in program.functions.values_mut() {
        replace_array_access_helper(&mut function.instructions, &mut index_var_counter);
    }
}

fn replace_array_access_helper(lines: &mut Vec<Line>, aux_var_counter: &mut usize) {
    for i in (0..lines.len()).rev() {
        match &mut lines[i] {
            Line::ArrayAccess {
                value,
                array,
                index,
            } => {
                let (value, array, index) = (value.clone(), array.clone(), index.clone());
                let ptr_var = Var {
                    name: format!("@aux_var_{}", aux_var_counter),
                };
                *aux_var_counter += 1;
                lines.insert(
                    i,
                    Line::Assignment {
                        var: ptr_var.clone(),
                        operation: HighLevelOperation::Add,
                        arg0: array.into(),
                        arg1: index,
                    },
                );
                match value {
                    VarOrConstant::Var(var) => {
                        lines[i + 1] = Line::RawAccess {
                            var,
                            index: ptr_var.into(),
                        };
                    }
                    VarOrConstant::Constant(cst) => {
                        let constant_var = Var {
                            name: format!("@aux_var_{}", aux_var_counter),
                        };
                        *aux_var_counter += 1;
                        lines[i + 1] = Line::Assignment {
                            var: constant_var.clone(),
                            operation: HighLevelOperation::Add,
                            arg0: cst.into(),
                            arg1: ConstantValue::Scalar(0).into(),
                        };
                        lines.insert(
                            i + 2,
                            Line::RawAccess {
                                var: constant_var,
                                index: ptr_var.into(),
                            },
                        );
                    }
                }
            }
            Line::ForLoop { body, .. } => {
                replace_array_access_helper(body, aux_var_counter);
            }
            Line::IfCondition {
                then_branch,
                else_branch,
                ..
            } => {
                replace_array_access_helper(then_branch, aux_var_counter);
                replace_array_access_helper(else_branch, aux_var_counter);
            }
            _ => {}
        }
    }
}

/// Replace "if Eq then A else B" by "if NotEq then B else A"
pub fn replace_if_eq(program: &mut Program) {
    for function in program.functions.values_mut() {
        replace_if_eq_helper(&mut function.instructions);
    }
}

pub fn replace_if_eq_helper(lines: &mut Vec<Line>) {
    for line in lines {
        match line {
            Line::IfCondition {
                condition,
                then_branch,
                else_branch,
            } => {
                replace_if_eq_helper(then_branch);
                replace_if_eq_helper(else_branch);
                if let Boolean::Equal { left, right } = condition {
                    let not_eq_condition = Boolean::Different {
                        left: left.clone(),
                        right: right.clone(),
                    };
                    *line = Line::IfCondition {
                        condition: not_eq_condition,
                        then_branch: else_branch.clone(),
                        else_branch: then_branch.clone(),
                    };
                }
            }
            Line::ForLoop { body, .. } => {
                replace_if_eq_helper(body);
            }
            _ => {}
        }
    }
}


pub fn remove_memory_raw_accesses_with_constants(program: &mut Program) {
    let mut aux_var_counter = 0;
    for function in program.functions.values_mut() {
        remove_memory_raw_accesses_with_constants_helper(&mut function.instructions, &mut aux_var_counter);
    }
}

pub fn remove_memory_raw_accesses_with_constants_helper(lines: &mut Vec<Line>, aux_var_counter: &mut usize) {
    for i in (0..lines.len()).rev() {
        match &mut lines[i] {
            Line::IfCondition {
                condition: _,
                then_branch,
                else_branch,
            } => {
                remove_memory_raw_accesses_with_constants_helper(then_branch, aux_var_counter);
                remove_memory_raw_accesses_with_constants_helper(else_branch, aux_var_counter);
            }
            Line::ForLoop { body, .. } => {
                remove_memory_raw_accesses_with_constants_helper(body, aux_var_counter);
            }
            Line::RawAccess { var, index } => {
                if let VarOrConstant::Constant(constant) = index {
                    let (var, index) = (var.clone(), constant.clone());
                    let var_to_constant = Var {
                        name: format!("@var_to_constant_{}", aux_var_counter),
                    };
                    *aux_var_counter += 1;
                    lines.insert(
                        i,
                        Line::Assignment {
                            var: var_to_constant.clone(),
                            operation: HighLevelOperation::Add,
                            arg0: index.into(),
                            arg1: ConstantValue::Scalar(0).into(),
                        },
                    );
                    lines[i + 1] = Line::RawAccess {
                        var: var.clone(),
                        index: var_to_constant.into(),
                    };
                }
            }
            _ => {}
        }
    }
}

