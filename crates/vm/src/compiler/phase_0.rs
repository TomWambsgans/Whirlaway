use crate::{bytecode::intermediate_bytecode::HighLevelOperation, lang::*};

/// Convert assert_not_eq(a, b) to if a == b then panic
pub fn replace_assert_not_eq(program: &mut Program) {
    for function in program.functions.values_mut() {
        for line in &mut function.instructions {
            if let Line::Assert(Boolean::Different { left, right }) = line {
                *line = Line::IfCondition {
                    condition: Boolean::Equal {
                        left: left.clone(),
                        right: right.clone(),
                    },
                    then_branch: vec![Line::Panic],
                    else_branch: vec![],
                };
            }
        }
    }
}

/// Convert arr[index] to ptr = arr + index; *ptr
pub fn replace_array_access(program: &mut Program) {
    let mut counter = 0;
    for function in program.functions.values_mut() {
        replace_array_access_in_lines(&mut function.instructions, &mut counter);
    }
}

fn replace_array_access_in_lines(lines: &mut Vec<Line>, counter: &mut usize) {
    for i in (0..lines.len()).rev() {
        match &mut lines[i] {
            Line::ArrayAccess { value, array, index } => {
                let (value, array, index) = (value.clone(), array.clone(), index.clone());
                
                // Create pointer variable: ptr = array + index
                let ptr_var = Var { name: format!("@aux_var_{}", counter) };
                *counter += 1;
                lines.insert(i, Line::Assignment {
                    var: ptr_var.clone(),
                    operation: HighLevelOperation::Add,
                    arg0: array.into(),
                    arg1: index,
                });

                // Replace with raw access
                match value {
                    VarOrConstant::Var(var) => {
                        lines[i + 1] = Line::RawAccess { var, index: ptr_var.into() };
                    }
                    VarOrConstant::Constant(cst) => {
                        // Constants need to be assigned to variables first
                        let const_var = Var { name: format!("@aux_var_{}", counter) };
                        *counter += 1;
                        lines[i + 1] = Line::Assignment {
                            var: const_var.clone(),
                            operation: HighLevelOperation::Add,
                            arg0: cst.into(),
                            arg1: ConstantValue::Scalar(0).into(),
                        };
                        lines.insert(i + 2, Line::RawAccess {
                            var: const_var,
                            index: ptr_var.into(),
                        });
                    }
                }
            }
            Line::ForLoop { body, .. } => replace_array_access_in_lines(body, counter),
            Line::IfCondition { then_branch, else_branch, .. } => {
                replace_array_access_in_lines(then_branch, counter);
                replace_array_access_in_lines(else_branch, counter);
            }
            _ => {}
        }
    }
}

/// Convert if a == b then X else Y to if a != b then Y else X
pub fn replace_if_eq(program: &mut Program) {
    for function in program.functions.values_mut() {
        replace_if_eq_in_lines(&mut function.instructions);
    }
}

fn replace_if_eq_in_lines(lines: &mut Vec<Line>) {
    for line in lines {
        match line {
            Line::IfCondition { condition, then_branch, else_branch } => {
                // Process nested blocks first
                replace_if_eq_in_lines(then_branch);
                replace_if_eq_in_lines(else_branch);
                
                // Flip equality condition
                if let Boolean::Equal { left, right } = condition {
                    *line = Line::IfCondition {
                        condition: Boolean::Different {
                            left: left.clone(),
                            right: right.clone(),
                        },
                        then_branch: else_branch.clone(),
                        else_branch: then_branch.clone(),
                    };
                }
            }
            Line::ForLoop { body, .. } => replace_if_eq_in_lines(body),
            _ => {}
        }
    }
}

pub fn remove_memory_raw_accesses_with_constants(program: &mut Program) {
    let mut counter = 0;
    for function in program.functions.values_mut() {
        remove_constant_accesses_in_lines(&mut function.instructions, &mut counter);
    }
}

fn remove_constant_accesses_in_lines(lines: &mut Vec<Line>, counter: &mut usize) {
    for i in (0..lines.len()).rev() {
        match &mut lines[i] {
            Line::RawAccess { var, index } => {
                if let VarOrConstant::Constant(constant) = index {
                    let (var, constant) = (var.clone(), constant.clone());
                    
                    // Create variable for constant: temp = constant + 0
                    let temp_var = Var { name: format!("@var_to_constant_{}", counter) };
                    *counter += 1;
                    lines.insert(i, Line::Assignment {
                        var: temp_var.clone(),
                        operation: HighLevelOperation::Add,
                        arg0: constant.into(),
                        arg1: ConstantValue::Scalar(0).into(),
                    });
                    
                    // Replace with variable access
                    lines[i + 1] = Line::RawAccess {
                        var,
                        index: temp_var.into(),
                    };
                }
            }
            Line::IfCondition { then_branch, else_branch, .. } => {
                remove_constant_accesses_in_lines(then_branch, counter);
                remove_constant_accesses_in_lines(else_branch, counter);
            }
            Line::ForLoop { body, .. } => remove_constant_accesses_in_lines(body, counter),
            _ => {}
        }
    }
}