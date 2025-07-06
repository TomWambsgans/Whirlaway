use crate::lang::*;

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
