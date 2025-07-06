use crate::lang::*;

/* 
Replace all assert_not_eq with if condition eq then panic
*/

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
