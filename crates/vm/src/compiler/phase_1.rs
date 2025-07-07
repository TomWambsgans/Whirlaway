use std::collections::HashSet;

use crate::{bytecode::high_level::*, lang::*};

/// Replace all loops with recursive function
pub fn replace_loops_with_recursion(program: &mut Program) {
    let mut loop_counter = 0;
    let mut new_functions = Vec::new();
    for function in program.functions.values_mut() {
        replace_loops_with_recursion_helper(
            &mut function.instructions,
            &mut loop_counter,
            &mut new_functions,
        );
    }
    // Add new functions to the program
    for new_function in new_functions {
        program
            .functions
            .insert(new_function.name.clone(), new_function);
    }
}

fn replace_loops_with_recursion_helper(
    lines: &mut Vec<Line>,
    loop_counter: &mut usize,
    new_functions: &mut Vec<Function>,
) {
    for line in &mut *lines {
        match line {
            Line::ForLoop {
                iterator,
                start,
                end,
                body,
            } => {
                let function_name = format!("____loop____{}", loop_counter);
                *loop_counter += 1;

                let (_, mut external_vars) = get_internal_and_external_variables(&body);
                // Remove the iterator from external variables
                external_vars.remove(iterator);
                let external_vars = external_vars.into_iter().collect::<Vec<_>>();

                let mut function_arguments = vec![iterator.clone()];
                function_arguments.extend(external_vars.clone());

                let mut first_call_arguments: Vec<VarOrConstant> = vec![start.clone().into()];
                first_call_arguments.extend(external_vars.iter().map(|var| var.clone().into()));

                let mut then_branch = std::mem::take(body);

                let incremented_iterator = Var { 
                    name: format!("____{}____incremented", iterator.name),
                };
                then_branch.push(Line::Assignment {
                    var: incremented_iterator.clone(),
                    operation: HighLevelOperation::Add,
                    arg0: iterator.clone().into(),
                    arg1: VarOrConstant::Constant(ConstantValue::Scalar(1)),
                });

                let mut recursive_call_arguments: Vec<VarOrConstant> =
                    vec![incremented_iterator.into()];
                recursive_call_arguments.extend(external_vars.iter().map(|var| var.clone().into()));

                then_branch.push(Line::FunctionCall {
                    function_name: function_name.clone(),
                    args: recursive_call_arguments,
                    return_data: vec![],
                });
                then_branch.push(Line::FunctionRet {
                    return_data: vec![],
                });

                let if_condition = Line::IfCondition {
                    condition: Boolean::Different {
                        left: iterator.clone().into(),
                        right: end.clone(),
                    },
                    then_branch,
                    else_branch: vec![Line::FunctionRet {
                        return_data: vec![],
                    }],
                };

                // Create a recursive function for the loop
                let recursive_function = Function {
                    name: function_name.clone(),
                    arguments: function_arguments,
                    n_returned_vars: 0,
                    instructions: vec![if_condition],
                };
                new_functions.push(recursive_function);

                // Replace the loop with a call to the recursive function
                *line = Line::FunctionCall {
                    function_name,
                    args: first_call_arguments,
                    return_data: vec![],
                };
            }
            Line::IfCondition {
                condition: _,
                then_branch,
                else_branch,
            } => {
                // Recursively process the then and else branches
                replace_loops_with_recursion_helper(then_branch, loop_counter, new_functions);
                replace_loops_with_recursion_helper(else_branch, loop_counter, new_functions);
            }

            Line::Assert(_)
            | Line::AssertEqExt { .. }
            | Line::FunctionRet { .. }
            | Line::FunctionCall { .. }
            | Line::Poseidon16 { .. }
            | Line::Poseidon24 { .. }
            | Line::MAlloc { .. }
            | Line::Panic
            | Line::Assignment { .. }
            | Line::RawAccess { .. } => {}
        }
    }
}

pub fn get_internal_and_external_variables(lines: &[Line]) -> (HashSet<Var>, HashSet<Var>) {
    let mut external_vars = HashSet::new();
    let mut internal_vars = HashSet::new();
    for line in lines {
        match line {
            Line::Assignment {
                var,
                operation: _,
                arg0,
                arg1,
            } => {
                internal_vars.insert(var.clone());
                if let VarOrConstant::Var(arg) = arg0 {
                    if !internal_vars.contains(&arg) {
                        external_vars.insert(arg.clone());
                    }
                }
                if let VarOrConstant::Var(arg) = arg1 {
                    if !internal_vars.contains(&arg) {
                        external_vars.insert(arg.clone());
                    }
                }
            }
            Line::IfCondition {
                condition,
                then_branch,
                else_branch,
            } => {
                match condition {
                    Boolean::Equal { left, right } | Boolean::Different { left, right } => {
                        if let VarOrConstant::Var(var) = left {
                            if !internal_vars.contains(&var) {
                                external_vars.insert(var.clone());
                            }
                        }
                        if let VarOrConstant::Var(var) = right {
                            if !internal_vars.contains(&var) {
                                external_vars.insert(var.clone());
                            }
                        }
                    }
                }

                let (internal_then_branch, external_then_branch) =
                    get_internal_and_external_variables(then_branch);
                let (internal_else_branch, external_else_branch) =
                    get_internal_and_external_variables(else_branch);
                let new_internal_vars: HashSet<Var> = internal_then_branch
                    .union(&internal_else_branch)
                    .cloned()
                    .collect();
                let new_external_vars: HashSet<Var> = external_then_branch
                    .union(&external_else_branch)
                    .filter(|var| {
                        !internal_vars.contains(var)
                    })
                    .cloned()
                    .collect();
                internal_vars.extend(new_internal_vars);
                external_vars.extend(new_external_vars);
            }
            Line::RawAccess { var, index } => {
                internal_vars.insert(var.clone());
                if let VarOrConstant::Var(arg) = index {
                    if !internal_vars.contains(&arg) {
                        external_vars.insert(arg.clone());
                    }
                }
            }
            Line::FunctionCall {
                function_name: _,
                args,
                return_data,
            } => {
                for arg in args {
                    if let VarOrConstant::Var(arg_var) = arg {
                        internal_vars.insert(arg_var.clone());
                    }
                }
                for arg in args {
                    if let VarOrConstant::Var(arg_var) = arg {
                        if !internal_vars.contains(&arg_var) {
                            external_vars.insert(arg_var.clone());
                        }
                    }
                }
                for var in return_data {
                    internal_vars.insert(var.clone());
                }
            }
            Line::Assert(condition) => match condition {
                Boolean::Equal { left, right } | Boolean::Different { left, right } => {
                    if let VarOrConstant::Var(var) = left {
                        if !internal_vars.contains(&var) {
                            external_vars.insert(var.clone());
                        }
                    }
                    if let VarOrConstant::Var(var) = right {
                        if !internal_vars.contains(&var) {
                            external_vars.insert(var.clone());
                        }
                    }
                }
            },
            Line::ForLoop {
                iterator,
                start,
                end,
                body,
            } => {
                // Get the external variables used in the loop
                let (_, mut new_external_vars) = get_internal_and_external_variables(&body);
                new_external_vars.remove(iterator);
                external_vars.extend(new_external_vars);

                if let VarOrConstant::Var(var) = start {
                    if !internal_vars.contains(&var) {
                        external_vars.insert(var.clone());
                    }
                }
                if let VarOrConstant::Var(var) = end {
                    if !internal_vars.contains(&var) {
                        external_vars.insert(var.clone());
                    }
                }
            }
            Line::AssertEqExt { left, right } => {
                if let VarOrConstant::Var(var) = left {
                    if !internal_vars.contains(&var) {
                        external_vars.insert(var.clone());
                    }
                }
                if let VarOrConstant::Var(var) = right {
                    if !internal_vars.contains(&var) {
                        external_vars.insert(var.clone());
                    }
                }
            }
            Line::FunctionRet { return_data } => {
                for ret in return_data {
                    if let VarOrConstant::Var(var) = ret {
                        if !internal_vars.contains(&var) {
                            external_vars.insert(var.clone());
                        }
                    }
                }
            }
            Line::MAlloc { var, size: _ } => {
                internal_vars.insert(var.clone());
            }
            Line::Panic => {}
            Line::Poseidon16 {
                arg0,
                arg1,
                res0,
                res1,
            } => {
                if let VarOrConstant::Var(arg) = arg0 {
                    if !internal_vars.contains(&arg) {
                        external_vars.insert(arg.clone());
                    }
                }
                if let VarOrConstant::Var(arg) = arg1 {
                    if !internal_vars.contains(&arg) {
                        external_vars.insert(arg.clone());
                    }
                }
                internal_vars.insert(res0.clone());
                internal_vars.insert(res1.clone());
            }
            Line::Poseidon24 {
                arg0,
                arg1,
                arg2,
                res0,
                res1,
                res2,
            } => {
                if let VarOrConstant::Var(arg) = arg0 {
                    if !internal_vars.contains(&arg) {
                        external_vars.insert(arg.clone());
                    }
                }
                if let VarOrConstant::Var(arg) = arg1 {
                    if !internal_vars.contains(&arg) {
                        external_vars.insert(arg.clone());
                    }
                }
                if let VarOrConstant::Var(arg) = arg2 {
                    if !internal_vars.contains(&arg) {
                        external_vars.insert(arg.clone());
                    }
                }
                internal_vars.insert(res0.clone());
                internal_vars.insert(res1.clone());
                internal_vars.insert(res2.clone());
            }
        }
    }
    (internal_vars, external_vars)
}
