use crate::{bytecode::intermediate_bytecode::*, lang::*};
use std::collections::BTreeSet;

/// Convert for loops into recursive function calls
pub fn replace_loops_with_recursion(program: &mut Program) {
    let mut counter = 0;
    let mut new_functions = Vec::new();

    for function in program.functions.values_mut() {
        convert_loops_in_lines(&mut function.instructions, &mut counter, &mut new_functions);
    }

    // Add generated functions to program
    for func in new_functions {
        program.functions.insert(func.name.clone(), func);
    }
}

fn convert_loops_in_lines(
    lines: &mut Vec<Line>,
    counter: &mut usize,
    new_functions: &mut Vec<Function>,
) {
    for line in lines {
        match line {
            Line::ForLoop {
                iterator,
                start,
                end,
                body,
            } => {
                // Recursively process nested loops first
                convert_loops_in_lines(body, counter, new_functions);

                let func_name = format!("@loop{}", counter);
                *counter += 1;

                // Find variables used inside loop but defined outside
                let (_, mut external_vars) = find_variable_usage(&body);

                // Include start/end variables if they're variables
                if let VarOrConstant::Var(var) = start {
                    external_vars.insert(var.clone());
                }
                if let VarOrConstant::Var(var) = end {
                    external_vars.insert(var.clone());
                }
                external_vars.remove(iterator); // Iterator is internal to loop

                let external_vars: Vec<_> = external_vars.into_iter().collect();

                // Create function arguments: iterator + external variables
                let mut func_args = vec![iterator.clone()];
                func_args.extend(external_vars.clone());

                // Create recursive function body
                let recursive_func = create_recursive_function(
                    func_name.clone(),
                    func_args,
                    iterator.clone(),
                    end.clone(),
                    std::mem::take(body),
                    &external_vars,
                );
                new_functions.push(recursive_func);

                // Replace loop with initial function call
                let mut call_args: Vec<VarOrConstant> = vec![start.clone()];
                call_args.extend(external_vars.iter().map(|v| v.clone().into()));

                *line = Line::FunctionCall {
                    function_name: func_name,
                    args: call_args,
                    return_data: vec![],
                };
            }
            Line::IfCondition {
                then_branch,
                else_branch,
                ..
            } => {
                convert_loops_in_lines(then_branch, counter, new_functions);
                convert_loops_in_lines(else_branch, counter, new_functions);
            }
            _ => {} // Other line types don't contain loops
        }
    }
}

fn create_recursive_function(
    name: String,
    args: Vec<Var>,
    iterator: Var,
    end: VarOrConstant,
    mut body: Vec<Line>,
    external_vars: &[Var],
) -> Function {
    // Add iterator increment
    let next_iter = Var {
        name: format!("@incremented_{}", iterator.name),
    };
    body.push(Line::Assignment {
        var: next_iter.clone(),
        operation: HighLevelOperation::Add,
        arg0: iterator.clone().into(),
        arg1: VarOrConstant::Constant(ConstantValue::Scalar(1)),
    });

    // Add recursive call
    let mut recursive_args: Vec<VarOrConstant> = vec![next_iter.into()];
    recursive_args.extend(external_vars.iter().map(|v| v.clone().into()));

    body.push(Line::FunctionCall {
        function_name: name.clone(),
        args: recursive_args,
        return_data: vec![],
    });
    body.push(Line::FunctionRet {
        return_data: vec![],
    });

    // Create conditional: if iterator != end then body else return
    let condition = Line::IfCondition {
        condition: Boolean::Different {
            left: iterator.into(),
            right: end,
        },
        then_branch: body,
        else_branch: vec![Line::FunctionRet {
            return_data: vec![],
        }],
    };

    Function {
        name,
        arguments: args,
        n_returned_vars: 0,
        instructions: vec![condition],
    }
}

/// Find variables defined inside vs used from outside a block of code
pub fn find_variable_usage(lines: &[Line]) -> (BTreeSet<Var>, BTreeSet<Var>) {
    let mut defined_vars = BTreeSet::new();
    let mut used_vars = BTreeSet::new();

    for line in lines {
        match line {
            Line::Assignment {
                var, arg0, arg1, ..
            } => {
                defined_vars.insert(var.clone());
                add_var_if_external(arg0, &defined_vars, &mut used_vars);
                add_var_if_external(arg1, &defined_vars, &mut used_vars);
            }
            Line::IfCondition {
                condition,
                then_branch,
                else_branch,
            } => {
                add_condition_vars(condition, &defined_vars, &mut used_vars);

                let (then_defined, then_used) = find_variable_usage(then_branch);
                let (else_defined, else_used) = find_variable_usage(else_branch);

                defined_vars.extend(then_defined.union(&else_defined).cloned());
                used_vars.extend(
                    then_used
                        .union(&else_used)
                        .filter(|v| !defined_vars.contains(v))
                        .cloned(),
                );
            }
            Line::RawAccess { var, index } => {
                defined_vars.insert(var.clone());
                add_var_if_external(index, &defined_vars, &mut used_vars);
            }
            Line::FunctionCall {
                args, return_data, ..
            } => {
                for arg in args {
                    add_var_if_external(arg, &defined_vars, &mut used_vars);
                }
                defined_vars.extend(return_data.iter().cloned());
            }
            Line::Assert(condition) => {
                add_condition_vars(condition, &defined_vars, &mut used_vars);
            }
            Line::FunctionRet { return_data } => {
                for ret in return_data {
                    add_var_if_external(ret, &defined_vars, &mut used_vars);
                }
            }
            Line::MAlloc { var, .. } => {
                defined_vars.insert(var.clone());
            }
            Line::Poseidon16 {
                arg0,
                arg1,
                res0,
                res1,
            } => {
                add_var_if_external(arg0, &defined_vars, &mut used_vars);
                add_var_if_external(arg1, &defined_vars, &mut used_vars);
                defined_vars.insert(res0.clone());
                defined_vars.insert(res1.clone());
            }
            Line::Poseidon24 {
                arg0,
                arg1,
                arg2,
                res0,
                res1,
                res2,
            } => {
                add_var_if_external(arg0, &defined_vars, &mut used_vars);
                add_var_if_external(arg1, &defined_vars, &mut used_vars);
                add_var_if_external(arg2, &defined_vars, &mut used_vars);
                defined_vars.insert(res0.clone());
                defined_vars.insert(res1.clone());
                defined_vars.insert(res2.clone());
            }
            Line::Print { content, .. } => {
                for var in content {
                    add_var_if_external(var, &defined_vars, &mut used_vars);
                }
            }
            Line::ForLoop { .. } | Line::ArrayAccess { .. } => {
                unreachable!("Should have been replaced earlier");
            }
            Line::Panic => {}
        }
    }

    (defined_vars, used_vars)
}

fn add_var_if_external(
    var_or_const: &VarOrConstant,
    defined: &BTreeSet<Var>,
    used: &mut BTreeSet<Var>,
) {
    if let VarOrConstant::Var(var) = var_or_const {
        if !defined.contains(var) {
            used.insert(var.clone());
        }
    }
}

fn add_condition_vars(condition: &Boolean, defined: &BTreeSet<Var>, used: &mut BTreeSet<Var>) {
    match condition {
        Boolean::Equal { left, right } | Boolean::Different { left, right } => {
            add_var_if_external(left, defined, used);
            add_var_if_external(right, defined, used);
        }
    }
}
