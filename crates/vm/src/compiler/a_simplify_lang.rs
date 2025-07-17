use std::collections::{BTreeMap, BTreeSet};

use crate::{
    bytecode::intermediate_bytecode::HighLevelOperation,
    lang::{Boolean, ConstExpression, Expression, Line, Program, Var, VarOrConstant},
};

#[derive(Debug, Clone)]
pub struct SimpleProgram {
    pub functions: BTreeMap<String, SimpleFunction>,
}

#[derive(Debug, Clone)]
pub struct SimpleFunction {
    pub name: String,
    pub arguments: Vec<Var>,
    pub n_returned_vars: usize,
    pub instructions: Vec<SimpleLine>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimpleLine {
    Assignment {
        var: Var,
        operation: HighLevelOperation,
        arg0: VarOrConstant,
        arg1: VarOrConstant,
    },
    RawAccess {
        var: Var,
        index: Var,
    }, // var = memory[index]
    IfNotZero {
        condition: VarOrConstant,
        then_branch: Vec<Self>,
        else_branch: Vec<Self>,
    },
    FunctionCall {
        function_name: String,
        args: Vec<VarOrConstant>,
        return_data: Vec<Var>,
    },
    FunctionRet {
        return_data: Vec<VarOrConstant>,
    },
    Poseidon16 {
        args: [VarOrConstant; 2],
        res: [Var; 2],
    },
    Poseidon24 {
        args: [VarOrConstant; 3],
        res: [Var; 3],
    },
    Print {
        line_info: String,
        content: Vec<VarOrConstant>,
    },
    MAlloc {
        var: Var,
        size: VarOrConstant,
    },
    Panic,
}

pub fn simplify_program(program: &Program) -> SimpleProgram {
    // println!("{}", program.to_string());
    let mut new_functions = BTreeMap::new();
    let mut counters = Counters::default();
    for (name, func) in &program.functions {
        let simplified_instructions =
            simplify_lines(&func.instructions, &mut counters, &mut new_functions);
        new_functions.insert(
            name.clone(),
            SimpleFunction {
                name: name.clone(),
                arguments: func.arguments.clone(),
                n_returned_vars: func.n_returned_vars,
                instructions: simplified_instructions,
            },
        );
    }
    SimpleProgram {
        functions: new_functions,
    }
}

#[derive(Debug, Clone, Default)]
struct Counters {
    aux_vars: usize,
    loops: usize,
    aux_arr: usize,
}

fn simplify_lines(
    lines: &[Line],
    counters: &mut Counters,
    new_functions: &mut BTreeMap<String, SimpleFunction>,
) -> Vec<SimpleLine> {
    let mut res = Vec::new();
    for line in lines {
        match line {
            Line::Assignment { var, value } => match value {
                Expression::Value(value) => {
                    res.push(SimpleLine::Assignment {
                        var: var.clone(),
                        operation: HighLevelOperation::Add,
                        arg0: value.clone(),
                        arg1: VarOrConstant::zero(),
                    });
                }
                Expression::ArrayAccess { array, index } => {
                    handle_array_assignment(
                        counters,
                        &mut res,
                        array.clone(),
                        index,
                        ArrayAccessType::VarIsAssigned(var.clone()),
                    );
                }
                Expression::Binary {
                    left,
                    operator,
                    right,
                } => {
                    let left = simplify_expr(left, &mut res, counters);
                    let right = simplify_expr(right, &mut res, counters);
                    res.push(SimpleLine::Assignment {
                        var: var.clone(),
                        operation: *operator,
                        arg0: left,
                        arg1: right,
                    });
                }
            },
            Line::ArrayAssign {
                array,
                index,
                value,
            } => {
                handle_array_assignment(
                    counters,
                    &mut res,
                    array.clone(),
                    index,
                    ArrayAccessType::ArrayIsAssigned(value.clone()),
                );
            }
            Line::Assert(boolean) => match boolean {
                Boolean::Different { left, right } => {
                    let left = simplify_expr(left, &mut res, counters);
                    let right = simplify_expr(right, &mut res, counters);
                    let diff_var = format!("@aux_var_{}", counters.aux_vars);
                    counters.aux_vars += 1;
                    res.push(SimpleLine::Assignment {
                        var: diff_var.clone(),
                        operation: HighLevelOperation::Sub,
                        arg0: left,
                        arg1: right,
                    });
                    res.push(SimpleLine::IfNotZero {
                        condition: diff_var.into(),
                        then_branch: vec![],
                        else_branch: vec![SimpleLine::Panic],
                    });
                }
                Boolean::Equal { left, right } => {
                    let left = simplify_expr(left, &mut res, counters);
                    let right = simplify_expr(right, &mut res, counters);
                    let (var, other) = if let VarOrConstant::Var(left) = left {
                        (left, right)
                    } else if let VarOrConstant::Var(right) = right {
                        (right, left)
                    } else {
                        unreachable!("Weird")
                    };
                    res.push(SimpleLine::Assignment {
                        var: var.clone(),
                        operation: HighLevelOperation::Add,
                        arg0: other.into(),
                        arg1: VarOrConstant::zero(),
                    });
                }
            },
            Line::IfCondition {
                condition,
                then_branch,
                else_branch,
            } => {
                // Transform if a == b then X else Y into if a != b then Y else X

                let (left, right, then_branch, else_branch) = match condition {
                    Boolean::Equal { left, right } => (left, right, else_branch, then_branch), // switched
                    Boolean::Different { left, right } => (left, right, then_branch, else_branch),
                };

                let left_simplified = simplify_expr(left, &mut res, counters);
                let right_simplified = simplify_expr(right, &mut res, counters);

                let diff_var = format!("@diff_{}", counters.aux_vars);
                counters.aux_vars += 1;
                res.push(SimpleLine::Assignment {
                    var: diff_var.clone(),
                    operation: HighLevelOperation::Sub,
                    arg0: left_simplified,
                    arg1: right_simplified,
                });

                let then_branch_simplified = simplify_lines(then_branch, counters, new_functions);
                let else_branch_simplified = simplify_lines(else_branch, counters, new_functions);

                res.push(SimpleLine::IfNotZero {
                    condition: diff_var.into(),
                    then_branch: then_branch_simplified,
                    else_branch: else_branch_simplified,
                });
            }
            Line::ForLoop {
                iterator,
                start,
                end,
                body,
            } => {
                let simplified_body = simplify_lines(body, counters, new_functions);

                let func_name = format!("@loop_{}", counters.loops);
                counters.loops += 1;

                // Find variables used inside loop but defined outside
                let (_, mut external_vars) = find_variable_usage(&body);

                // Include variables in start/end
                for expr in [start, end] {
                    for var in vars_in_expression(expr) {
                        external_vars.insert(var);
                    }
                }
                external_vars.remove(iterator); // Iterator is internal to loop

                let mut external_vars: Vec<_> = external_vars.into_iter().collect();

                let start_simplified = simplify_expr(start, &mut res, counters);
                let end_simplified = simplify_expr(end, &mut res, counters);

                for (simplified, original) in [
                    (start_simplified.clone(), start.clone()),
                    (end_simplified.clone(), end.clone()),
                ] {
                    if !matches!(original, Expression::Value(_)) {
                        // the simplified var is auxiliary
                        if let VarOrConstant::Var(var) = simplified {
                            external_vars.push(var);
                        }
                    }
                }

                // Create function arguments: iterator + external variables
                let mut func_args = vec![iterator.clone()];
                func_args.extend(external_vars.clone());

                // Create recursive function body
                let recursive_func = create_recursive_function(
                    func_name.clone(),
                    func_args,
                    iterator.clone(),
                    end_simplified,
                    simplified_body,
                    &external_vars,
                );
                new_functions.insert(func_name.clone(), recursive_func);

                // Replace loop with initial function call
                let mut call_args = vec![start_simplified];
                call_args.extend(external_vars.iter().map(|v| v.clone().into()));

                res.push(SimpleLine::FunctionCall {
                    function_name: func_name,
                    args: call_args,
                    return_data: vec![],
                });
            }
            Line::FunctionCall {
                function_name,
                args,
                return_data,
            } => {
                let simplified_args = args
                    .iter()
                    .map(|arg| simplify_expr(arg, &mut res, counters))
                    .collect::<Vec<_>>();
                res.push(SimpleLine::FunctionCall {
                    function_name: function_name.clone(),
                    args: simplified_args,
                    return_data: return_data.clone(),
                });
            }
            Line::FunctionRet { return_data } => {
                let simplified_return_data = return_data
                    .iter()
                    .map(|ret| simplify_expr(ret, &mut res, counters))
                    .collect::<Vec<_>>();
                res.push(SimpleLine::FunctionRet {
                    return_data: simplified_return_data,
                });
            }
            Line::Poseidon16 { args, res: ret } => {
                let simplified_args = [
                    simplify_expr(&args[0], &mut res, counters),
                    simplify_expr(&args[1], &mut res, counters),
                ];
                res.push(SimpleLine::Poseidon16 {
                    args: simplified_args,
                    res: ret.clone(),
                });
            }
            Line::Poseidon24 { args, res: ret } => {
                let simplified_args = [
                    simplify_expr(&args[0], &mut res, counters),
                    simplify_expr(&args[1], &mut res, counters),
                    simplify_expr(&args[2], &mut res, counters),
                ];
                res.push(SimpleLine::Poseidon24 {
                    args: simplified_args,
                    res: ret.clone(),
                });
            }
            Line::Print { line_info, content } => {
                let simplified_content = content
                    .iter()
                    .map(|var| simplify_expr(var, &mut res, counters))
                    .collect::<Vec<_>>();
                res.push(SimpleLine::Print {
                    line_info: line_info.clone(),
                    content: simplified_content,
                });
            }
            Line::MAlloc { var, size } => {
                let simplified_size = simplify_expr(size, &mut res, counters);
                res.push(SimpleLine::MAlloc {
                    var: var.clone(),
                    size: simplified_size,
                });
            }
            Line::Panic => {
                res.push(SimpleLine::Panic);
            }
        }
    }

    res
}

fn simplify_expr(
    expr: &Expression,
    lines: &mut Vec<SimpleLine>,
    counters: &mut Counters,
) -> VarOrConstant {
    match expr {
        Expression::Value(value) => return value.clone(),
        Expression::ArrayAccess { array, index } => {
            let aux_arr = format!("@aux_arr_{}", counters.aux_arr);
            counters.aux_arr += 1;
            handle_array_assignment(
                counters,
                lines,
                array.clone(),
                index,
                ArrayAccessType::VarIsAssigned(aux_arr.clone()),
            );
            return VarOrConstant::Var(aux_arr);
        }
        Expression::Binary {
            left,
            operator,
            right,
        } => {
            let left_var = simplify_expr(left, lines, counters);
            let right_var = simplify_expr(right, lines, counters);

            if let (VarOrConstant::Constant(left_cst), VarOrConstant::Constant(right_cst)) =
                (&left_var, &right_var)
            {
                return VarOrConstant::Constant(ConstExpression::Binary {
                    left: Box::new(left_cst.clone()),
                    operator: *operator,
                    right: Box::new(right_cst.clone()),
                });
            }

            let aux_var = format!("@aux_var_{}", counters.aux_vars);
            counters.aux_vars += 1;
            lines.push(SimpleLine::Assignment {
                var: aux_var.clone(),
                operation: *operator,
                arg0: left_var,
                arg1: right_var,
            });
            return VarOrConstant::Var(aux_var);
        }
    }
}

/// Returns (internal_vars, external_vars)
pub fn find_variable_usage(lines: &[Line]) -> (BTreeSet<Var>, BTreeSet<Var>) {
    let mut internal_vars = BTreeSet::new();
    let mut external_vars = BTreeSet::new();

    let on_new_expr =
        |expr: &Expression, internal_vars: &BTreeSet<Var>, external_vars: &mut BTreeSet<Var>| {
            for var in vars_in_expression(expr) {
                if !internal_vars.contains(&var) {
                    external_vars.insert(var);
                }
            }
        };

    let on_new_condition =
        |condition: &Boolean, internal_vars: &BTreeSet<Var>, external_vars: &mut BTreeSet<Var>| {
            let (Boolean::Equal { left, right } | Boolean::Different { left, right }) = condition;
            on_new_expr(left, internal_vars, external_vars);
            on_new_expr(right, internal_vars, external_vars);
        };

    for line in lines {
        match line {
            Line::Assignment { var, value } => {
                on_new_expr(value, &internal_vars, &mut external_vars);
                internal_vars.insert(var.clone());
            }
            Line::IfCondition {
                condition,
                then_branch,
                else_branch,
            } => {
                on_new_condition(condition, &internal_vars, &mut external_vars);

                let (then_internal, then_external) = find_variable_usage(then_branch);
                let (else_internal, else_external) = find_variable_usage(else_branch);

                internal_vars.extend(then_internal.union(&else_internal).cloned());
                external_vars.extend(
                    then_external
                        .union(&else_external)
                        .filter(|v| !internal_vars.contains(*v))
                        .cloned(),
                );
            }
            Line::FunctionCall {
                args, return_data, ..
            } => {
                for arg in args {
                    on_new_expr(arg, &internal_vars, &mut external_vars);
                }
                internal_vars.extend(return_data.iter().cloned());
            }
            Line::Assert(condition) => {
                on_new_condition(condition, &internal_vars, &mut external_vars);
            }
            Line::FunctionRet { return_data } => {
                for ret in return_data {
                    on_new_expr(ret, &internal_vars, &mut external_vars);
                }
            }
            Line::MAlloc { var, .. } => {
                internal_vars.insert(var.clone());
            }
            Line::Poseidon16 { args, res } => {
                for arg in args {
                    on_new_expr(arg, &internal_vars, &mut external_vars);
                }
                for r in res {
                    internal_vars.insert(r.clone());
                }
            }
            Line::Poseidon24 { args, res } => {
                for arg in args {
                    on_new_expr(arg, &internal_vars, &mut external_vars);
                }
                for r in res {
                    internal_vars.insert(r.clone());
                }
            }
            Line::Print { content, .. } => {
                for var in content {
                    on_new_expr(var, &internal_vars, &mut external_vars);
                }
            }
            Line::ForLoop {
                iterator,
                start,
                end,
                body,
            } => {
                let (body_internal, body_external) = find_variable_usage(body);
                internal_vars.extend(body_internal);
                internal_vars.insert(iterator.clone());
                external_vars.extend(body_external.difference(&internal_vars).cloned());
                on_new_expr(start, &internal_vars, &mut external_vars);
                on_new_expr(end, &internal_vars, &mut external_vars);
            }
            Line::ArrayAssign {
                array,
                index,
                value,
            } => {
                on_new_expr(&array.clone().into(), &internal_vars, &mut external_vars);
                on_new_expr(index, &internal_vars, &mut external_vars);
                on_new_expr(value, &internal_vars, &mut external_vars);
            }
            Line::Panic => {}
        }
    }

    (internal_vars, external_vars)
}

fn vars_in_expression(expr: &Expression) -> BTreeSet<Var> {
    let mut vars = BTreeSet::new();
    match expr {
        Expression::Value(value) => {
            if let VarOrConstant::Var(var) = value {
                vars.insert(var.clone());
            }
        }
        Expression::ArrayAccess { array, index } => {
            vars.insert(array.clone());
            vars.extend(vars_in_expression(index));
        }
        Expression::Binary { left, right, .. } => {
            vars.extend(vars_in_expression(left));
            vars.extend(vars_in_expression(right));
        }
    }
    vars
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArrayAccessType {
    VarIsAssigned(Var),          // var = array[index]
    ArrayIsAssigned(Expression), // array[index] = expr
}

fn handle_array_assignment(
    counters: &mut Counters,
    res: &mut Vec<SimpleLine>,
    array: Var,
    index: &Expression,
    access_type: ArrayAccessType,
) {
    // Create pointer variable: ptr = array + index
    let ptr_var = format!("@aux_var_{}", counters.aux_vars);
    counters.aux_vars += 1;

    let simplified_index = simplify_expr(index, res, counters);

    res.push(SimpleLine::Assignment {
        var: ptr_var.clone(),
        operation: HighLevelOperation::Add,
        arg0: array.clone().into(),
        arg1: simplified_index,
    });

    let value_simplified = match access_type {
        ArrayAccessType::VarIsAssigned(var) => VarOrConstant::Var(var.clone()),
        ArrayAccessType::ArrayIsAssigned(expr) => simplify_expr(&expr, res, counters),
    };

    // Replace with raw access
    match value_simplified {
        VarOrConstant::Var(var) => {
            res.push(SimpleLine::RawAccess {
                var,
                index: ptr_var.into(),
            });
        }
        VarOrConstant::Constant(cst) => {
            // Constants need to be assigned to variables first
            let const_var = format!("@aux_var_{}", counters.aux_vars);
            counters.aux_vars += 1;
            res.push(SimpleLine::Assignment {
                var: const_var.clone(),
                operation: HighLevelOperation::Add,
                arg0: cst.into(),
                arg1: VarOrConstant::zero(),
            });
            res.push(SimpleLine::RawAccess {
                var: const_var,
                index: ptr_var.into(),
            });
        }
    }
}

fn create_recursive_function(
    name: String,
    args: Vec<Var>,
    iterator: Var,
    end: VarOrConstant,
    mut body: Vec<SimpleLine>,
    external_vars: &[Var],
) -> SimpleFunction {
    // Add iterator increment
    let next_iter = format!("@incremented_{}", iterator);
    body.push(SimpleLine::Assignment {
        var: next_iter.clone(),
        operation: HighLevelOperation::Add,
        arg0: iterator.clone().into(),
        arg1: VarOrConstant::one(),
    });

    // Add recursive call
    let mut recursive_args: Vec<VarOrConstant> = vec![next_iter.into()];
    recursive_args.extend(external_vars.iter().map(|v| v.clone().into()));

    body.push(SimpleLine::FunctionCall {
        function_name: name.clone(),
        args: recursive_args,
        return_data: vec![],
    });
    body.push(SimpleLine::FunctionRet {
        return_data: vec![],
    });

    let diff_var = format!("@diff_{}", iterator);

    let instructions = vec![
        SimpleLine::Assignment {
            var: diff_var.clone(),
            operation: HighLevelOperation::Sub,
            arg0: iterator.into(),
            arg1: end,
        },
        SimpleLine::IfNotZero {
            condition: diff_var.into(),
            then_branch: body,
            else_branch: vec![SimpleLine::FunctionRet {
                return_data: vec![],
            }],
        },
    ];

    SimpleFunction {
        name,
        arguments: args,
        n_returned_vars: 0,
        instructions,
    }
}

impl ToString for SimpleLine {
    fn to_string(&self) -> String {
        self.to_string_with_indent(0)
    }
}

impl SimpleLine {
    fn to_string_with_indent(&self, indent: usize) -> String {
        let spaces = "    ".repeat(indent);
        let line_str = match self {
            SimpleLine::Assignment {
                var,
                operation,
                arg0,
                arg1,
            } => {
                format!(
                    "{} = {} {} {}",
                    var.to_string(),
                    arg0.to_string(),
                    operation.to_string(),
                    arg1.to_string()
                )
            }
            SimpleLine::RawAccess { var, index } => {
                format!("{} = memory[{}]", var.to_string(), index.to_string())
            }
            SimpleLine::IfNotZero {
                condition,
                then_branch,
                else_branch,
            } => {
                let then_str = then_branch
                    .iter()
                    .map(|line| line.to_string_with_indent(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");

                let else_str = else_branch
                    .iter()
                    .map(|line| line.to_string_with_indent(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");

                if else_branch.is_empty() {
                    format!(
                        "if {} != 0 {{\n{}\n{}}}",
                        condition.to_string(),
                        then_str,
                        spaces
                    )
                } else {
                    format!(
                        "if {} != 0 {{\n{}\n{}}} else {{\n{}\n{}}}",
                        condition.to_string(),
                        then_str,
                        spaces,
                        else_str,
                        spaces
                    )
                }
            }
            SimpleLine::FunctionCall {
                function_name,
                args,
                return_data,
            } => {
                let args_str = args
                    .iter()
                    .map(|arg| arg.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let return_data_str = return_data
                    .iter()
                    .map(|var| var.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");

                if return_data.is_empty() {
                    format!("{}({})", function_name, args_str)
                } else {
                    format!("{} = {}({})", return_data_str, function_name, args_str)
                }
            }
            SimpleLine::FunctionRet { return_data } => {
                let return_data_str = return_data
                    .iter()
                    .map(|arg| arg.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("return {}", return_data_str)
            }
            SimpleLine::Poseidon16 {
                args: [arg0, arg1],
                res: [res0, res1],
            } => {
                format!(
                    "{}, {} = poseidon16({}, {})",
                    res0.to_string(),
                    res1.to_string(),
                    arg0.to_string(),
                    arg1.to_string()
                )
            }
            SimpleLine::Poseidon24 {
                args: [arg0, arg1, arg2],
                res: [res0, res1, res2],
            } => {
                format!(
                    "{}, {}, {} = poseidon24({}, {}, {})",
                    res0.to_string(),
                    res1.to_string(),
                    res2.to_string(),
                    arg0.to_string(),
                    arg1.to_string(),
                    arg2.to_string()
                )
            }
            SimpleLine::Print {
                line_info: _,
                content,
            } => {
                let content_str = content
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("print({})", content_str)
            }
            SimpleLine::MAlloc { var, size } => {
                format!("{} = malloc({})", var.to_string(), size.to_string())
            }
            SimpleLine::Panic => "panic".to_string(),
        };
        format!("{}{}", spaces, line_str)
    }
}

impl ToString for SimpleFunction {
    fn to_string(&self) -> String {
        let args_str = self
            .arguments
            .iter()
            .map(|arg| arg.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let instructions_str = self
            .instructions
            .iter()
            .map(|line| line.to_string_with_indent(1))
            .collect::<Vec<_>>()
            .join("\n");

        if self.instructions.is_empty() {
            format!(
                "fn {}({}) -> {} {{}}",
                self.name, args_str, self.n_returned_vars
            )
        } else {
            format!(
                "fn {}({}) -> {} {{\n{}\n}}",
                self.name, args_str, self.n_returned_vars, instructions_str
            )
        }
    }
}

impl ToString for SimpleProgram {
    fn to_string(&self) -> String {
        let mut result = String::new();
        for (i, function) in self.functions.values().enumerate() {
            if i > 0 {
                result.push('\n');
            }
            result.push_str(&function.to_string());
        }
        result
    }
}
