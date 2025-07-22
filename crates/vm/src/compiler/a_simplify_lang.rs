use crate::{
    bytecode::intermediate_bytecode::HighLevelOperation,
    lang::{
        Boolean, ConstExpression, ConstMallocLabel, Expression, Line, Program, SimpleExpr, Var,
    },
};
use p3_field::PrimeField64;
use std::collections::{BTreeMap, BTreeSet};

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
pub enum VarOrConstMallocAccess {
    Var(Var),
    ConstMallocAccess {
        malloc_label: ConstMallocLabel,
        offset: ConstExpression,
    },
}

impl From<VarOrConstMallocAccess> for SimpleExpr {
    fn from(var_or_const: VarOrConstMallocAccess) -> Self {
        match var_or_const {
            VarOrConstMallocAccess::Var(var) => SimpleExpr::Var(var),
            VarOrConstMallocAccess::ConstMallocAccess {
                malloc_label,
                offset,
            } => SimpleExpr::ConstMallocAccess {
                malloc_label,
                offset,
            },
        }
    }
}

impl TryInto<VarOrConstMallocAccess> for SimpleExpr {
    type Error = ();

    fn try_into(self) -> Result<VarOrConstMallocAccess, Self::Error> {
        match self {
            SimpleExpr::Var(var) => Ok(VarOrConstMallocAccess::Var(var)),
            SimpleExpr::ConstMallocAccess {
                malloc_label,
                offset,
            } => Ok(VarOrConstMallocAccess::ConstMallocAccess {
                malloc_label,
                offset,
            }),
            _ => Err(()),
        }
    }
}

impl From<Var> for VarOrConstMallocAccess {
    fn from(var: Var) -> Self {
        Self::Var(var)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimpleLine {
    Assignment {
        var: VarOrConstMallocAccess,
        operation: HighLevelOperation,
        arg0: SimpleExpr,
        arg1: SimpleExpr,
    },
    RawAccess {
        res: SimpleExpr,
        index: Var,
        shift: ConstExpression,
    }, // res = memory[index + shift]
    IfNotZero {
        condition: SimpleExpr,
        then_branch: Vec<Self>,
        else_branch: Vec<Self>,
    },
    FunctionCall {
        function_name: String,
        args: Vec<SimpleExpr>,
        return_data: Vec<Var>,
    },
    FunctionRet {
        return_data: Vec<SimpleExpr>,
    },
    Poseidon16 {
        args: [SimpleExpr; 2],
        res: [Var; 2],
    },
    Poseidon24 {
        args: [SimpleExpr; 3],
        res: [Var; 3],
    },
    ExtensionMul {
        args: [SimpleExpr; 3],
    },
    Panic,
    // Hints
    DecomposeBits {
        var: Var, // a pointer to 31 field elements, containing the bits of "to_decompose"
        to_decompose: SimpleExpr,
    },
    Print {
        line_info: String,
        content: Vec<SimpleExpr>,
    },
    HintMAlloc {
        var: Var,
        size: SimpleExpr,
        vectorized: bool,
    },
    ConstMalloc {
        // always not vectorized
        var: Var,
        size: ConstExpression,
        label: ConstMallocLabel,
    },
}

pub fn simplify_program(program: &Program) -> SimpleProgram {
    // println!("{}", program.to_string());
    let mut new_functions = BTreeMap::new();
    let mut counters = Counters::default();
    let mut const_malloc = ConstMalloc::default();
    for (name, func) in &program.functions {
        let mut array_manager = ArrayManager::default();
        let simplified_instructions = simplify_lines(
            &func.instructions,
            &mut counters,
            &mut new_functions,
            false,
            &mut array_manager,
            &mut const_malloc,
        );
        new_functions.insert(
            name.clone(),
            SimpleFunction {
                name: name.clone(),
                arguments: func.arguments.clone(),
                n_returned_vars: func.n_returned_vars,
                instructions: simplified_instructions,
            },
        );
        const_malloc.map.clear();
    }
    SimpleProgram {
        functions: new_functions,
    }
}

#[derive(Debug, Clone, Default)]
struct Counters {
    aux_vars: usize,
    loops: usize,
}

#[derive(Debug, Clone, Default)]
struct ArrayManager {
    counter: usize,
    aux_vars: BTreeMap<(Var, Expression), Var>, // (array, index) -> aux_var
    valid: BTreeSet<Var>,                       // currently valid aux vars
}

#[derive(Debug, Clone, Default)]
pub struct ConstMalloc {
    counter: usize,
    map: BTreeMap<Var, ConstMallocLabel>,
    if_else_branch: bool, // ware we in an if/else branch?
}

impl ArrayManager {
    fn get_aux_var(&mut self, array: &Var, index: &Expression) -> Var {
        if let Some(var) = self.aux_vars.get(&(array.clone(), index.clone())) {
            return var.clone();
        }
        let new_var = format!("@arr_aux_{}", self.counter);
        self.counter += 1;
        self.aux_vars
            .insert((array.clone(), index.clone()), new_var.clone());
        new_var
    }
}

fn simplify_lines(
    lines: &[Line],
    counters: &mut Counters,
    new_functions: &mut BTreeMap<String, SimpleFunction>,
    in_a_loop: bool,
    array_manager: &mut ArrayManager,
    const_malloc: &mut ConstMalloc,
) -> Vec<SimpleLine> {
    let mut res = Vec::new();
    for line in lines {
        match line {
            Line::Assignment { var, value } => match value {
                Expression::Value(value) => {
                    res.push(SimpleLine::Assignment {
                        var: var.clone().into(),
                        operation: HighLevelOperation::Add,
                        arg0: value.clone(),
                        arg1: SimpleExpr::zero(),
                    });
                }
                Expression::ArrayAccess { array, index } => {
                    handle_array_assignment(
                        counters,
                        &mut res,
                        array.clone(),
                        index,
                        ArrayAccessType::VarIsAssigned(var.clone()),
                        array_manager,
                        const_malloc,
                    );
                }
                Expression::Binary {
                    left,
                    operation,
                    right,
                } => {
                    let left = simplify_expr(left, &mut res, counters, array_manager, const_malloc);
                    let right =
                        simplify_expr(right, &mut res, counters, array_manager, const_malloc);
                    res.push(SimpleLine::Assignment {
                        var: var.clone().into(),
                        operation: *operation,
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
                    array_manager,
                    const_malloc,
                );
            }
            Line::Assert(boolean) => match boolean {
                Boolean::Different { left, right } => {
                    let left = simplify_expr(left, &mut res, counters, array_manager, const_malloc);
                    let right =
                        simplify_expr(right, &mut res, counters, array_manager, const_malloc);
                    let diff_var = format!("@aux_var_{}", counters.aux_vars);
                    counters.aux_vars += 1;
                    res.push(SimpleLine::Assignment {
                        var: diff_var.clone().into(),
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
                    let left = simplify_expr(left, &mut res, counters, array_manager, const_malloc);
                    let right =
                        simplify_expr(right, &mut res, counters, array_manager, const_malloc);
                    let (var, other) = if let SimpleExpr::Var(left) = left {
                        (left, right)
                    } else if let SimpleExpr::Var(right) = right {
                        (right, left)
                    } else {
                        unreachable!("Weird")
                    };
                    res.push(SimpleLine::Assignment {
                        var: var.clone().into(),
                        operation: HighLevelOperation::Add,
                        arg0: other.into(),
                        arg1: SimpleExpr::zero(),
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

                let left_simplified =
                    simplify_expr(left, &mut res, counters, array_manager, const_malloc);
                let right_simplified =
                    simplify_expr(right, &mut res, counters, array_manager, const_malloc);

                let diff_var = format!("@diff_{}", counters.aux_vars);
                counters.aux_vars += 1;
                res.push(SimpleLine::Assignment {
                    var: diff_var.clone().into(),
                    operation: HighLevelOperation::Sub,
                    arg0: left_simplified,
                    arg1: right_simplified,
                });

                let mut then_const_malloc = const_malloc.clone();
                let mut else_const_malloc = const_malloc.clone();
                then_const_malloc.if_else_branch = true;
                else_const_malloc.if_else_branch = true;

                let mut array_manager_then = array_manager.clone();
                let then_branch_simplified = simplify_lines(
                    then_branch,
                    counters,
                    new_functions,
                    in_a_loop,
                    &mut array_manager_then,
                    &mut then_const_malloc,
                );
                let mut array_manager_else = array_manager_then.clone();
                array_manager_else.valid = array_manager.valid.clone(); // Crucial: remove the access added in the IF branch

                let else_branch_simplified = simplify_lines(
                    else_branch,
                    counters,
                    new_functions,
                    in_a_loop,
                    &mut array_manager_else,
                    &mut else_const_malloc,
                );

                *array_manager = array_manager_else.clone();
                // keep the intersection both branches
                array_manager.valid = array_manager
                    .valid
                    .intersection(&array_manager_then.valid)
                    .cloned()
                    .collect();

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
                unroll,
            } => {
                if *unroll {
                    let (internal_variables, _) = find_variable_usage(&body);
                    let mut unrolled_lines = Vec::new();
                    let start_evaluated = start.naive_eval().unwrap().as_canonical_u64() as usize;
                    let end_evaluated = end.naive_eval().unwrap().as_canonical_u64() as usize;

                    for i in start_evaluated..end_evaluated {
                        let mut body_copy = body.clone();
                        replace_vars_for_unroll(&mut body_copy, iterator, i, &internal_variables);
                        let mut loop_const_malloc = ConstMalloc::default(); // don't give access to current const_malloc segments
                        loop_const_malloc.counter = const_malloc.counter;
                        unrolled_lines.extend(simplify_lines(
                            &body_copy,
                            counters,
                            new_functions,
                            in_a_loop,
                            array_manager,
                            &mut loop_const_malloc,
                        ));
                        const_malloc.counter = loop_const_malloc.counter;
                    }
                    res.extend(unrolled_lines);
                    continue;
                }

                let mut loop_const_malloc = ConstMalloc::default();
                loop_const_malloc.counter = const_malloc.counter;
                let valid_aux_vars_in_array_manager_before = array_manager.valid.clone();
                array_manager.valid.clear();
                let simplified_body = simplify_lines(
                    body,
                    counters,
                    new_functions,
                    true,
                    array_manager,
                    &mut loop_const_malloc,
                );
                const_malloc.counter = loop_const_malloc.counter;
                array_manager.valid = valid_aux_vars_in_array_manager_before; // restore the valid aux vars

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

                let start_simplified =
                    simplify_expr(start, &mut res, counters, array_manager, const_malloc);
                let end_simplified =
                    simplify_expr(end, &mut res, counters, array_manager, const_malloc);

                for (simplified, original) in [
                    (start_simplified.clone(), start.clone()),
                    (end_simplified.clone(), end.clone()),
                ] {
                    if !matches!(original, Expression::Value(_)) {
                        // the simplified var is auxiliary
                        if let SimpleExpr::Var(var) = simplified {
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
                    .map(|arg| simplify_expr(arg, &mut res, counters, array_manager, const_malloc))
                    .collect::<Vec<_>>();
                res.push(SimpleLine::FunctionCall {
                    function_name: function_name.clone(),
                    args: simplified_args,
                    return_data: return_data.clone(),
                });
            }
            Line::FunctionRet { return_data } => {
                assert!(
                    !in_a_loop,
                    "Function return inside a loop is not currently supported"
                );
                let simplified_return_data = return_data
                    .iter()
                    .map(|ret| simplify_expr(ret, &mut res, counters, array_manager, const_malloc))
                    .collect::<Vec<_>>();
                res.push(SimpleLine::FunctionRet {
                    return_data: simplified_return_data,
                });
            }
            Line::Poseidon16 { args, res: ret } => {
                let simplified_args = [
                    simplify_expr(&args[0], &mut res, counters, array_manager, const_malloc),
                    simplify_expr(&args[1], &mut res, counters, array_manager, const_malloc),
                ];
                res.push(SimpleLine::Poseidon16 {
                    args: simplified_args,
                    res: ret.clone(),
                });
            }
            Line::Poseidon24 { args, res: ret } => {
                let simplified_args = [
                    simplify_expr(&args[0], &mut res, counters, array_manager, const_malloc),
                    simplify_expr(&args[1], &mut res, counters, array_manager, const_malloc),
                    simplify_expr(&args[2], &mut res, counters, array_manager, const_malloc),
                ];
                res.push(SimpleLine::Poseidon24 {
                    args: simplified_args,
                    res: ret.clone(),
                });
            }
            Line::ExtensionMul { args } => {
                let simplified_args = [
                    simplify_expr(&args[0], &mut res, counters, array_manager, const_malloc),
                    simplify_expr(&args[1], &mut res, counters, array_manager, const_malloc),
                    simplify_expr(&args[2], &mut res, counters, array_manager, const_malloc),
                ];
                res.push(SimpleLine::ExtensionMul {
                    args: simplified_args,
                });
            }
            Line::Print { line_info, content } => {
                let simplified_content = content
                    .iter()
                    .map(|var| simplify_expr(var, &mut res, counters, array_manager, const_malloc))
                    .collect::<Vec<_>>();
                res.push(SimpleLine::Print {
                    line_info: line_info.clone(),
                    content: simplified_content,
                });
            }
            Line::Break => {
                assert!(in_a_loop, "Break statement outside of a loop");
                res.push(SimpleLine::FunctionRet {
                    return_data: vec![],
                });
            }
            Line::MAlloc {
                var,
                size,
                vectorized,
            } => {
                let simplified_size =
                    simplify_expr(size, &mut res, counters, array_manager, const_malloc);
                match simplified_size {
                    SimpleExpr::Constant(const_size)
                        if !*vectorized && !const_malloc.if_else_branch =>
                    {
                        // TODO do this optimization even if we are in an if/else branch
                        let label = const_malloc.counter;
                        const_malloc.counter += 1;
                        const_malloc.map.insert(var.clone(), label);
                        res.push(SimpleLine::ConstMalloc {
                            var: var.clone(),
                            size: const_size,
                            label,
                        });
                    }
                    _ => {
                        res.push(SimpleLine::HintMAlloc {
                            var: var.clone(),
                            size: simplified_size,
                            vectorized: *vectorized,
                        });
                    }
                }
            }
            Line::DecomposeBits { var, to_decompose } => {
                let simplified_to_decompose = simplify_expr(
                    &to_decompose,
                    &mut res,
                    counters,
                    array_manager,
                    const_malloc,
                );
                res.push(SimpleLine::DecomposeBits {
                    var: var.clone(),
                    to_decompose: simplified_to_decompose,
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
    array_manager: &mut ArrayManager,
    const_malloc: &ConstMalloc,
) -> SimpleExpr {
    match expr {
        Expression::Value(value) => return value.clone(),
        Expression::ArrayAccess { array, index } => {
            if let Some(label) = const_malloc.map.get(array) {
                if let Ok(offset) = ConstExpression::try_from(*index.clone()) {
                    return SimpleExpr::ConstMallocAccess {
                        malloc_label: *label,
                        offset,
                    };
                }
            }

            let aux_arr = array_manager.get_aux_var(array, index); // auxiliary var to store m[array + index]

            if !array_manager.valid.insert(aux_arr.clone()) {
                return SimpleExpr::Var(aux_arr.clone());
            }

            handle_array_assignment(
                counters,
                lines,
                array.clone(),
                index,
                ArrayAccessType::VarIsAssigned(aux_arr.clone()),
                array_manager,
                const_malloc,
            );
            return SimpleExpr::Var(aux_arr);
        }
        Expression::Binary {
            left,
            operation,
            right,
        } => {
            let left_var = simplify_expr(left, lines, counters, array_manager, const_malloc);
            let right_var = simplify_expr(right, lines, counters, array_manager, const_malloc);

            if let (SimpleExpr::Constant(left_cst), SimpleExpr::Constant(right_cst)) =
                (&left_var, &right_var)
            {
                return SimpleExpr::Constant(ConstExpression::Binary {
                    left: Box::new(left_cst.clone()),
                    operation: *operation,
                    right: Box::new(right_cst.clone()),
                });
            }

            let aux_var = format!("@aux_var_{}", counters.aux_vars);
            counters.aux_vars += 1;
            lines.push(SimpleLine::Assignment {
                var: aux_var.clone().into(),
                operation: *operation,
                arg0: left_var,
                arg1: right_var,
            });
            return SimpleExpr::Var(aux_var);
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
            Line::MAlloc { var, size, .. } => {
                on_new_expr(size, &internal_vars, &mut external_vars);
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
            Line::ExtensionMul { args } => {
                for arg in args {
                    on_new_expr(arg, &internal_vars, &mut external_vars);
                }
            }
            Line::Print { content, .. } => {
                for var in content {
                    on_new_expr(var, &internal_vars, &mut external_vars);
                }
            }
            Line::DecomposeBits { var, to_decompose } => {
                on_new_expr(&to_decompose, &internal_vars, &mut external_vars);
                internal_vars.insert(var.clone());
            }
            Line::ForLoop {
                iterator,
                start,
                end,
                body,
                unroll: _,
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
            Line::Panic | Line::Break => {}
        }
    }

    (internal_vars, external_vars)
}

fn vars_in_expression(expr: &Expression) -> BTreeSet<Var> {
    let mut vars = BTreeSet::new();
    match expr {
        Expression::Value(value) => {
            if let SimpleExpr::Var(var) = value {
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
    array_manager: &mut ArrayManager,
    const_malloc: &ConstMalloc,
) {
    let simplified_index = simplify_expr(index, res, counters, array_manager, const_malloc);

    if let SimpleExpr::Constant(offset) = simplified_index.clone() {
        if let Some(label) = const_malloc.map.get(&array) {
            if let ArrayAccessType::ArrayIsAssigned(Expression::Binary {
                left,
                operation,
                right,
            }) = access_type
            {
                let arg0 = simplify_expr(&left, res, counters, array_manager, const_malloc);
                let arg1 = simplify_expr(&right, res, counters, array_manager, const_malloc);
                res.push(SimpleLine::Assignment {
                    var: VarOrConstMallocAccess::ConstMallocAccess {
                        malloc_label: label.clone(),
                        offset,
                    },
                    operation,
                    arg0,
                    arg1,
                });
                return;
            }
        }
    }

    let value_simplified = match access_type {
        ArrayAccessType::VarIsAssigned(var) => SimpleExpr::Var(var.clone()),
        ArrayAccessType::ArrayIsAssigned(expr) => {
            simplify_expr(&expr, res, counters, array_manager, const_malloc)
        }
    };

    // TODO opti: in some case we could use ConstMallocAccess

    let (index_var, shift) = match simplified_index {
        SimpleExpr::Constant(c) => (array, c),
        _ => {
            // Create pointer variable: ptr = array + index
            let ptr_var = format!("@aux_var_{}", counters.aux_vars);
            counters.aux_vars += 1;
            res.push(SimpleLine::Assignment {
                var: ptr_var.clone().into(),
                operation: HighLevelOperation::Add,
                arg0: array.clone().into(),
                arg1: simplified_index.into(),
            });
            (ptr_var, ConstExpression::zero())
        }
    };

    res.push(SimpleLine::RawAccess {
        res: value_simplified.into(),
        index: index_var.into(),
        shift,
    });
}

fn create_recursive_function(
    name: String,
    args: Vec<Var>,
    iterator: Var,
    end: SimpleExpr,
    mut body: Vec<SimpleLine>,
    external_vars: &[Var],
) -> SimpleFunction {
    // Add iterator increment
    let next_iter = format!("@incremented_{}", iterator);
    body.push(SimpleLine::Assignment {
        var: next_iter.clone().into(),
        operation: HighLevelOperation::Add,
        arg0: iterator.clone().into(),
        arg1: SimpleExpr::one(),
    });

    // Add recursive call
    let mut recursive_args: Vec<SimpleExpr> = vec![next_iter.into()];
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
            var: diff_var.clone().into(),
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

fn replace_vars_for_unroll_in_expr(
    expr: &mut Expression,
    iterator: &Var,
    iterator_value: usize,
    internal_vars: &BTreeSet<Var>,
) {
    match expr {
        Expression::Value(value_expr) => match value_expr {
            SimpleExpr::Var(var) => {
                if var == iterator {
                    *value_expr = SimpleExpr::Constant(ConstExpression::from(iterator_value));
                } else if internal_vars.contains(var) {
                    *var = format!("@unrolled_{}_{}", iterator_value, var).into();
                }
            }
            SimpleExpr::Constant(_) | SimpleExpr::ConstMallocAccess { .. } => {}
        },
        Expression::ArrayAccess { array, index } => {
            assert!(array != iterator, "Weird");
            if internal_vars.contains(array) {
                *array = format!("@unrolled_{}_{}", iterator_value, array).into();
            }
            replace_vars_for_unroll_in_expr(index, iterator, iterator_value, internal_vars);
        }
        Expression::Binary { left, right, .. } => {
            replace_vars_for_unroll_in_expr(left, iterator, iterator_value, internal_vars);
            replace_vars_for_unroll_in_expr(right, iterator, iterator_value, internal_vars);
        }
    }
}

fn replace_vars_for_unroll(
    lines: &mut [Line],
    iterator: &Var,
    iterator_value: usize,
    internal_vars: &BTreeSet<Var>,
) {
    for line in lines {
        match line {
            Line::Assignment { var, value } => {
                assert!(var != iterator, "Weird");
                *var = format!("@unrolled_{}_{}", iterator_value, var).into();
                replace_vars_for_unroll_in_expr(value, iterator, iterator_value, internal_vars);
            }
            Line::ArrayAssign {
                // array[index] = value
                array,
                index,
                value,
            } => {
                assert!(array != iterator, "Weird");
                if internal_vars.contains(array) {
                    *array = format!("@unrolled_{}_{}", iterator_value, array).into();
                }
                replace_vars_for_unroll_in_expr(index, iterator, iterator_value, internal_vars);
                replace_vars_for_unroll_in_expr(value, iterator, iterator_value, internal_vars);
            }
            Line::Assert(Boolean::Equal { left, right } | Boolean::Different { left, right }) => {
                replace_vars_for_unroll_in_expr(left, iterator, iterator_value, internal_vars);
                replace_vars_for_unroll_in_expr(right, iterator, iterator_value, internal_vars);
            }
            Line::IfCondition {
                condition: Boolean::Equal { left, right } | Boolean::Different { left, right },
                then_branch,
                else_branch,
            } => {
                replace_vars_for_unroll_in_expr(left, iterator, iterator_value, internal_vars);
                replace_vars_for_unroll_in_expr(right, iterator, iterator_value, internal_vars);
                replace_vars_for_unroll(then_branch, iterator, iterator_value, internal_vars);
                replace_vars_for_unroll(else_branch, iterator, iterator_value, internal_vars);
            }
            Line::ForLoop {
                iterator: other_iterator,
                start,
                end,
                body,
                unroll: _,
            } => {
                assert!(other_iterator != iterator);
                *other_iterator = format!("@unrolled_{}_{}", iterator_value, other_iterator).into();
                replace_vars_for_unroll_in_expr(start, iterator, iterator_value, internal_vars);
                replace_vars_for_unroll_in_expr(end, iterator, iterator_value, internal_vars);
                replace_vars_for_unroll(body, iterator, iterator_value, internal_vars);
            }
            Line::FunctionCall {
                function_name: _,
                args,
                return_data,
            } => {
                // Function calls are not unrolled, so we don't need to change them
                for arg in args {
                    replace_vars_for_unroll_in_expr(arg, iterator, iterator_value, internal_vars);
                }
                for ret in return_data {
                    *ret = format!("@unrolled_{}_{}", iterator_value, ret).into();
                }
            }
            Line::FunctionRet { return_data } => {
                for ret in return_data {
                    replace_vars_for_unroll_in_expr(ret, iterator, iterator_value, internal_vars);
                }
            }
            Line::Poseidon16 { args, res } => {
                for arg in args {
                    replace_vars_for_unroll_in_expr(arg, iterator, iterator_value, internal_vars);
                }
                for r in res {
                    *r = format!("@unrolled_{}_{}", iterator_value, r).into();
                }
            }
            Line::Poseidon24 { args, res } => {
                for arg in args {
                    replace_vars_for_unroll_in_expr(arg, iterator, iterator_value, internal_vars);
                }
                for r in res {
                    *r = format!("@unrolled_{}_{}", iterator_value, r).into();
                }
            }
            Line::ExtensionMul { args } => {
                for arg in args {
                    replace_vars_for_unroll_in_expr(arg, iterator, iterator_value, internal_vars);
                }
            }
            Line::Break => {}
            Line::Panic => {}
            Line::Print { line_info, content } => {
                // Print statements are not unrolled, so we don't need to change them
                *line_info += &format!(" (unrolled {})", iterator_value);
                for var in content {
                    replace_vars_for_unroll_in_expr(var, iterator, iterator_value, internal_vars);
                }
            }
            Line::MAlloc {
                var,
                size,
                vectorized: _,
            } => {
                assert!(var != iterator, "Weird");
                *var = format!("@unrolled_{}_{}", iterator_value, var).into();
                replace_vars_for_unroll_in_expr(size, iterator, iterator_value, internal_vars);
                // vectorized is not changed
            }
            Line::DecomposeBits { var, to_decompose } => {
                assert!(var != iterator, "Weird");
                *var = format!("@unrolled_{}_{}", iterator_value, var).into();
                replace_vars_for_unroll_in_expr(
                    to_decompose,
                    iterator,
                    iterator_value,
                    internal_vars,
                );
            }
        }
    }
}

impl ToString for SimpleLine {
    fn to_string(&self) -> String {
        self.to_string_with_indent(0)
    }
}

impl ToString for VarOrConstMallocAccess {
    fn to_string(&self) -> String {
        match self {
            VarOrConstMallocAccess::Var(var) => var.to_string(),
            VarOrConstMallocAccess::ConstMallocAccess {
                malloc_label,
                offset,
            } => {
                format!(
                    "ConstMallocAccess({}, {})",
                    malloc_label,
                    offset.to_string()
                )
            }
        }
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
            SimpleLine::DecomposeBits {
                var: result,
                to_decompose,
            } => {
                format!(
                    "{} = decompose_bits({})",
                    result.to_string(),
                    to_decompose.to_string()
                )
            }
            SimpleLine::RawAccess { res, index, shift } => {
                format!(
                    "{} = memory[{} + {}]",
                    res.to_string(),
                    index.to_string(),
                    shift.to_string()
                )
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
            SimpleLine::ExtensionMul {
                args: [arg0, arg1, arg2],
            } => {
                format!(
                    "extension_mul({}, {}, {})",
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
            SimpleLine::HintMAlloc {
                var,
                size,
                vectorized,
            } => {
                let alloc_type = if *vectorized {
                    "malloc_vectorized"
                } else {
                    "malloc"
                };
                format!("{} = {}({})", var.to_string(), alloc_type, size.to_string())
            }
            SimpleLine::ConstMalloc {
                var,
                size,
                label: _,
            } => {
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
