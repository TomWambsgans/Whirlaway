use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::HashMap;

use crate::bytecode::Operation;
use crate::lang::*;

#[derive(Parser)]
#[grammar = "grammar.pest"]
struct LangParser;

pub fn parse_program(input: &str) -> Result<Program, pest::error::Error<Rule>> {
    let input = remove_comments(input);
    let mut pairs = LangParser::parse(Rule::program, &input)?;
    let program_pair = pairs.next().unwrap();

    let mut functions = HashMap::new();
    let mut main_function = None;

    for pair in program_pair.into_inner() {
        match pair.as_rule() {
            Rule::function => {
                let func = parse_function(pair);
                if func.name == "main" {
                    main_function = Some(func);
                } else {
                    functions.insert(func.name.clone(), func);
                }
            }
            Rule::EOI => break,
            _ => unreachable!(),
        }
    }

    Ok(Program {
        main_function: main_function.expect("No main function found"),
        functions,
    })
}

fn remove_comments(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            if let Some(pos) = line.find("//") {
                line[..pos].trim_end()
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_function(pair: Pair<Rule>) -> Function {
    let mut inner = pair.into_inner();

    // Skip "fn" keyword and get function name
    let name = inner.next().unwrap().as_str().to_string();

    // Parse parameters
    let mut arguments = Vec::new();
    let mut n_returned_vars = 0;
    let mut instructions = Vec::new();

    for pair in inner {
        match pair.as_rule() {
            Rule::parameter_list => {
                arguments = parse_parameter_list(pair);
            }
            Rule::return_count => {
                n_returned_vars = parse_return_count(pair);
            }
            Rule::statement => {
                instructions.push(parse_statement(pair));
            }
            _ => {}
        }
    }

    Function {
        name,
        arguments,
        n_returned_vars,
        instructions,
    }
}

fn parse_parameter_list(pair: Pair<Rule>) -> Vec<Var> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::identifier)
        .map(|p| Var {
            name: p.as_str().to_string(),
        })
        .collect()
}

fn parse_return_count(pair: Pair<Rule>) -> usize {
    let number_pair = pair.into_inner().next().unwrap();
    number_pair.as_str().parse().unwrap()
}

fn parse_statement(pair: Pair<Rule>) -> Line {
    let inner_pair = pair.into_inner().next().unwrap();

    match inner_pair.as_rule() {
        Rule::single_assignment => parse_single_assignment(inner_pair),
        Rule::constant_assignment => parse_constant_assignment(inner_pair),
        Rule::raw_memory_access => parse_raw_memory_access(inner_pair),
        Rule::if_statement => parse_if_statement(inner_pair),
        Rule::for_statement => parse_for_statement(inner_pair),
        Rule::return_statement => parse_return_statement(inner_pair),
        Rule::function_call => parse_function_call(inner_pair),
        Rule::assert_eq_statement => parse_assert_eq(inner_pair),
        Rule::assert_not_eq_statement => parse_assert_not_eq(inner_pair),
        Rule::assert_eq_ext_statement => parse_assert_eq_ext(inner_pair),
        _ => unreachable!(),
    }
}

fn parse_single_assignment(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let var_name = inner.next().unwrap().as_str().to_string();
    let expression = inner.next().unwrap();

    if let Some(binary_expr) = expression.clone().into_inner().next() {
        if binary_expr.as_rule() == Rule::binary_expression {
            let mut expr_inner = binary_expr.into_inner();
            let arg0 = parse_var_or_constant(expr_inner.next().unwrap());
            let op = parse_binary_operator(expr_inner.next().unwrap());
            let arg1 = parse_var_or_constant(expr_inner.next().unwrap());

            return Line::Assignment {
                var: Var { name: var_name },
                operation: op,
                arg0,
                arg1,
            };
        }
    }

    // If not a binary expression, create a simple assignment (e.g., x = y)
    let value = parse_var_or_constant(expression.into_inner().next().unwrap());
    Line::Assignment {
        var: Var { name: var_name },
        operation: Operation::Add, // Default operation for simple assignment
        arg0: value,
        arg1: VarOrConstant::Constant(ConstantValue::Scalar(0)),
    }
}

fn parse_constant_assignment(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let var_name = inner.next().unwrap().as_str().to_string();
    let constant = parse_constant_value(inner.next().unwrap());

    Line::Assignment {
        var: Var { name: var_name },
        operation: Operation::Add,
        arg0: VarOrConstant::Constant(constant),
        arg1: VarOrConstant::Constant(ConstantValue::Scalar(0)),
    }
}

fn parse_raw_memory_access(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let var_name = inner.next().unwrap().as_str().to_string();
    let index = parse_var_or_constant(inner.next().unwrap());

    Line::RawAccess {
        var: Var { name: var_name },
        index,
    }
}

fn parse_if_statement(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let condition = parse_condition(inner.next().unwrap());

    let mut then_branch = Vec::new();
    let mut else_branch = Vec::new();

    for pair in inner {
        match pair.as_rule() {
            Rule::statement => {
                then_branch.push(parse_statement(pair));
            }
            Rule::else_clause => {
                for stmt_pair in pair.into_inner() {
                    if stmt_pair.as_rule() == Rule::statement {
                        else_branch.push(parse_statement(stmt_pair));
                    }
                }
            }
            _ => {}
        }
    }

    Line::IfCondition {
        condition,
        then_branch,
        else_branch,
    }
}

fn parse_for_statement(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let iterator = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };
    let start = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };
    let end = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };

    let mut body = Vec::new();
    for pair in inner {
        if pair.as_rule() == Rule::statement {
            body.push(parse_statement(pair));
        }
    }

    Line::ForLoop {
        iterator,
        start,
        end,
        body,
    }
}

fn parse_return_statement(pair: Pair<Rule>) -> Line {
    let mut return_data = Vec::new();

    for inner_pair in pair.into_inner() {
        if inner_pair.as_rule() == Rule::tuple_expression {
            for var_pair in inner_pair.into_inner() {
                if let VarOrConstant::Var(var) = parse_var_or_constant(var_pair) {
                    return_data.push(var);
                }
            }
        }
    }

    Line::FunctionRet { return_data }
}

fn parse_function_call(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let mut return_data = Vec::new();
    let mut function_name = String::new();
    let mut args = Vec::new();

    for pair in inner {
        match pair.as_rule() {
            Rule::function_res => {
                let var_list = pair.into_inner().next().unwrap();
                return_data = parse_var_list(var_list);
            }
            Rule::identifier => {
                function_name = pair.as_str().to_string();
            }
            Rule::var_list => {
                args = parse_var_list(pair);
            }
            _ => {}
        }
    }

    // Handle special function names
    match function_name.as_str() {
        "poseidon16" => {
            if args.len() >= 2 && return_data.len() >= 2 {
                Line::Poseidon16 {
                    arg0: args[0].clone(),
                    arg1: args[1].clone(),
                    res0: return_data[0].clone(),
                    res1: return_data[1].clone(),
                }
            } else {
                Line::FunctionCall {
                    function_name,
                    args,
                    return_data,
                }
            }
        }
        "poseidon24" => {
            if args.len() >= 3 && return_data.len() >= 3 {
                Line::Poseidon24 {
                    arg0: args[0].clone(),
                    arg1: args[1].clone(),
                    arg2: args[2].clone(),
                    res0: return_data[0].clone(),
                    res1: return_data[1].clone(),
                    res2: return_data[2].clone(),
                }
            } else {
                Line::FunctionCall {
                    function_name,
                    args,
                    return_data,
                }
            }
        }
        _ => Line::FunctionCall {
            function_name,
            args,
            return_data,
        },
    }
}

fn parse_var_list(pair: Pair<Rule>) -> Vec<Var> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::identifier)
        .map(|p| Var {
            name: p.as_str().to_string(),
        })
        .collect()
}

fn parse_assert_eq(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let left = parse_var_or_constant(inner.next().unwrap());
    let right = parse_var_or_constant(inner.next().unwrap());

    Line::Assert(Boolean::Equal { left, right })
}

fn parse_assert_not_eq(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let left = parse_var_or_constant(inner.next().unwrap());
    let right = parse_var_or_constant(inner.next().unwrap());

    Line::Assert(Boolean::Different { left, right })
}

fn parse_assert_eq_ext(pair: Pair<Rule>) -> Line {
    let mut inner = pair.into_inner();
    let left = parse_var_or_constant(inner.next().unwrap());
    let right = parse_var_or_constant(inner.next().unwrap());

    Line::AssertEqExt { left, right }
}

fn parse_condition(pair: Pair<Rule>) -> Boolean {
    let inner_pair = pair.into_inner().next().unwrap();

    match inner_pair.as_rule() {
        Rule::condition_eq => {
            let mut inner = inner_pair.into_inner();
            let left = parse_var_or_constant(inner.next().unwrap());
            let right = parse_var_or_constant(inner.next().unwrap());
            Boolean::Equal { left, right }
        }
        Rule::condition_diff => {
            let mut inner = inner_pair.into_inner();
            let left = parse_var_or_constant(inner.next().unwrap());
            let right = parse_var_or_constant(inner.next().unwrap());
            Boolean::Different { left, right }
        }
        _ => unreachable!(),
    }
}

fn parse_var_or_constant(pair: Pair<Rule>) -> VarOrConstant {
    let inner_pair = pair.into_inner().next().unwrap();

    match inner_pair.as_rule() {
        Rule::identifier => VarOrConstant::Var(Var {
            name: inner_pair.as_str().to_string(),
        }),
        Rule::constant_value => VarOrConstant::Constant(parse_constant_value(inner_pair)),
        _ => unreachable!(),
    }
}

fn parse_constant_value(pair: Pair<Rule>) -> ConstantValue {
    let value = pair.as_str();

    if value == "public_input_start" {
        ConstantValue::PublicInputStart
    } else {
        ConstantValue::Scalar(value.parse().unwrap())
    }
}

fn parse_binary_operator(pair: Pair<Rule>) -> Operation {
    match pair.as_str() {
        "+" => Operation::Add,
        "*" => Operation::Mul,
        "-" => Operation::Sub,
        "/" => Operation::Div,
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser() {
        let program = r#"
fn main() {
// this a comment
    a = 5;
    b = 6;
    c = a + b;
    assert c == d;
    if c != b { // this a comment
        d = 1;
        e = 9;
        f = d * e;
    } else {
        f = 8;
    }
    assert f != g;
    oo = memory[8];
    assert_ext oo == f;
    x = 8;
    y = 9;

    gh = memory[7];
    hh = memory[gh];

    (xx, yy) = poseidon16(x, y);
    (xxx, yyy, zzz) = poseidon24(x, y, b);

    assert_eq_ext(a, b);

    k = public_input_start;

    for i in a..b {
        assert i != d;
    }

    (i, j, k) = my_function1(b, b, a);
}

fn my_function1(a, b, c) -> 2 {
    d = a + b;
    e = b + c;
    if e == e {
        return (0, 0);
    }
    if d != e {
        return (d, e);
    } else {
        return (e, d);
    }
}

    "#;

        let parsed = parse_program(program).unwrap();
        dbg!("{:?}", parsed);
    }
}
