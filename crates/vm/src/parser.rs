use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::BTreeMap;

use crate::bytecode::intermediate_bytecode::*;
use crate::lang::*;

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct LangParser;

#[derive(Debug)]
pub enum ParseError {
    PestError(pest::error::Error<Rule>),
    SemanticError(String),
}

impl From<pest::error::Error<Rule>> for ParseError {
    fn from(error: pest::error::Error<Rule>) -> Self {
        ParseError::PestError(error)
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::PestError(e) => write!(f, "Parse error: {}", e),
            ParseError::SemanticError(e) => write!(f, "Semantic error: {}", e),
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse_program(input: &str) -> Result<Program, ParseError> {
    let input = remove_comments(input);
    let mut pairs = LangParser::parse(Rule::program, &input)?;
    let program_pair = pairs.next().unwrap();

    let mut constants = BTreeMap::new();
    let mut functions = BTreeMap::new();
    let mut trash_var_count = 0;

    for pair in program_pair.into_inner() {
        match pair.as_rule() {
            Rule::constant_declaration => {
                let (name, value) = parse_constant_declaration(pair)?;
                constants.insert(name, value);
            }
            Rule::function => {
                let function = parse_function(pair, &constants, &mut trash_var_count)?;
                functions.insert(function.name.clone(), function);
            }
            Rule::EOI => break,
            _ => {}
        }
    }

    Ok(Program { functions })
}

fn remove_comments(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            if let Some(pos) = line.find("//") {
                &line[..pos]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_constant_declaration(pair: Pair<Rule>) -> Result<(String, usize), ParseError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let value = inner
        .next()
        .unwrap()
        .as_str()
        .parse()
        .map_err(|_| ParseError::SemanticError("Invalid constant value".to_string()))?;
    Ok((name, value))
}

fn parse_function(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
    trash_var_count: &mut usize,
) -> Result<Function, ParseError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();

    let mut arguments = Vec::new();
    let mut n_returned_vars = 0;
    let mut instructions = Vec::new();

    for pair in inner {
        match pair.as_rule() {
            Rule::parameter_list => {
                for param in pair.into_inner() {
                    if param.as_rule() == Rule::identifier {
                        arguments.push(param.as_str().to_string());
                    }
                }
            }
            Rule::return_count => {
                let count_str = pair.into_inner().next().unwrap().as_str();
                n_returned_vars = constants
                    .get(count_str)
                    .copied()
                    .or_else(|| count_str.parse().ok())
                    .ok_or_else(|| ParseError::SemanticError("Invalid return count".to_string()))?;
            }
            Rule::statement => {
                instructions.push(parse_statement(pair, constants, trash_var_count)?);
            }
            _ => {}
        }
    }

    Ok(Function {
        name,
        arguments,
        n_returned_vars,
        instructions,
    })
}

fn parse_statement(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
    trash_var_count: &mut usize,
) -> Result<Line, ParseError> {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::single_assignment => parse_assignment(inner, constants),
        Rule::array_assign => parse_array_assign(inner, constants),
        Rule::if_statement => parse_if_statement(inner, constants, trash_var_count),
        Rule::for_statement => parse_for_statement(inner, constants, trash_var_count),
        Rule::return_statement => parse_return_statement(inner, constants),
        Rule::function_call => parse_function_call(inner, constants, trash_var_count),
        Rule::assert_eq_statement => parse_assert_eq(inner, constants),
        Rule::assert_not_eq_statement => parse_assert_not_eq(inner, constants),
        _ => Err(ParseError::SemanticError("Unknown statement".to_string())),
    }
}

fn parse_if_statement(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
    trash_var_count: &mut usize,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let condition = parse_condition(inner.next().unwrap(), constants)?;

    let mut then_branch = Vec::new();
    let mut else_branch = Vec::new();

    for item in inner {
        match item.as_rule() {
            Rule::statement => then_branch.push(parse_statement(item, constants, trash_var_count)?),
            Rule::else_clause => {
                for else_item in item.into_inner() {
                    if else_item.as_rule() == Rule::statement {
                        else_branch.push(parse_statement(else_item, constants, trash_var_count)?);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(Line::IfCondition {
        condition,
        then_branch,
        else_branch,
    })
}

fn parse_assignment(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let var = inner.next().unwrap().as_str().to_string();
    let expr = inner.next().unwrap();
    let value = parse_expression(expr, constants)?;

    Ok(Line::Assignment { var, value })
}

fn parse_array_assign(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let array = inner.next().unwrap().as_str().to_string();
    let index = parse_expression(inner.next().unwrap(), constants)?;
    let value = parse_expression(inner.next().unwrap(), constants)?;
    Ok(Line::ArrayAssign {
        array,
        index,
        value,
    })
}

fn parse_for_statement(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
    trash_var_count: &mut usize,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let iterator = inner.next().unwrap().as_str().to_string();
    let start = parse_expression(inner.next().unwrap(), constants)?;
    let end = parse_expression(inner.next().unwrap(), constants)?;

    let mut body = Vec::new();
    for item in inner {
        if item.as_rule() == Rule::statement {
            body.push(parse_statement(item, constants, trash_var_count)?);
        }
    }

    Ok(Line::ForLoop {
        iterator,
        start,
        end,
        body,
    })
}

fn parse_return_statement(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut return_data = Vec::new();
    for item in pair.into_inner() {
        if item.as_rule() == Rule::tuple_expression {
            for tuple_item in item.into_inner() {
                return_data.push(parse_expression(tuple_item, constants)?);
            }
        }
    }
    Ok(Line::FunctionRet { return_data })
}
fn parse_expression(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Expression, ParseError> {
    match pair.as_rule() {
        Rule::expression => parse_expression(pair.into_inner().next().unwrap(), constants),
        Rule::add_expr => parse_binary_expr(pair, constants, HighLevelOperation::Add),
        Rule::sub_expr => parse_binary_expr(pair, constants, HighLevelOperation::Sub),
        Rule::mul_expr => parse_binary_expr(pair, constants, HighLevelOperation::Mul),
        Rule::div_expr => parse_binary_expr(pair, constants, HighLevelOperation::Div),
        Rule::primary => parse_primary(pair, constants),
        _ => Err(ParseError::SemanticError("Invalid expression".to_string())),
    }
}

fn parse_array_access(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Expression, ParseError> {
    let mut inner = pair.into_inner();
    let array = inner.next().unwrap().as_str().to_string();
    let index = parse_expression(inner.next().unwrap(), constants)?;
    Ok(Expression::ArrayAccess {
        array,
        index: Box::new(index),
    })
}

fn parse_binary_expr(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
    operator: HighLevelOperation,
) -> Result<Expression, ParseError> {
    let mut inner = pair.into_inner();
    let mut expr = parse_expression(inner.next().unwrap(), constants)?;

    while let Some(right) = inner.next() {
        let right_expr = parse_expression(right, constants)?;
        expr = Expression::Binary {
            left: Box::new(expr),
            operator,
            right: Box::new(right_expr),
        };
    }

    Ok(expr)
}

fn parse_primary(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Expression, ParseError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::expression => parse_expression(inner, constants),
        Rule::var_or_constant => Ok(Expression::Value(parse_var_or_constant(inner, constants)?)),
        Rule::array_access_expr => parse_array_access(inner, constants),
        _ => Err(ParseError::SemanticError(
            "Invalid primary expression".to_string(),
        )),
    }
}

fn parse_argument_list(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Vec<Expression>, ParseError> {
    pair.into_inner()
        .map(|item| parse_expression(item, constants))
        .collect()
}
fn parse_function_call(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
    trash_var_count: &mut usize,
) -> Result<Line, ParseError> {
    let inner = pair.clone().into_inner();
    let mut return_data = Vec::new();
    let mut function_name = String::new();
    let mut args = Vec::new();

    for item in inner {
        match item.as_rule() {
            Rule::function_res => {
                for res_item in item.into_inner() {
                    if res_item.as_rule() == Rule::var_list {
                        return_data = parse_var_list(res_item, constants)?
                            .into_iter()
                            .filter_map(|v| {
                                if let VarOrConstant::Var(var) = v {
                                    Some(var)
                                } else {
                                    None
                                }
                            })
                            .collect();
                    }
                }
            }
            Rule::identifier => function_name = item.as_str().to_string(),
            Rule::argument_list => args = parse_argument_list(item, constants)?,
            _ => {}
        }
    }

    for var in &mut return_data {
        if var == "_" {
            *trash_var_count += 1;
            *var = format!("@trash_{}", trash_var_count);
        }
    }

    match function_name.as_str() {
        "poseidon16" => {
            assert!(
                args.len() == 2 && return_data.len() == 2,
                "Invalid poseidon16 call"
            );
            Ok(Line::Poseidon16 {
                args: [args[0].clone(), args[1].clone()],
                res: [return_data[0].clone(), return_data[1].clone()],
            })
        }
        "poseidon24" => {
            assert!(
                args.len() == 3 && return_data.len() == 3,
                "Invalid poseidon24 call"
            );
            Ok(Line::Poseidon24 {
                args: [args[0].clone(), args[1].clone(), args[2].clone()],
                res: [
                    return_data[0].clone(),
                    return_data[1].clone(),
                    return_data[2].clone(),
                ],
            })
        }
        "malloc" => {
            assert!(
                args.len() == 1 && return_data.len() == 1,
                "Invalid malloc call"
            );
            Ok(Line::MAlloc {
                var: return_data[0].clone(),
                size: args[0].clone(),
            })
        }
        "print" => {
            assert!(
                return_data.is_empty(),
                "Print function should not return values"
            );
            Ok(Line::Print {
                line_info: pair.as_str().to_string(),
                content: args,
            })
        }
        _ => Ok(Line::FunctionCall {
            function_name,
            args,
            return_data,
        }),
    }
}

fn parse_condition(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Boolean, ParseError> {
    let inner = pair.into_inner().next().unwrap();
    let mut parts = inner.clone().into_inner();
    let left = parse_expression(parts.next().unwrap(), constants)?;
    let right = parse_expression(parts.next().unwrap(), constants)?;

    match inner.as_rule() {
        Rule::condition_eq => Ok(Boolean::Equal { left, right }),
        Rule::condition_diff => Ok(Boolean::Different { left, right }),
        _ => unreachable!(),
    }
}

fn parse_assert_eq(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let left = parse_expression(inner.next().unwrap(), constants)?;
    let right = parse_expression(inner.next().unwrap(), constants)?;
    Ok(Line::Assert(Boolean::Equal { left, right }))
}

fn parse_assert_not_eq(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let left = parse_expression(inner.next().unwrap(), constants)?;
    let right = parse_expression(inner.next().unwrap(), constants)?;
    Ok(Line::Assert(Boolean::Different { left, right }))
}

fn parse_var_or_constant(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<VarOrConstant, ParseError> {
    let text = pair.as_str();

    match pair.as_rule() {
        Rule::var_or_constant => {
            return parse_var_or_constant(pair.into_inner().next().unwrap(), constants);
        }
        Rule::identifier | Rule::constant_value => match text {
            "public_input_start" => Ok(VarOrConstant::Constant(ConstExpression::Value(
                ConstantValue::PublicInputStart,
            ))),
            "pointer_to_zero_vector" => Ok(VarOrConstant::Constant(ConstExpression::Value(
                ConstantValue::PointerToZeroVector,
            ))),
            _ => {
                if let Some(value) = constants.get(text) {
                    Ok(VarOrConstant::Constant(ConstExpression::Value(
                        ConstantValue::Scalar(*value),
                    )))
                } else if let Ok(value) = text.parse::<usize>() {
                    Ok(VarOrConstant::Constant(ConstExpression::Value(
                        ConstantValue::Scalar(value),
                    )))
                } else {
                    Ok(VarOrConstant::Var(text.to_string()))
                }
            }
        },
        _ => Err(ParseError::SemanticError(
            "Expected identifier or constant".to_string(),
        )),
    }
}

fn parse_var_list(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Vec<VarOrConstant>, ParseError> {
    pair.into_inner()
        .map(|item| parse_var_or_constant(item, constants))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser() {
        let program = r#"

// This is a comment

const A = 10000;
const B = 20000;
// Another comment

fn main() {
// this a comment

    c = a + b;
    assert c == d;
    if c != b { // this a comment
        d = 1;
        e = 9;
        f = d * ((a - b) + ((h / 1) + d));
    } else {
        f = 8;
    }
    assert f != g;
    oo = memory[B];
    x = 8;
    y = 9;
    uuu = y[9];
    vvv = y[uuu];

    gh = memory[7];
    hh = memory[gh];

    print(hh);

    xx, yy = poseidon16(x, y);
    xxx, yyy, zzz = poseidon24(7, y, b);

    k = public_input_start;

    for i in a..(b + 9) * ( 7 - 7 ) {
        assert i != d;
    }

    i, j, k = my_function1(b, b, a);
}

fn my_function1(a, b, c) -> 2 {
    d = a + b;
    e = b + c;
    if e == e {
        return 0, 0;
    }
    if d != e {
        return d, e;
    } else {
        return e, d;
    }
}
    "#;

        let parsed = parse_program(program).unwrap();
        println!("{}", parsed.to_string());
    }
}
