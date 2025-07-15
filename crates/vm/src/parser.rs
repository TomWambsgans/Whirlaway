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

    for pair in program_pair.into_inner() {
        match pair.as_rule() {
            Rule::constant_declaration => {
                let (name, value) = parse_constant_declaration(pair)?;
                constants.insert(name, value);
            }
            Rule::function => {
                let function = parse_function(pair, &constants)?;
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
                        arguments.push(Var {
                            name: param.as_str().to_string(),
                        });
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
                instructions.push(parse_statement(pair, constants)?);
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
) -> Result<Line, ParseError> {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::single_assignment => parse_assignment(inner, constants),
        Rule::array_access => parse_array_access(inner, constants),
        Rule::array_assign => parse_array_assign(inner, constants),
        Rule::if_statement => parse_if_statement(inner, constants),
        Rule::for_statement => parse_for_statement(inner, constants),
        Rule::return_statement => parse_return_statement(inner, constants),
        Rule::function_call => parse_function_call(inner, constants),
        Rule::assert_eq_statement => parse_assert_eq(inner, constants),
        Rule::assert_not_eq_statement => parse_assert_not_eq(inner, constants),
        _ => Err(ParseError::SemanticError("Unknown statement".to_string())),
    }
}

fn parse_assignment(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let var = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };
    let expr = inner.next().unwrap().into_inner().next().unwrap();

    match expr.as_rule() {
        Rule::binary_expression => {
            let mut expr_inner = expr.into_inner();
            let arg0 = parse_var_or_constant(expr_inner.next().unwrap(), constants)?;
            let op_str = expr_inner.next().unwrap().as_str();
            let arg1 = parse_var_or_constant(expr_inner.next().unwrap(), constants)?;

            let operation = match op_str {
                "+" => HighLevelOperation::Add,
                "-" => HighLevelOperation::Sub,
                "*" => HighLevelOperation::Mul,
                "/" => HighLevelOperation::Div,
                _ => return Err(ParseError::SemanticError("Unknown operator".to_string())),
            };

            Ok(Line::Assignment {
                var,
                operation,
                arg0,
                arg1,
            })
        }
        Rule::var_or_constant => {
            let value = parse_var_or_constant(expr, constants)?;
            Ok(Line::Assignment {
                var,
                operation: HighLevelOperation::Add,
                arg0: value,
                arg1: VarOrConstant::Constant(ConstantValue::Scalar(0)),
            })
        }
        _ => Err(ParseError::SemanticError("Invalid expression".to_string())),
    }
}

fn parse_array_access(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let value = Var {
        name: inner.next().unwrap().as_str().to_string(),
    }
    .into();
    let array = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };
    let index = parse_var_or_constant(inner.next().unwrap(), constants)?;
    Ok(Line::ArrayAccess {
        value,
        array,
        index,
    })
}

fn parse_array_assign(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let array = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };
    let index = parse_var_or_constant(inner.next().unwrap(), constants)?;
    let value = parse_var_or_constant(inner.next().unwrap(), constants)?;
    Ok(Line::ArrayAccess {
        value,
        array,
        index,
    })
}

fn parse_if_statement(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let condition = parse_condition(inner.next().unwrap(), constants)?;

    let mut then_branch = Vec::new();
    let mut else_branch = Vec::new();

    for item in inner {
        match item.as_rule() {
            Rule::statement => then_branch.push(parse_statement(item, constants)?),
            Rule::else_clause => {
                for else_item in item.into_inner() {
                    if else_item.as_rule() == Rule::statement {
                        else_branch.push(parse_statement(else_item, constants)?);
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

fn parse_for_statement(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let iterator = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };
    let start = parse_var_or_constant(inner.next().unwrap(), constants)?;
    let end = parse_var_or_constant(inner.next().unwrap(), constants)?;

    let mut body = Vec::new();
    for item in inner {
        if item.as_rule() == Rule::statement {
            body.push(parse_statement(item, constants)?);
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
                if tuple_item.as_rule() == Rule::var_or_constant {
                    return_data.push(parse_var_or_constant(tuple_item, constants)?);
                }
            }
        }
    }
    Ok(Line::FunctionRet { return_data })
}

fn parse_function_call(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
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
            Rule::var_list => args = parse_var_list(item, constants)?,
            _ => {}
        }
    }

    match function_name.as_str() {
        "poseidon16" => Ok(Line::Poseidon16 {
            arg0: args[0].clone(),
            arg1: args[1].clone(),
            res0: return_data[0].clone(),
            res1: return_data[1].clone(),
        }),
        "poseidon24" => Ok(Line::Poseidon24 {
            arg0: args[0].clone(),
            arg1: args[1].clone(),
            arg2: args[2].clone(),
            res0: return_data[0].clone(),
            res1: return_data[1].clone(),
            res2: return_data[2].clone(),
        }),
        "malloc" => Ok(Line::MAlloc {
            var: return_data[0].clone(),
            size: args[0].as_constant().unwrap(),
        }),
        "print" => Ok(Line::Print {
            line_info: pair.as_str().to_string(),
            content: args,
        }),
        _ => Ok(Line::FunctionCall {
            function_name,
            args,
            return_data,
        }),
    }
}

fn parse_assert_eq(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let left = parse_var_or_constant(inner.next().unwrap(), constants)?;
    let right = parse_var_or_constant(inner.next().unwrap(), constants)?;
    Ok(Line::Assert(Boolean::Equal { left, right }))
}

fn parse_assert_not_eq(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let left = parse_var_or_constant(inner.next().unwrap(), constants)?;
    let right = parse_var_or_constant(inner.next().unwrap(), constants)?;
    Ok(Line::Assert(Boolean::Different { left, right }))
}

fn parse_condition(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<Boolean, ParseError> {
    let inner = pair.into_inner().next().unwrap();
    let mut parts = inner.clone().into_inner();
    let left = parse_var_or_constant(parts.next().unwrap(), constants)?;
    let right = parse_var_or_constant(parts.next().unwrap(), constants)?;

    match inner.as_rule() {
        Rule::condition_eq => Ok(Boolean::Equal { left, right }),
        Rule::condition_diff => Ok(Boolean::Different { left, right }),
        _ => unreachable!(),
    }
}

fn parse_var_or_constant(
    pair: Pair<Rule>,
    constants: &BTreeMap<String, usize>,
) -> Result<VarOrConstant, ParseError> {
    let inner = pair.into_inner().next().unwrap();
    let text = inner.as_str();

    match inner.as_rule() {
        Rule::identifier | Rule::constant_value => match text {
            "public_input_start" => Ok(VarOrConstant::Constant(ConstantValue::PublicInputStart)),
            "pointer_to_zero_vector" => {
                Ok(VarOrConstant::Constant(ConstantValue::PointerToZeroVector))
            }
            _ => {
                if let Some(value) = constants.get(text) {
                    Ok(VarOrConstant::Constant(ConstantValue::Scalar(*value)))
                } else if let Ok(value) = text.parse::<usize>() {
                    Ok(VarOrConstant::Constant(ConstantValue::Scalar(value)))
                } else {
                    Ok(VarOrConstant::Var(Var {
                        name: text.to_string(),
                    }))
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
