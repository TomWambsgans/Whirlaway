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
    NotAConstant(String),
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
            ParseError::NotAConstant(e) => write!(f, "Not a constant: {}", e),
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone)]
struct ParseContext {
    constants: BTreeMap<String, usize>,
}

impl ParseContext {
    fn new() -> Self {
        Self {
            constants: BTreeMap::new(),
        }
    }

    fn add_constant(&mut self, name: String, value: usize) {
        self.constants.insert(name, value);
    }

    fn resolve_constant(&self, name: &str) -> Option<usize> {
        self.constants.get(name).copied()
    }
}

pub fn parse_program(input: &str) -> Result<Program, ParseError> {
    assert!(
        !input.contains("@"),
        "@ is a reserved character and should not be used in the program"
    );

    let input = remove_comments(input);
    let mut pairs = LangParser::parse(Rule::program, &input)?;
    let program_pair = pairs.next().unwrap();

    let mut context = ParseContext::new();
    let mut functions = BTreeMap::new();

    for pair in program_pair.into_inner() {
        match pair.as_rule() {
            Rule::constant_declaration => {
                parse_constant_declaration(pair, &mut context)?;
            }
            Rule::function => {
                let function = parse_function(pair, &context)?;
                functions.insert(function.name.clone(), function);
            }
            Rule::EOI => break,
            _ => {}
        }
    }

    let main_function = functions.get("main").expect("Main function not found");
    assert!(
        main_function.n_returned_vars == 0,
        "Main function should not return any values"
    );
    assert!(
        main_function.arguments.is_empty(),
        "Main function should not have any arguments"
    );

    Ok(Program { functions })
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

fn parse_constant_declaration(
    pair: Pair<Rule>,
    context: &mut ParseContext,
) -> Result<(), ParseError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let value = inner
        .next()
        .unwrap()
        .as_str()
        .parse::<usize>()
        .map_err(|_| ParseError::SemanticError("Invalid constant value".to_string()))?;

    context.add_constant(name, value);
    Ok(())
}

fn parse_function(pair: Pair<Rule>, context: &ParseContext) -> Result<Function, ParseError> {
    let mut inner = pair.into_inner();

    let name = inner.next().unwrap().as_str().to_string();

    let mut arguments = Vec::new();
    let mut n_returned_vars = 0;
    let mut instructions = Vec::new();

    for pair in inner {
        match pair.as_rule() {
            Rule::parameter_list => {
                arguments = parse_parameter_list(pair, context)?;
            }
            Rule::return_count => {
                n_returned_vars = parse_return_count(pair, context)?;
            }
            Rule::statement => {
                instructions.push(parse_statement(pair, context)?);
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

fn parse_parameter_list(pair: Pair<Rule>, _context: &ParseContext) -> Result<Vec<Var>, ParseError> {
    let mut params = Vec::new();
    for param_pair in pair.into_inner() {
        if param_pair.as_rule() == Rule::identifier {
            params.push(Var {
                name: param_pair.as_str().to_string(),
            });
        }
    }
    Ok(params)
}

fn parse_return_count(pair: Pair<Rule>, context: &ParseContext) -> Result<usize, ParseError> {
    let inner = pair.into_inner().next().unwrap();
    let count_str = inner.as_str();

    if let Some(value) = context.resolve_constant(count_str) {
        Ok(value)
    } else {
        count_str
            .parse::<usize>()
            .map_err(|_| ParseError::SemanticError("Invalid return count".to_string()))
    }
}

fn parse_statement(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::single_assignment => parse_single_assignment(inner, context),
        Rule::raw_memory_access => parse_raw_memory_access(inner, context),
        Rule::array_access => parse_array_access(inner, context),
        Rule::array_assign => parse_array_assign(inner, context),
        Rule::if_statement => parse_if_statement(inner, context),
        Rule::for_statement => parse_for_statement(inner, context),
        Rule::return_statement => parse_return_statement(inner, context),
        Rule::function_call => parse_function_call(inner, context),
        Rule::assert_eq_statement => parse_assert_eq_statement(inner, context),
        Rule::assert_not_eq_statement => parse_assert_not_eq_statement(inner, context),
        _ => Err(ParseError::SemanticError(format!(
            "Unknown statement type: {:?}",
            inner.as_rule()
        ))),
    }
}

fn parse_single_assignment(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let var_name = inner.next().unwrap().as_str().to_string();
    let expression = inner.next().unwrap();

    let var = Var { name: var_name };

    match expression.as_rule() {
        Rule::expression => {
            let expr_inner = expression.into_inner().next().unwrap();
            match expr_inner.as_rule() {
                Rule::binary_expression => {
                    let (op, arg0, arg1) = parse_binary_expression(expr_inner, context)?;
                    Ok(Line::Assignment {
                        var,
                        operation: op,
                        arg0,
                        arg1,
                    })
                }
                Rule::var_or_constant => {
                    let value = parse_var_or_constant(expr_inner, context)?;
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
        _ => Err(ParseError::SemanticError("Expected expression".to_string())),
    }
}

fn parse_raw_memory_access(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let var_name = inner.next().unwrap().as_str().to_string();
    let index = inner.next().unwrap();

    let var = Var { name: var_name };
    let index_value = parse_var_or_constant(index, context)?;

    Ok(Line::RawAccess {
        var,
        index: index_value,
    })
}

fn parse_array_access(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let value = Var {
        name: inner.next().unwrap().as_str().to_string(),
    }
    .into();
    let array = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };
    let index = parse_var_or_constant(inner.next().unwrap(), context)?;

    Ok(Line::ArrayAccess {
        value,
        array,
        index,
    })
}

fn parse_array_assign(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let array = Var {
        name: inner.next().unwrap().as_str().to_string(),
    };
    let index = parse_var_or_constant(inner.next().unwrap(), context)?;
    let value = parse_var_or_constant(inner.next().unwrap(), context)?;

    Ok(Line::ArrayAccess {
        value,
        array,
        index,
    })
}

fn parse_if_statement(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let condition = parse_condition(inner.next().unwrap(), context)?;

    let mut then_branch = Vec::new();
    let mut else_branch = Vec::new();
    let mut in_else = false;

    for item in inner {
        match item.as_rule() {
            Rule::statement => {
                if in_else {
                    else_branch.push(parse_statement(item, context)?);
                } else {
                    then_branch.push(parse_statement(item, context)?);
                }
            }
            Rule::else_clause => {
                in_else = true;
                for else_item in item.into_inner() {
                    if else_item.as_rule() == Rule::statement {
                        else_branch.push(parse_statement(else_item, context)?);
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

fn parse_for_statement(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let iterator_name = inner.next().unwrap().as_str().to_string();
    let start = parse_var_or_constant(inner.next().unwrap(), context)?;
    let end = parse_var_or_constant(inner.next().unwrap(), context)?;

    let mut body = Vec::new();
    for item in inner {
        if item.as_rule() == Rule::statement {
            body.push(parse_statement(item, context)?);
        }
    }

    Ok(Line::ForLoop {
        iterator: Var {
            name: iterator_name,
        },
        start,
        end,
        body,
    })
}

fn parse_return_statement(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let mut return_data = Vec::new();

    for item in pair.into_inner() {
        if item.as_rule() == Rule::tuple_expression {
            for tuple_item in item.into_inner() {
                if tuple_item.as_rule() == Rule::var_or_constant {
                    return_data.push(parse_var_or_constant(tuple_item, context)?);
                }
            }
        }
    }

    Ok(Line::FunctionRet { return_data })
}

fn parse_function_call(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let line_str = pair.as_str();

    let inner = pair.into_inner();
    let mut return_data = Vec::new();
    let mut function_name = String::new();
    let mut args = Vec::new();

    for item in inner {
        match item.as_rule() {
            Rule::function_res => {
                for res_item in item.into_inner() {
                    if res_item.as_rule() == Rule::var_list {
                        return_data = parse_var_list(res_item, context)?
                            .into_iter()
                            .map(|var| {
                                if let VarOrConstant::Var(var) = var {
                                    var
                                } else {
                                    panic!("Expected variable in function return data");
                                }
                            })
                            .collect();
                    }
                }
            }
            Rule::identifier => {
                function_name = item.as_str().to_string();
            }
            Rule::var_list => {
                args = parse_var_list(item, context)?;
            }
            _ => {}
        }
    }

    match function_name.as_str() {
        "poseidon16" => {
            assert_eq!(args.len(), 2, "poseidon16 requires  2 arguments");
            assert_eq!(return_data.len(), 2, "poseidon16 requires 2 return values");
            Ok(Line::Poseidon16 {
                arg0: args[0].clone(),
                arg1: args[1].clone(),
                res0: return_data[0].clone(),
                res1: return_data[1].clone(),
            })
        }
        "poseidon24" => {
            assert_eq!(args.len(), 3, "poseidon24 requires 3 arguments");
            assert_eq!(return_data.len(), 3, "poseidon24 requires 3 return values");
            Ok(Line::Poseidon24 {
                arg0: args[0].clone(),
                arg1: args[1].clone(),
                arg2: args[2].clone(),
                res0: return_data[0].clone(),
                res1: return_data[1].clone(),
                res2: return_data[2].clone(),
            })
        }
        "malloc" => {
            assert_eq!(args.len(), 1, "malloc requires 1 argument");
            assert_eq!(return_data.len(), 1, "malloc requires 1 return value");
            Ok(Line::MAlloc {
                var: return_data[0].clone(),
                size: args[0].as_constant().unwrap(),
            })
        }
        "print" => {
            assert!(return_data.is_empty(), "print should not return values");
            Ok(Line::Print {
                line_info: format!("{}", line_str),
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

fn parse_assert_eq_statement(pair: Pair<Rule>, context: &ParseContext) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let left = parse_var_or_constant(inner.next().unwrap(), context)?;
    let right = parse_var_or_constant(inner.next().unwrap(), context)?;

    Ok(Line::Assert(Boolean::Equal { left, right }))
}

fn parse_assert_not_eq_statement(
    pair: Pair<Rule>,
    context: &ParseContext,
) -> Result<Line, ParseError> {
    let mut inner = pair.into_inner();
    let left = parse_var_or_constant(inner.next().unwrap(), context)?;
    let right = parse_var_or_constant(inner.next().unwrap(), context)?;

    Ok(Line::Assert(Boolean::Different { left, right }))
}

fn parse_condition(pair: Pair<Rule>, context: &ParseContext) -> Result<Boolean, ParseError> {
    let inner_pair = pair.into_inner().next().unwrap();

    match inner_pair.as_rule() {
        Rule::condition_eq => {
            let mut inner = inner_pair.into_inner();
            let left = parse_var_or_constant(inner.next().unwrap(), context)?;
            let right = parse_var_or_constant(inner.next().unwrap(), context)?;
            Ok(Boolean::Equal { left, right })
        }
        Rule::condition_diff => {
            let mut inner = inner_pair.into_inner();
            let left = parse_var_or_constant(inner.next().unwrap(), context)?;
            let right = parse_var_or_constant(inner.next().unwrap(), context)?;
            Ok(Boolean::Different { left, right })
        }
        _ => unreachable!(),
    }
}

fn parse_binary_expression(
    pair: Pair<Rule>,
    context: &ParseContext,
) -> Result<(HighLevelOperation, VarOrConstant, VarOrConstant), ParseError> {
    let mut inner = pair.into_inner();
    let left = parse_var_or_constant(inner.next().unwrap(), context)?;
    let operator = inner.next().unwrap().as_str();
    let right = parse_var_or_constant(inner.next().unwrap(), context)?;

    let operation = match operator {
        "+" => HighLevelOperation::Add,
        "-" => HighLevelOperation::Sub,
        "*" => HighLevelOperation::Mul,
        "/" => HighLevelOperation::Div,
        _ => {
            return Err(ParseError::SemanticError(format!(
                "Unknown binary operator: {}",
                operator
            )));
        }
    };

    Ok((operation, left, right))
}

fn parse_var_or_constant(
    pair: Pair<Rule>,
    context: &ParseContext,
) -> Result<VarOrConstant, ParseError> {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::identifier => {
            let name = inner.as_str();
            if let Ok(constant) = parse_constant_value(name, context) {
                Ok(VarOrConstant::Constant(constant))
            } else {
                Ok(VarOrConstant::Var(Var {
                    name: name.to_string(),
                }))
            }
        }
        Rule::constant_value => Ok(VarOrConstant::Constant(parse_constant_value(
            inner.as_str(),
            context,
        )?)),
        other => Err(ParseError::SemanticError(format!(
            "Expected identifier or constant value, found: {:?}",
            other
        ))),
    }
}

fn parse_constant_value(value: &str, context: &ParseContext) -> Result<ConstantValue, ParseError> {
    if value == "public_input_start" {
        return Ok(ConstantValue::PublicInputStart);
    } else if value == "pointer_to_zero_vector" {
        return Ok(ConstantValue::PointerToZeroVector);
    } else if let Some(value) = context.resolve_constant(value) {
        return Ok(ConstantValue::Scalar(value));
    } else {
        return Ok(ConstantValue::Scalar(
            value
                .parse()
                .or_else(|_| Err(ParseError::NotAConstant(value.to_string())))?,
        ));
    }
}

fn parse_var_list(
    pair: Pair<Rule>,
    _context: &ParseContext,
) -> Result<Vec<VarOrConstant>, ParseError> {
    let mut vars = Vec::new();
    for item in pair.into_inner() {
        assert_eq!(item.as_rule(), Rule::var_or_constant);
        vars.push(parse_var_or_constant(item, _context)?);
    }
    Ok(vars)
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
    a = A;
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

    assert_eq_ext(a, b);

    k = public_input_start;

    for i in a..b {
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
        dbg!("{:?}", parsed);
    }
}
