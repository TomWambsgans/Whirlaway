use p3_field::PrimeCharacteristicRing;
use std::collections::BTreeMap;

use crate::{
    F,
    bytecode::{bytecode::Label, intermediate_bytecode::HighLevelOperation},
};

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: BTreeMap<String, Function>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<Var>,
    pub n_returned_vars: usize,
    pub instructions: Vec<Line>,
}

pub type Var = String;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VarOrConstant {
    Var(Var),
    Constant(ConstExpression),
}

impl VarOrConstant {
    pub fn zero() -> Self {
        Self::scalar(0)
    }

    pub fn one() -> Self {
        Self::scalar(1)
    }

    pub fn scalar(scalar: usize) -> Self {
        Self::Constant(ConstantValue::Scalar(scalar).into())
    }
}

impl From<ConstantValue> for VarOrConstant {
    fn from(constant: ConstantValue) -> Self {
        Self::Constant(constant.into())
    }
}

impl From<ConstExpression> for VarOrConstant {
    fn from(constant: ConstExpression) -> Self {
        Self::Constant(constant)
    }
}

impl From<Var> for VarOrConstant {
    fn from(var: Var) -> Self {
        Self::Var(var)
    }
}

impl VarOrConstant {
    pub fn as_var(&self) -> Option<Var> {
        match self {
            Self::Var(var) => Some(var.clone()),
            Self::Constant(_) => None,
        }
    }

    pub fn as_constant(&self) -> Option<ConstExpression> {
        match self {
            Self::Var(_) => None,
            Self::Constant(constant) => Some(constant.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Boolean {
    Equal { left: Expression, right: Expression },
    Different { left: Expression, right: Expression },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstantValue {
    Scalar(usize),
    PublicInputStart,
    PointerToZeroVector, // In the memory of chunks of 8 field elements
    FunctionSize { function_name: Label },
    Label(Label),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstExpression {
    Value(ConstantValue),
    Binary {
        left: Box<Self>,
        operator: HighLevelOperation,
        right: Box<Self>,
    },
}

impl From<usize> for ConstExpression {
    fn from(value: usize) -> Self {
        ConstExpression::Value(ConstantValue::Scalar(value))
    }
}

impl ConstExpression {
    pub fn zero() -> Self {
        Self::scalar(0)
    }

    pub fn one() -> Self {
        Self::scalar(1)
    }

    pub fn label(label: Label) -> Self {
        Self::Value(ConstantValue::Label(label))
    }

    pub fn scalar(scalar: usize) -> Self {
        Self::Value(ConstantValue::Scalar(scalar))
    }

    pub fn function_size(function_name: Label) -> Self {
        Self::Value(ConstantValue::FunctionSize { function_name })
    }
    pub fn eval_with<EvalFn>(&self, func: &EvalFn) -> F
    where
        EvalFn: Fn(&ConstantValue) -> F,
    {
        match self {
            Self::Value(value) => func(value),
            Self::Binary {
                left,
                operator,
                right,
            } => operator.eval(left.eval_with(func), right.eval_with(func)),
        }
    }

    pub fn naive_eval(&self) -> F {
        self.eval_with(&|value| match value {
            ConstantValue::Scalar(scalar) => F::from_usize(*scalar),
            _ => panic!("Naive evaluation only supports scalar constants"),
        })
    }
}

impl From<ConstantValue> for ConstExpression {
    fn from(value: ConstantValue) -> Self {
        Self::Value(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expression {
    Value(VarOrConstant),
    ArrayAccess {
        array: Var,
        index: Box<Expression>,
    },
    Binary {
        left: Box<Self>,
        operator: HighLevelOperation,
        right: Box<Self>,
    },
}

impl From<VarOrConstant> for Expression {
    fn from(value: VarOrConstant) -> Self {
        Self::Value(value)
    }
}

impl From<Var> for Expression {
    fn from(var: Var) -> Self {
        Self::Value(var.into())
    }
}

impl Expression {
    pub fn naive_eval(&self) -> F {
        match self {
            Expression::Value(value) => value.as_constant().map_or_else(
                || panic!("Naive evaluation only supports constant values"),
                |const_expr| const_expr.naive_eval(),
            ),
            Expression::ArrayAccess { array, index } => {
                panic!(
                    "Naive evaluation does not support array access: {}[{}]",
                    array,
                    index.to_string()
                );
            }
            Expression::Binary {
                left,
                operator,
                right,
            } => operator.eval(left.naive_eval(), right.naive_eval()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Line {
    Assignment {
        var: Var,
        value: Expression,
    },
    ArrayAssign {
        // array[index] = value
        array: Var,
        index: Expression,
        value: Expression,
    },
    Assert(Boolean),
    IfCondition {
        condition: Boolean,
        then_branch: Vec<Self>,
        else_branch: Vec<Self>,
    },
    ForLoop {
        iterator: Var,
        start: Expression,
        end: Expression,
        body: Vec<Self>,
        unroll: bool,
    },
    FunctionCall {
        function_name: String,
        args: Vec<Expression>,
        return_data: Vec<Var>,
    },
    FunctionRet {
        return_data: Vec<Expression>,
    },
    Poseidon16 {
        args: [Expression; 2],
        res: [Var; 2],
    },
    Poseidon24 {
        args: [Expression; 3],
        res: [Var; 3],
    },
    Break,
    Panic,
    // Hints:
    Print {
        line_info: String,
        content: Vec<Expression>,
    },
    MAlloc {
        var: Var,
        size: Expression,
        vectorized: bool,
    },
    DecomposeBits {
        var: Var, // a pointer to 31 field elements, containing the bits of "to_decompose"
        to_decompose: Expression,
    },
}

impl ToString for Expression {
    fn to_string(&self) -> String {
        match self {
            Expression::Value(val) => val.to_string(),
            Expression::ArrayAccess { array, index } => {
                format!("{}[{}]", array, index.to_string())
            }
            Expression::Binary {
                left,
                operator,
                right,
            } => {
                format!(
                    "({} {} {})",
                    left.to_string(),
                    operator.to_string(),
                    right.to_string()
                )
            }
        }
    }
}

impl Line {
    fn to_string_with_indent(&self, indent: usize) -> String {
        let spaces = "    ".repeat(indent);
        let line_str = match self {
            Line::Assignment { var, value } => {
                format!("{} = {}", var.to_string(), value.to_string())
            }
            Line::ArrayAssign {
                array,
                index,
                value,
            } => {
                format!(
                    "{}[{}] = {}",
                    array.to_string(),
                    index.to_string(),
                    value.to_string()
                )
            }
            Line::Assert(condition) => format!("assert {}", condition.to_string()),
            Line::IfCondition {
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
                        "if {} {{\n{}\n{}}}",
                        condition.to_string(),
                        then_str,
                        spaces
                    )
                } else {
                    format!(
                        "if {} {{\n{}\n{}}} else {{\n{}\n{}}}",
                        condition.to_string(),
                        then_str,
                        spaces,
                        else_str,
                        spaces
                    )
                }
            }
            Line::ForLoop {
                iterator,
                start,
                end,
                body,
                unroll,
            } => {
                let body_str = body
                    .iter()
                    .map(|line| line.to_string_with_indent(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "for {} in {}..{} {}{{\n{}\n{}}}",
                    iterator.to_string(),
                    start.to_string(),
                    end.to_string(),
                    if *unroll { "unroll " } else { "" },
                    body_str,
                    spaces
                )
            }
            Line::FunctionCall {
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
            Line::FunctionRet { return_data } => {
                let return_data_str = return_data
                    .iter()
                    .map(|arg| arg.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("return {}", return_data_str)
            }
            Line::Poseidon16 {
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
            Line::Poseidon24 {
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
            Line::Print {
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
            Line::MAlloc {
                var,
                size,
                vectorized,
            } => {
                if *vectorized {
                    format!(
                        "{} = malloc_vectorized({})",
                        var.to_string(),
                        size.to_string()
                    )
                } else {
                    format!("{} = malloc({})", var.to_string(), size.to_string())
                }
            }
            Line::DecomposeBits { var, to_decompose } => {
                format!(
                    "{} = decompose_bits({})",
                    var.to_string(),
                    to_decompose.to_string()
                )
            }
            Line::Break => "break".to_string(),
            Line::Panic => "panic".to_string(),
        };
        format!("{}{}", spaces, line_str)
    }
}

impl ToString for Boolean {
    fn to_string(&self) -> String {
        match self {
            Boolean::Equal { left, right } => {
                format!("{} == {}", left.to_string(), right.to_string())
            }
            Boolean::Different { left, right } => {
                format!("{} != {}", left.to_string(), right.to_string())
            }
        }
    }
}

impl ToString for ConstantValue {
    fn to_string(&self) -> String {
        match self {
            ConstantValue::Scalar(scalar) => scalar.to_string(),
            ConstantValue::PublicInputStart => "@public_input_start".to_string(),
            ConstantValue::PointerToZeroVector => "@pointer_to_zero_vector".to_string(),
            ConstantValue::FunctionSize { function_name } => {
                format!("@function_size_{}", function_name)
            }
            ConstantValue::Label(label) => label.to_string(),
        }
    }
}

impl ToString for VarOrConstant {
    fn to_string(&self) -> String {
        match self {
            VarOrConstant::Var(var) => var.to_string(),
            VarOrConstant::Constant(constant) => constant.to_string(),
        }
    }
}

impl ToString for ConstExpression {
    fn to_string(&self) -> String {
        match self {
            ConstExpression::Value(value) => value.to_string(),
            ConstExpression::Binary {
                left,
                operator,
                right,
            } => {
                format!(
                    "({} {} {})",
                    left.to_string(),
                    operator.to_string(),
                    right.to_string()
                )
            }
        }
    }
}

impl ToString for Line {
    fn to_string(&self) -> String {
        self.to_string_with_indent(0)
    }
}

impl ToString for Program {
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

impl ToString for Function {
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
