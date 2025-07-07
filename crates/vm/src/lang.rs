use std::collections::BTreeMap;

use crate::bytecode::intermediate_bytecode::HighLevelOperation;

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VarOrConstant {
    Var(Var),
    Constant(ConstantValue),
}

impl From<ConstantValue> for VarOrConstant {
    fn from(constant: ConstantValue) -> Self {
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

    pub fn as_constant(&self) -> Option<ConstantValue> {
        match self {
            Self::Var(_) => None,
            Self::Constant(constant) => Some(constant.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Boolean {
    Equal {
        left: VarOrConstant,
        right: VarOrConstant,
    },
    Different {
        left: VarOrConstant,
        right: VarOrConstant,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstantValue {
    Scalar(usize),
    PublicInputStart,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Line {
    Assignment {
        var: Var,
        operation: HighLevelOperation,
        arg0: VarOrConstant,
        arg1: VarOrConstant,
    },
    RawAccess {
        var: Var,
        index: VarOrConstant,
    }, // var = memory[index]
    Assert(Boolean),
    IfCondition {
        condition: Boolean,
        then_branch: Vec<Line>,
        else_branch: Vec<Line>,
    },
    ForLoop {
        iterator: Var,
        start: VarOrConstant,
        end: VarOrConstant,
        body: Vec<Line>,
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
        arg0: VarOrConstant,
        arg1: VarOrConstant,
        res0: Var,
        res1: Var,
        // 4 pointers in the memory of chunks of 8 field elements
    },
    Poseidon24 {
        arg0: VarOrConstant,
        arg1: VarOrConstant,
        arg2: VarOrConstant,
        res0: Var,
        res1: Var,
        res2: Var,
        // 6 pointers in the memory of chunks of 8 field elements
    },
    Print {
        line_info: String,
        content: Vec<VarOrConstant>,
    },
    AssertEqExt {
        left: VarOrConstant,
        right: VarOrConstant,
        // 2 pointers in the memory of chunks of 8 field elements
    },
    MAlloc {
        var: Var,
        size: ConstantValue,
    },
    Panic,
}

impl Line {
    fn to_string_with_indent(&self, indent: usize) -> String {
        let spaces = "    ".repeat(indent);
        let line_str = match self {
            Line::Assignment {
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
            Line::RawAccess { var, index } => {
                format!("{} = memory[{}]", var.to_string(), index.to_string())
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
            } => {
                let body_str = body
                    .iter()
                    .map(|line| line.to_string_with_indent(indent + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "for {} in {}..{} {{\n{}\n{}}}",
                    iterator.to_string(),
                    start.to_string(),
                    end.to_string(),
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
                arg0,
                arg1,
                res0,
                res1,
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
                arg0,
                arg1,
                arg2,
                res0,
                res1,
                res2,
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
            Line::AssertEqExt { left, right } => {
                format!("assert_eq_ext({}, {})", left.to_string(), right.to_string())
            }
            Line::MAlloc { var, size } => {
                format!("{} = malloc({})", var.to_string(), size.to_string())
            }
            Line::Panic => "panic".to_string(),
        };
        format!("{}{}", spaces, line_str)
    }
}

impl ToString for Var {
    fn to_string(&self) -> String {
        self.name.clone()
    }
}

impl ToString for ConstantValue {
    fn to_string(&self) -> String {
        match self {
            ConstantValue::Scalar(scalar) => scalar.to_string(),
            ConstantValue::PublicInputStart => "public_input_start".to_string(),
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

impl ToString for Line {
    fn to_string(&self) -> String {
        self.to_string_with_indent(0)
    }
}

impl ToString for HighLevelOperation {
    fn to_string(&self) -> String {
        match self {
            HighLevelOperation::Add => "+".to_string(),
            HighLevelOperation::Mul => "*".to_string(),
            HighLevelOperation::Sub => "-".to_string(),
            HighLevelOperation::Div => "/".to_string(),
        }
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
            format!("fn {}({}) -> {} {{}}", self.name, args_str, self.n_returned_vars)
        } else {
            format!(
                "fn {}({}) -> {} {{\n{}\n}}",
                self.name, args_str, self.n_returned_vars, instructions_str
            )
        }
    }
}