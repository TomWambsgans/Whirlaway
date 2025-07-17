use std::collections::BTreeMap;

use crate::{bytecode::bytecode::Operation, lang::ConstExpression};

pub type Label = String;

#[derive(Debug, Clone)]
pub struct IntermediateBytecode {
    pub bytecode: BTreeMap<Label, Vec<IntermediateInstruction>>,
    pub memory_size_per_function: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntermediateValue {
    Constant(ConstExpression),
    Fp,
    MemoryAfterFp { shift: usize }, // m[fp + shift]
}

impl From<ConstExpression> for IntermediateValue {
    fn from(value: ConstExpression) -> Self {
        IntermediateValue::Constant(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntermediaryMemOrFpOrConstant {
    MemoryAfterFp { shift: usize }, // m[fp + shift]
    Fp,
    Constant(ConstExpression),
}

impl IntermediateValue {
    pub fn label(label: Label) -> Self {
        Self::Constant(ConstExpression::label(label))
    }

    pub fn as_constant(&self) -> Option<ConstExpression> {
        if let IntermediateValue::Constant(c) = self {
            Some(c.clone())
        } else {
            None
        }
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, IntermediateValue::Constant(_))
    }

    pub fn is_fp(&self) -> bool {
        matches!(self, IntermediateValue::Fp)
    }

    pub fn is_mem_after_fp(&self) -> bool {
        matches!(self, IntermediateValue::MemoryAfterFp { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HighLevelOperation {
    Add,
    Mul,
    Sub,
    Div, // in the end everything compiles to either Add or Mul
}

#[derive(Debug, Clone)]
pub enum IntermediateInstruction {
    Computation {
        operation: Operation,
        arg_a: IntermediateValue,
        arg_b: IntermediateValue,
        res: IntermediateValue,
    },
    Deref {
        shift_0: usize,
        shift_1: usize,
        res: IntermediaryMemOrFpOrConstant,
    }, // res = m[m[fp + shift_0]]
    Panic,
    Jump {
        dest: IntermediateValue,
        updated_fp: Option<IntermediateValue>,
    },
    JumpIfNotZero {
        condition: IntermediateValue,
        dest: IntermediateValue,
        updated_fp: Option<IntermediateValue>,
    },
    Poseidon2_16 {
        shift: usize,
    }, /*
       Poseidon2(m[8 * m[fp + shift]] .. 8 * (1 + m[fp + shift])] | m[8 * m[fp + shift + 1]] .. 8 * (1 + m[fp + shift + 1])])
       = m[8 * m[fp + shift + 2]] .. 8 * (1 + m[fp + shift + 2])] | m[8 * m[fp + shift + 3]] .. 8 * (1 + m[fp + shift + 3])]
       */
    Poseidon2_24 {
        shift: usize,
    }, // same as above, but with 24 field elements

    // HINTS (does not appears in the final bytecode)
    RequestMemory {
        shift: usize,          // m[fp + shift] where the hint will be stored
        size: IntermediateValue, // the hint
        vectorized: bool, // if true, will be 8-alligned, and the returned pointer will be "divied" by 8 (i.e. everything is in chunks of 8 field elements)
    },

    Print {
        line_info: String, // information about the line where the print occurs
        content: Vec<IntermediateValue>, // values to print
    },
}

impl IntermediateInstruction {
    pub fn computation(
        operation: HighLevelOperation,
        arg_a: IntermediateValue,
        arg_b: IntermediateValue,
        res: IntermediateValue,
    ) -> Self {
        match operation {
            HighLevelOperation::Add => Self::Computation {
                operation: Operation::Add,
                arg_a,
                arg_b,
                res,
            },
            HighLevelOperation::Mul => Self::Computation {
                operation: Operation::Mul,
                arg_a,
                arg_b,
                res,
            },
            HighLevelOperation::Sub => Self::Computation {
                operation: Operation::Add,
                arg_a: res,
                arg_b: arg_b,
                res: arg_a,
            },
            HighLevelOperation::Div => Self::Computation {
                operation: Operation::Mul,
                arg_a: res,
                arg_b: arg_b,
                res: arg_a,
            },
        }
    }

    pub fn equality(left: IntermediateValue, right: IntermediateValue) -> Self {
        Self::Computation {
            operation: Operation::Add,
            arg_a: left,
            arg_b: IntermediateValue::Constant(ConstExpression::zero()),
            res: right,
        }
    }
}

impl ToString for IntermediateValue {
    fn to_string(&self) -> String {
        match self {
            IntermediateValue::Constant(value) => value.to_string(),
            IntermediateValue::Fp => "fp".to_string(),
            IntermediateValue::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
        }
    }
}

impl ToString for IntermediaryMemOrFpOrConstant {
    fn to_string(&self) -> String {
        match self {
            Self::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
            Self::Fp => "fp".to_string(),
            Self::Constant(c) => format!("{}", c.to_string()),
        }
    }
}

impl ToString for IntermediateInstruction {
    fn to_string(&self) -> String {
        match self {
            IntermediateInstruction::Deref {
                shift_0,
                shift_1,
                res,
            } => format!("{} = m[m[fp + {}] + {}]", res.to_string(), shift_0, shift_1),
            IntermediateInstruction::Computation {
                operation,
                arg_a,
                arg_b,
                res,
            } => {
                format!(
                    "{} = {} {} {}",
                    res.to_string(),
                    arg_a.to_string(),
                    operation.to_string(),
                    arg_b.to_string()
                )
            }
            IntermediateInstruction::Panic => "panic".to_string(),
            IntermediateInstruction::Jump { dest, updated_fp } => {
                if let Some(fp) = updated_fp {
                    format!("jump {} with fp = {}", dest.to_string(), fp.to_string())
                } else {
                    format!("jump {}", dest.to_string())
                }
            }
            IntermediateInstruction::JumpIfNotZero {
                condition,
                dest,
                updated_fp,
            } => {
                if let Some(fp) = updated_fp {
                    format!(
                        "jump_if_not_zero {} to {} with fp = {}",
                        condition.to_string(),
                        dest.to_string(),
                        fp.to_string()
                    )
                } else {
                    format!(
                        "jump_if_not_zero {} to {}",
                        condition.to_string(),
                        dest.to_string()
                    )
                }
            }
            IntermediateInstruction::Poseidon2_16 { shift } => format!(
                "poseidon2_16 m[8 * m[fp + {}] .. 8 * (1 + m[fp + {}])] | m[8 * m[fp + {} + 1]] .. 8 * (1 + m[fp + {} + 1])]",
                shift, shift, shift, shift
            ),
            IntermediateInstruction::Poseidon2_24 { shift } => format!(
                "poseidon2_24 m[8 * m[fp + {}] .. 8 * (1 + m[fp + {}])] | m[8 * m[fp + {} + 1]] .. 8 * (1 + m[fp + {} + 1])]",
                shift, shift, shift, shift
            ),
            IntermediateInstruction::RequestMemory {
                shift,
                size,
                vectorized,
            } => format!(
                "request_memory m[fp + {}] = {} {}",
                shift,
                size.to_string(),
                if *vectorized { "# vectorized" } else { "" }
            ),
            IntermediateInstruction::Print { line_info, content } => format!(
                "print {}: {}",
                line_info,
                content
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
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

impl ToString for IntermediateBytecode {
    fn to_string(&self) -> String {
        let mut res = String::new();
        for (label, instructions) in &self.bytecode {
            res.push_str(&format!("\n{}:\n", label));
            for instruction in instructions {
                res.push_str(&format!("  {}\n", instruction.to_string()));
            }
        }
        res.push_str("\nMemory size per function:\n");
        for (function_name, size) in &self.memory_size_per_function {
            res.push_str(&format!("{}: {}\n", function_name, size));
        }
        res
    }
}
