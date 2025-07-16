use std::collections::BTreeMap;

use crate::{
    F,
    bytecode::intermediate_bytecode::{HighLevelOperation, IntermediateValue},
};

pub type Label = String;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub hints: BTreeMap<usize, Vec<Hint>>, // pc -> hints
    pub public_input_start: usize,
    pub starting_frame_memory: usize,
    pub ending_pc: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemOrConstant {
    Constant(usize),
    MemoryAfterFp { shift: usize }, // m[fp + shift]
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemOrFpOrConstant {
    MemoryAfterFp { shift: usize }, // m[fp + shift]
    Fp,
    Constant(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemOrFp {
    MemoryAfterFp { shift: usize }, // m[fp + shift]
    Fp,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operation {
    Add,
    Mul,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Instruction {
    Computation {
        operation: Operation,
        arg_a: MemOrConstant,
        arg_b: MemOrFp,
        res: MemOrConstant,
    },
    Deref {
        shift_0: usize,
        shift_1: usize,
        res: MemOrFpOrConstant,
    }, // res = m[m[fp + shift_0] + shift_1]
    JumpIfNotZero {
        condition: MemOrConstant,
        dest: MemOrConstant,
        updated_fp: MemOrFp,
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
}

impl Operation {
    pub fn compute(&self, a: F, b: F) -> F {
        match self {
            Operation::Add => a + b,
            Operation::Mul => a * b,
        }
    }

    pub fn inverse_compute(&self, a: F, b: F) -> F {
        match self {
            Operation::Add => a - b,
            Operation::Mul => a / b,
        }
    }
}

impl TryFrom<HighLevelOperation> for Operation {
    type Error = String;

    fn try_from(value: HighLevelOperation) -> Result<Self, Self::Error> {
        match value {
            HighLevelOperation::Add => Ok(Operation::Add),
            HighLevelOperation::Mul => Ok(Operation::Mul),
            _ => Err(format!("Cannot convert {:?} to +/x", value)),
        }
    }
}

impl TryFrom<IntermediateValue> for MemOrFp {
    type Error = String;

    fn try_from(value: IntermediateValue) -> Result<Self, Self::Error> {
        match value {
            IntermediateValue::MemoryAfterFp { shift } => Ok(MemOrFp::MemoryAfterFp { shift }),
            IntermediateValue::Fp => Ok(MemOrFp::Fp),
            _ => Err(format!("Cannot convert {:?} to MemOrFp", value)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Hint {
    RequestMemory {
        shift: usize, // m[fp + shift] where the hint will be stored
        size: usize,  // the hint
        vectorized: bool,
    },

    Print {
        line_info: String,
        content: Vec<MemOrConstant>,
    },
}

impl ToString for Bytecode {
    fn to_string(&self) -> String {
        let mut pc = 0;
        let mut res = String::new();
        for instruction in &self.instructions {
            for hint in self.hints.get(&pc).unwrap_or(&Vec::new()) {
                res.push_str(&format!("hint: {}\n", hint.to_string()));
            }
            res.push_str(&format!("{:>4}: {}\n", pc, instruction.to_string()));
            pc += 1;
        }
        return res;
    }
}

impl ToString for MemOrConstant {
    fn to_string(&self) -> String {
        match self {
            Self::Constant(c) => format!("{}", c),
            Self::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
        }
    }
}

impl ToString for MemOrFp {
    fn to_string(&self) -> String {
        match self {
            Self::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
            Self::Fp => "fp".to_string(),
        }
    }
}

impl ToString for MemOrFpOrConstant {
    fn to_string(&self) -> String {
        match self {
            Self::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
            Self::Fp => "fp".to_string(),
            Self::Constant(c) => format!("{}", c),
        }
    }
}

impl ToString for Operation {
    fn to_string(&self) -> String {
        match self {
            Self::Add => "+".to_string(),
            Self::Mul => "x".to_string(),
        }
    }
}

impl ToString for Instruction {
    fn to_string(&self) -> String {
        match self {
            Self::Computation {
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
            Self::Deref {
                shift_0,
                shift_1,
                res,
            } => {
                format!("{} = m[m[fp + {}] + {}]", res.to_string(), shift_0, shift_1)
            }
            Self::JumpIfNotZero {
                condition,
                dest,
                updated_fp,
            } => {
                format!(
                    "if {} != 0 jump to {} with next(fp) = {}",
                    condition.to_string(),
                    dest.to_string(),
                    updated_fp.to_string()
                )
            }
            Self::Poseidon2_16 { shift } => {
                format!("Poseidon2_16(m[{}..+4])", shift)
            }
            Self::Poseidon2_24 { shift } => {
                format!("Poseidon2_24(m[{}..+6])", shift)
            }
        }
    }
}

impl ToString for Hint {
    fn to_string(&self) -> String {
        match self {
            Self::RequestMemory {
                shift,
                size,
                vectorized,
            } => {
                format!(
                    "m[fp + {}] = malloc({}) {}",
                    shift,
                    size,
                    if *vectorized { "# vectorized" } else { "" }
                )
            }
            Self::Print { line_info, content } => {
                format!(
                    "print({}) for \"{}\"",
                    content
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<String>>()
                        .join(", "),
                    line_info,
                )
            }
        }
    }
}
