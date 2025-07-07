use std::collections::BTreeMap;

use crate::F;

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
pub enum Value {
    Constant(usize),
    Fp,
    MemoryAfterFp { shift: usize }, // m[fp + shift]
    DirectMemory { shift: usize },  // m[shift]
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
        arg_a: Value,
        arg_b: Value,
        res: Value,
    },
    MemoryPointerEq {
        shift_0: usize,
        shift_1: usize,
        res: Value,
    }, // res = m[m[fp + shift_0] + shift_1]
    JumpIfNotZero {
        condition: Value,
        dest: Value,
        updated_fp: Value,
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
    ExtComputation {
        operation: Operation,
        arg_a: Value, // pointer (to the memory of chunks of 8 field elements)
        arg_b: Value, // same
        res: Value,   // same
    },
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Hint {
    RequestMemory {
        shift: usize, // m[fp + shift] where the hint will be stored
        size: usize,  // the hint
    },

    Print {
        line_info: String,
        content: Vec<Value>,
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

impl ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Self::Constant(c) => format!("{}", c),
            Self::Fp => "fp".to_string(),
            Self::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
            Self::DirectMemory { shift } => format!("m[{}]", shift),
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
            Self::MemoryPointerEq {
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
            Self::ExtComputation {
                operation,
                arg_a,
                arg_b,
                res,
            } => {
                format!(
                    "m[{}..+8] = m[{}..+8] {} m[{}..+8] # Extension Field",
                    res.to_string(),
                    arg_a.to_string(),
                    operation.to_string(),
                    arg_b.to_string()
                )
            }
        }
    }
}

impl ToString for Hint {
    fn to_string(&self) -> String {
        match self {
            Self::RequestMemory { shift, size } => {
                format!("m[fp + {}] = malloc({})", shift, size)
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
