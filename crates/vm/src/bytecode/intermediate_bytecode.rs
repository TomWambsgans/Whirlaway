use std::collections::BTreeMap;

pub type Label = String;

#[derive(Debug, Clone)]
pub struct HighLevelBytecode {
    pub bytecode: BTreeMap<Label, Vec<HighLevelInstruction>>,
    pub memory_size_per_function: BTreeMap<String, usize>,
}

#[derive(Debug, Clone)]
pub enum HighLevelValue {
    Label(Label),
    Constant(usize),
    PublicInputStart,
    PointerToZeroVector, // in the memory of chunks of 8 field elements
    Fp,
    MemoryAfterFp { shift: usize }, // m[fp + shift]
    MemoryPointer { shift: usize }, // m[m[fp + shift]]
    ShiftedMemoryPointer { shift_0: usize, shift_1: usize }, // m[m[fp + shift_0] + shift_1]
    DirectMemory { shift: usize },  // m[shift]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HighLevelOperation {
    Add,
    Mul,
    Sub,
    Div, // in the end everything compiles to either Add or Mul
}

#[derive(Debug, Clone)]
pub enum HighLevelInstruction {
    Computation {
        operation: HighLevelOperation,
        arg_a: HighLevelValue,
        arg_b: HighLevelValue,
        res: HighLevelValue,
    },
    Eq {
        left: HighLevelValue,
        right: HighLevelValue,
    },
    Jump {
        dest: HighLevelValue,
        updated_fp: Option<HighLevelValue>,
    },
    JumpIfNotZero {
        condition: HighLevelValue,
        dest: HighLevelValue,
        updated_fp: Option<HighLevelValue>,
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
        operation: HighLevelOperation,
        arg_a: HighLevelValue, // pointer (to the memory of chunks of 8 field elements)
        arg_b: HighLevelValue, // same
        res: HighLevelValue,   // same
    },

    // HINTS (does not appears in the final bytecode)
    RequestMemory {
        shift: usize,             // m[fp + shift] where the hint will be stored
        size: HighLevelMetaValue, // the hint
        vectorized: bool, // if true, will be 8-alligned, and the returned pointer will be "divied" by 8 (i.e. everything is in chunks of 8 field elements)
    },

    Print {
        line_info: String,            // information about the line where the print occurs
        content: Vec<HighLevelValue>, // values to print
    },
}

#[derive(Debug, Clone)]
pub enum HighLevelMetaValue {
    Constant(usize),
    FunctionSize { function_name: Label },
}

impl ToString for HighLevelValue {
    fn to_string(&self) -> String {
        match self {
            HighLevelValue::Label(label) => label.clone(),
            HighLevelValue::Constant(value) => value.to_string(),
            HighLevelValue::PublicInputStart => "public_input_start".to_string(),
            HighLevelValue::PointerToZeroVector => "pointer_to_zero_vector".to_string(),
            HighLevelValue::Fp => "fp".to_string(),
            HighLevelValue::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
            HighLevelValue::MemoryPointer { shift } => format!("m[m[fp + {}]]", shift),
            HighLevelValue::ShiftedMemoryPointer { shift_0, shift_1 } => {
                format!("m[m[fp + {}] + {}]", shift_0, shift_1)
            }
            HighLevelValue::DirectMemory { shift } => format!("m[{}]", shift),
        }
    }
}

impl ToString for HighLevelInstruction {
    fn to_string(&self) -> String {
        match self {
            HighLevelInstruction::Computation {
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
            HighLevelInstruction::Eq { left, right } => {
                format!("{} == {}", left.to_string(), right.to_string())
            }
            HighLevelInstruction::Jump { dest, updated_fp } => {
                if let Some(fp) = updated_fp {
                    format!("jump {} with fp = {}", dest.to_string(), fp.to_string())
                } else {
                    format!("jump {}", dest.to_string())
                }
            }
            HighLevelInstruction::JumpIfNotZero {
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
            HighLevelInstruction::Poseidon2_16 { shift } => format!(
                "poseidon2_16 m[8 * m[fp + {}] .. 8 * (1 + m[fp + {}])] | m[8 * m[fp + {} + 1]] .. 8 * (1 + m[fp + {} + 1])]",
                shift, shift, shift, shift
            ),
            HighLevelInstruction::Poseidon2_24 { shift } => format!(
                "poseidon2_24 m[8 * m[fp + {}] .. 8 * (1 + m[fp + {}])] | m[8 * m[fp + {} + 1]] .. 8 * (1 + m[fp + {} + 1])]",
                shift, shift, shift, shift
            ),
            HighLevelInstruction::ExtComputation {
                operation,
                arg_a,
                arg_b,
                res,
            } => {
                format!(
                    "{} = ext_computation({}, {}, {})",
                    res.to_string(),
                    arg_a.to_string(),
                    arg_b.to_string(),
                    operation.to_string()
                )
            }
            HighLevelInstruction::RequestMemory {
                shift,
                size,
                vectorized,
            } => format!(
                "request_memory m[fp + {}] = {} {}",
                shift,
                size.to_string(),
                if *vectorized { "# vectorized" } else { "" }
            ),
            HighLevelInstruction::Print { line_info, content } => format!(
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

impl ToString for HighLevelMetaValue {
    fn to_string(&self) -> String {
        match self {
            HighLevelMetaValue::Constant(value) => value.to_string(),
            HighLevelMetaValue::FunctionSize { function_name } => {
                format!("function_size({})", function_name)
            }
        }
    }
}

impl ToString for HighLevelBytecode {
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
