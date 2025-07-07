use std::collections::BTreeMap;

use crate::lang::ConstantValue;

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
    FpAssign {
        value: HighLevelValue,
    },
    Jump {
        dest: HighLevelValue,
        updated_fp: Option<HighLevelValue>,
    },
    JumpIfNotZero {
        condition: HighLevelValue,
        dest: HighLevelValue,
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

    // META INSTRUCTIONS (provides useful hints to run the program, but does not appears in the final bytecode)
    RequestMemory {
        shift: usize,             // m[fp + shift] where the hint will be stored
        size: HighLevelMetaValue, // the hint
    },
}

#[derive(Debug, Clone)]
pub enum HighLevelMetaValue {
    Constant(usize),
    FunctionSize { function_name: Label },
}

impl ToString for ConstantValue {
    fn to_string(&self) -> String {
        match self {
            ConstantValue::Scalar(value) => value.to_string(),
            ConstantValue::PublicInputStart => "public_input_start".to_string(),
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

impl ToString for HighLevelValue {
    fn to_string(&self) -> String {
        match self {
            HighLevelValue::Label(label) => label.clone(),
            HighLevelValue::Constant(value) => value.to_string(),
            HighLevelValue::PublicInputStart => "public_input_start".to_string(),
            HighLevelValue::PointerToZeroVector => "null_pointer_EF".to_string(),
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
            HighLevelInstruction::FpAssign { value } => format!("fp = {}", value.to_string()),
            HighLevelInstruction::Jump { dest, updated_fp } => {
                if let Some(fp) = updated_fp {
                    format!("jump to {} with fp = {}", dest.to_string(), fp.to_string())
                } else {
                    format!("jump to {}", dest.to_string())
                }
            }
            HighLevelInstruction::JumpIfNotZero { condition, dest } => format!(
                "if {} != 0 jump to {}",
                condition.to_string(),
                dest.to_string()
            ),
            HighLevelInstruction::Poseidon2_16 { shift } => {
                format!("poseidon2_16 m[8 * m[fp + {}]] .. ]", shift)
            }
            HighLevelInstruction::Poseidon2_24 { shift } => {
                format!("poseidon2_24 m[8 * m[fp + {}]] .. ]", shift)
            }
            HighLevelInstruction::ExtComputation {
                operation,
                arg_a,
                arg_b,
                res,
            } => {
                format!(
                    "ext_computation {} = {} {} {}",
                    res.to_string(),
                    arg_a.to_string(),
                    operation.to_string(),
                    arg_b.to_string()
                )
            }
            HighLevelInstruction::RequestMemory { shift, size } => format!(
                "# hint: m[fp + {}] = malloc({})",
                shift,
                match size {
                    HighLevelMetaValue::Constant(len) => len.to_string(),
                    HighLevelMetaValue::FunctionSize { function_name } => {
                        format!("function_size({})", function_name)
                    }
                },
            ),
        }
    }
}

impl ToString for HighLevelBytecode {
    fn to_string(&self) -> String {
        let mut result = String::new();
        let main = self
            .bytecode
            .get("@function_main")
            .unwrap_or_else(|| panic!("No main function found in the compiled program"));
        result.push_str("@function_main:\n");
        for instruction in main {
            result.push_str(&instruction.to_string());
            result.push('\n');
        }
        for (label, instructions) in &self.bytecode {
            if label == "@function_main" {
                continue;
            }
            result.push_str(&format!("\n{}:\n", label));
            for instruction in instructions {
                result.push_str(&instruction.to_string());
                result.push('\n');
            }
        }
        result
    }
}
