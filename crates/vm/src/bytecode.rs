use std::collections::HashMap;

use crate::lang::ConstantValue;

pub type Label = String;

#[derive(Debug, Clone)]
pub struct CompiledProgram(pub HashMap<Label, Vec<Instruction>>);

#[derive(Debug, Clone)]
pub enum Value {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Mul,
    Sub,
    Div, // in the end everything compiles to either Add or Mul
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Computation {
        operation: Operation,
        arg_a: Value,
        arg_b: Value,
        res: Value,
    },
    Eq {
        left: Value,
        right: Value,
    },
    FpAssign {
        value: Value,
    },
    Jump {
        dest: Value,
        updated_fp: Option<Value>,
    },
    JumpIfNotZero {
        condition: Value,
        dest: Value,
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
    // Meta instructions (provides useful hints to run the program, but does not appears in the final bytecode)
    RequestMemory {
        shift: usize,    // m[fp + shift] where the hint will be stored
        size: MetaValue, // the hint
    },
}

#[derive(Debug, Clone)]
pub enum MetaValue {
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

impl ToString for MetaValue {
    fn to_string(&self) -> String {
        match self {
            MetaValue::Constant(value) => value.to_string(),
            MetaValue::FunctionSize { function_name } => {
                format!("function_size({})", function_name)
            }
        }
    }
}

impl ToString for Operation {
    fn to_string(&self) -> String {
        match self {
            Operation::Add => "+".to_string(),
            Operation::Mul => "*".to_string(),
            Operation::Sub => "-".to_string(),
            Operation::Div => "/".to_string(),
        }
    }
}

impl ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Value::Label(label) => label.clone(),
            Value::Constant(value) => value.to_string(),
            Value::PublicInputStart => "public_input_start".to_string(),
            Value::PointerToZeroVector => "null_pointer_EF".to_string(),
            Value::Fp => "fp".to_string(),
            Value::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
            Value::MemoryPointer { shift } => format!("m[m[fp + {}]]", shift),
            Value::ShiftedMemoryPointer { shift_0, shift_1 } => {
                format!("m[m[fp + {}] + {}]", shift_0, shift_1)
            }
            Value::DirectMemory { shift } => format!("m[{}]", shift),
        }
    }
}

impl ToString for Instruction {
    fn to_string(&self) -> String {
        match self {
            Instruction::Computation {
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
            Instruction::Eq { left, right } => {
                format!("{} == {}", left.to_string(), right.to_string())
            }
            Instruction::FpAssign { value } => format!("fp = {}", value.to_string()),
            Instruction::Jump { dest, updated_fp } => {
                if let Some(fp) = updated_fp {
                    format!("jump to {} with fp = {}", dest.to_string(), fp.to_string())
                } else {
                    format!("jump to {}", dest.to_string())
                }
            }
            Instruction::JumpIfNotZero { condition, dest } => format!(
                "if {} != 0 jump to {}",
                condition.to_string(),
                dest.to_string()
            ),
            Instruction::Poseidon2_16 { shift } => {
                format!("poseidon2_16 m[8 * m[fp + {}]] .. ]", shift)
            }
            Instruction::Poseidon2_24 { shift } => {
                format!("poseidon2_24 m[8 * m[fp + {}]] .. ]", shift)
            }
            Instruction::ExtComputation {
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
            Instruction::RequestMemory { shift, size } => format!(
                "# hint: m[fp + {}] = malloc({})",
                shift,
                match size {
                    MetaValue::Constant(len) => len.to_string(),
                    MetaValue::FunctionSize { function_name } => {
                        format!("function_size({})", function_name)
                    }
                },
            ),
        }
    }
}

impl ToString for CompiledProgram {
    fn to_string(&self) -> String {
        let mut result = String::new();
        let main = self
            .0
            .get("@function_main")
            .unwrap_or_else(|| panic!("No main function found in the compiled program"));
        result.push_str("@function_main:\n");
        for instruction in main {
            result.push_str(&instruction.to_string());
            result.push('\n');
        }
        for (label, instructions) in &self.0 {
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
