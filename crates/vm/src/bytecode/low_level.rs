pub type Label = String;

#[derive(Debug, Clone)]
pub struct Bytecode(pub Vec<Instruction>);

#[derive(Debug, Clone)]
pub enum Value {
    Constant(usize),
    Fp,
    NextFp,
    MemoryAfterFp { shift: usize }, // m[fp + shift]
    DirectMemory { shift: usize },  // m[shift]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Mul,
}

#[derive(Debug, Clone)]
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

    // META INSTRUCTIONS (provides useful hints to run the program, but does not appears in the final bytecode)
    
    RequestMemory {
        shift: usize, // m[fp + shift] where the hint will be stored
        size: usize,  // the hint
    },
}

impl ToString for Bytecode {
    fn to_string(&self) -> String {
        self.0.iter().enumerate()
            .map(|(i, instruction)| format!("{}: {}", i, instruction.to_string()))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

impl ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Value::Constant(c) => format!("{}", c),
            Value::Fp => "fp".to_string(),
            Value::NextFp => "next(fp)".to_string(),
            Value::MemoryAfterFp { shift } => format!("m[fp + {}]", shift),
            Value::DirectMemory { shift } => format!("m[{}]", shift),
        }
    }
}

impl ToString for Operation {
    fn to_string(&self) -> String {
        match self {
            Operation::Add => "+".to_string(),
            Operation::Mul => "x".to_string(),
        }
    }
}

impl ToString for Instruction {
    fn to_string(&self) -> String {
        match self {
            Instruction::Computation { operation, arg_a, arg_b, res } => {
                format!("{} = {} {} {}", res.to_string(), arg_a.to_string(), operation.to_string(), arg_b.to_string())
            }
            Instruction::MemoryPointerEq { shift_0, shift_1, res } => {
                format!("{} = m[m[fp + {}] + {}]", res.to_string(), shift_0, shift_1)
            }
            Instruction::JumpIfNotZero { condition, dest, updated_fp } => {
                format!("if {} != 0 jump to {} with next(fp) = {}", condition.to_string(), dest.to_string(), updated_fp.to_string())
            }
            Instruction::Poseidon2_16 { shift } => {
                format!("Poseidon2_16(m[{}..+4])", shift)
            }
            Instruction::Poseidon2_24 { shift } => {
                format!("Poseidon2_24(m[{}..+6])", shift)
            }
            Instruction::ExtComputation { operation, arg_a, arg_b, res } => {
                format!("m[{}..+8] = m[{}..+8] {} m[{}..+8] # Extension Field", 
                        res.to_string(), arg_a.to_string(), operation.to_string(), arg_b.to_string())
            }
            Instruction::RequestMemory { shift, size } => {
                format!("# hint: m[fp + {}] = malloc({})", shift, size)
            }
        }
    }
}