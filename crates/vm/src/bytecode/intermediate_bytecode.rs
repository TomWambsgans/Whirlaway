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

    // META INSTRUCTIONS (provides useful hints to run the program, but does not appears in the final bytecode)
    RequestMemory {
        shift: usize,             // m[fp + shift] where the hint will be stored
        size: HighLevelMetaValue, // the hint
    },

    Print {
        line_info: String, // information about the line where the print occurs
        content: Vec<HighLevelValue>, // values to print
    }
}

#[derive(Debug, Clone)]
pub enum HighLevelMetaValue {
    Constant(usize),
    FunctionSize { function_name: Label },
}
