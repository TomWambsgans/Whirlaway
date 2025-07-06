use crate::lang::ConstantValue;

pub type Label = String;

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

#[derive(Debug, Clone, Copy)]
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
        updated_fp: Option<Value>
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
        shift: usize, // m[fp + shift] where the hint will be stored
        size: MetaValue, // the hint
    },
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    Constant(ConstantValue),
    FunctionSize { function_name: Label },
}
