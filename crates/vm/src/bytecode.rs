pub type Label = String;

#[derive(Debug, Clone)]
pub enum Value {
    Label(Label),
    Constant(usize),
    PublicInputStart,
    Fp,
    MemoryAfterFp { shift: usize }, // m[fp + shift]
    MemoryPointer { shift: usize }, // m[m[fp + shift]]
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
}
