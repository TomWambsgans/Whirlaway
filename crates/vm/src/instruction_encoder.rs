use crate::{
    F, N_INSTRUCTION_FIELDS,
    bytecode::bytecode::{Instruction, MemOrConstant, MemOrFp, MemOrFpOrConstant, Operation},
};
use p3_field::PrimeCharacteristicRing;

impl Instruction {
    pub fn field_representation(&self) -> [F; N_INSTRUCTION_FIELDS] {
        let mut fields = [F::ZERO; N_INSTRUCTION_FIELDS];
        match self {
            Self::Computation {
                operation,
                arg_a,
                arg_c,
                res,
            } => {
                match operation {
                    Operation::Add => {
                        fields[6] = F::ONE;
                    }
                    Operation::Mul => {
                        fields[7] = F::ONE;
                    }
                }

                set_nu_a(&mut fields, arg_a);
                set_nu_b(&mut fields, res);
                set_nu_c(&mut fields, arg_c);
            }
            Self::Deref {
                shift_0,
                shift_1,
                res,
            } => {
                fields[3] = F::ZERO; // flag_A = 0
                fields[0] = F::from_usize(*shift_0);
                fields[5] = F::ONE; // flag_C = 1
                fields[2] = F::from_usize(*shift_1);
                match res {
                    MemOrFpOrConstant::Constant(cst) => {
                        fields[10] = F::ONE; // AUX = 1
                        fields[4] = F::ONE; // flag_B = 1
                        fields[1] = *cst;
                    }
                    MemOrFpOrConstant::MemoryAfterFp { offset } => {
                        fields[10] = F::ONE; // AUX = 1
                        fields[4] = F::ZERO; // flag_B = 0
                        fields[1] = F::from_usize(*offset);
                    }
                    MemOrFpOrConstant::Fp => {
                        fields[10] = F::ZERO; // AUX = 0
                        fields[4] = F::ONE; // flag_B = 1
                    }
                }
            }
            Self::JumpIfNotZero {
                condition,
                dest,
                updated_fp,
            } => {
                fields[9] = F::ONE; // JUZ = 1
                set_nu_a(&mut fields, condition);
                set_nu_b(&mut fields, dest);
                set_nu_c(&mut fields, updated_fp);
            }
            Self::Poseidon2_16 { arg_a, arg_b, res } => {
                fields[11] = F::ONE; // POSEIDON_16 = 1
                set_nu_a(&mut fields, arg_a);
                set_nu_b(&mut fields, arg_b);
                set_nu_c(&mut fields, res);
            }
            Self::Poseidon2_24 { arg_a, arg_b, res } => {
                fields[12] = F::ONE; // POSEIDON_24 = 1
                set_nu_a(&mut fields, arg_a);
                set_nu_b(&mut fields, arg_b);
                set_nu_c(&mut fields, res);
            }
            Self::DotProductExtensionExtension {
                arg0,
                arg1,
                res,
                size,
            } => {
                fields[13] = F::ONE; // DOT_PRODUCT_EXTENSION = 1
                set_nu_a(&mut fields, arg0);
                set_nu_b(&mut fields, arg1);
                set_nu_c(&mut fields, res);
                fields[10] = F::from_usize(*size); // AUX stores size
            }
            Self::DotProductBaseExtension {
                arg_base,
                arg_ext,
                res,
                size,
            } => {
                fields[14] = F::ONE; // DOT_PRODUCT_BASE_EXTENSION = 1
                set_nu_a(&mut fields, arg_base);
                set_nu_b(&mut fields, arg_ext);
                set_nu_c(&mut fields, res);
                fields[10] = F::from_usize(*size); // AUX stores size
            }
        }
        fields
    }
}

fn set_nu_a(fields: &mut [F; N_INSTRUCTION_FIELDS], a: &MemOrConstant) {
    match a {
        MemOrConstant::Constant(cst) => {
            fields[3] = F::ONE;
            fields[0] = *cst;
        }
        MemOrConstant::MemoryAfterFp { offset } => {
            fields[3] = F::ZERO;
            fields[0] = F::from_usize(*offset);
        }
    }
}

fn set_nu_b(fields: &mut [F; N_INSTRUCTION_FIELDS], b: &MemOrConstant) {
    match b {
        MemOrConstant::Constant(cst) => {
            fields[4] = F::ONE;
            fields[1] = *cst;
        }
        MemOrConstant::MemoryAfterFp { offset } => {
            fields[4] = F::ZERO;
            fields[1] = F::from_usize(*offset);
        }
    }
}

fn set_nu_c(fields: &mut [F; N_INSTRUCTION_FIELDS], c: &MemOrFp) {
    match c {
        MemOrFp::Fp => {
            fields[5] = F::ONE;
        }
        MemOrFp::MemoryAfterFp { offset } => {
            fields[5] = F::ZERO;
            fields[2] = F::from_usize(*offset);
        }
    }
}
