use crate::{bytecode::*, compiler::phase_2::first_compile_pass, lang::*};

pub fn compile_program(program: Program) -> Result<CompiledProgram, String> {
    let mut bytecode = first_compile_pass(program)?;
    use_only_add_and_mul_ops(&mut bytecode);
    return Ok(bytecode);
}

fn use_only_add_and_mul_ops(program: &mut CompiledProgram) {
    for (_, instructions) in &mut program.0 {
        use_only_add_and_mul_ops_helper(instructions);
    }
}

fn use_only_add_and_mul_ops_helper(instructions: &mut Vec<Instruction>) {
    for instruction in instructions {
        if let Instruction::Computation {
            operation,
            arg_a,
            arg_b,
            res,
        } = instruction
        {
            if *operation == Operation::Div {
                (*operation, *res, *arg_a, *arg_b) =
                    (Operation::Mul, arg_a.clone(), res.clone(), arg_b.clone());
            }
            if *operation == Operation::Sub {
                (*operation, *res, *arg_a, *arg_b) =
                    (Operation::Add, arg_a.clone(), res.clone(), arg_b.clone());
            }
        }
    }
}
