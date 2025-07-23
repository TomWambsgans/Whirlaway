use crate::{
    bytecode::bytecode::Bytecode,
    compiler::{
        a_simplify_lang::simplify_program,
        b_compile_intermediate::compile_to_intermediate_bytecode,
        c_compile_final::compile_to_low_level_bytecode,
    },
    parser::parse_program,
};

mod a_simplify_lang;
mod b_compile_intermediate;
mod c_compile_final;

use a_simplify_lang::SimpleProgram;

pub fn compile_program(program: &str) -> Bytecode {
    let parsed_program = parse_program(program).unwrap();
    // println!("Parsed program: {}", parsed_program.to_string());
    let simple_program = simplify_program(parsed_program);
    // println!("Simplified program: {}", simple_program.to_string());
    let intermediate_bytecode = compile_to_intermediate_bytecode(simple_program).unwrap();
    // println!("Intermediate Bytecode:\n\n{}", intermediate_bytecode.to_string());
    let compiled = compile_to_low_level_bytecode(intermediate_bytecode).unwrap();
    // println!("Compiled Program:\n\n{}", compiled.to_string());
    compiled
}
