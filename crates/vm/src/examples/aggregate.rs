use crate::{compiler::compile_to_low_level_bytecode, parser::parse_program};

#[test]
fn compile_aggregate_program() {
    let program_str = include_str!("aggregate.vm");
    let parsed_program = parse_program(program_str).unwrap();
    let compiled = compile_to_low_level_bytecode(parsed_program).unwrap();
    println!("Compiled Program:\n\n{}", compiled.to_string());
    // dbg!(compiled);
}