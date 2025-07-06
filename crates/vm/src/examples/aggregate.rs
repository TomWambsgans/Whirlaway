use crate::{compiler::compile_program, parser::parse_program};

#[test]
fn compile_aggregate_program() {
    let program_str = include_str!("aggregate.vm");
    let parsed_program = parse_program(program_str).unwrap();
    let compiled = compile_program(parsed_program).unwrap();
    dbg!(compiled);
}