use crate::parser::parse_program;

#[test]
fn compile_aggregate_program() {
    let program_str = include_str!("aggregate.vm");
    let parsed_program = parse_program(program_str).unwrap();
    dbg!(parsed_program);
}