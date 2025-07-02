// use std::collections::HashMap;

// use crate::{bytecode::*, lang::*};

// struct Compiler {
//     var_count: usize,
//     vars_in_scope: HashMap<Var, usize>, // var = m[fp + index]
//     current_shift: usize,
//     bytecode: HashMap<Label, Vec<Instruction>>,
//     condition_labels: usize,
// }

// impl Compiler {
//     fn new() -> Self {
//         Compiler {
//             var_count: 0,
//             vars_in_scope: HashMap::new(),
//             current_shift: 0,
//             bytecode: HashMap::new(),
//             condition_labels: 0,
//         }
//     }

//     fn add_var(&mut self, name: String) -> Var {
//         let var = Var { index: self.var_count, name };
//         self.vars_in_scope.insert(var.clone(), self.current_shift);
//         self.var_count += 1;
//         self.current_shift += 1;
//         var
//     }
// }

// pub fn compile_function(program: Program) -> Result<Vec<Instruction>, String> {
//     todo!()
// }

// fn compile_lines(lines: &[Line], compiler: &mut Compiler) -> Result<Vec<Instruction>, String> {
//     let mut res = Vec::new();
//     for (i, line) in lines.iter().enumerate() {
//         match line {
//             Line::Assignment { var, operation, arg0, arg1 } => {
//                 let shift0 = *compiler
//                     .vars_in_scope
//                     .get(arg0)
//                     .ok_or_else(|| format!("Variable {} not found", arg0.name))?;
//                 let shift1 = *compiler
//                     .vars_in_scope
//                     .get(arg1)
//                     .ok_or_else(|| format!("Variable {} not found", arg1.name))?;
//                 let new_var = compiler.add_var(var.name.clone());
//                 let new_shift = compiler.vars_in_scope.get(&new_var).unwrap();
//                 let instruction = Instruction::Computation {
//                     operation: *operation,
//                     arg_a: Value::MemoryAfterFp { shift: shift0 },
//                     arg_b: Value::MemoryAfterFp { shift: shift1 },
//                     res: Value::MemoryAfterFp { shift: *new_shift },
//                 };
//                 res.push(instruction);
//             }
//             Line::ConstantAssignment { var, value } => {
//                 let new_var = compiler.add_var(var.name.clone());
//                 let new_shift = compiler.vars_in_scope.get(&new_var).unwrap();
//                 let instruction = Instruction::Eq {
//                     left: Value::MemoryAfterFp { shift: *new_shift },
//                     right: match value {
//                         ConstantValue::Scalar(scalar) => Value::Constant(*scalar),
//                         ConstantValue::PublicInputStart => Value::PublicInputStart,
//                     },
//                 };
//                 res.push(instruction);
//             }
//             Line::RawAccess { var, index } => {
//                 let new_var = compiler.add_var(var.name.clone());
//                 let new_shift = compiler.vars_in_scope.get(&new_var).unwrap();
//                 let instruction = Instruction::Eq {
//                     left: Value::MemoryAfterFp { shift: *new_shift },
//                     right: Value::DirectMemory { shift: *index },
//                 };
//                 res.push(instruction);
//             }
//             Line::AssertEq { left, right } => {
//                 let shift_left = compiler
//                     .vars_in_scope
//                     .get(left)
//                     .ok_or_else(|| format!("Variable {} not found", left.name))?;
//                 let shift_right = compiler
//                     .vars_in_scope
//                     .get(right)
//                     .ok_or_else(|| format!("Variable {} not found", right.name))?;
//                 let instruction = Instruction::Eq {
//                     left: Value::MemoryAfterFp { shift: *shift_left },
//                     right: Value::MemoryAfterFp { shift: *shift_right },
//                 };
//                 res.push(instruction);
//             }
//             Line::IfCondition {
//                 condition,
//                 then_branch,
//                 else_branch,
//             } => match condition {
//                 Condition::NotZero { var } => {
//                     let label_if = format!("if_not_zero_{}", compiler.condition_labels);
//                     let label_else = format!("else_{}", compiler.condition_labels);
//                     let label_end = format!("end_{}", compiler.condition_labels);
//                     compiler.condition_labels += 1;

//                     let mut if_compiled = compile_lines(then_branch, compiler)?;
//                     if_compiled.push(Instruction::Jump {
//                         dest: Value::Label(label_end.clone()),
//                     });
//                     compiler.bytecode.insert(label_if.clone(), if_compiled);

//                     let mut else_compiled = compile_lines(else_branch, compiler)?;
//                     else_compiled.push(Instruction::Jump {
//                         dest: Value::Label(label_end.clone()),
//                     });
//                     compiler.bytecode.insert(label_else.clone(), else_compiled);

//                     let end_lines = compile_lines(&lines[i + 1..], compiler)?;
//                     compiler.bytecode.insert(label_end.clone(), end_lines);

//                     let shift = compiler
//                         .vars_in_scope
//                         .get(var)
//                         .ok_or_else(|| format!("Variable {} not found", var.name))?;
//                     res.push(Instruction::JumpIfNotZero {
//                         condition: Value::MemoryAfterFp { shift: *shift },
//                         dest: Value::Label(label_if),
//                     });
//                     res.push(Instruction::Jump {
//                         dest: Value::Label(label_else),
//                     });
//                     return Ok(res);
//                 }
//             },
//             Line::ForLoop { iterator, start, end, body } => {}
//             Line::FunctionCall {
//                 function_name,
//                 args,
//                 return_data,
//             } => {}
//             Line::Poseidon16 { arg0, arg1, res0, res1 } => {}
//             _ => todo!(),
//         }
//     }
//     Ok(res)
// }
