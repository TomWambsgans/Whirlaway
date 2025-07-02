use std::collections::HashMap;

use crate::bytecode::Operation;

#[derive(Debug, Clone)]
pub struct Program {
    pub main_function: Function,
    pub functions: HashMap<String, Function>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<Var>,
    pub n_returned_vars: usize,
    pub instructions: Vec<Line>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Var {
    pub name: String,
}

#[derive(Debug, Clone)]
pub enum VarOrConstant {
    Var(Var),
    Constant(ConstantValue),
}

#[derive(Debug, Clone)]
pub enum Boolean {
    Equal {
        left: VarOrConstant,
        right: VarOrConstant,
    },
    Different {
        left: VarOrConstant,
        right: VarOrConstant,
    },
}

#[derive(Debug, Clone)]
pub enum ConstantValue {
    Scalar(usize),
    PublicInputStart,
}

#[derive(Debug, Clone)]
pub enum Line {
    Assignment {
        var: Var,
        operation: Operation,
        arg0: VarOrConstant,
        arg1: VarOrConstant,
    },
    RawAccess {
        var: Var,
        index: VarOrConstant,
    }, // var = memory[index]
    Assert(Boolean),
    IfCondition {
        condition: Boolean,
        then_branch: Vec<Line>,
        else_branch: Vec<Line>,
    },
    ForLoop {
        iterator: Var,
        start: Var,
        end: Var,
        body: Vec<Line>,
    },
    FunctionCall {
        function_name: String,
        args: Vec<Var>,
        return_data: Vec<Var>,
    },
    FunctionRet {
        return_data: Vec<Var>,
    },
    Poseidon16 {
        arg0: Var,
        arg1: Var,
        res0: Var,
        res1: Var,
        // 4 pointers in the memory of chunks of 8 field elements
    },
    Poseidon24 {
        arg0: Var,
        arg1: Var,
        arg2: Var,
        res0: Var,
        res1: Var,
        res2: Var,
        // 6 pointers in the memory of chunks of 8 field elements
    },
    AssertEqExt {
        left: VarOrConstant,
        right: VarOrConstant,
        // 2 pointers in the memory of chunks of 8 field elements
    },
    // 3 pointers in the memory of chunks of 8 field elements
}
