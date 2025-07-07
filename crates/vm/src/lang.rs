use std::collections::HashMap;

use crate::bytecode::high_level::HighLevelOperation;

#[derive(Debug, Clone)]
pub struct Program {
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

impl From<ConstantValue> for VarOrConstant {
    fn from(constant: ConstantValue) -> Self {
        Self::Constant(constant)
    }
}

impl From<Var> for VarOrConstant {
    fn from(var: Var) -> Self {
        Self::Var(var)
    }
}

impl VarOrConstant {
    pub fn as_var(&self) -> Option<Var> {
        match self {
            Self::Var(var) => Some(var.clone()),
            Self::Constant(_) => None,
        }
    }

    pub fn as_constant(&self) -> Option<ConstantValue> {
        match self {
            Self::Var(_) => None,
            Self::Constant(constant) => Some(constant.clone()),
        }
    }
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
        operation: HighLevelOperation,
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
        start: VarOrConstant,
        end: VarOrConstant,
        body: Vec<Line>,
    },
    FunctionCall {
        function_name: String,
        args: Vec<VarOrConstant>,
        return_data: Vec<Var>,
    },
    FunctionRet {
        return_data: Vec<VarOrConstant>,
    },
    Poseidon16 {
        arg0: VarOrConstant,
        arg1: VarOrConstant,
        res0: Var,
        res1: Var,
        // 4 pointers in the memory of chunks of 8 field elements
    },
    Poseidon24 {
        arg0: VarOrConstant,
        arg1: VarOrConstant,
        arg2: VarOrConstant,
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
    MAlloc {
        var: Var,
        size: ConstantValue,
    },
    Panic,
}
