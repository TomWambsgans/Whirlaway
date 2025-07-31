#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Precompile {
    pub name: PrecompileName,
    pub n_inputs: usize,
    pub n_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrecompileName {
    Poseidon16,
    Poseidon24,
    MulExtension,
    AddExtension
}

impl ToString for PrecompileName {
    fn to_string(&self) -> String {
        match self {
            PrecompileName::Poseidon16 => "poseidon16",
            PrecompileName::Poseidon24 => "poseidon24",
            PrecompileName::MulExtension => "mul_extension",
            PrecompileName::AddExtension => "add_extension",
        }
        .to_string()
    }
}

pub const POSEIDON_16: Precompile = Precompile {
    name: PrecompileName::Poseidon16,
    n_inputs: 2,
    n_outputs: 2,
};

pub const POSEIDON_24: Precompile = Precompile {
    name: PrecompileName::Poseidon24,
    n_inputs: 3,
    n_outputs: 3,
};

pub const MUL_EXTENSION: Precompile = Precompile {
    name: PrecompileName::MulExtension,
    n_inputs: 3,
    n_outputs: 0,
};

pub const ADD_EXTENSION: Precompile = Precompile {
    name: PrecompileName::AddExtension,
    n_inputs: 3,
    n_outputs: 0,
};

pub const PRECOMPILES: [Precompile; 4] = [POSEIDON_16, POSEIDON_24, MUL_EXTENSION, ADD_EXTENSION];
