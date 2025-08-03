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
    DotProductExtensionExtension,
    DotProductBaseExtension,
}

impl ToString for PrecompileName {
    fn to_string(&self) -> String {
        match self {
            PrecompileName::Poseidon16 => "poseidon16",
            PrecompileName::Poseidon24 => "poseidon24",
            PrecompileName::DotProductExtensionExtension => "dot_product_extension_extension",
            PrecompileName::DotProductBaseExtension => "dot_product_base_extension",
        }
        .to_string()
    }
}

pub const POSEIDON_16: Precompile = Precompile {
    name: PrecompileName::Poseidon16,
    n_inputs: 2,
    n_outputs: 1,
};

pub const POSEIDON_24: Precompile = Precompile {
    name: PrecompileName::Poseidon24,
    n_inputs: 2,
    n_outputs: 1,
};

pub const DOT_PRODUCT_EXTENSION_EXTENSION: Precompile = Precompile {
    name: PrecompileName::DotProductExtensionExtension,
    n_inputs: 4,
    n_outputs: 0,
};

pub const DOT_PRODUCT_BASE_EXTENSION: Precompile = Precompile {
    name: PrecompileName::DotProductBaseExtension,
    n_inputs: 4,
    n_outputs: 0,
};

pub const PRECOMPILES: [Precompile; 4] = [
    POSEIDON_16,
    POSEIDON_24,
    DOT_PRODUCT_EXTENSION_EXTENSION,
    DOT_PRODUCT_BASE_EXTENSION,
];
