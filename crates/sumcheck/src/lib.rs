#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod prove;
use p3_field::Field;
pub use prove::*;

mod verify;
use utils::log2_up;
pub use verify::*;

#[cfg(test)]
mod test;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SumcheckGrinding {
    Custom(usize),
    Auto { security_bits: usize },
    None,
}

impl SumcheckGrinding {
    pub fn pow_bits<EF: Field>(&self, degree: usize) -> usize {
        match self {
            Self::Custom(pow_bits) => *pow_bits,
            Self::Auto { security_bits } => {
                security_bits.saturating_sub(EF::bits().saturating_sub(log2_up(degree + 1)))
            }
            Self::None => 0,
        }
    }
}
