use std::{fmt::Display, marker::PhantomData, str::FromStr};

use serde::Serialize;

pub fn default_max_pow(num_variables: usize, log_inv_rate: usize) -> usize {
    num_variables + log_inv_rate - 3
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum SoundnessType {
    UniqueDecoding,
    ProvableList,
    ConjectureList,
}

impl Display for SoundnessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                SoundnessType::ProvableList => "without conjecture",
                SoundnessType::ConjectureList => "conjectured",
                SoundnessType::UniqueDecoding => "unique decoding",
            }
        )
    }
}

impl FromStr for SoundnessType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "ProvableList" {
            Ok(SoundnessType::ProvableList)
        } else if s == "ConjectureList" {
            Ok(SoundnessType::ConjectureList)
        } else if s == "UniqueDecoding" {
            Ok(SoundnessType::UniqueDecoding)
        } else {
            Err(format!("Invalid soundness specification: {}", s))
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MultivariateParameters<F> {
    pub(crate) num_variables: usize,
    _field: PhantomData<F>,
}

impl<F> MultivariateParameters<F> {
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            _field: PhantomData,
        }
    }
}

impl<F> Display for MultivariateParameters<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Number of variables: {}", self.num_variables)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FoldingFactor {
    Constant(usize),                       // Use the same folding factor for all rounds
    ConstantFromSecondRound(usize, usize), // Use the same folding factor for all rounds, but the first round uses a different folding factor
}

impl FoldingFactor {
    pub fn at_round(&self, round: usize) -> usize {
        match self {
            FoldingFactor::Constant(factor) => *factor,
            FoldingFactor::ConstantFromSecondRound(first_round_factor, factor) => {
                if round == 0 {
                    *first_round_factor
                } else {
                    *factor
                }
            }
        }
    }

    pub fn as_constant(&self) -> Option<usize> {
        match self {
            FoldingFactor::Constant(factor) => Some(*factor),
            FoldingFactor::ConstantFromSecondRound(_, _) => None,
        }
    }

    pub fn check_validity(&self, num_variables: usize) -> Result<(), String> {
        match self {
            FoldingFactor::Constant(factor) => {
                if *factor > num_variables {
                    Err(format!(
                        "Folding factor {} is greater than the number of variables {}. Polynomial too small, just send it directly.",
                        factor, num_variables
                    ))
                } else if *factor == 0 {
                    // We should at least fold some time
                    Err(format!("Folding factor shouldn't be zero."))
                } else {
                    Ok(())
                }
            }
            FoldingFactor::ConstantFromSecondRound(first_round_factor, factor) => {
                if *first_round_factor > num_variables {
                    Err(format!(
                        "First round folding factor {} is greater than the number of variables {}. Polynomial too small, just send it directly.",
                        first_round_factor, num_variables
                    ))
                } else if *factor > num_variables {
                    Err(format!(
                        "Folding factor {} is greater than the number of variables {}. Polynomial too small, just send it directly.",
                        factor, num_variables
                    ))
                } else if *factor == 0 || *first_round_factor == 0 {
                    // We should at least fold some time
                    Err(format!("Folding factor shouldn't be zero."))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Compute the number of WHIR rounds and the number of rounds in the final
    /// sumcheck.
    pub fn compute_number_of_rounds(&self, num_variables: usize) -> (usize, usize) {
        match self {
            FoldingFactor::Constant(factor) => {
                // It's checked that factor > 0 and factor <= num_variables
                let final_sumcheck_rounds = num_variables % factor;
                (
                    (num_variables - final_sumcheck_rounds) / factor - 1,
                    final_sumcheck_rounds,
                )
            }
            FoldingFactor::ConstantFromSecondRound(first_round_factor, factor) => {
                let nv_except_first_round = num_variables - *first_round_factor;
                if nv_except_first_round < *factor {
                    // This case is equivalent to Constant(first_round_factor)
                    return (0, nv_except_first_round);
                }
                let final_sumcheck_rounds = nv_except_first_round % *factor;
                (
                    // No need to minus 1 because the initial round is already
                    // excepted out
                    (nv_except_first_round - final_sumcheck_rounds) / factor,
                    final_sumcheck_rounds,
                )
            }
        }
    }

    /// Compute folding_factor(0) + ... + folding_factor(n_rounds)
    pub fn total_number(&self, n_rounds: usize) -> usize {
        match self {
            FoldingFactor::Constant(factor) => {
                // It's checked that factor > 0 and factor <= num_variables
                factor * (n_rounds + 1)
            }
            FoldingFactor::ConstantFromSecondRound(first_round_factor, factor) => {
                first_round_factor + factor * n_rounds
            }
        }
    }
}

#[derive(Clone)]
pub struct WhirParameters {
    pub starting_log_inv_rate: usize,
    pub folding_factor: FoldingFactor,
    pub soundness_type: SoundnessType,
    pub security_level: usize,
    pub pow_bits: usize,
    pub cuda: bool,
}

impl WhirParameters {
    pub fn standard(
        soundness_type: SoundnessType,
        security_bits: usize,
        log_inv_rate: usize,
        cuda: bool,
    ) -> Self {
        Self {
            security_level: security_bits,
            pow_bits: 16,
            folding_factor: FoldingFactor::Constant(4),
            soundness_type,
            starting_log_inv_rate: log_inv_rate,
            cuda,
        }
    }
}
