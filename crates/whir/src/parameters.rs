use std::{fmt::Display, str::FromStr};

use serde::Serialize;

const MAX_NUM_VARIABLES_TO_SEND_COEFFS: usize = 6;

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

    /// Computes the number of WHIR rounds and the number of rounds in the final sumcheck.
    #[must_use]
    pub fn compute_number_of_rounds(&self, num_variables: usize) -> (usize, usize) {
        match self {
            FoldingFactor::Constant(factor) => {
                if num_variables <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                    // the first folding is mandatory in the current implem (TODO don't fold, send directly the polynomial)
                    return (0, num_variables - factor);
                }

                // Starting from `num_variables`, each round reduces the number of variables by `factor`. As soon as the
                // number of variables is less of equal than `MAX_NUM_VARIABLES_TO_SEND_COEFFS`, we stop folding and the
                // prover sends directly the coefficients of the polynomial.
                let num_rounds =
                    (num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(*factor);
                let final_sumcheck_rounds = num_variables - num_rounds * factor;
                // The -1 accounts for the fact that the last round does not require another folding.
                (num_rounds - 1, final_sumcheck_rounds)
            }
            FoldingFactor::ConstantFromSecondRound(first_round_factor, factor) => {
                // Compute the number of variables remaining after the first round.
                let nv_except_first_round = num_variables - *first_round_factor;
                if nv_except_first_round < MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                    // This case is equivalent to Constant(first_round_factor)
                    // the first folding is mandatory in the current implem (TODO don't fold, send directly the polynomial)
                    return (0, nv_except_first_round);
                }
                // Starting from `num_variables`, the first round reduces the number of variables by `first_round_factor`,
                // and the next ones by `factor`. As soon as the number of variables is less of equal than
                // `MAX_NUM_VARIABLES_TO_SEND_COEFFS`, we stop folding and the prover sends directly the coefficients of the polynomial.
                let num_rounds =
                    (nv_except_first_round - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(*factor);
                let final_sumcheck_rounds = nv_except_first_round - num_rounds * factor;
                // No need to minus 1 because the initial round is already excepted out
                (num_rounds, final_sumcheck_rounds)
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

#[derive(Clone, Debug)]
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
        folding_factor: FoldingFactor,
        cuda: bool,
    ) -> Self {
        Self {
            security_level: security_bits,
            pow_bits: 16,
            folding_factor,
            soundness_type,
            starting_log_inv_rate: log_inv_rate,
            cuda,
        }
    }
}
