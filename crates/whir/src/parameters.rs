use core::panic;
use derive_more::Deref;
use p3_field::{ExtensionField, Field};
use serde::Serialize;
use std::{f64::consts::LOG2_10, fmt::Debug, marker::PhantomData};

const MAX_NUM_VARIABLES_TO_SEND_COEFFS: usize = 6;

#[derive(Clone)]
pub struct WhirConfigBuilder {
    pub soundness_type: SoundnessType,
    pub security_level: usize,
    pub starting_log_inv_rate: usize,
    pub folding_factor: FoldingFactor,
    pub target_pow_bits: usize,
    pub cuda: bool,
}

impl WhirConfigBuilder {
    pub fn standard(
        soundness_type: SoundnessType,
        security_level: usize,
        starting_log_inv_rate: usize,
        folding_factor: FoldingFactor,
        cuda: bool,
    ) -> Self {
        Self {
            soundness_type,
            security_level,
            starting_log_inv_rate,
            folding_factor,
            cuda,
            target_pow_bits: 16,
        }
    }
}

#[derive(Clone, Deref)]
pub struct WhirConfig<F: Field, EF: ExtensionField<F>> {
    #[deref]
    builder: WhirConfigBuilder,
    pub num_variables: usize,
    _coeffs_field: PhantomData<F>,
    _opening_field: PhantomData<EF>,

    pub(crate) committment_ood_samples: usize,
    // The WHIR protocol can prove either:
    // 1. The commitment is a valid low degree polynomial. In that case, the
    //    initial statement is set to false.
    // 2. The commitment is a valid folded polynomial, and an additional
    //    polynomial evaluation statement. In that case, the initial statement
    //    is set to true.
    pub(crate) starting_folding_pow_bits: usize,

    pub(crate) round_parameters: Vec<RoundConfig>,

    pub(crate) final_queries: usize,
    pub(crate) final_pow_bits: usize,
    pub(crate) final_log_inv_rate: usize,
    pub(crate) final_sumcheck_rounds: usize,
    pub(crate) final_folding_pow_bits: usize,
}

#[derive(Clone)]
pub(crate) struct RoundConfig {
    pub(crate) pow_bits: usize,
    pub(crate) folding_pow_bits: usize,
    pub(crate) num_queries: usize,
    pub(crate) ood_samples: usize,
    pub(crate) log_inv_rate: usize,
}

impl WhirConfigBuilder {
    pub fn build<F: Field, EF: ExtensionField<F>>(
        &self,
        num_variables: usize,
    ) -> WhirConfig<F, EF> {
        self.folding_factor.check_validity(num_variables).unwrap();

        let protocol_security_level = self.security_level.saturating_sub(self.target_pow_bits);

        let (num_rounds, final_sumcheck_rounds) =
            self.folding_factor.compute_number_of_rounds(num_variables);

        let field_size_bits = EF::bits() as usize;

        let committment_ood_samples = ood_samples(
            self.security_level,
            self.soundness_type,
            num_variables,
            self.starting_log_inv_rate,
            compute_log_eta(self.soundness_type, self.starting_log_inv_rate),
            field_size_bits,
        );

        let starting_folding_pow_bits = folding_pow_bits(
            self.security_level,
            self.soundness_type,
            field_size_bits,
            num_variables,
            self.starting_log_inv_rate,
            compute_log_eta(self.soundness_type, self.starting_log_inv_rate),
        );

        let mut round_parameters = Vec::with_capacity(num_rounds);
        let mut num_variables_new = num_variables - self.folding_factor.at_round(0);
        let mut log_inv_rate = self.starting_log_inv_rate;
        for round in 0..num_rounds {
            // Queries are set w.r.t. to old rate, while the rest to the new rate
            let next_rate = log_inv_rate + (self.folding_factor.at_round(round) - 1);

            let log_next_eta = compute_log_eta(self.soundness_type, next_rate);
            let num_queries = queries(self.soundness_type, protocol_security_level, log_inv_rate);

            let ood_samples = ood_samples(
                self.security_level,
                self.soundness_type,
                num_variables_new,
                next_rate,
                log_next_eta,
                field_size_bits,
            );

            let query_error = rbr_queries(self.soundness_type, log_inv_rate, num_queries);
            let combination_error = rbr_soundness_queries_combination(
                self.soundness_type,
                field_size_bits,
                num_variables_new,
                next_rate,
                log_next_eta,
                ood_samples,
                num_queries,
            );

            let pow_bits = 0_f64
                .max(self.security_level as f64 - (query_error.min(combination_error)))
                .ceil() as usize;

            let folding_pow_bits = folding_pow_bits(
                self.security_level,
                self.soundness_type,
                field_size_bits,
                num_variables_new,
                next_rate,
                log_next_eta,
            );

            round_parameters.push(RoundConfig {
                ood_samples,
                num_queries,
                pow_bits,
                folding_pow_bits,
                log_inv_rate,
            });

            num_variables_new -= self.folding_factor.at_round(round + 1);
            log_inv_rate = next_rate;
        }

        let final_queries = queries(self.soundness_type, protocol_security_level, log_inv_rate);

        let final_pow_bits = 0_f64
            .max(
                self.security_level as f64
                    - rbr_queries(self.soundness_type, log_inv_rate, final_queries),
            )
            .ceil() as usize;

        let final_folding_pow_bits = 0_f64
            .max(self.security_level as f64 - (field_size_bits - 1) as f64)
            .ceil() as usize;

        WhirConfig {
            builder: self.clone(),
            num_variables,
            _coeffs_field: PhantomData,
            _opening_field: PhantomData,
            committment_ood_samples,
            starting_folding_pow_bits,
            round_parameters,
            final_queries,
            final_pow_bits,
            final_sumcheck_rounds,
            final_folding_pow_bits,
            final_log_inv_rate: log_inv_rate,
        }
    }
}

fn compute_log_eta(soundness_type: SoundnessType, log_inv_rate: usize) -> f64 {
    // Ask me how I did this? At the time, only God and I knew. Now only God knows
    match soundness_type {
        SoundnessType::ProvableList => -(0.5 * log_inv_rate as f64 + LOG2_10 + 1.),
        SoundnessType::UniqueDecoding => 0.,
        SoundnessType::ConjectureList => -(log_inv_rate as f64 + 1.),
    }
}

fn list_size_bits(
    soundness_type: SoundnessType,
    num_variables: usize,
    log_inv_rate: usize,
    log_eta: f64,
) -> f64 {
    match soundness_type {
        SoundnessType::ConjectureList => (num_variables + log_inv_rate) as f64 - log_eta,
        SoundnessType::ProvableList => {
            let log_inv_sqrt_rate: f64 = log_inv_rate as f64 / 2.;
            log_inv_sqrt_rate - (1. + log_eta)
        }
        SoundnessType::UniqueDecoding => 0.0,
    }
}

fn rbr_ood_sample(
    soundness_type: SoundnessType,
    num_variables: usize,
    log_inv_rate: usize,
    log_eta: f64,
    field_size_bits: usize,
    ood_samples: usize,
) -> f64 {
    let list_size_bits = list_size_bits(soundness_type, num_variables, log_inv_rate, log_eta);

    let error = 2. * list_size_bits + (num_variables * ood_samples) as f64;
    (ood_samples * field_size_bits) as f64 + 1. - error
}

fn ood_samples(
    security_level: usize, // We don't do PoW for OOD
    soundness_type: SoundnessType,
    num_variables: usize,
    log_inv_rate: usize,
    log_eta: f64,
    field_size_bits: usize,
) -> usize {
    if matches!(soundness_type, SoundnessType::UniqueDecoding) {
        0
    } else {
        for ood_samples in 1..64 {
            if rbr_ood_sample(
                soundness_type,
                num_variables,
                log_inv_rate,
                log_eta,
                field_size_bits,
                ood_samples,
            ) >= security_level as f64
            {
                return ood_samples;
            }
        }

        panic!("Could not find an appropriate number of OOD samples");
    }
}

// Compute the proximity gaps term of the fold
fn rbr_soundness_fold_prox_gaps(
    soundness_type: SoundnessType,
    field_size_bits: usize,
    num_variables: usize,
    log_inv_rate: usize,
    log_eta: f64,
) -> f64 {
    // Recall, at each round we are only folding by two at a time
    let error = match soundness_type {
        SoundnessType::ConjectureList => (num_variables + log_inv_rate) as f64 - log_eta,
        SoundnessType::ProvableList => {
            LOG2_10 + 3.5 * log_inv_rate as f64 + 2. * num_variables as f64
        }
        SoundnessType::UniqueDecoding => (num_variables + log_inv_rate) as f64,
    };

    field_size_bits as f64 - error
}

fn rbr_soundness_fold_sumcheck(
    soundness_type: SoundnessType,
    field_size_bits: usize,
    num_variables: usize,
    log_inv_rate: usize,
    log_eta: f64,
) -> f64 {
    let list_size = list_size_bits(soundness_type, num_variables, log_inv_rate, log_eta);

    field_size_bits as f64 - (list_size + 1.)
}

fn folding_pow_bits(
    security_level: usize,
    soundness_type: SoundnessType,
    field_size_bits: usize,
    num_variables: usize,
    log_inv_rate: usize,
    log_eta: f64,
) -> usize {
    let prox_gaps_error = rbr_soundness_fold_prox_gaps(
        soundness_type,
        field_size_bits,
        num_variables,
        log_inv_rate,
        log_eta,
    );
    let sumcheck_error = rbr_soundness_fold_sumcheck(
        soundness_type,
        field_size_bits,
        num_variables,
        log_inv_rate,
        log_eta,
    );

    let error = prox_gaps_error.min(sumcheck_error);

    0_f64.max(security_level as f64 - error).ceil() as usize
}

// Used to select the number of queries
fn queries(
    soundness_type: SoundnessType,
    protocol_security_level: usize,
    log_inv_rate: usize,
) -> usize {
    let num_queries_f = match soundness_type {
        SoundnessType::UniqueDecoding => {
            let rate = 1. / ((1 << log_inv_rate) as f64);
            let denom = (0.5 * (1. + rate)).log2();

            -(protocol_security_level as f64) / denom
        }
        SoundnessType::ProvableList => (2 * protocol_security_level) as f64 / log_inv_rate as f64,
        SoundnessType::ConjectureList => protocol_security_level as f64 / log_inv_rate as f64,
    };
    num_queries_f.ceil() as usize
}

// This is the bits of security of the query step
fn rbr_queries(soundness_type: SoundnessType, log_inv_rate: usize, num_queries: usize) -> f64 {
    let num_queries = num_queries as f64;

    match soundness_type {
        SoundnessType::UniqueDecoding => {
            let rate = 1. / ((1 << log_inv_rate) as f64);
            let denom = -(0.5 * (1. + rate)).log2();

            num_queries * denom
        }
        SoundnessType::ProvableList => num_queries * 0.5 * log_inv_rate as f64,
        SoundnessType::ConjectureList => num_queries * log_inv_rate as f64,
    }
}

fn rbr_soundness_queries_combination(
    soundness_type: SoundnessType,
    field_size_bits: usize,
    num_variables: usize,
    log_inv_rate: usize,
    log_eta: f64,
    ood_samples: usize,
    num_queries: usize,
) -> f64 {
    let list_size = list_size_bits(soundness_type, num_variables, log_inv_rate, log_eta);

    let log_combination = ((ood_samples + num_queries) as f64).log2();

    field_size_bits as f64 - (log_combination + list_size + 1.)
}

impl<F: Field, EF: ExtensionField<F>> WhirConfig<F, EF> {
    pub fn n_rounds(&self) -> usize {
        self.round_parameters.len()
    }
    pub fn check_pow_bits(&self) -> bool {
        [
            self.starting_folding_pow_bits,
            self.final_pow_bits,
            self.final_folding_pow_bits,
        ]
        .into_iter()
        .all(|x| x <= self.target_pow_bits)
            && self.round_parameters.iter().all(|r| {
                r.pow_bits <= self.target_pow_bits && r.folding_pow_bits <= self.target_pow_bits
            })
    }
}

impl<F: Field, EF: ExtensionField<F>> Debug for WhirConfig<F, EF> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "num variables: {}", self.num_variables)?;
        writeln!(f, ", folding factor: {:?}", self.folding_factor)?;
        writeln!(
            f,
            "Security level: {} bits using {:?} security and {} bits of PoW",
            self.security_level, self.soundness_type, self.target_pow_bits
        )?;

        writeln!(
            f,
            "initial_folding_pow_bits: {}",
            self.starting_folding_pow_bits
        )?;
        for r in &self.round_parameters {
            r.fmt(f)?;
        }

        writeln!(
            f,
            "final_queries: {}, final_rate: 2^-{}, final_pow_bits: {}, final_folding_pow_bits: {}",
            self.final_queries,
            self.final_log_inv_rate,
            self.final_pow_bits,
            self.final_folding_pow_bits,
        )?;

        writeln!(f, "------------------------------------")?;
        writeln!(f, "Round by round soundness analysis:")?;
        writeln!(f, "------------------------------------")?;

        let field_size_bits = EF::bits() as usize;
        let log_eta = compute_log_eta(self.soundness_type, self.starting_log_inv_rate);
        let mut num_variables = self.num_variables;

        if self.committment_ood_samples > 0 {
            writeln!(
                f,
                "{:.1} bits -- OOD commitment",
                rbr_ood_sample(
                    self.soundness_type,
                    num_variables,
                    self.starting_log_inv_rate,
                    log_eta,
                    field_size_bits,
                    self.committment_ood_samples
                )
            )?;
        }

        let prox_gaps_error = rbr_soundness_fold_prox_gaps(
            self.soundness_type,
            field_size_bits,
            num_variables,
            self.starting_log_inv_rate,
            log_eta,
        );
        let sumcheck_error = rbr_soundness_fold_sumcheck(
            self.soundness_type,
            field_size_bits,
            num_variables,
            self.starting_log_inv_rate,
            log_eta,
        );
        writeln!(
            f,
            "{:.1} bits -- (x{}) prox gaps: {:.1}, sumcheck: {:.1}, pow: {:.1}",
            prox_gaps_error.min(sumcheck_error) + self.starting_folding_pow_bits as f64,
            self.folding_factor.at_round(0),
            prox_gaps_error,
            sumcheck_error,
            self.starting_folding_pow_bits,
        )?;

        num_variables -= self.folding_factor.at_round(0);

        for (round, r) in self.round_parameters.iter().enumerate() {
            let next_rate = r.log_inv_rate + (self.folding_factor.at_round(round) - 1);
            let log_eta = compute_log_eta(self.soundness_type, next_rate);

            if r.ood_samples > 0 {
                writeln!(
                    f,
                    "{:.1} bits -- OOD sample",
                    rbr_ood_sample(
                        self.soundness_type,
                        num_variables,
                        next_rate,
                        log_eta,
                        field_size_bits,
                        r.ood_samples
                    )
                )?;
            }

            let query_error = rbr_queries(self.soundness_type, r.log_inv_rate, r.num_queries);
            let combination_error = rbr_soundness_queries_combination(
                self.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_eta,
                r.ood_samples,
                r.num_queries,
            );
            writeln!(
                f,
                "{:.1} bits -- query error: {:.1}, combination: {:.1}, pow: {:.1}",
                query_error.min(combination_error) + r.pow_bits as f64,
                query_error,
                combination_error,
                r.pow_bits,
            )?;

            let prox_gaps_error = rbr_soundness_fold_prox_gaps(
                self.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_eta,
            );
            let sumcheck_error = rbr_soundness_fold_sumcheck(
                self.soundness_type,
                field_size_bits,
                num_variables,
                next_rate,
                log_eta,
            );

            writeln!(
                f,
                "{:.1} bits -- (x{}) prox gaps: {:.1}, sumcheck: {:.1}, pow: {:.1}",
                prox_gaps_error.min(sumcheck_error) + r.folding_pow_bits as f64,
                self.folding_factor.at_round(round + 1),
                prox_gaps_error,
                sumcheck_error,
                r.folding_pow_bits,
            )?;

            num_variables -= self.folding_factor.at_round(round + 1);
        }

        let query_error = rbr_queries(
            self.soundness_type,
            self.final_log_inv_rate,
            self.final_queries,
        );
        writeln!(
            f,
            "{:.1} bits -- query error: {:.1}, pow: {:.1}",
            query_error + self.final_pow_bits as f64,
            query_error,
            self.final_pow_bits,
        )?;

        if self.final_sumcheck_rounds > 0 {
            let combination_error = field_size_bits as f64 - 1.;
            writeln!(
                f,
                "{:.1} bits -- (x{}) combination: {:.1}, pow: {:.1}",
                combination_error + self.final_pow_bits as f64,
                self.final_sumcheck_rounds,
                combination_error,
                self.final_folding_pow_bits,
            )?;
        }

        Ok(())
    }
}

impl Debug for RoundConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Num_queries: {}, rate: 2^-{}, pow_bits: {}, ood_samples: {}, folding_pow: {}",
            self.num_queries,
            self.log_inv_rate,
            self.pow_bits,
            self.ood_samples,
            self.folding_pow_bits,
        )
    }
}

#[derive(Clone, Copy, Serialize)]
pub enum SoundnessType {
    UniqueDecoding,
    ProvableList,
    ConjectureList,
}

impl Debug for SoundnessType {
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
