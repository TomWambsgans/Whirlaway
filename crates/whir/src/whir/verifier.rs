use algebra::pols::{CoefficientListHost, UnivariatePolynomial};
use fiat_shamir::{FsError, FsVerifier};
use merkle_tree::MultiPath;
use p3_field::{ExtensionField, Field, TwoAdicField};
use std::iter;
use tracing::instrument;
use utils::{KeccakDigest, powers};
use utils::{eq_extension, multilinear_point_from_univariate};

use super::{Statement, parameters::WhirConfig};
use crate::whir::fs_utils::get_challenge_stir_queries;

pub struct Verifier<F: TwoAdicField, EF: ExtensionField<F>> {
    params: WhirConfig<F, EF>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhirError {
    Fs(FsError),
    MerkleTree,
    Decoding,
    SumMismatch,
    FoldingMismatch,
}
impl From<FsError> for WhirError {
    fn from(e: FsError) -> Self {
        WhirError::Fs(e)
    }
}

#[derive(Clone)]
pub struct ParsedCommitment<F, D> {
    root: D,
    ood_points: Vec<F>,
    ood_answers: Vec<F>,
}

#[derive(Clone)]
struct ParsedProof<F: Field> {
    initial_combination_randomness: Vec<F>,
    initial_sumcheck_rounds: Vec<(UnivariatePolynomial<F>, F)>,
    rounds: Vec<ParsedRound<F>>,
    final_randomness_points: Vec<F>,
    final_randomness_answers: Vec<Vec<F>>,
    final_folding_randomness: Vec<F>,
    final_sumcheck_rounds: Vec<(UnivariatePolynomial<F>, F)>,
    final_sumcheck_randomness: Vec<F>,
    final_coefficients: CoefficientListHost<F>,
}

#[derive(Debug, Clone, Hash)]
struct ParsedRound<F: Field> {
    folding_randomness: Vec<F>,
    ood_points: Vec<F>,
    ood_answers: Vec<F>,
    stir_challenges_points: Vec<F>,
    stir_challenges_answers: Vec<Vec<F>>,
    combination_randomness: Vec<F>,
    sumcheck_rounds: Vec<(UnivariatePolynomial<F>, F)>,
}

impl<F: TwoAdicField, EF: ExtensionField<F> + TwoAdicField> Verifier<F, EF> {
    pub fn new(params: WhirConfig<F, EF>) -> Self {
        Verifier { params }
    }

    pub fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<ParsedCommitment<EF, KeccakDigest>, WhirError> {
        let root = KeccakDigest(fs_verifier.next_bytes(32)?.try_into().unwrap()); // TODO avoid harcoding 32

        let mut ood_points = vec![EF::ZERO; self.params.committment_ood_samples];
        let mut ood_answers = vec![EF::ZERO; self.params.committment_ood_samples];
        if self.params.committment_ood_samples > 0 {
            ood_points = fs_verifier.challenge_scalars::<EF>(self.params.committment_ood_samples);
            ood_answers = fs_verifier.next_scalars(self.params.committment_ood_samples)?;
        }

        Ok(ParsedCommitment {
            root,
            ood_points,
            ood_answers,
        })
    }

    fn merkle_verified_answers<IF: ExtensionField<F>>(
        &self,
        r: usize,
        fs_verifier: &mut FsVerifier,
        domain_size: usize,
        prev_root: &KeccakDigest,
        stir_challenges_indexes: &[usize],
    ) -> Result<Vec<Vec<IF>>, WhirError> {
        let merkle_proof = MultiPath::from_bytes(&fs_verifier.next_variable_bytes()?)
            .ok_or(WhirError::Decoding)?;

        let answers: Vec<Vec<IF>> = fs_verifier.next_scalar_matrix(None).unwrap();

        let merkle_tree_height =
            domain_size.trailing_zeros() as usize - self.params.folding_factor.at_round(r);
        if !merkle_proof.verify(&prev_root, &answers, merkle_tree_height)
            || merkle_proof.leaf_indexes != stir_challenges_indexes
        {
            return Err(WhirError::MerkleTree);
        }

        Ok(answers)
    }

    fn parse_round<IF: ExtensionField<F>>(
        &self,
        r: usize,
        fs_verifier: &mut FsVerifier,
        domain_size: &mut usize,
        rounds: &mut Vec<ParsedRound<EF>>,
        exp_domain_gen: &mut EF,
        prev_root: &mut KeccakDigest,
        folding_randomness: &mut Vec<EF>,
        domain_gen: &mut EF,
    ) -> Result<(), WhirError>
    where
        EF: ExtensionField<IF>,
    {
        let round_params = &self.params.round_parameters[r];

        let new_root = KeccakDigest(fs_verifier.next_bytes(32)?.try_into().unwrap()); // TODO avoid harcoding 32

        let mut ood_points = vec![EF::ZERO; round_params.ood_samples];
        let mut ood_answers = vec![EF::ZERO; round_params.ood_samples];
        if round_params.ood_samples > 0 {
            ood_points = fs_verifier.challenge_scalars::<EF>(round_params.ood_samples);
            ood_answers = fs_verifier.next_scalars(round_params.ood_samples)?;
        }

        let stir_challenges_indexes = get_challenge_stir_queries(
            *domain_size,
            self.params.folding_factor.at_round(r),
            round_params.num_queries,
            fs_verifier,
        );

        let stir_challenges_points = stir_challenges_indexes
            .iter()
            .map(|index| exp_domain_gen.exp_u64(*index as u64))
            .collect();

        let answers = self.merkle_verified_answers::<IF>(
            r,
            fs_verifier,
            *domain_size,
            prev_root,
            &stir_challenges_indexes,
        )?;
        fs_verifier.challenge_pow(round_params.pow_bits)?;

        let combination_randomness_gen = fs_verifier.challenge_scalars::<EF>(1)[0];
        let combination_randomness = powers(
            combination_randomness_gen,
            stir_challenges_indexes.len() + round_params.ood_samples,
        );

        let mut sumcheck_rounds = Vec::with_capacity(self.params.folding_factor.at_round(r + 1));
        for _ in 0..self.params.folding_factor.at_round(r + 1) {
            let sumcheck_poly = UnivariatePolynomial::new(fs_verifier.next_scalars(3)?);
            let folding_randomness_single = fs_verifier.challenge_scalars::<EF>(1)[0];
            sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            fs_verifier.challenge_pow(round_params.folding_pow_bits)?;
        }

        let new_folding_randomness = sumcheck_rounds.iter().map(|&(_, r)| r).collect::<Vec<_>>();

        rounds.push(ParsedRound {
            folding_randomness: std::mem::take(folding_randomness),
            ood_points,
            ood_answers,
            stir_challenges_points,
            stir_challenges_answers: answers
                .into_iter()
                .map(|v| v.into_iter().map(|x| EF::from(x)).collect())
                .collect(),
            combination_randomness,
            sumcheck_rounds,
        });

        *folding_randomness = new_folding_randomness;

        *prev_root = new_root;
        *domain_gen = domain_gen.square();
        *exp_domain_gen = domain_gen.exp_u64(1 << self.params.folding_factor.at_round(r + 1));
        *domain_size /= 2;
        Ok(())
    }

    fn parse_proof(
        &self,
        fs_verifier: &mut FsVerifier,
        parsed_commitment: &ParsedCommitment<EF, KeccakDigest>,
        statement: &Statement<EF>, // Will be needed later
    ) -> Result<ParsedProof<EF>, WhirError> {
        let mut initial_sumcheck_rounds = Vec::new();
        let mut final_folding_randomness: Vec<EF>;
        let initial_combination_randomness;
        // Derive combination randomness and first sumcheck polynomial
        let combination_randomness_gen = fs_verifier.challenge_scalars::<EF>(1)[0];
        initial_combination_randomness = powers(
            combination_randomness_gen,
            parsed_commitment.ood_points.len() + statement.points.len(),
        );

        // Initial sumcheck
        initial_sumcheck_rounds.reserve_exact(self.params.folding_factor.at_round(0));
        for _ in 0..self.params.folding_factor.at_round(0) {
            let sumcheck_poly = UnivariatePolynomial::new(fs_verifier.next_scalars::<EF>(3)?);
            let folding_randomness_single = fs_verifier.challenge_scalars::<EF>(1)[0];
            initial_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            fs_verifier.challenge_pow(self.params.starting_folding_pow_bits)?;
        }

        final_folding_randomness = initial_sumcheck_rounds.iter().map(|&(_, r)| r).collect();

        let mut prev_root = parsed_commitment.root.clone();
        let mut domain_gen =
            EF::two_adic_generator(self.params.num_variables + self.params.starting_log_inv_rate);
        let mut exp_domain_gen = domain_gen.exp_u64(1 << self.params.folding_factor.at_round(0));
        let mut domain_size = 1 << (self.params.num_variables + self.params.starting_log_inv_rate);
        let mut rounds = vec![];

        if self.params.n_rounds() > 0 {
            self.parse_round::<F>(
                0,
                fs_verifier,
                &mut domain_size,
                &mut rounds,
                &mut exp_domain_gen,
                &mut prev_root,
                &mut final_folding_randomness,
                &mut domain_gen,
            )?;
        }
        for r in 1..self.params.n_rounds() {
            self.parse_round::<EF>(
                r,
                fs_verifier,
                &mut domain_size,
                &mut rounds,
                &mut exp_domain_gen,
                &mut prev_root,
                &mut final_folding_randomness,
                &mut domain_gen,
            )?;
        }

        let final_coefficients =
            fs_verifier.next_scalars(1 << self.params.final_sumcheck_rounds)?;
        let final_coefficients = CoefficientListHost::new(final_coefficients);

        // Final queries verify
        let stir_challenges_indexes = get_challenge_stir_queries(
            domain_size,
            self.params.folding_factor.at_round(self.params.n_rounds()),
            self.params.final_queries,
            fs_verifier,
        );

        let final_randomness_points = stir_challenges_indexes
            .iter()
            .map(|index| exp_domain_gen.exp_u64(*index as u64))
            .collect();

        let final_randomness_answers = if self.params.n_rounds() == 0 {
            self.merkle_verified_answers::<F>(
                self.params.n_rounds(),
                fs_verifier,
                domain_size,
                &prev_root,
                &stir_challenges_indexes,
            )?
            .into_iter()
            .map(|v| v.into_iter().map(EF::from).collect())
            .collect()
        } else {
            self.merkle_verified_answers::<EF>(
                self.params.n_rounds(),
                fs_verifier,
                domain_size,
                &prev_root,
                &stir_challenges_indexes,
            )?
        };

        fs_verifier.challenge_pow(self.params.final_pow_bits)?;

        let mut final_sumcheck_rounds = Vec::with_capacity(self.params.final_sumcheck_rounds);
        for _ in 0..self.params.final_sumcheck_rounds {
            let sumcheck_poly = UnivariatePolynomial::new(fs_verifier.next_scalars::<EF>(3)?);
            let folding_randomness_single = fs_verifier.challenge_scalars::<EF>(1)[0];
            final_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            fs_verifier.challenge_pow(self.params.final_folding_pow_bits)?;
        }
        let final_sumcheck_randomness = final_sumcheck_rounds.iter().map(|&(_, r)| r).collect();

        Ok(ParsedProof {
            initial_combination_randomness,
            initial_sumcheck_rounds,
            rounds,
            final_folding_randomness,
            final_randomness_points,
            final_randomness_answers,
            final_sumcheck_rounds,
            final_sumcheck_randomness,
            final_coefficients,
        })
    }

    fn compute_v_poly(
        &self,
        parsed_commitment: &ParsedCommitment<EF, KeccakDigest>,
        statement: &Statement<EF>,
        proof: &ParsedProof<EF>,
    ) -> EF {
        let mut num_variables = self.params.num_variables;

        let mut folding_randomness = proof
            .rounds
            .iter()
            .map(|r| &r.folding_randomness)
            .chain(iter::once(&proof.final_folding_randomness))
            .chain(iter::once(&proof.final_sumcheck_randomness))
            .flatten()
            .copied()
            .collect::<Vec<EF>>();

        let mut value = parsed_commitment
            .ood_points
            .iter()
            .map(|ood_point| multilinear_point_from_univariate(*ood_point, num_variables))
            .chain(statement.points.clone())
            .zip(&proof.initial_combination_randomness)
            .map(|(point, randomness): (Vec<EF>, _)| {
                *randomness * eq_extension(&point, &folding_randomness)
            })
            .sum();

        for (round, round_proof) in proof.rounds.iter().enumerate() {
            num_variables -= self.params.folding_factor.at_round(round);
            folding_randomness =
                folding_randomness[folding_randomness.len() - num_variables..].to_vec();

            let ood_points = &round_proof.ood_points;
            let stir_challenges_points = &round_proof.stir_challenges_points;
            let stir_challenges: Vec<Vec<EF>> = ood_points
                .iter()
                .chain(stir_challenges_points)
                .cloned()
                .map(|univariate| {
                    multilinear_point_from_univariate(univariate, num_variables)
                    // TODO:
                    // Maybe refactor outside
                })
                .collect();

            let sum_of_claims: EF = stir_challenges
                .into_iter()
                .map(|point: Vec<EF>| eq_extension(&point, &folding_randomness))
                .zip(&round_proof.combination_randomness)
                .map(|(point, rand)| point * *rand)
                .sum();

            value += sum_of_claims;
        }

        value
    }

    fn compute_folds_helped(&self, parsed: &ParsedProof<EF>) -> Vec<Vec<EF>> {
        let mut result = Vec::new();

        for round in &parsed.rounds {
            let evaluations: Vec<_> = round
                .stir_challenges_answers
                .iter()
                .map(|answers| {
                    let mut folding_randomness_rev = round.folding_randomness.clone();
                    folding_randomness_rev.reverse();
                    CoefficientListHost::new(answers.to_vec()).evaluate(&folding_randomness_rev)
                })
                .collect();
            result.push(evaluations);
        }

        // Final round
        let evaluations: Vec<_> = parsed
            .final_randomness_answers
            .iter()
            .map(|answers| {
                let mut folding_randomness_rev = parsed.final_folding_randomness.clone();
                folding_randomness_rev.reverse();
                CoefficientListHost::new(answers.to_vec()).evaluate(&folding_randomness_rev)
            })
            .collect();
        result.push(evaluations);

        result
    }

    #[instrument(name = "whir: verify", skip_all)]
    pub fn verify(
        &self,
        fs_verifier: &mut FsVerifier,
        parsed_commitment: &ParsedCommitment<EF, KeccakDigest>,
        statement: Statement<EF>,
    ) -> Result<(), WhirError> {
        // We first do a pass in which we rederive all the FS challenges
        // Then we will check the algebraic part (so to optimise inversions)
        let parsed = self.parse_proof(fs_verifier, parsed_commitment, &statement)?;

        let computed_folds = self.compute_folds_helped(&parsed);

        let mut prev: Option<(UnivariatePolynomial<EF>, EF)> = None;
        if let Some(round) = parsed.initial_sumcheck_rounds.first() {
            // Check the first polynomial
            let (mut prev_poly, mut randomness) = round.clone();
            if prev_poly.sum_over_hypercube()
                != parsed_commitment
                    .ood_answers
                    .iter()
                    .copied()
                    .map(EF::from)
                    .chain(statement.evaluations.clone())
                    .zip(&parsed.initial_combination_randomness)
                    .map(|(ans, rand)| ans * *rand)
                    .sum()
            {
                return Err(WhirError::SumMismatch);
            }

            // Check the rest of the rounds
            for (sumcheck_poly, new_randomness) in &parsed.initial_sumcheck_rounds[1..] {
                if sumcheck_poly.sum_over_hypercube() != prev_poly.eval(&randomness) {
                    return Err(WhirError::SumMismatch);
                }
                prev_poly = sumcheck_poly.clone();
                randomness = *new_randomness;
            }

            prev = Some((prev_poly, randomness));
        }

        for (round, folds) in parsed.rounds.iter().zip(&computed_folds) {
            let (sumcheck_poly, new_randomness) = &round.sumcheck_rounds[0].clone();

            let values = round.ood_answers.iter().copied().chain(folds.clone());

            let prev_eval = if let Some((prev_poly, randomness)) = prev {
                prev_poly.eval(&randomness)
            } else {
                EF::ZERO
            };
            let claimed_sum = prev_eval
                + values
                    .zip(&round.combination_randomness)
                    .map(|(val, rand)| val * *rand)
                    .sum::<EF>();

            if sumcheck_poly.sum_over_hypercube() != claimed_sum {
                return Err(WhirError::SumMismatch);
            }

            prev = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &round.sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev.unwrap();
                if sumcheck_poly.sum_over_hypercube() != prev_poly.eval(&randomness) {
                    return Err(WhirError::SumMismatch);
                }
                prev = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }

        // Check the foldings computed from the proof match the evaluations of the polynomial
        let final_folds = &computed_folds[computed_folds.len() - 1];
        let final_coefficients_univariate =
            UnivariatePolynomial::new(parsed.final_coefficients.reverse_vars().coeffs);
        let mut final_evaluations = Vec::new();
        for point in &parsed.final_randomness_points {
            // interpret the coeffs as a univariate polynomial
            final_evaluations.push(final_coefficients_univariate.eval(point));
        }

        if !final_folds
            .iter()
            .zip(final_evaluations)
            .all(|(&fold, eval)| fold == eval)
        {
            return Err(WhirError::FoldingMismatch);
        }

        // Check the final sumchecks
        if self.params.final_sumcheck_rounds > 0 {
            let prev_sumcheck_poly_eval = if let Some((prev_poly, randomness)) = prev {
                prev_poly.eval(&randomness)
            } else {
                EF::ZERO
            };
            let (sumcheck_poly, new_randomness) = &parsed.final_sumcheck_rounds[0].clone();
            let claimed_sum = prev_sumcheck_poly_eval;

            if sumcheck_poly.sum_over_hypercube() != claimed_sum {
                return Err(WhirError::SumMismatch);
            }

            prev = Some((sumcheck_poly.clone(), *new_randomness));

            // Check the rest of the round
            for (sumcheck_poly, new_randomness) in &parsed.final_sumcheck_rounds[1..] {
                let (prev_poly, randomness) = prev.unwrap();
                if sumcheck_poly.sum_over_hypercube() != prev_poly.eval(&randomness) {
                    return Err(WhirError::SumMismatch);
                }
                prev = Some((sumcheck_poly.clone(), *new_randomness));
            }
        }
        let prev_sumcheck_poly_eval = if let Some((prev_poly, randomness)) = prev {
            prev_poly.eval(&randomness)
        } else {
            EF::ZERO
        };

        // Check the final sumcheck evaluation
        let evaluation_of_v_poly = self.compute_v_poly(&parsed_commitment, &statement, &parsed);
        assert_eq!(
            prev_sumcheck_poly_eval,
            evaluation_of_v_poly
                * parsed
                    .final_coefficients
                    .evaluate(&parsed.final_sumcheck_randomness)
        );

        Ok(())
    }
}
