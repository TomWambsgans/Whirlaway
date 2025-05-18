use crate::fs_utils::get_challenge_stir_queries;
use algebra::pols::{CoefficientListHost, UnivariatePolynomial};
use fiat_shamir::{FsError, FsVerifier};
use merkle_tree::MultiPath;
use p3_field::PrimeCharacteristicRing;
use p3_field::{ExtensionField, Field, TwoAdicField};
use std::iter;
use tracing::instrument;
use utils::{Evaluation, KeccakDigest, embed_vec_vec, powers};
use utils::{eq_extension, multilinear_point_from_univariate};

use super::parameters::WhirConfig;

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
pub struct ParsedCommitment<EF: Field> {
    root: KeccakDigest,
    ood_points: Vec<Vec<EF>>,
    ood_answers: Vec<EF>,
}

#[derive(Clone)]
struct ParsedProof<EF: Field> {
    initial_combination_randomness: Vec<EF>,
    initial_sumcheck_rounds: Vec<(UnivariatePolynomial<EF>, EF)>,
    rounds: Vec<ParsedRound<EF>>,
    final_randomness_points: Vec<EF>,
    final_randomness_answers: Vec<Vec<EF>>,
    final_folding_randomness: Vec<EF>,
    final_sumcheck_rounds: Vec<(UnivariatePolynomial<EF>, EF)>,
    final_sumcheck_randomness: Vec<EF>,
    final_coefficients: CoefficientListHost<EF>,
}

#[derive(Debug, Clone, Hash)]
struct ParsedRound<EF: Field> {
    folding_randomness: Vec<EF>,
    ood_points: Vec<Vec<EF>>, // TODO there is an embedding for rounds > 0
    ood_answers: Vec<EF>,
    stir_challenges_points: Vec<EF>,
    stir_challenges_answers: Vec<Vec<EF>>, // TODO there is an embedding for round == 0
    combination_randomness: Vec<EF>,
    sumcheck_rounds: Vec<(UnivariatePolynomial<EF>, EF)>,
}

impl<F: TwoAdicField, EF: ExtensionField<F> + TwoAdicField> WhirConfig<F, EF>
where
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
{
    pub fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<ParsedCommitment<EF>, WhirError> {
        let root = KeccakDigest(fs_verifier.next_bytes(32)?.try_into().unwrap()); // TODO avoid harcoding 32

        let mut ood_points = vec![];
        let mut ood_answers = vec![];
        if self.committment_ood_samples > 0 {
            ood_points = (0..self.committment_ood_samples)
                .map(|_| fs_verifier.challenge_scalars::<EF>(self.num_variables))
                .collect::<Vec<_>>();
            ood_answers = fs_verifier.next_scalars::<EF>(self.committment_ood_samples)?;
        }

        Ok(ParsedCommitment {
            root,
            ood_points,
            ood_answers,
        })
    }

    fn merkle_verified_answers<IF: Field>(
        &self,
        r: usize,
        fs_verifier: &mut FsVerifier,
        domain_size: usize,
        prev_root: &KeccakDigest,
        stir_challenges_indexes: &[usize],
    ) -> Result<Vec<Vec<IF>>, WhirError> {
        let merkle_proof = MultiPath::from_bytes(&fs_verifier.next_variable_bytes()?)
            .ok_or(WhirError::Decoding)?;

        let answers: Vec<Vec<IF>> = fs_verifier.next_scalar_matrix(None)?;

        let merkle_tree_height =
            domain_size.trailing_zeros() as usize - self.folding_factor.at_round(r);
        if !merkle_proof.verify(&prev_root, &answers, merkle_tree_height)
            || merkle_proof.leaf_indexes != stir_challenges_indexes
        {
            return Err(WhirError::MerkleTree);
        }

        Ok(answers)
    }

    fn parse_round<CoeffField: ExtensionField<F>>(
        &self,
        r: usize,
        fs: &mut FsVerifier,
        domain_size: &mut usize,
        rounds: &mut Vec<ParsedRound<EF>>,
        exp_domain_gen: &mut EF,
        prev_root: &mut KeccakDigest,
        folding_randomness: &mut Vec<EF>,
        domain_gen: &mut EF,
        num_variables_at_round: usize,
    ) -> Result<(), WhirError>
    where
        EF: ExtensionField<CoeffField>,
    {
        let round_params = &self.round_parameters[r];

        let new_root = KeccakDigest(fs.next_bytes(32)?.try_into().unwrap());
        let mut ood_points = vec![];
        let mut ood_answers = vec![];
        if round_params.ood_samples > 0 {
            ood_points = (0..round_params.ood_samples)
                .map(|_| fs.challenge_scalars::<F>(num_variables_at_round))
                .collect::<Vec<_>>();
            ood_answers = fs.next_scalars::<EF>(round_params.ood_samples)?;
        }

        let stir_challenges_indexes = get_challenge_stir_queries(
            *domain_size,
            self.folding_factor.at_round(r),
            round_params.num_queries,
            fs,
        );

        let stir_challenges_points = stir_challenges_indexes
            .iter()
            .map(|index| exp_domain_gen.exp_u64(*index as u64))
            .collect::<Vec<EF>>();

        let stir_challenges_answers = self.merkle_verified_answers::<CoeffField>(
            r,
            fs,
            *domain_size,
            prev_root,
            &stir_challenges_indexes,
        )?;
        fs.challenge_pow(round_params.pow_bits)?;

        let n_claims = stir_challenges_indexes.len() + round_params.ood_samples;

        let combination_randomness_gen = fs.challenge_scalars::<EF>(1)[0];
        let combination_randomness = powers(combination_randomness_gen, n_claims);

        let mut sumcheck_rounds = Vec::with_capacity(self.folding_factor.at_round(r + 1));
        for _ in 0..self.folding_factor.at_round(r + 1) {
            let sumcheck_poly = UnivariatePolynomial::new(fs.next_scalars(3)?);
            let folding_randomness_single = fs.challenge_scalars::<EF>(1)[0];
            sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            fs.challenge_pow(round_params.folding_pow_bits)?;
        }

        let new_folding_randomness = sumcheck_rounds.iter().map(|&(_, r)| r).collect::<Vec<_>>();

        rounds.push(ParsedRound {
            folding_randomness: std::mem::take(folding_randomness),
            ood_points: embed_vec_vec(&ood_points),
            ood_answers,
            stir_challenges_points,
            stir_challenges_answers: embed_vec_vec(&stir_challenges_answers),
            combination_randomness,
            sumcheck_rounds,
        });

        *folding_randomness = new_folding_randomness;

        let domain_reduction_factor = if r == 0 {
            self.innitial_domain_reduction_factor
        } else {
            1
        };

        *prev_root = new_root;
        *domain_gen = domain_gen.exp_u64(1 << domain_reduction_factor);
        *exp_domain_gen = domain_gen.exp_u64(1 << self.folding_factor.at_round(r + 1));

        *domain_size >>= domain_reduction_factor;
        Ok(())
    }

    fn parse_proof(
        &self,
        fs: &mut FsVerifier,
        parsed_commitment: &ParsedCommitment<EF>,
        statement: &Vec<Evaluation<EF>>, // Will be needed later
    ) -> Result<ParsedProof<EF>, WhirError> {
        let mut initial_sumcheck_rounds = Vec::new();
        let mut final_folding_randomness: Vec<EF>;
        let initial_combination_randomness;
        // Derive combination randomness and first sumcheck polynomial
        let n_initial_claims = parsed_commitment.ood_points.len() + statement.len();

        let combination_randomness_gen = fs.challenge_scalars::<EF>(1)[0];
        initial_combination_randomness = powers(combination_randomness_gen, n_initial_claims);

        // Initial sumcheck
        initial_sumcheck_rounds.reserve_exact(self.folding_factor.at_round(0));
        for _ in 0..self.folding_factor.at_round(0) {
            let sumcheck_poly = UnivariatePolynomial::new(fs.next_scalars::<EF>(3)?);
            let folding_randomness_single = fs.challenge_scalars::<EF>(1)[0];
            initial_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            fs.challenge_pow(self.starting_folding_pow_bits)?;
        }

        final_folding_randomness = initial_sumcheck_rounds.iter().map(|&(_, r)| r).collect();

        let mut prev_root = parsed_commitment.root.clone();
        let mut domain_gen =
            EF::two_adic_generator(self.num_variables + self.starting_log_inv_rate);
        let mut exp_domain_gen = domain_gen.exp_u64(1 << self.folding_factor.at_round(0));
        let mut domain_size = 1 << (self.num_variables + self.starting_log_inv_rate);
        let mut rounds = vec![];

        let mut num_variables_at_round = self.num_variables;

        num_variables_at_round -= self.folding_factor.at_round(0);
        if self.n_rounds() > 0 {
            self.parse_round::<F>(
                0,
                fs,
                &mut domain_size,
                &mut rounds,
                &mut exp_domain_gen,
                &mut prev_root,
                &mut final_folding_randomness,
                &mut domain_gen,
                num_variables_at_round,
            )?;
        }

        for r in 1..self.n_rounds() {
            num_variables_at_round -= self.folding_factor.at_round(r);
            self.parse_round::<EF>(
                r,
                fs,
                &mut domain_size,
                &mut rounds,
                &mut exp_domain_gen,
                &mut prev_root,
                &mut final_folding_randomness,
                &mut domain_gen,
                num_variables_at_round,
            )?;
        }

        let final_coefficients = fs.next_scalars(1 << self.final_sumcheck_rounds)?;
        let final_coefficients = CoefficientListHost::new(final_coefficients);

        // Final queries verify
        let stir_challenges_indexes = get_challenge_stir_queries(
            domain_size,
            self.folding_factor.at_round(self.n_rounds()),
            self.final_queries,
            fs,
        );

        let final_randomness_points = stir_challenges_indexes
            .iter()
            .map(|index| exp_domain_gen.exp_u64(*index as u64))
            .collect::<Vec<EF>>();

        let final_randomness_answers = if self.n_rounds() == 0 {
            embed_vec_vec(&self.merkle_verified_answers::<F>(
                self.n_rounds(),
                fs,
                domain_size,
                &prev_root,
                &stir_challenges_indexes,
            )?)
        } else {
            self.merkle_verified_answers::<EF>(
                self.n_rounds(),
                fs,
                domain_size,
                &prev_root,
                &stir_challenges_indexes,
            )?
        };

        fs.challenge_pow(self.final_pow_bits)?;

        let mut final_sumcheck_rounds = Vec::with_capacity(self.final_sumcheck_rounds);
        for _ in 0..self.final_sumcheck_rounds {
            let sumcheck_poly = UnivariatePolynomial::new(fs.next_scalars::<EF>(3)?);
            let folding_randomness_single = fs.challenge_scalars::<EF>(1)[0];
            final_sumcheck_rounds.push((sumcheck_poly, folding_randomness_single));

            fs.challenge_pow(self.final_folding_pow_bits)?;
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
        parsed_commitment: &ParsedCommitment<EF>,
        statement: &Vec<Evaluation<EF>>,
        proof: &ParsedProof<EF>,
    ) -> EF {
        let mut num_variables = self.num_variables;

        let statement_points = statement
            .iter()
            .map(|eval| eval.point.clone())
            .collect::<Vec<_>>();

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
            .cloned()
            .chain(statement_points.clone())
            .zip(&proof.initial_combination_randomness)
            .map(|(point, randomness)| eq_extension(&point, &folding_randomness) * *randomness)
            .sum::<EF>();

        for (round, round_proof) in proof.rounds.iter().enumerate() {
            num_variables -= self.folding_factor.at_round(round);
            folding_randomness =
                folding_randomness[folding_randomness.len() - num_variables..].to_vec();

            let ood_points = &round_proof.ood_points;
            let stir_challenges_points = &round_proof.stir_challenges_points;
            let stir_challenges: Vec<Vec<EF>> = ood_points
                .iter()
                .cloned()
                .chain(stir_challenges_points.iter().map(|univariate| {
                    multilinear_point_from_univariate(EF::from(*univariate), num_variables)
                }))
                .collect();

            let sum_of_claims = stir_challenges
                .into_iter()
                .map(|point| eq_extension(&point, &folding_randomness))
                .zip(&round_proof.combination_randomness)
                .map(|(point, rand)| point * *rand)
                .sum::<EF>();

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
        parsed_commitment: &ParsedCommitment<EF>,
        statement: Vec<Evaluation<EF>>,
        fs: &mut FsVerifier,
    ) -> Result<(), WhirError> {
        // We first do a pass in which we rederive all the FS challenges
        // Then we will check the algebraic part (so to optimise inversions)
        let parsed = self.parse_proof(fs, parsed_commitment, &statement)?;

        let statement_evals = statement.iter().map(|eval| eval.value).collect::<Vec<_>>();

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
                    .chain(statement_evals.clone())
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
            UnivariatePolynomial::<EF>::new(parsed.final_coefficients.reverse_vars().coeffs);
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
        if self.final_sumcheck_rounds > 0 {
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
        if prev_sumcheck_poly_eval
            != evaluation_of_v_poly
                * parsed
                    .final_coefficients
                    .evaluate(&parsed.final_sumcheck_randomness)
        {
            return Err(WhirError::SumMismatch);
        }

        Ok(())
    }
}
