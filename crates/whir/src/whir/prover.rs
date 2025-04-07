use super::{Statement, committer::Witness, parameters::WhirConfig};
use crate::{
    domain::Domain,
    poly_utils::{coeffs::CoefficientList, fold::restructure_evaluations},
    utils::expand_from_coeff_maybe_with_cuda,
};
use algebra::{
    field_utils::{dot_product, multilinear_point_from_univariate},
    pols::{MultilinearPolynomial, TransparentPolynomial},
    utils::expand_randomness,
};
use fiat_shamir::FsProver;
use merkle_tree::MerkleTree;
use p3_field::{ExtensionField, Field, TwoAdicField};
use tracing::instrument;

use crate::whir::fs_utils::get_challenge_stir_queries;

pub struct Prover<F: TwoAdicField, EF: ExtensionField<F>>(pub WhirConfig<F, EF>);

impl<F: TwoAdicField, EF: ExtensionField<F>> Prover<F, EF> {
    fn validate_parameters(&self) -> bool {
        self.0.mv_parameters.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<EF>) -> bool {
        if statement.points.len() != statement.evaluations.len() {
            return false;
        }
        if !statement
            .points
            .iter()
            .all(|point| point.len() == self.0.mv_parameters.num_variables)
        {
            return false;
        }
        true
    }

    fn validate_witness(&self, witness: &Witness<EF>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        witness.polynomial.num_variables() == self.0.mv_parameters.num_variables
    }

    #[instrument(name = "whir: prove", skip_all)]
    pub fn prove(
        &self,
        fs_prover: &mut FsProver,
        statement: Statement<EF>,
        witness: Witness<EF>,
    ) -> Option<()> {
        assert!(self.validate_parameters());
        assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));

        let initial_claims: Vec<_> = witness
            .ood_points
            .into_iter()
            .map(|ood_point| {
                multilinear_point_from_univariate(ood_point, self.0.mv_parameters.num_variables)
            })
            .chain(statement.points)
            .collect();
        let initial_answers: Vec<_> = witness
            .ood_answers
            .into_iter()
            .chain(statement.evaluations)
            .collect();

        let sumcheck_mles;
        let folding_randomness = {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let combination_randomness_gen = fs_prover.challenge_scalars::<EF>(1)[0];
            let combination_randomness =
                expand_randomness(combination_randomness_gen, initial_claims.len());

            let nodes = vec![
                witness.polynomial.clone().reverse_vars().into_evals(), // TODO: Avoid clone
                randomized_eq_extensions(&initial_claims, &combination_randomness),
            ];
            let n_rounds = Some(self.0.folding_factor.at_round(0));
            let pow_bits = self.0.starting_folding_pow_bits;
            let sum = dot_product(&initial_answers, &combination_randomness);
            let mut folding_randomness;
            (folding_randomness, sumcheck_mles) = sumcheck::prove(
                nodes,
                &[
                    (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                        .fix_computation(false),
                ],
                &[EF::ONE],
                None,
                false,
                fs_prover,
                Some(sum),
                n_rounds,
                pow_bits,
            );
            folding_randomness.reverse();
            folding_randomness
        };

        let round_state = RoundState {
            domain: self.0.starting_domain.clone(),
            round: 0,
            sumcheck_mles,
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
        };

        self.round(fs_prover, round_state)
    }

    fn round(&self, fs_prover: &mut FsProver, mut round_state: RoundState<F, EF>) -> Option<()> {
        // Fold the coefficients
        let folded_coefficients = round_state
            .coefficients
            .fold(&round_state.folding_randomness);

        let num_variables = self.0.mv_parameters.num_variables
            - self.0.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.num_variables());

        // Base case
        if round_state.round == self.0.n_rounds() {
            // Directly send coefficients of the polynomial to the verifier.
            fs_prover.add_scalars(folded_coefficients.coeffs());

            // Final verifier queries and answers. The indices are over the
            // *folded* domain.
            let final_challenge_indexes = get_challenge_stir_queries(
                round_state.domain.size(), // The size of the *original* domain before folding
                self.0.folding_factor.at_round(round_state.round), // The folding factor we used to fold the previous polynomial
                self.0.final_queries,
                fs_prover,
            );

            let merkle_proof = round_state
                .prev_merkle
                .generate_multi_proof(final_challenge_indexes.clone());
            // Every query requires opening these many in the previous Merkle tree
            let fold_size = 1 << self.0.folding_factor.at_round(round_state.round);
            let answers = final_challenge_indexes
                .into_iter()
                .map(|i| {
                    round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec()
                })
                .collect::<Vec<_>>();

            fs_prover.add_variable_bytes(&merkle_proof.to_bytes());
            fs_prover.add_scalar_matrix(&answers, false);

            // PoW
            if self.0.final_pow_bits > 0 {
                fs_prover.challenge_pow(self.0.final_pow_bits);
            }

            // Final sumcheck
            if self.0.final_sumcheck_rounds > 0 {
                let n_rounds = Some(self.0.final_sumcheck_rounds);
                let pow_bits = self.0.final_folding_pow_bits;
                (_, round_state.sumcheck_mles) = sumcheck::prove(
                    round_state.sumcheck_mles,
                    &[
                        (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                            .fix_computation(false),
                    ],
                    &[EF::ONE],
                    None,
                    false,
                    fs_prover,
                    None,
                    n_rounds,
                    pow_bits,
                ); // TODO sum could be known, currently it is recomputed
            }

            return Some(());
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2);
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let evals = expand_from_coeff_maybe_with_cuda::<F, EF>(
            folded_coefficients.coeffs(),
            expansion,
            self.0.cuda,
        );
        let folded_evals = restructure_evaluations::<F, EF>(
            evals,
            new_domain.backing_domain.group_gen_inv(),
            self.0.folding_factor.at_round(round_state.round + 1),
        );

        let merkle_tree = MerkleTree::new(
            &folded_evals,
            1 << self.0.folding_factor.at_round(round_state.round + 1),
        );

        fs_prover.add_bytes(&merkle_tree.root());

        // OOD Samples
        let mut ood_points = vec![EF::ZERO; round_params.ood_samples];
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        if round_params.ood_samples > 0 {
            ood_points = fs_prover.challenge_scalars::<EF>(round_params.ood_samples);
            ood_answers.extend(ood_points.iter().map(|ood_point| {
                folded_coefficients.evaluate(&multilinear_point_from_univariate(
                    *ood_point,
                    num_variables,
                ))
            }));
            fs_prover.add_scalars(&ood_answers);
        }

        // STIR queries
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain.size(), // Current domain size *before* folding
            self.0.folding_factor.at_round(round_state.round), // Current fold factor
            round_params.num_queries,
            fs_prover,
        );
        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.0.folding_factor.at_round(round_state.round));
        let stir_challenges: Vec<_> = ood_points
            .into_iter()
            .chain(
                stir_challenges_indexes
                    .iter()
                    .map(|i| EF::from(domain_scaled_gen.exp_u64(*i as u64))),
            )
            .map(|univariate| multilinear_point_from_univariate(univariate, num_variables))
            .collect();

        let merkle_proof = round_state
            .prev_merkle
            .generate_multi_proof(stir_challenges_indexes.clone());
        let fold_size = 1 << self.0.folding_factor.at_round(round_state.round);
        let answers: Vec<_> = stir_challenges_indexes
            .iter()
            .map(|i| round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec())
            .collect();
        // Evaluate answers in the folding randomness.
        let mut stir_evaluations = ood_answers.clone();
        stir_evaluations.extend(answers.iter().map(|answers| {
            // The oracle values have been linearly
            // transformed such that they are exactly the coefficients of the
            // multilinear polynomial whose evaluation at the folding randomness
            // is just the folding of f evaluated at the folded point.
            CoefficientList::new(answers.to_vec()).evaluate(&round_state.folding_randomness)
        }));

        fs_prover.add_variable_bytes(&merkle_proof.to_bytes());
        fs_prover.add_scalar_matrix(&answers, false);
        // PoW
        if round_params.pow_bits > 0 {
            fs_prover.challenge_pow(round_params.pow_bits);
        }

        // Randomness for combination
        let combination_randomness_gen = fs_prover.challenge_scalars::<EF>(1)[0];
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        round_state.sumcheck_mles[1] +=
            randomized_eq_extensions(&stir_challenges, &combination_randomness).into();

        let mut folding_randomness;
        (folding_randomness, round_state.sumcheck_mles) = sumcheck::prove(
            round_state.sumcheck_mles,
            &[
                (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                    .fix_computation(false),
            ],
            &[EF::ONE],
            None,
            false,
            fs_prover,
            None, // TODO sum could be known, currently it is recomputed
            Some(self.0.folding_factor.at_round(round_state.round + 1)),
            round_params.folding_pow_bits,
        );
        folding_randomness.reverse();

        let round_state = RoundState {
            round: round_state.round + 1,
            domain: new_domain,
            sumcheck_mles: round_state.sumcheck_mles,
            folding_randomness,
            coefficients: folded_coefficients, // TODO: Is this redundant with `sumcheck_prover.coeff` ?
            prev_merkle: merkle_tree,
            prev_merkle_answers: folded_evals,
        };

        self.round(fs_prover, round_state)
    }
}

struct RoundState<F: TwoAdicField, EF: ExtensionField<F>> {
    round: usize,
    domain: Domain<F>,
    sumcheck_mles: Vec<MultilinearPolynomial<EF>>,
    folding_randomness: Vec<EF>,
    coefficients: CoefficientList<EF>,
    prev_merkle: MerkleTree<EF>,
    prev_merkle_answers: Vec<EF>,
}

fn randomized_eq_extensions<F: Field>(
    eq_points: &[Vec<F>],
    randomized_coefs: &[F],
) -> MultilinearPolynomial<F> {
    assert_eq!(eq_points.len(), randomized_coefs.len());
    assert!(
        eq_points
            .iter()
            .all(|point| point.len() == eq_points[0].len())
    );
    let mut res = MultilinearPolynomial::zero(eq_points[0].len());
    for (initial_claim, randomness_coef) in eq_points.iter().zip(randomized_coefs) {
        let mut initial_claim = initial_claim.clone();
        initial_claim.reverse();
        let mut eq_mle = MultilinearPolynomial::eq_mle(&initial_claim);
        eq_mle = eq_mle.scale(*randomness_coef);
        res += eq_mle;
    }
    res
}
