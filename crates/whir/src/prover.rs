use crate::fs_utils::get_challenge_stir_queries;

use super::{committer::Witness, parameters::WhirConfig};
use algebra::pols::{CoefficientList, Multilinear};
use arithmetic_circuit::TransparentPolynomial;
use fiat_shamir::FsProver;
use merkle_tree::MerkleTree;
use p3_field::{ExtensionField, PrimeCharacteristicRing};
use p3_field::{Field, TwoAdicField};
use rand::distr::{Distribution, StandardUniform};
use sumcheck::SumcheckGrinding;
use tracing::instrument;
use utils::{Evaluation, multilinear_point_from_univariate};
use utils::{dot_product_1, dot_product_2, powers};

impl<F: TwoAdicField + Ord, EF: ExtensionField<F>> WhirConfig<F, EF>
where
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    EF: ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield> + TwoAdicField + Ord,
    F::PrimeSubfield: TwoAdicField,
    StandardUniform: Distribution<EF>,
    StandardUniform: Distribution<F>,
{
    fn validate_parameters(&self) -> bool {
        self.num_variables
            == self.folding_factor.total_number(self.n_rounds()) + self.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Vec<Evaluation<EF>>) -> bool {
        if !statement
            .iter()
            .all(|point| point.point.len() == self.num_variables)
        {
            return false;
        }
        true
    }

    fn validate_witness(&self, witness: &Witness<F, EF>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        witness.polynomial.n_vars == self.num_variables
    }

    #[instrument(name = "whir: prove", skip_all)]
    pub fn open(&self, witness: Witness<F, EF>, statement: Vec<Evaluation<EF>>, fs: &mut FsProver) {
        assert!(self.validate_parameters());
        debug_assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));

        let statement_points = statement
            .iter()
            .map(|eval| eval.point.clone())
            .collect::<Vec<_>>();
        let statement_evals = statement.iter().map(|eval| eval.value).collect::<Vec<_>>();

        let n_initial_claims = witness.ood_points.len() + statement.len();

        let sumcheck_mles;
        let hypercube_sum;
        let folding_randomness;
        {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.

            let combination_randomness_gen = fs.challenge_scalars::<EF>(1)[0];
            let combination_randomness = powers(combination_randomness_gen, n_initial_claims);

            let liner_comb = randomized_eq_extensions::<EF, EF>(
                &witness.ood_points,
                &statement_points,
                &combination_randomness,
            );

            // TODO avoid embedding
            let embedd_lagrange_polynomial = witness.lagrange_polynomial.embed::<EF>();

            let nodes = vec![&embedd_lagrange_polynomial, &liner_comb];
            let initial_folding_factor = Some(self.folding_factor.at_round(0));
            let pow_bits = self.starting_folding_pow_bits;
            let sum = dot_product_2(
                &combination_randomness,
                (&witness.ood_answers, &statement_evals),
            );
            (folding_randomness, sumcheck_mles, hypercube_sum) = sumcheck::prove::<F, EF, EF, _>(
                1,
                &nodes,
                &[
                    (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                        .fix_computation(false),
                ],
                &[EF::ONE],
                None,
                false,
                fs,
                sum,
                initial_folding_factor,
                SumcheckGrinding::Custom(pow_bits),
                None,
            );
        }

        let round_state = RoundState::<F, EF> {
            round: 0,
            sumcheck_mles,
            hypercube_sum,
            domain_size: 1 << (self.num_variables + self.starting_log_inv_rate),
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
        };

        self.round::<F>(fs, round_state);
    }

    fn round<CoeffsField: Field>(
        &self,
        fs: &mut FsProver,
        mut round_state: RoundState<CoeffsField, EF>,
    ) where
        EF: ExtensionField<CoeffsField>,
    {
        // Fold the coefficients
        let _fold_span = tracing::info_span!("whir folding").entered();
        let folded_coefficients = round_state
            .coefficients
            .whir_fold(&round_state.folding_randomness);

        std::mem::drop(_fold_span);

        let num_variables =
            self.num_variables - self.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.n_vars);

        // Base case
        if round_state.round == self.n_rounds() {
            // Directly send coefficients of the polynomial to the verifier.
            fs.add_scalars(&folded_coefficients.coeffs);

            // Final verifier queries and answers. The indices are over the
            // *folded* domain.
            let final_challenge_indexes = get_challenge_stir_queries(
                round_state.domain_size, // The size of the *original* domain before folding
                self.folding_factor.at_round(round_state.round), // The folding factor we used to fold the previous polynomial
                self.final_queries,
                fs,
            );

            let merkle_proof = round_state
                .prev_merkle
                .generate_multi_proof(final_challenge_indexes.clone());
            // Every query requires opening these many in the previous Merkle tree
            let fold_size = 1 << self.folding_factor.at_round(round_state.round);
            let answers = final_challenge_indexes
                .into_iter()
                .map(|i| {
                    round_state.prev_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec()
                })
                .collect::<Vec<_>>();

            fs.add_variable_bytes(&merkle_proof.to_bytes());
            fs.add_scalar_matrix(&answers, false);

            fs.challenge_pow(self.final_pow_bits);

            // Final sumcheck
            if self.final_sumcheck_rounds > 0 {
                let n_rounds = Some(self.final_sumcheck_rounds);
                let pow_bits = self.final_folding_pow_bits;
                (_, round_state.sumcheck_mles, _) = sumcheck::prove::<F, _, _, _>(
                    1,
                    &round_state.sumcheck_mles,
                    &[
                        (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                            .fix_computation(false),
                    ],
                    &[EF::ONE],
                    None,
                    false,
                    fs,
                    round_state.hypercube_sum,
                    n_rounds,
                    SumcheckGrinding::Custom(pow_bits),
                    None,
                );
            }

            return;
        }

        let round_params = &self.round_parameters[round_state.round];

        let domain_reduction_factor = if round_state.round == 0 {
            self.innitial_domain_reduction_factor
        } else {
            1
        };
        let expansion =
            (round_state.domain_size / folded_coefficients.n_coefs()) >> domain_reduction_factor;

        let folded_evals = folded_coefficients.expand_from_coeff_and_restructure(
            expansion,
            self.folding_factor.at_round(round_state.round + 1),
        );

        let merkle_tree = MerkleTree::new(
            &folded_evals,
            1 << self.folding_factor.at_round(round_state.round + 1),
        );

        fs.add_bytes(&merkle_tree.root().0);

        // OOD Samples
        let mut ood_points = vec![];
        let mut ood_answers = vec![];
        if round_params.ood_samples > 0 {
            ood_points = (0..round_params.ood_samples)
                .map(|_| fs.challenge_scalars::<F>(num_variables)) // Now that the coeffs are in EF we can sample OOD in F, result stays in EF
                .collect::<Vec<_>>();
            ood_answers = ood_points
                .iter()
                .map(|ood_point| round_state.sumcheck_mles[0].evaluate_in_small_field(ood_point))
                .collect::<Vec<_>>();

            fs.add_scalars(&ood_answers);
        }

        // STIR queries
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain_size, // Current domain size *before* folding
            self.folding_factor.at_round(round_state.round), // Current fold factor
            round_params.num_queries,
            fs,
        );
        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = F::two_adic_generator(
            round_state.domain_size.trailing_zeros() as usize
                - self.folding_factor.at_round(round_state.round),
        );
        let stir_challenges_base: Vec<Vec<F>> = stir_challenges_indexes
            .iter()
            .map(|i| {
                multilinear_point_from_univariate(
                    domain_scaled_gen.exp_u64(*i as u64),
                    num_variables,
                )
            })
            .collect();

        let merkle_proof = round_state
            .prev_merkle
            .generate_multi_proof(stir_challenges_indexes.clone());
        let fold_size = 1 << self.folding_factor.at_round(round_state.round);
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
            let mut folding_randomness_rev = round_state.folding_randomness.clone();
            folding_randomness_rev.reverse();
            CoefficientList::new(answers.to_vec()).evaluate(&folding_randomness_rev)
        }));

        fs.add_variable_bytes(&merkle_proof.to_bytes());
        fs.add_scalar_matrix(&answers, false);

        fs.challenge_pow(round_params.pow_bits);

        // Randomness for combination
        let n_claims = round_params.ood_samples + stir_challenges_base.len();

        let combination_randomness_gen = fs.challenge_scalars::<EF>(1)[0];
        let combination_randomness = powers(combination_randomness_gen, n_claims);

        round_state.sumcheck_mles[1].add_assign::<EF>(&randomized_eq_extensions::<F, EF>(
            &[ood_points.clone(), stir_challenges_base.clone()].concat(),
            &[],
            &combination_randomness,
        ));
        let folding_randomness;
        let hypercube_sum;
        let sumcheck_mles;
        (folding_randomness, sumcheck_mles, hypercube_sum) = sumcheck::prove::<F, _, _, _>(
            1,
            &round_state.sumcheck_mles,
            &[
                (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                    .fix_computation(true),
            ],
            &[EF::ONE],
            None,
            false,
            fs,
            round_state.hypercube_sum + dot_product_1(&combination_randomness, &stir_evaluations),
            Some(self.folding_factor.at_round(round_state.round + 1)),
            SumcheckGrinding::Custom(round_params.folding_pow_bits),
            None,
        );

        let round_state = RoundState {
            round: round_state.round + 1,
            domain_size: round_state.domain_size >> domain_reduction_factor,
            sumcheck_mles,
            hypercube_sum,
            folding_randomness,
            coefficients: folded_coefficients,
            prev_merkle: merkle_tree,
            prev_merkle_answers: folded_evals,
        };

        self.round::<EF>(fs, round_state);
    }
}

struct RoundState<CoeffsField: Field, EF: ExtensionField<CoeffsField>> {
    round: usize,
    sumcheck_mles: Vec<Multilinear<EF>>,
    hypercube_sum: EF,
    domain_size: usize,
    folding_randomness: Vec<EF>,
    coefficients: CoefficientList<CoeffsField>,
    prev_merkle: MerkleTree<CoeffsField>,
    prev_merkle_answers: Vec<CoeffsField>,
}

#[instrument(name = "randomized_eq_extensions", skip_all)]
fn randomized_eq_extensions<F: Field, EF: ExtensionField<F>>(
    eq_points_base: &[Vec<F>],
    eq_points_ext: &[Vec<EF>],
    randomized_coefs: &[EF],
) -> Multilinear<EF> {
    assert_eq!(
        eq_points_ext.len() + eq_points_base.len(),
        randomized_coefs.len()
    );
    let n_vars = if eq_points_base.len() > 0 {
        eq_points_base[0].len()
    } else {
        eq_points_ext[0].len()
    };
    assert!(eq_points_ext.iter().all(|point| point.len() == n_vars));
    assert!(eq_points_base.iter().all(|point| point.len() == n_vars));

    // TODO to limit memory usage, use intermediate sums

    let mut all_eq_mles_base = Vec::new();
    for point in eq_points_base {
        all_eq_mles_base.push(Multilinear::eq_mle(point));
    }

    let mut all_eq_mles_ext = Vec::new();
    for point in eq_points_ext {
        all_eq_mles_ext.push(Multilinear::eq_mle(point));
    }

    let mut res = if eq_points_ext.is_empty() {
        Multilinear::<EF>::zero(n_vars)
    } else {
        Multilinear::<EF>::linear_comb_in_large_field::<EF, _>(
            &all_eq_mles_ext,
            &randomized_coefs[eq_points_base.len()..],
        )
    };

    if all_eq_mles_base.len() > 0 {
        let res_base = Multilinear::linear_comb_in_large_field(
            &all_eq_mles_base,
            &randomized_coefs[..eq_points_base.len()],
        );

        res.add_assign::<EF>(&res_base);
    }

    res
}
