use crate::fs_utils::get_challenge_stir_queries;

use super::{committer::Witness, parameters::WhirConfig};
use algebra::pols::{CoefficientList, CoefficientListHost, Multilinear, MultilinearsVec};
use arithmetic_circuit::TransparentPolynomial;
use cuda_engine::{HostOrDeviceBuffer, cuda_sync};
use fiat_shamir::FsProver;
use merkle_tree::MerkleTree;
use p3_field::{ExtensionField, PrimeCharacteristicRing};
use p3_field::{Field, TwoAdicField};
use sumcheck::SumcheckGrinding;
use tracing::instrument;
use utils::multilinear_point_from_univariate;
use utils::{MyExtensionField, Statement, dot_product_1, dot_product_2, log2_up, powers};

pub struct Prover<F, RCF>(pub WhirConfig<F, RCF>);

impl<F: TwoAdicField + Ord, RCF: Field> Prover<F, RCF>
where
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield> + MyExtensionField<RCF>,
    RCF: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    F::PrimeSubfield: TwoAdicField,
{
    fn validate_parameters(&self) -> bool {
        self.0.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<F>) -> bool {
        if statement.points.len() != statement.evaluations.len() {
            return false;
        }
        if !statement
            .points
            .iter()
            .all(|point| point.len() == self.0.num_variables)
        {
            return false;
        }
        true
    }

    fn validate_witness(&self, witness: &Witness<F>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        witness.polynomial.n_vars() == self.0.num_variables
    }

    #[instrument(name = "whir: prove", skip_all)]
    pub fn prove(&self, fs_prover: &mut FsProver, statement: Statement<F>, witness: Witness<F>) {
        assert!(self.validate_parameters());
        debug_assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));

        let n_initial_claims = witness.ood_points.len() + statement.points.len();

        let sumcheck_mles;
        let hypercube_sum;
        let folding_randomness;
        {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            fs_prover.challenge_pow(
                self.0
                    .security_level
                    .saturating_sub(RCF::bits() - log2_up(n_initial_claims)),
            );
            let combination_randomness_gen = fs_prover.challenge_scalars::<RCF>(1)[0];
            let combination_randomness = powers(combination_randomness_gen, n_initial_claims);

            let liner_comb = randomized_eq_extensions::<F, RCF>(
                &witness.ood_points,
                &statement.points,
                &combination_randomness,
                self.0.cuda,
            );

            let nodes = vec![&witness.lagrange_polynomial, &liner_comb];
            let initial_folding_factor = Some(self.0.folding_factor.at_round(0));
            let pow_bits = self.0.starting_folding_pow_bits;
            let sum = dot_product_2(
                &combination_randomness,
                (&witness.ood_answers, &statement.evaluations),
            );
            (folding_randomness, sumcheck_mles, hypercube_sum) =
                sumcheck::prove::<F::PrimeSubfield, _, _, _>(
                    1,
                    &nodes,
                    &[
                        (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                            .fix_computation(false),
                    ],
                    &[F::ONE],
                    None,
                    false,
                    fs_prover,
                    sum,
                    initial_folding_factor,
                    SumcheckGrinding::Custom(pow_bits),
                    None,
                );
        }
        let round_state = RoundState::<F> {
            round: 0,
            sumcheck_mles,
            hypercube_sum,
            domain_size: 1 << (self.0.num_variables + self.0.starting_log_inv_rate),
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
        };

        self.round(fs_prover, round_state);
    }

    fn round(&self, fs_prover: &mut FsProver, mut round_state: RoundState<F>) {
        // Fold the coefficients
        let _fold_span = tracing::info_span!("whir folding").entered();
        let folded_coefficients = round_state
            .coefficients
            .whir_fold(&round_state.folding_randomness);
        cuda_sync();
        std::mem::drop(_fold_span);

        let num_variables =
            self.0.num_variables - self.0.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.n_vars());

        // Base case
        if round_state.round == self.0.n_rounds() {
            // Directly send coefficients of the polynomial to the verifier.
            fs_prover.add_scalars(&folded_coefficients.transfer_to_host().coeffs);

            // Final verifier queries and answers. The indices are over the
            // *folded* domain.
            let final_challenge_indexes = get_challenge_stir_queries(
                round_state.domain_size, // The size of the *original* domain before folding
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
                    round_state
                        .prev_merkle_answers
                        .slice(i * fold_size..(i + 1) * fold_size)
                })
                .collect::<Vec<_>>();
            cuda_sync();
            fs_prover.add_variable_bytes(&merkle_proof.to_bytes());
            fs_prover.add_scalar_matrix(&answers, false);

            fs_prover.challenge_pow(self.0.final_pow_bits);

            // Final sumcheck
            if self.0.final_sumcheck_rounds > 0 {
                let n_rounds = Some(self.0.final_sumcheck_rounds);
                let pow_bits = self.0.final_folding_pow_bits;
                (_, round_state.sumcheck_mles, _) = sumcheck::prove::<F::PrimeSubfield, _, _, _>(
                    1,
                    &round_state.sumcheck_mles,
                    &[
                        (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                            .fix_computation(false),
                    ],
                    &[F::ONE],
                    None,
                    false,
                    fs_prover,
                    round_state.hypercube_sum,
                    n_rounds,
                    SumcheckGrinding::Custom(pow_bits),
                    None,
                );
            }

            return;
        }

        let round_params = &self.0.round_parameters[round_state.round];

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let expansion = round_state.domain_size / (2 * folded_coefficients.n_coefs());

        let folded_evals = folded_coefficients.expand_from_coeff_and_restructure(
            expansion,
            self.0.folding_factor.at_round(round_state.round + 1),
        );

        let merkle_tree = MerkleTree::new(
            &folded_evals,
            1 << self.0.folding_factor.at_round(round_state.round + 1),
        );

        fs_prover.add_bytes(&merkle_tree.root().0);

        // OOD Samples
        let mut ood_points = vec![];
        let mut ood_answers = vec![];
        if round_params.ood_samples > 0 {
            ood_points = (0..round_params.ood_samples)
                .map(|_| fs_prover.challenge_scalars::<F::PrimeSubfield>(num_variables))
                .collect::<Vec<_>>();
            ood_answers = ood_points
                .iter()
                .map(|ood_point| round_state.sumcheck_mles[0].evaluate_in_small_field(ood_point))
                .collect::<Vec<_>>();
            cuda_sync();
            fs_prover.add_scalars(&ood_answers);
        }

        // STIR queries
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain_size, // Current domain size *before* folding
            self.0.folding_factor.at_round(round_state.round), // Current fold factor
            round_params.num_queries,
            fs_prover,
        );
        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = F::PrimeSubfield::two_adic_generator(
            round_state.domain_size.trailing_zeros() as usize
                - self.0.folding_factor.at_round(round_state.round),
        );
        let stir_challenges_base: Vec<Vec<F::PrimeSubfield>> = stir_challenges_indexes
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
        let fold_size = 1 << self.0.folding_factor.at_round(round_state.round);
        let answers: Vec<_> = stir_challenges_indexes
            .iter()
            .map(|i| {
                round_state
                    .prev_merkle_answers
                    .slice(i * fold_size..(i + 1) * fold_size)
            })
            .collect();
        cuda_sync();
        // Evaluate answers in the folding randomness.
        let mut stir_evaluations = ood_answers.clone();
        stir_evaluations.extend(answers.iter().map(|answers| {
            // The oracle values have been linearly
            // transformed such that they are exactly the coefficients of the
            // multilinear polynomial whose evaluation at the folding randomness
            // is just the folding of f evaluated at the folded point.
            let mut folding_randomness_rev = round_state.folding_randomness.clone();
            folding_randomness_rev.reverse();
            CoefficientListHost::new(answers.to_vec()).evaluate(&folding_randomness_rev)
        }));

        fs_prover.add_variable_bytes(&merkle_proof.to_bytes());
        fs_prover.add_scalar_matrix(&answers, false);

        fs_prover.challenge_pow(round_params.pow_bits);

        // Randomness for combination
        let n_claims = round_params.ood_samples + stir_challenges_base.len();
        fs_prover.challenge_pow(
            self.0
                .security_level
                .saturating_sub(RCF::bits() - log2_up(n_claims)),
        );
        let combination_randomness_gen = fs_prover.challenge_scalars::<RCF>(1)[0];
        let combination_randomness = powers(combination_randomness_gen, n_claims);

        round_state.sumcheck_mles[1].add_assign::<F>(&randomized_eq_extensions::<F, RCF>(
            &[ood_points.clone(), stir_challenges_base.clone()].concat(),
            &[],
            &combination_randomness,
            self.0.cuda,
        ));
        let folding_randomness;
        let hypercube_sum;
        let sumcheck_mles;
        (folding_randomness, sumcheck_mles, hypercube_sum) =
            sumcheck::prove::<F::PrimeSubfield, _, _, _>(
                1,
                &round_state.sumcheck_mles,
                &[
                    (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                        .fix_computation(true),
                ],
                &[F::ONE],
                None,
                false,
                fs_prover,
                round_state.hypercube_sum
                    + dot_product_1(&combination_randomness, &stir_evaluations),
                Some(self.0.folding_factor.at_round(round_state.round + 1)),
                SumcheckGrinding::Custom(round_params.folding_pow_bits),
                None,
            );

        let round_state = RoundState {
            round: round_state.round + 1,
            domain_size: round_state.domain_size / 2,
            sumcheck_mles,
            hypercube_sum,
            folding_randomness,
            coefficients: folded_coefficients, // TODO: Is this redundant with `sumcheck_prover.coeff` ?
            prev_merkle: merkle_tree,
            prev_merkle_answers: folded_evals,
        };

        self.round(fs_prover, round_state);
    }
}

struct RoundState<F: Field> {
    round: usize,
    sumcheck_mles: Vec<Multilinear<F>>,
    hypercube_sum: F,
    domain_size: usize,
    folding_randomness: Vec<F>,
    coefficients: CoefficientList<F>,
    prev_merkle: MerkleTree<F>,
    prev_merkle_answers: HostOrDeviceBuffer<F>,
}

#[instrument(name = "randomized_eq_extensions", skip_all)]
fn randomized_eq_extensions<
    F: Field + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield> + MyExtensionField<RCF>,
    RCF: ExtensionField<F::PrimeSubfield>,
>(
    eq_points_base: &[Vec<F::PrimeSubfield>],
    eq_points_ext: &[Vec<F>],
    randomized_coefs: &[RCF],
    cuda: bool,
) -> Multilinear<F> {
    assert_eq!(
        eq_points_ext.len() + eq_points_base.len(),
        randomized_coefs.len()
    );
    let n_vars = eq_points_base[0].len();
    assert!(eq_points_ext.iter().all(|point| point.len() == n_vars));
    assert!(eq_points_base.iter().all(|point| point.len() == n_vars));

    // TODO to limit GPU memory usage, use intermediate sums

    let mut all_eq_mles_base = Vec::new();
    for point in eq_points_base {
        all_eq_mles_base.push(Multilinear::eq_mle(point, cuda));
    }

    let mut all_eq_mles_ext = Vec::new();
    for point in eq_points_ext {
        all_eq_mles_ext.push(Multilinear::eq_mle(point, cuda));
    }

    let mut res = if eq_points_ext.is_empty() {
        Multilinear::zero(n_vars, cuda)
    } else {
        MultilinearsVec::<F>::from(all_eq_mles_ext)
            .as_ref()
            .linear_comb_in_small_field(&randomized_coefs[eq_points_base.len()..])
    };

    let res_base = MultilinearsVec::from(all_eq_mles_base)
        .as_ref()
        .linear_comb_in_large_field(&randomized_coefs[..eq_points_base.len()]);

    res.add_assign::<RCF>(&res_base);

    cuda_sync();
    res
}
