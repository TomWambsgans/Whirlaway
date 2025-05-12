use std::any::TypeId;

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
use utils::{Statement, powers};
use utils::{dot_product, multilinear_point_from_univariate};

pub struct Prover<F: Field, EF: ExtensionField<F>>(pub WhirConfig<F, EF>);

impl<
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield> + TwoAdicField + Ord,
    EF: ExtensionField<F>
        + ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>
        + TwoAdicField
        + Ord,
> Prover<F, EF>
where
    F::PrimeSubfield: TwoAdicField,
{
    fn validate_parameters(&self) -> bool {
        self.0.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &Statement<EF>) -> bool {
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

    fn validate_witness(&self, witness: &Witness<F, EF>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        witness.polynomial.n_vars() == self.0.num_variables
    }

    #[instrument(name = "whir: prove", skip_all)]
    pub fn prove(
        &self,
        fs_prover: &mut FsProver,
        statement: Statement<EF>,
        witness: Witness<F, EF>,
    ) -> Option<()> {
        assert!(self.validate_parameters());
        debug_assert!(self.validate_statement(&statement));
        assert!(self.validate_witness(&witness));

        let initial_claims = witness
            .ood_points
            .into_iter()
            .map(|ood_point| multilinear_point_from_univariate(ood_point, self.0.num_variables))
            .chain(statement.points)
            .collect::<Vec<_>>();
        let initial_answers = witness
            .ood_answers
            .into_iter()
            .chain(statement.evaluations)
            .collect::<Vec<_>>();

        let sumcheck_mles;
        let hypercube_sum;
        let folding_randomness;
        {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let combination_randomness_gen = fs_prover.challenge_scalars::<EF>(1)[0];
            let combination_randomness = powers(combination_randomness_gen, initial_claims.len());

            let liner_comb = randomized_eq_extensions::<F, EF>(
                &[],
                &initial_claims,
                &combination_randomness,
                self.0.cuda,
            );
            let embbedded_lagrange_polynomial = if TypeId::of::<F>() == TypeId::of::<EF>() {
                unsafe { std::mem::transmute::<_, &Multilinear<EF>>(&witness.lagrange_polynomial) }
            } else {
                &witness.lagrange_polynomial.embed::<EF>()
            };

            let nodes = vec![&embbedded_lagrange_polynomial, &liner_comb];
            let initial_folding_factor = Some(self.0.folding_factor.at_round(0));
            let pow_bits = self.0.starting_folding_pow_bits;
            let sum = dot_product(&initial_answers, &combination_randomness);
            (folding_randomness, sumcheck_mles, hypercube_sum) =
                sumcheck::prove::<F::PrimeSubfield, _, _, _>(
                    1,
                    &nodes,
                    &[
                        (TransparentPolynomial::Node(0) * TransparentPolynomial::Node(1))
                            .fix_computation(false),
                    ],
                    &[EF::ONE],
                    None,
                    false,
                    fs_prover,
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
            domain_size: 1 << (self.0.num_variables + self.0.starting_log_inv_rate),
            folding_randomness,
            coefficients: witness.polynomial,
            prev_merkle: witness.merkle_tree,
            prev_merkle_answers: witness.merkle_leaves,
        };

        self.round::<F>(fs_prover, round_state)
    }

    fn round<IF: ExtensionField<F>>(
        &self,
        fs_prover: &mut FsProver,
        mut round_state: RoundState<IF, EF>,
    ) -> Option<()>
    where
        EF: ExtensionField<IF>,
    {
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

            fs_prover.challenge_pow(self.0.final_pow_bits, self.0.cuda);

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
                    &[EF::ONE],
                    None,
                    false,
                    fs_prover,
                    round_state.hypercube_sum,
                    n_rounds,
                    SumcheckGrinding::Custom(pow_bits),
                    None,
                );
            }

            return Some(());
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
        let mut ood_points = vec![EF::ZERO; round_params.ood_samples];
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        if round_params.ood_samples > 0 {
            ood_points = fs_prover.challenge_scalars::<EF>(round_params.ood_samples);
            ood_answers.extend(ood_points.iter().map(|ood_point| {
                round_state.sumcheck_mles[0].evaluate(&multilinear_point_from_univariate(
                    *ood_point,
                    num_variables,
                ))
            }));
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
        let domain_scaled_gen = F::two_adic_generator(
            round_state.domain_size.trailing_zeros() as usize
                - self.0.folding_factor.at_round(round_state.round),
        );
        let stir_challenges_ext: Vec<Vec<EF>> = ood_points
            .into_iter()
            .map(|univariate| multilinear_point_from_univariate(univariate, num_variables))
            .collect();
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

        fs_prover.challenge_pow(round_params.pow_bits, self.0.cuda);

        // Randomness for combination
        let combination_randomness_gen = fs_prover.challenge_scalars::<EF>(1)[0];
        let combination_randomness = powers(
            combination_randomness_gen,
            stir_challenges_ext.len() + stir_challenges_base.len(),
        );

        round_state.sumcheck_mles[1] += &randomized_eq_extensions::<F, EF>(
            &stir_challenges_base,
            &stir_challenges_ext,
            &combination_randomness,
            self.0.cuda,
        );
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
                &[EF::ONE],
                None,
                false,
                fs_prover,
                round_state.hypercube_sum + dot_product(&combination_randomness, &stir_evaluations),
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

        self.round::<EF>(fs_prover, round_state)
    }
}

struct RoundState<F: Field, EF: ExtensionField<F>> {
    round: usize,
    sumcheck_mles: Vec<Multilinear<EF>>,
    hypercube_sum: EF,
    domain_size: usize,
    folding_randomness: Vec<EF>,
    coefficients: CoefficientList<F>,
    prev_merkle: MerkleTree<F>,
    prev_merkle_answers: HostOrDeviceBuffer<F>,
}

#[instrument(name = "randomized_eq_extensions", skip_all)]
fn randomized_eq_extensions<F: Field, EF: ExtensionField<F>>(
    eq_points_base: &[Vec<F>],
    eq_points_ext: &[Vec<EF>],
    randomized_coefs: &[EF],
    cuda: bool,
) -> Multilinear<EF> {
    assert_eq!(
        eq_points_ext.len() + eq_points_base.len(),
        randomized_coefs.len()
    );
    let n_vars = eq_points_ext[0].len();
    assert!(eq_points_ext.iter().all(|point| point.len() == n_vars));
    assert!(eq_points_base.iter().all(|point| point.len() == n_vars));

    // TODO to limit GPU memory usage, use intermediate sums

    let _span = tracing::info_span!("eq_mle (EF)", count = eq_points_ext.len()).entered();
    let mut all_eq_mles_ext = Vec::new();
    for point in eq_points_ext {
        all_eq_mles_ext.push(Multilinear::eq_mle(point, cuda));
    }
    cuda_sync();
    std::mem::drop(_span);

    let _span = tracing::info_span!("eq_mle (F)", count = eq_points_base.len()).entered();
    let mut all_eq_mles_base = Vec::new();
    for point in eq_points_base {
        all_eq_mles_base.push(Multilinear::eq_mle(point, cuda));
    }
    cuda_sync();
    std::mem::drop(_span);

    let _span =
        tracing::info_span!("linear combination (EF)", count = eq_points_ext.len()).entered();

    if TypeId::of::<F>() == TypeId::of::<EF>() {
        let drained = std::mem::take(&mut all_eq_mles_ext);
        let transmuted = unsafe { std::mem::transmute::<_, Vec<Multilinear<EF>>>(drained) };
        all_eq_mles_ext.extend(transmuted);
    }

    let mut res = MultilinearsVec::from(all_eq_mles_ext)
        .as_ref()
        .linear_comb(&randomized_coefs[0..eq_points_ext.len()]);
    cuda_sync();
    std::mem::drop(_span);

    if eq_points_base.len() > 0 {
        let _span =
            tracing::info_span!("linear combination (F)", count = eq_points_ext.len()).entered();
        let res_base = MultilinearsVec::from(all_eq_mles_base)
            .as_ref()
            .linear_comb(&randomized_coefs[eq_points_ext.len()..]);
        cuda_sync();
        std::mem::drop(_span);

        let _span = tracing::info_span!("sum", count = eq_points_ext.len()).entered();
        res += &res_base;
        cuda_sync();
    }
    res
}
