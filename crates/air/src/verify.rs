use p3_air::Air;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Packable, TwoAdicField, cyclic_subgroup_known_order, dot_product};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};
use sumcheck::{SumcheckComputation, SumcheckError, SumcheckGrinding};
use tracing::instrument;
use utils::{ConstraintFolder, fold_multilinear_in_large_field, log2_up};
use whir_p3::{
    fiat_shamir::{errors::ProofError, verifier::VerifierState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::reader::CommitmentReader,
        statement::{Statement, weights::Weights},
        verifier::Verifier,
    },
};

use crate::{
    AirSettings,
    utils::{column_down, column_up, matrix_down_lde, matrix_up_lde},
};

use super::table::AirTable;

#[derive(Debug, Clone)]
pub enum AirVerifError {
    InvalidPcsCommitment,
    InvalidPcsOpening,
    Fs(ProofError),
    Sumcheck(SumcheckError),
    InvalidBoundaryCondition,
    SumMismatch,
}

impl From<ProofError> for AirVerifError {
    fn from(e: ProofError) -> Self {
        Self::Fs(e)
    }
}

impl From<SumcheckError> for AirVerifError {
    fn from(e: SumcheckError) -> Self {
        Self::Sumcheck(e)
    }
}

impl<
    'a,
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    A: Air<ConstraintFolder<'a, F, EF, EF>>,
> AirTable<F, EF, A>
{
    #[instrument(name = "air table: verify", skip_all)]
    pub fn verify<H, C, Challenger, const DIGEST_ELEMS: usize>(
        &self,
        settings: &AirSettings,
        merkle_hash: H,
        merkle_compress: C,
        verifier_state: &mut VerifierState<EF, F, Challenger, DIGEST_ELEMS>,
        log_length: usize,
    ) -> Result<(), AirVerifError>
    where
        StandardUniform: Distribution<EF> + Distribution<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]> + Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2> + Sync,
        [F; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        F: Eq + Packable,
    {
        let whir_params = self.build_whir_params(settings, merkle_hash, merkle_compress);

        let commitment_reader = CommitmentReader::new(&whir_params);
        let whir_verifier = Verifier::new(&whir_params);
        let parsed_commitment = commitment_reader
            .parse_commitment::<DIGEST_ELEMS>(verifier_state)
            .map_err(|_| AirVerifError::InvalidPcsCommitment)?;

        let pow_witness = verifier_state.proof_data.pow_witnesses.remove(0);
        assert!(
            verifier_state.challenger.check_witness(
                settings
                    .security_bits
                    .saturating_sub(EF::bits().saturating_sub(log2_up(self.n_constraints))),
                pow_witness
            ),
            "Witness does not satisfy the PoW condition"
        );

        let constraints_batching_scalar = verifier_state.challenger.sample_algebra_element();

        let pow_witness = verifier_state.proof_data.pow_witnesses.remove(0);
        assert!(
            verifier_state.challenger.check_witness(
                settings
                    .security_bits
                    .saturating_sub(EF::bits().saturating_sub(self.log_length)),
                pow_witness
            ),
            "Witness does not satisfy the PoW condition"
        );

        let mut zerocheck_challenges = vec![EF::ZERO; log_length - settings.univariate_skips + 1];
        for challenge in &mut zerocheck_challenges {
            *challenge = verifier_state.challenger.sample_algebra_element();
        }

        let (sc_sum, outer_sumcheck_challenge) =
            sumcheck::verify_with_univariate_skip::<EF, F, Challenger, DIGEST_ELEMS>(
                verifier_state,
                self.constraint_degree + 1,
                log_length,
                settings.univariate_skips,
                SumcheckGrinding::Auto {
                    security_bits: settings.security_bits,
                },
            )?;
        if sc_sum != EF::ZERO {
            return Err(AirVerifError::SumMismatch);
        }

        let witness_shifted_evals = verifier_state.proof_data.piop.remove(0);

        for witness_shifted_eval in &witness_shifted_evals {
            verifier_state
                .challenger
                .observe_algebra_element(*witness_shifted_eval);
        }

        let (witness_up, witness_down) = witness_shifted_evals.split_at(self.n_witness_columns());
        let outer_selector_evals = self
            .univariate_selectors
            .iter()
            .map(|s| s.evaluate(outer_sumcheck_challenge.point[0]))
            .collect::<Vec<_>>();
        let preprocessed_up = self
            .preprocessed_columns
            .iter()
            .map(|c| {
                fold_multilinear_in_large_field(&column_up(c), &outer_selector_evals).evaluate(
                    &MultilinearPoint(outer_sumcheck_challenge.point[1..].to_vec()),
                )
            })
            .collect::<Vec<_>>();
        let preprocessed_down = self
            .preprocessed_columns
            .iter()
            .map(|c| {
                fold_multilinear_in_large_field(&column_down(c), &outer_selector_evals).evaluate(
                    &MultilinearPoint(outer_sumcheck_challenge.point[1..].to_vec()),
                )
            })
            .collect::<Vec<_>>();

        let global_point = [
            preprocessed_up,
            witness_up.to_vec(),
            preprocessed_down,
            witness_down.to_vec(),
        ]
        .concat();

        let global_constraint_eval = SumcheckComputation::eval(
            &self.air,
            &global_point,
            &cyclic_subgroup_known_order(constraints_batching_scalar, self.n_constraints)
                .collect::<Vec<_>>(),
        );

        let zerocheck_selector_evals = self
            .univariate_selectors
            .iter()
            .map(|s| s.evaluate(zerocheck_challenges[0]));
        if dot_product::<EF, _, _>(
            zerocheck_selector_evals.clone(),
            outer_selector_evals.iter().copied(),
        ) * MultilinearPoint(zerocheck_challenges[1..].to_vec()).eq_poly_outside(
            &MultilinearPoint(outer_sumcheck_challenge.point[1..].to_vec()),
        ) * global_constraint_eval
            != outer_sumcheck_challenge.value
        {
            return Err(AirVerifError::SumMismatch);
        }

        let pow_witness = verifier_state.proof_data.pow_witnesses.remove(0);
        assert!(
            verifier_state.challenger.check_witness(
                settings
                    .security_bits
                    .saturating_sub(EF::bits().saturating_sub(log2_up(self.n_witness_columns()))),
                pow_witness
            ),
            "Witness does not satisfy the PoW condition"
        );

        let mut columns_batching_scalars = vec![EF::ZERO; self.log_n_witness_columns()];
        for challenge in &mut columns_batching_scalars {
            *challenge = verifier_state.challenger.sample_algebra_element();
        }

        let alpha: EF = verifier_state.challenger.sample_algebra_element();

        let sub_evals = verifier_state.proof_data.piop.remove(0);
        for sub_eval in &sub_evals {
            verifier_state.challenger.observe_algebra_element(*sub_eval);
        }

        if dot_product::<EF, _, _>(
            sub_evals.iter().copied(),
            outer_selector_evals.iter().copied(),
        ) != dot_product::<EF, _, _>(
            witness_up.iter().copied(),
            EvaluationsList::eval_eq(&columns_batching_scalars).evals()[..self.n_witness_columns()]
                .iter()
                .copied(),
        ) + dot_product::<EF, _, _>(
            witness_down.iter().copied(),
            EvaluationsList::eval_eq(&columns_batching_scalars).evals()[..self.n_witness_columns()]
                .iter()
                .copied(),
        ) * alpha
        {
            return Err(AirVerifError::SumMismatch);
        }

        let mut epsilons = vec![EF::ZERO; settings.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = verifier_state.challenger.sample_algebra_element();
        }

        let (batched_inner_sum, inner_sumcheck_challenge) =
            sumcheck::verify::<EF, F, Challenger, DIGEST_ELEMS>(
                verifier_state,
                log_length,
                2,
                SumcheckGrinding::Auto {
                    security_bits: settings.security_bits,
                },
            )?;

        if batched_inner_sum
            != EvaluationsList::new(sub_evals).evaluate(&MultilinearPoint(epsilons.clone()))
        {
            return Err(AirVerifError::SumMismatch);
        }

        let matrix_lde_point = [
            epsilons.clone(),
            outer_sumcheck_challenge.point[1..].to_vec(),
            inner_sumcheck_challenge.point.clone(),
        ]
        .concat();
        let up = matrix_up_lde(&matrix_lde_point);
        let down = matrix_down_lde(&matrix_lde_point);

        let expected_final_value = inner_sumcheck_challenge.value / (up + alpha * down);

        let final_point = [
            columns_batching_scalars.clone(),
            inner_sumcheck_challenge.point,
        ]
        .concat();

        let mut statement = Statement::<EF>::new(final_point.len());
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(final_point)),
            expected_final_value,
        );
        whir_verifier
            .verify(verifier_state, &parsed_commitment, &statement)
            .map_err(|_| AirVerifError::InvalidPcsOpening)?;

        Ok(())
    }
}
