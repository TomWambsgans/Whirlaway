use multi_pcs::pcs::PCS;
use p3_air::Air;
use p3_field::{ExtensionField, TwoAdicField, cyclic_subgroup_known_order, dot_product};
use p3_uni_stark::SymbolicAirBuilder;
use std::fmt::Debug;
use sumcheck::{SumcheckComputation, SumcheckError};
use tracing::instrument;
use utils::Evaluation;
use utils::{ConstraintFolder, fold_multilinear_in_large_field};
use utils::{FSVerifier, PF};
use whir_p3::fiat_shamir::FSChallenger;
use whir_p3::poly::evals::eval_eq;
use whir_p3::{
    fiat_shamir::errors::ProofError,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
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
    EF: TwoAdicField + ExtensionField<PF<EF>>,
    A: Air<SymbolicAirBuilder<PF<EF>>> + for<'a> Air<ConstraintFolder<'a, EF, EF>>,
> AirTable<EF, A>
{
    #[instrument(name = "air table: verify", skip_all)]
    pub fn verify(
        &self,
        settings: &AirSettings,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        log_length: usize,
        pcs: &impl PCS<PF<EF>, EF>,
    ) -> Result<(), AirVerifError>
    where
        PF<EF>: TwoAdicField,
    {
        let num_variables = self.log_length + self.log_n_witness_columns();
        let parsed_commitment = pcs
            .parse_commitment(verifier_state, num_variables)
            .map_err(|_| AirVerifError::InvalidPcsCommitment)?;

        let constraints_batching_scalar = verifier_state.sample();

        let mut zerocheck_challenges = vec![EF::ZERO; log_length - settings.univariate_skips + 1];
        for challenge in &mut zerocheck_challenges {
            *challenge = verifier_state.sample();
        }

        let (sc_sum, outer_sumcheck_challenge) = sumcheck::verify_with_univariate_skip::<EF>(
            verifier_state,
            self.constraint_degree + 1,
            log_length,
            settings.univariate_skips,
        )?;
        if sc_sum != EF::ZERO {
            return Err(AirVerifError::SumMismatch);
        }

        let outer_selector_evals = self
            .univariate_selectors
            .iter()
            .map(|s| s.evaluate(outer_sumcheck_challenge.point[0]))
            .collect::<Vec<_>>();

        if self.air.structured() {
            self.verify_structured_columns(
                verifier_state,
                settings,
                pcs,
                &parsed_commitment,
                &outer_sumcheck_challenge,
                &outer_selector_evals,
                &zerocheck_challenges,
                constraints_batching_scalar,
                log_length,
            )
        } else {
            self.verify_unstructured_columns(
                verifier_state,
                settings,
                pcs,
                &parsed_commitment,
                &outer_sumcheck_challenge,
                &outer_selector_evals,
                &zerocheck_challenges,
                constraints_batching_scalar,
            )
        }
    }

    fn verify_unstructured_columns<Pcs: PCS<PF<EF>, EF>>(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        settings: &AirSettings,
        pcs: &Pcs,
        parsed_commitment: &Pcs::ParsedCommitment,
        outer_sumcheck_challenge: &Evaluation<EF>,
        outer_selector_evals: &[EF],
        zerocheck_challenges: &[EF],
        constraints_batching_scalar: EF,
    ) -> Result<(), AirVerifError>
    where
        PF<EF>: TwoAdicField,
    {
        let witness_evals = verifier_state.next_extension_scalars_vec(self.n_witness_columns())?;
        let preprocessed_evals = self
            .preprocessed_columns
            .iter()
            .map(|c| {
                fold_multilinear_in_large_field(c, &outer_selector_evals).evaluate(
                    &MultilinearPoint(outer_sumcheck_challenge.point[1..].to_vec()),
                )
            })
            .collect::<Vec<_>>();

        let global_point = [preprocessed_evals, witness_evals.clone()].concat();

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

        let mut columns_batching_scalars = vec![EF::ZERO; self.log_n_witness_columns()];
        for challenge in &mut columns_batching_scalars {
            *challenge = verifier_state.sample();
        }

        let sub_evals =
            verifier_state.next_extension_scalars_vec(1 << settings.univariate_skips)?;

        if dot_product::<EF, _, _>(
            sub_evals.iter().copied(),
            outer_selector_evals.iter().copied(),
        ) != dot_product::<EF, _, _>(
            witness_evals.iter().copied(),
            eval_eq(&columns_batching_scalars)[..self.n_witness_columns()]
                .iter()
                .copied(),
        ) {
            return Err(AirVerifError::SumMismatch);
        }

        let mut epsilons = vec![EF::ZERO; settings.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = verifier_state.sample();
        }

        let [final_value] = verifier_state.next_extension_scalars_const()?;

        if final_value != sub_evals.evaluate(&MultilinearPoint(epsilons.clone())) {
            return Err(AirVerifError::SumMismatch);
        }

        let final_point = MultilinearPoint(
            [
                columns_batching_scalars,
                epsilons.clone(),
                outer_sumcheck_challenge.point[1..].to_vec(),
            ]
            .concat(),
        );

        let statement = vec![Evaluation {
            point: final_point,
            value: final_value,
        }];
        pcs.verify(verifier_state, &parsed_commitment, &statement)
            .map_err(|_| AirVerifError::InvalidPcsOpening)?;

        return Ok(());
    }

    fn verify_structured_columns<Pcs: PCS<PF<EF>, EF>>(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        settings: &AirSettings,
        pcs: &Pcs,
        parsed_commitment: &Pcs::ParsedCommitment,
        outer_sumcheck_challenge: &Evaluation<EF>,
        outer_selector_evals: &[EF],
        zerocheck_challenges: &[EF],
        constraints_batching_scalar: EF,
        log_length: usize,
    ) -> Result<(), AirVerifError>
    where
        PF<EF>: TwoAdicField,
    {
        let witness_up = verifier_state.next_extension_scalars_vec(self.n_witness_columns())?;
        let witness_down = verifier_state.next_extension_scalars_vec(self.n_witness_columns())?;

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
            witness_up.clone(),
            preprocessed_down,
            witness_down.clone(),
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

        let mut columns_batching_scalars = vec![EF::ZERO; self.log_n_witness_columns()];
        for challenge in &mut columns_batching_scalars {
            *challenge = verifier_state.sample();
        }

        let alpha: EF = verifier_state.sample();

        let sub_evals =
            verifier_state.next_extension_scalars_vec(1 << settings.univariate_skips)?;

        if dot_product::<EF, _, _>(
            sub_evals.iter().copied(),
            outer_selector_evals.iter().copied(),
        ) != dot_product::<EF, _, _>(
            witness_up.iter().copied(),
            eval_eq(&columns_batching_scalars)[..self.n_witness_columns()]
                .iter()
                .copied(),
        ) + dot_product::<EF, _, _>(
            witness_down.iter().copied(),
            eval_eq(&columns_batching_scalars)[..self.n_witness_columns()]
                .iter()
                .copied(),
        ) * alpha
        {
            return Err(AirVerifError::SumMismatch);
        }

        let mut epsilons = vec![EF::ZERO; settings.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = verifier_state.sample();
        }

        let (batched_inner_sum, inner_sumcheck_challenge) =
            sumcheck::verify::<EF>(verifier_state, log_length, 2)?;

        if batched_inner_sum != sub_evals.evaluate(&MultilinearPoint(epsilons.clone())) {
            return Err(AirVerifError::SumMismatch);
        }

        let matrix_lde_point = [
            epsilons.clone(),
            outer_sumcheck_challenge.point[1..].to_vec(),
            inner_sumcheck_challenge.point.0.clone(),
        ]
        .concat();
        let up = matrix_up_lde(&matrix_lde_point);
        let down = matrix_down_lde(&matrix_lde_point);

        let expected_final_value = inner_sumcheck_challenge.value / (up + alpha * down);

        let final_point = [
            columns_batching_scalars.clone(),
            inner_sumcheck_challenge.point.0,
        ]
        .concat();

        let statement = vec![Evaluation {
            point: MultilinearPoint(final_point),
            value: expected_final_value,
        }];
        pcs.verify(verifier_state, &parsed_commitment, &statement)
            .map_err(|_| AirVerifError::InvalidPcsOpening)?;

        Ok(())
    }
}
