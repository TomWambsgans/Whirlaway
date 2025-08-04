use p3_air::Air;
use p3_field::{ExtensionField, TwoAdicField, cyclic_subgroup_known_order, dot_product};
use p3_uni_stark::SymbolicAirBuilder;
use std::fmt::Debug;
use std::ops::Range;
use sumcheck::{SumcheckComputation, SumcheckError};
use tracing::instrument;
use utils::{ConstraintFolder, shift_range};
use utils::{Evaluation, log2_up};
use utils::{FSVerifier, PF};
use whir_p3::fiat_shamir::FSChallenger;
use whir_p3::poly::evals::eval_eq;
use whir_p3::{
    fiat_shamir::errors::ProofError,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
};

use crate::utils::{matrix_down_lde, matrix_up_lde};

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
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        log_n_rows: usize,
        column_groups: &[Range<usize>],
    ) -> Result<Vec<Evaluation<EF>>, AirVerifError> {
        let constraints_batching_scalar = verifier_state.sample();

        let mut zerocheck_challenges = vec![EF::ZERO; log_n_rows - self.univariate_skips + 1];
        for challenge in &mut zerocheck_challenges {
            *challenge = verifier_state.sample();
        }

        let (sc_sum, outer_sumcheck_challenge) = sumcheck::verify_with_univariate_skip::<EF>(
            verifier_state,
            self.constraint_degree + 1,
            log_n_rows,
            self.univariate_skips,
        )?;
        if sc_sum != EF::ZERO {
            return Err(AirVerifError::SumMismatch);
        }

        let outer_selector_evals = self
            .univariate_selectors
            .iter()
            .map(|s| s.evaluate(outer_sumcheck_challenge.point[0]))
            .collect::<Vec<_>>();

        let mut evaluations_remaining_to_verify = vec![];
        let mut global_point = EF::zero_vec(if self.air.structured() {
            2 * self.n_columns()
        } else {
            self.n_columns()
        });

        for group in column_groups {
            if self.air.structured() {
                let (evaluation_remaining_to_verify, witness_evals_up, witness_evals_down) = self
                    .verify_structured_columns(
                    verifier_state,
                    &outer_sumcheck_challenge,
                    &outer_selector_evals,
                    log_n_rows,
                    group.len(),
                )?;
                evaluations_remaining_to_verify.push(evaluation_remaining_to_verify);
                global_point[group.clone()].copy_from_slice(&witness_evals_up);
                global_point[shift_range(group.clone(), self.n_columns())]
                    .copy_from_slice(&witness_evals_down);
            } else {
                let (evaluation_remaining_to_verify, witness_evals) = self
                    .verify_unstructured_columns(
                        verifier_state,
                        &outer_sumcheck_challenge,
                        &outer_selector_evals,
                        group.len(),
                    )?;
                evaluations_remaining_to_verify.push(evaluation_remaining_to_verify);
                global_point[group.clone()].copy_from_slice(&witness_evals);
            }
        }

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

        Ok(evaluations_remaining_to_verify)
    }

    fn verify_unstructured_columns(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        outer_sumcheck_challenge: &Evaluation<EF>,
        outer_selector_evals: &[EF],
        n_columns: usize,
    ) -> Result<(Evaluation<EF>, Vec<EF>), AirVerifError> {
        let log_n_columns = log2_up(n_columns);
        let witness_evals = verifier_state.next_extension_scalars_vec(n_columns)?;

        let mut columns_batching_scalars = vec![EF::ZERO; log_n_columns];
        for challenge in &mut columns_batching_scalars {
            *challenge = verifier_state.sample();
        }

        let sub_evals = verifier_state.next_extension_scalars_vec(1 << self.univariate_skips)?;

        if dot_product::<EF, _, _>(
            sub_evals.iter().copied(),
            outer_selector_evals.iter().copied(),
        ) != dot_product::<EF, _, _>(
            witness_evals.iter().copied(),
            eval_eq(&columns_batching_scalars)[..n_columns]
                .iter()
                .copied(),
        ) {
            return Err(AirVerifError::SumMismatch);
        }

        let mut epsilons = vec![EF::ZERO; self.univariate_skips];
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

        Ok((
            Evaluation {
                point: final_point,
                value: final_value,
            },
            witness_evals,
        ))
    }

    fn verify_structured_columns(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        outer_sumcheck_challenge: &Evaluation<EF>,
        outer_selector_evals: &[EF],
        log_n_rows: usize,
        n_columns: usize,
    ) -> Result<(Evaluation<EF>, Vec<EF>, Vec<EF>), AirVerifError> {
        let log_n_columns = log2_up(n_columns);

        let witness_up = verifier_state.next_extension_scalars_vec(n_columns)?;
        let witness_down = verifier_state.next_extension_scalars_vec(n_columns)?;

        let mut columns_batching_scalars = vec![EF::ZERO; log_n_columns];
        for challenge in &mut columns_batching_scalars {
            *challenge = verifier_state.sample();
        }

        let alpha: EF = verifier_state.sample();

        let sub_evals = verifier_state.next_extension_scalars_vec(1 << self.univariate_skips)?;

        if dot_product::<EF, _, _>(
            sub_evals.iter().copied(),
            outer_selector_evals.iter().copied(),
        ) != dot_product::<EF, _, _>(
            witness_up.iter().copied(),
            eval_eq(&columns_batching_scalars)[..n_columns]
                .iter()
                .copied(),
        ) + dot_product::<EF, _, _>(
            witness_down.iter().copied(),
            eval_eq(&columns_batching_scalars)[..n_columns]
                .iter()
                .copied(),
        ) * alpha
        {
            return Err(AirVerifError::SumMismatch);
        }

        let mut epsilons = vec![EF::ZERO; self.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = verifier_state.sample();
        }

        let (batched_inner_sum, inner_sumcheck_challenge) =
            sumcheck::verify::<EF>(verifier_state, log_n_rows, 2)?;

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

        Ok((
            Evaluation {
                point: MultilinearPoint(final_point),
                value: expected_final_value,
            },
            witness_up,
            witness_down,
        ))
    }
}
