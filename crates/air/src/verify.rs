use p3_field::{ExtensionField, cyclic_subgroup_known_order, dot_product};
use p3_util::log2_ceil_usize;
use std::ops::Range;
use sumcheck::SumcheckComputation;
use tracing::instrument;
use utils::univariate_selectors;
use utils::{Evaluation, from_end};
use utils::{FSVerifier, PF};
use whir_p3::fiat_shamir::FSChallenger;
use whir_p3::poly::evals::eval_eq;
use whir_p3::{
    fiat_shamir::errors::ProofError,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
};

use crate::MyAir;
use crate::utils::{matrix_down_lde, matrix_up_lde};

use super::table::AirTable;

impl<EF: ExtensionField<PF<EF>>, A: MyAir<EF>> AirTable<EF, A> {
    #[instrument(name = "air table: verify", skip_all)]
    pub fn verify(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        log_n_rows: usize,
        column_groups: &[Range<usize>],
    ) -> Result<Vec<Evaluation<EF>>, ProofError> {
        let constraints_batching_scalar = verifier_state.sample();

        let zerocheck_challenges =
            verifier_state.sample_vec(log_n_rows + 1 - self.univariate_skips);

        let (sc_sum, outer_sumcheck_challenge) = sumcheck::verify_with_univariate_skip::<EF>(
            verifier_state,
            self.air.degree() + 1,
            log_n_rows,
            self.univariate_skips,
        )?;
        if sc_sum != EF::ZERO {
            return Err(ProofError::InvalidProof);
        }

        let outer_selector_evals = univariate_selectors::<PF<EF>>(self.univariate_skips)
            .iter()
            .map(|s| s.evaluate(outer_sumcheck_challenge.point[0]))
            .collect::<Vec<_>>();

        let all_inner_sums =
            verifier_state.next_extension_scalars_vec(if self.air.structured() {
                2 * self.n_columns()
            } else {
                self.n_columns()
            })?;

        let global_constraint_eval = SumcheckComputation::eval(
            &self.air,
            &all_inner_sums,
            &cyclic_subgroup_known_order(constraints_batching_scalar, self.n_constraints)
                .collect::<Vec<_>>(),
        );

        let zerocheck_selector_evals = univariate_selectors::<PF<EF>>(self.univariate_skips)
            .iter()
            .map(|s| s.evaluate(zerocheck_challenges[0]))
            .collect::<Vec<_>>();
        if dot_product::<EF, _, _>(
            zerocheck_selector_evals.into_iter(),
            outer_selector_evals.iter().copied(),
        ) * MultilinearPoint(zerocheck_challenges[1..].to_vec()).eq_poly_outside(
            &MultilinearPoint(outer_sumcheck_challenge.point[1..].to_vec()),
        ) * global_constraint_eval
            != outer_sumcheck_challenge.value
        {
            return Err(ProofError::InvalidProof);
        }

        if self.air.structured() {
            self.verify_structured_columns(
                verifier_state,
                &all_inner_sums,
                column_groups,
                &outer_sumcheck_challenge,
                &outer_selector_evals,
                log_n_rows,
            )
        } else {
            self.verify_unstructured_columns(
                verifier_state,
                &all_inner_sums,
                column_groups,
                &outer_sumcheck_challenge,
                &outer_selector_evals,
            )
        }
    }

    fn verify_unstructured_columns(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        all_inner_sums: &[EF],
        column_groups: &[Range<usize>],
        outer_sumcheck_challenge: &Evaluation<EF>,
        outer_selector_evals: &[EF],
    ) -> Result<Vec<Evaluation<EF>>, ProofError> {
        let max_columns_per_group = Iterator::max(column_groups.iter().map(|g| g.len())).unwrap();
        let log_max_columns_per_group = log2_ceil_usize(max_columns_per_group);
        let columns_batching_scalars = verifier_state.sample_vec(log_max_columns_per_group);

        let mut all_sub_evals = vec![];
        for group in column_groups {
            let sub_evals =
                verifier_state.next_extension_scalars_vec(1 << self.univariate_skips)?;

            if dot_product::<EF, _, _>(
                sub_evals.iter().copied(),
                outer_selector_evals.iter().copied(),
            ) != dot_product::<EF, _, _>(
                all_inner_sums[group.clone()].iter().copied(),
                eval_eq(&from_end(
                    &columns_batching_scalars,
                    log2_ceil_usize(group.len()),
                ))[..group.len()]
                    .iter()
                    .copied(),
            ) {
                return Err(ProofError::InvalidProof);
            }

            all_sub_evals.push(sub_evals);
        }

        let epsilons = MultilinearPoint(verifier_state.sample_vec(self.univariate_skips));

        let mut evaluations_remaining_to_verify = vec![];
        for (i, group) in column_groups.iter().enumerate() {
            let final_value = all_sub_evals[i].evaluate(&epsilons);
            let final_point = MultilinearPoint(
                [
                    from_end(&columns_batching_scalars, log2_ceil_usize(group.len())).to_vec(),
                    epsilons.0.clone(),
                    outer_sumcheck_challenge.point[1..].to_vec(),
                ]
                .concat(),
            );
            evaluations_remaining_to_verify.push(Evaluation {
                point: final_point,
                value: final_value,
            });
        }

        Ok(evaluations_remaining_to_verify)
    }

    fn verify_structured_columns(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
        all_inner_sums: &[EF],
        column_groups: &[Range<usize>],
        outer_sumcheck_challenge: &Evaluation<EF>,
        outer_selector_evals: &[EF],
        log_n_rows: usize,
    ) -> Result<Vec<Evaluation<EF>>, ProofError> {
        let max_columns_per_group = Iterator::max(column_groups.iter().map(|g| g.len())).unwrap();
        let log_max_columns_per_group = log2_ceil_usize(max_columns_per_group);
        let columns_batching_scalars = verifier_state.sample_vec(log_max_columns_per_group);

        let alpha = verifier_state.sample();

        let all_witness_up = &all_inner_sums[..self.n_columns()];
        let all_witness_down = &all_inner_sums[self.n_columns()..];
        assert_eq!(all_witness_up.len(), all_witness_down.len());

        let mut all_sub_evals = vec![];
        for group in column_groups {
            let sub_evals =
                verifier_state.next_extension_scalars_vec(1 << self.univariate_skips)?;

            let witness_up = &all_witness_up[group.clone()];
            let witness_down = &all_witness_down[group.clone()];

            if dot_product::<EF, _, _>(
                sub_evals.iter().copied(),
                outer_selector_evals.iter().copied(),
            ) != dot_product::<EF, _, _>(
                witness_up.iter().copied(),
                eval_eq(&from_end(
                    &columns_batching_scalars,
                    log2_ceil_usize(group.len()),
                ))[..group.len()]
                    .iter()
                    .copied(),
            ) + dot_product::<EF, _, _>(
                witness_down.iter().copied(),
                eval_eq(&from_end(
                    &columns_batching_scalars,
                    log2_ceil_usize(group.len()),
                ))[..group.len()]
                    .iter()
                    .copied(),
            ) * alpha
            {
                return Err(ProofError::InvalidProof);
            }

            all_sub_evals.push(sub_evals);
        }

        let epsilons = MultilinearPoint(verifier_state.sample_vec(self.univariate_skips));

        let (
            all_batched_inner_sums,
            inner_sumcheck_challenge_point,
            inner_sumcheck_challenge_values,
        ) = sumcheck::verify_in_parallel(
            verifier_state,
            vec![log_n_rows; column_groups.len()],
            vec![2; column_groups.len()],
            true,
        )?;

        for (batched_inner_sum, sub_evals) in all_batched_inner_sums.into_iter().zip(all_sub_evals)
        {
            if batched_inner_sum != sub_evals.evaluate(&epsilons) {
                return Err(ProofError::InvalidProof);
            }
        }

        let mut evaluations_remaining_to_verify = vec![];
        for (group, inner_sumcheck_challenge_value) in
            column_groups.iter().zip(inner_sumcheck_challenge_values)
        {
            let matrix_lde_point = [
                epsilons.0.clone(),
                outer_sumcheck_challenge.point[1..].to_vec(),
                inner_sumcheck_challenge_point.0.clone(),
            ]
            .concat();
            let up = matrix_up_lde(&matrix_lde_point);
            let down = matrix_down_lde(&matrix_lde_point);

            let final_value = inner_sumcheck_challenge_value / (up + alpha * down);

            let final_point = [
                from_end(&columns_batching_scalars, log2_ceil_usize(group.len())).to_vec(),
                inner_sumcheck_challenge_point.0.clone(),
            ]
            .concat();

            evaluations_remaining_to_verify.push(Evaluation {
                point: MultilinearPoint(final_point),
                value: final_value,
            });
        }
        Ok(evaluations_remaining_to_verify)
    }
}
