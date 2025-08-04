use p3_air::Air;
use p3_field::PackedValue;
use p3_field::{ExtensionField, TwoAdicField, cyclic_subgroup_known_order};
use p3_uni_stark::SymbolicAirBuilder;
use sumcheck::ProductComputation;
use tracing::{info_span, instrument};
use utils::{
    ConstraintFolder, ConstraintFolderPackedBase, Evaluation, FSProver, PFPacking,
    add_multilinears, log2_up, multilinears_linear_combination, shift_range,
};
use utils::{ConstraintFolderPackedExtension, PF};
use whir_p3::fiat_shamir::FSChallenger;
use whir_p3::poly::evals::{eval_eq, fold_multilinear, scale_poly};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

use crate::witness::AirWitness;
use crate::{
    uni_skip_utils::{matrix_down_folded, matrix_up_folded},
    utils::{column_down, column_up, columns_up_and_down},
};

use super::table::AirTable;

/* Multi Column CCS (SuperSpartan)

cf https://eprint.iacr.org/2023/552.pdf and https://solvable.group/posts/super-air/#fnref:1

*/

impl<EF, A> AirTable<EF, A>
where
    EF: TwoAdicField + ExtensionField<PF<EF>>,
    PF<EF>: TwoAdicField,
    A: Air<SymbolicAirBuilder<PF<EF>>>
        + for<'a> Air<ConstraintFolder<'a, PF<EF>, EF>>
        + for<'a> Air<ConstraintFolder<'a, EF, EF>>
        + for<'a> Air<ConstraintFolderPackedBase<'a, EF>>
        + for<'a> Air<ConstraintFolderPackedExtension<'a, EF>>,
{
    #[instrument(name = "air: prove", skip_all)]
    pub fn prove<'a>(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        witness: AirWitness<'a, PF<EF>>,
    ) -> Vec<Evaluation<EF>> {
        assert!(
            self.univariate_skips < witness.log_n_rows(),
            "TODO handle the case UNIVARIATE_SKIPS >= log_length"
        );
        let log_length = witness.log_n_rows();

        let constraints_batching_scalar = prover_state.sample();

        let constraints_batching_scalars =
            cyclic_subgroup_known_order(constraints_batching_scalar, self.n_constraints)
                .collect::<Vec<_>>();

        let mut zerocheck_challenges = vec![EF::ZERO; log_length + 1 - self.univariate_skips];
        for challenge in &mut zerocheck_challenges {
            *challenge = prover_state.sample();
        }

        let columns_up_and_down_opt = if self.air.structured() {
            Some(columns_up_and_down(&witness))
        } else {
            None
        };

        let columns_for_zero_check = if self.air.structured() {
            columns_up_and_down_opt
                .as_ref()
                .unwrap()
                .iter()
                .map(|c| c.as_slice())
                .collect::<Vec<_>>()
        } else {
            witness.cols.clone()
        };

        let columns_for_zero_check = columns_for_zero_check
            .iter()
            .map(|col| PFPacking::<EF>::pack_slice(col))
            .collect::<Vec<_>>();

        let (outer_sumcheck_challenge, all_inner_sums, _) =
            info_span!("zerocheck").in_scope(|| {
                sumcheck::prove_base_packed::<EF, _>(
                    self.univariate_skips,
                    columns_for_zero_check,
                    &self.air,
                    self.constraint_degree,
                    &constraints_batching_scalars,
                    Some((&zerocheck_challenges, None)),
                    true,
                    prover_state,
                    EF::ZERO,
                    None,
                    true,
                )
            });

        let mut evaluations_remaining_to_prove = vec![];
        for group in &witness.column_groups {
            if self.air.structured() {
                let columns = &witness.cols[group.clone()];
                let inner_sums_up = all_inner_sums[group.clone()].to_vec();
                let inner_sums_down =
                    all_inner_sums[shift_range(group.clone(), self.n_columns())].to_vec();
                evaluations_remaining_to_prove.push(self.open_structured_columns(
                    prover_state,
                    &columns,
                    &inner_sums_up,
                    &inner_sums_down,
                    &outer_sumcheck_challenge,
                ));
            } else {
                let columns = &witness.cols[group.clone()];
                let inner_sums = &all_inner_sums[group.clone()];
                evaluations_remaining_to_prove.push(self.open_unstructured_columns(
                    prover_state,
                    &columns,
                    &inner_sums,
                    &outer_sumcheck_challenge,
                ));
            }
        }
        evaluations_remaining_to_prove
    }

    #[instrument(skip_all)]
    fn open_unstructured_columns(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        columns: &[&[PF<EF>]],
        all_inner_sums: &[EF],
        outer_sumcheck_challenge: &[EF],
    ) -> Evaluation<EF> {
        assert_eq!(columns.len(), all_inner_sums.len());
        let log_n_columns = log2_up(columns.len());
        prover_state.add_extension_scalars(all_inner_sums);

        let mut columns_batching_scalars = vec![EF::ZERO; log_n_columns];
        for challenge in &mut columns_batching_scalars {
            *challenge = prover_state.sample();
        }

        let batched_column = multilinears_linear_combination(
            columns,
            &eval_eq(&columns_batching_scalars)[..columns.len()],
        );

        // TODO opti
        let sub_evals = fold_multilinear(
            &batched_column,
            &MultilinearPoint(outer_sumcheck_challenge[1..].to_vec()),
        );

        prover_state.add_extension_scalars(&sub_evals);

        let mut epsilons = vec![EF::ZERO; self.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = prover_state.sample();
        }

        let point =
            MultilinearPoint([epsilons.clone(), outer_sumcheck_challenge[1..].to_vec()].concat());

        let final_value = sub_evals.evaluate(&MultilinearPoint(epsilons));

        prover_state.add_extension_scalar(final_value);

        Evaluation {
            point: MultilinearPoint([columns_batching_scalars, point.0].concat()),
            value: final_value,
        }
    }

    #[instrument(skip_all)]
    fn open_structured_columns(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        columns: &[&[PF<EF>]],
        inner_sums_up: &[EF],
        inner_sums_down: &[EF],
        outer_sumcheck_challenge: &[EF],
    ) -> Evaluation<EF> {
        assert_eq!(columns.len(), inner_sums_up.len());
        assert_eq!(columns.len(), inner_sums_down.len());
        let log_n_columns = log2_up(columns.len());

        prover_state.add_extension_scalars(&inner_sums_up);
        prover_state.add_extension_scalars(&inner_sums_down);

        let mut columns_batching_scalars = vec![EF::ZERO; log_n_columns];
        for challenge in &mut columns_batching_scalars {
            *challenge = prover_state.sample();
        }

        let batched_column = multilinears_linear_combination(
            columns,
            &eval_eq(&columns_batching_scalars)[..columns.len()],
        );

        let alpha = prover_state.sample();

        let batched_column_mixed = add_multilinears(
            &column_up(&batched_column),
            &scale_poly(&column_down(&batched_column), alpha),
        );

        // TODO opti
        let sub_evals = fold_multilinear(
            &batched_column_mixed,
            &MultilinearPoint(outer_sumcheck_challenge[1..].to_vec()),
        );

        prover_state.add_extension_scalars(&sub_evals);

        let mut epsilons = vec![EF::ZERO; self.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = prover_state.sample();
        }

        let point = [epsilons.clone(), outer_sumcheck_challenge[1..].to_vec()].concat();
        let mles_for_inner_sumcheck = vec![
            add_multilinears(
                &matrix_up_folded(&point),
                &scale_poly(&matrix_down_folded(&point), alpha),
            ),
            batched_column,
        ];

        // TODO do not recompute
        let inner_sum = info_span!("inner sum evaluation")
            .in_scope(|| batched_column_mixed.evaluate(&MultilinearPoint(point.clone())));

        let (inner_challenges, inner_evals, _) = sumcheck::prove_generic::<EF, EF, _>(
            1,
            mles_for_inner_sumcheck
                .iter()
                .map(|m| m.as_slice())
                .collect::<Vec<_>>(),
            &ProductComputation,
            2,
            &[EF::ONE],
            None,
            false,
            prover_state,
            inner_sum,
            None,
            false,
        );

        let final_point = [columns_batching_scalars.clone(), inner_challenges.0].concat();

        let packed_value = inner_evals[1];

        Evaluation {
            point: MultilinearPoint(final_point),
            value: packed_value,
        }
    }
}
