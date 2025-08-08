use p3_field::PackedValue;
use p3_field::{ExtensionField, cyclic_subgroup_known_order};
use p3_util::log2_ceil_usize;
use sumcheck::{MleGroupOwned, MleGroupRef, ProductComputation};
use tracing::{info_span, instrument};
use utils::PF;
use utils::{
    Evaluation, FSProver, PFPacking, add_multilinears, from_end, multilinears_linear_combination,
};
use whir_p3::fiat_shamir::FSChallenger;
use whir_p3::poly::evals::{eval_eq, fold_multilinear, scale_poly};
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

use crate::MyAir;
use crate::witness::AirWitness;
use crate::{
    uni_skip_utils::{matrix_down_folded, matrix_up_folded},
    utils::{column_down, column_up, columns_up_and_down},
};

use super::table::AirTable;

/*

cf https://eprint.iacr.org/2023/552.pdf and https://solvable.group/posts/super-air/#fnref:1

*/

impl<EF: ExtensionField<PF<EF>>, A: MyAir<EF>> AirTable<EF, A> {
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

        let zerocheck_challenges = prover_state.sample_vec(log_length + 1 - self.univariate_skips);

        let columns_for_zero_check: MleGroup<EF> = if self.air.structured() {
            MleGroupOwned::Base(columns_up_and_down(&witness)).into()
        } else {
            MleGroupRef::Base(witness.cols.clone()).into()
        };

        let columns_for_zero_check_packed = columns_for_zero_check.by_ref().pack();

        let (outer_sumcheck_challenge, all_inner_sums, _) =
            info_span!("zerocheck").in_scope(|| {
                sumcheck::prove::<EF, _>(
                    self.univariate_skips,
                    columns_for_zero_check_packed,
                    &self.air,
                    &constraints_batching_scalars,
                    Some((zerocheck_challenges, None)),
                    true,
                    prover_state,
                    EF::ZERO,
                    None,
                )
            });

        prover_state.add_extension_scalars(&all_inner_sums);

        if self.air.structured() {
            self.open_structured_columns(prover_state, &witness, &outer_sumcheck_challenge)
        } else {
            self.open_unstructured_columns(prover_state, &witness, &outer_sumcheck_challenge)
        }
    }

    #[instrument(skip_all)]
    fn open_unstructured_columns<'a>(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        witness: &AirWitness<'a, PF<EF>>,
        outer_sumcheck_challenge: &[EF],
    ) -> Vec<Evaluation<EF>> {
        let columns_batching_scalars = prover_state.sample_vec(witness.log_max_columns_per_group());

        let mut all_sub_evals = vec![];
        for group in &witness.column_groups {
            let batched_column = multilinears_linear_combination(
                &witness.cols[group.clone()],
                &eval_eq(&from_end(
                    &columns_batching_scalars,
                    log2_ceil_usize(group.len()),
                ))[..group.len()],
            );

            // TODO opti
            let sub_evals = fold_multilinear(
                &batched_column,
                &MultilinearPoint(outer_sumcheck_challenge[1..].to_vec()),
            );

            prover_state.add_extension_scalars(&sub_evals);
            all_sub_evals.push(sub_evals);
        }

        let epsilons = MultilinearPoint(prover_state.sample_vec(self.univariate_skips));

        let point = [epsilons.0.clone(), outer_sumcheck_challenge[1..].to_vec()].concat();

        let mut evaluations_remaining_to_prove = vec![];

        for (group, sub_evals) in witness.column_groups.iter().zip(all_sub_evals) {
            evaluations_remaining_to_prove.push(Evaluation {
                point: MultilinearPoint(
                    [
                        from_end(&columns_batching_scalars, log2_ceil_usize(group.len())).to_vec(),
                        point.clone(),
                    ]
                    .concat(),
                ),
                value: sub_evals.evaluate(&epsilons),
            });
        }
        evaluations_remaining_to_prove
    }

    #[instrument(skip_all)]
    fn open_structured_columns<'a>(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        witness: &AirWitness<'a, PF<EF>>,
        outer_sumcheck_challenge: &[EF],
    ) -> Vec<Evaluation<EF>> {
        let columns_batching_scalars = prover_state.sample_vec(witness.log_max_columns_per_group());
        let alpha = prover_state.sample();

        let mut all_inner_mles = vec![];
        let mut all_inner_sums = vec![];
        let mut all_batched_columns = vec![];
        let mut all_batched_columns_mixed = vec![];
        for group in &witness.column_groups {
            let batched_column = multilinears_linear_combination(
                &witness.cols[group.clone()],
                &eval_eq(&from_end(
                    &columns_batching_scalars,
                    log2_ceil_usize(group.len()),
                ))[..group.len()],
            );
            all_batched_columns.push(batched_column.clone());
            let batched_column_mixed = add_multilinears(
                &column_up(&batched_column),
                &scale_poly(&column_down(&batched_column), alpha),
            );
            all_batched_columns_mixed.push(batched_column_mixed.clone());

            // TODO opti
            let sub_evals = fold_multilinear(
                &batched_column_mixed,
                &MultilinearPoint(outer_sumcheck_challenge[1..].to_vec()),
            );

            prover_state.add_extension_scalars(&sub_evals);
        }

        let epsilons = prover_state.sample_vec(self.univariate_skips);

        for (batched_column, batched_column_mixed) in all_batched_columns
            .into_iter()
            .zip(all_batched_columns_mixed)
        {
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

            all_inner_mles.push(MleGroupOwned::Extension(mles_for_inner_sumcheck));
            all_inner_sums.push(inner_sum);
        }
        let n_groups = witness.column_groups.len();
        let (inner_challenges, all_inner_evals, _) = sumcheck::prove_in_parallel_1::<EF, _, _>(
            vec![1; n_groups],
            all_inner_mles,
            vec![&ProductComputation; n_groups],
            vec![&[]; n_groups],
            vec![None; n_groups],
            vec![false; n_groups],
            prover_state,
            all_inner_sums,
            vec![None; n_groups],
            true,
        );

        let mut evaluations = vec![];
        for i in 0..n_groups {
            let group = &witness.column_groups[i];
            let point = MultilinearPoint(
                [
                    from_end(&columns_batching_scalars, log2_ceil_usize(group.len())).to_vec(),
                    inner_challenges.0.clone(),
                ]
                .concat(),
            );
            let value = all_inner_evals[i][1];
            evaluations.push(Evaluation { point, value });
        }
        evaluations
    }
}
