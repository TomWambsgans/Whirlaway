use multi_pcs::pcs::PCS;
use p3_air::Air;
use p3_field::{BasedVectorSpace, ExtensionField, TwoAdicField, cyclic_subgroup_known_order};
use sumcheck::ProductComputation;
use tracing::{Level, info_span, instrument, span};
use utils::{
    ConstraintFolder, ConstraintFolderPackedBase, Evaluation, FSProver, add_multilinears,
    multilinears_linear_combination, packed_multilinear,
};
use utils::{ConstraintFolderPackedExtension, PF};
use whir_p3::fiat_shamir::FSChallenger;
use whir_p3::{
    dft::EvalsDft,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
};

use crate::{
    AirSettings,
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
    A: for<'a> Air<ConstraintFolder<'a, PF<EF>, EF>>
        + for<'a> Air<ConstraintFolder<'a, EF, EF>>
        + for<'a> Air<ConstraintFolderPackedBase<'a, EF>>
        + for<'a> Air<ConstraintFolderPackedExtension<'a, EF>>,
{
    #[instrument(name = "air: prove", skip_all)]
    pub fn prove(
        &self,
        settings: &AirSettings,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        witness: Vec<EvaluationsList<PF<EF>>>,
        pcs: &impl PCS<PF<EF>, EF>,
        dft: &EvalsDft<PF<EF>>,
    ) where
        PF<EF>: TwoAdicField,
    {
        assert!(
            settings.univariate_skips < self.log_length,
            "TODO handle the case UNIVARIATE_SKIPS >= log_length"
        );
        let log_length = self.log_length;
        assert!(witness.iter().all(|w| w.num_variables() == log_length));

        // 1) Commit to the witness columns

        // TODO avoid cloning (use a row major matrix for the witness)

        let packed_pol = packed_multilinear(&witness);

        let ext_dim = <EF as BasedVectorSpace<PF<EF>>>::DIMENSION;
        assert!(ext_dim.is_power_of_two());

        let packed_witness = pcs.commit(&dft, prover_state, &packed_pol);

        let constraints_batching_scalar = prover_state.sample();

        let constraints_batching_scalars =
            cyclic_subgroup_known_order(constraints_batching_scalar, self.n_constraints)
                .collect::<Vec<_>>();

        let mut zerocheck_challenges = vec![EF::ZERO; log_length + 1 - settings.univariate_skips];
        for challenge in &mut zerocheck_challenges {
            *challenge = prover_state.sample();
        }

        let preprocessed_and_witness = self
            .preprocessed_columns
            .iter()
            .chain(&witness)
            .collect::<Vec<_>>();

        let my_columns_up_and_down = columns_up_and_down(&preprocessed_and_witness);
        let my_columns_up_and_down = my_columns_up_and_down
            .iter()
            .map(|m| m.as_slice())
            .collect::<Vec<_>>();

        let (zerocheck_challenges, all_inner_sums, _) = info_span!("zerocheck").in_scope(|| {
            sumcheck::prove_base_packed::<EF, _>(
                settings.univariate_skips,
                my_columns_up_and_down,
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

        let _span = span!(Level::INFO, "inner sumchecks").entered();

        let inner_sums_up = &all_inner_sums[self.n_preprocessed_columns()..self.n_columns];
        let inner_sums_down = &all_inner_sums[self.n_columns + self.n_preprocessed_columns()..];

        prover_state.add_extension_scalars(&inner_sums_up);
        prover_state.add_extension_scalars(&inner_sums_down);

        let mut columns_batching_scalars = vec![EF::ZERO; self.log_n_witness_columns()];
        for challenge in &mut columns_batching_scalars {
            *challenge = prover_state.sample();
        }

        let batched_column = multilinears_linear_combination(
            &witness,
            &EvaluationsList::eval_eq(&columns_batching_scalars).evals()[..witness.len()],
        );

        let alpha = prover_state.sample();

        let batched_column_mixed = add_multilinears(
            &column_up(&batched_column),
            &EvaluationsList::new(column_down(&batched_column))
                .scale(alpha)
                .into_evals(),
        );

        // TODO opti
        let sub_evals =
            &batched_column_mixed.fold(&MultilinearPoint(zerocheck_challenges[1..].to_vec()));

        prover_state.add_extension_scalars(sub_evals);

        let mut epsilons = vec![EF::ZERO; settings.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = prover_state.sample();
        }

        let point = [epsilons.clone(), zerocheck_challenges[1..].to_vec()].concat();
        let mles_for_inner_sumcheck = vec![
            add_multilinears(
                &matrix_up_folded(&point),
                &matrix_down_folded(&point).scale(alpha),
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
                .map(|m| m.evals())
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

        std::mem::drop(_span);

        let statement = vec![Evaluation {
            point: MultilinearPoint(final_point),
            value: packed_value,
        }];
        pcs.open(&dft, prover_state, &statement, packed_witness, &packed_pol);
    }
}
