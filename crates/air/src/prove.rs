use multi_pcs::pcs::PCS;
use p3_air::Air;
use p3_field::PackedValue;
use p3_field::{ExtensionField, TwoAdicField, cyclic_subgroup_known_order};
use p3_uni_stark::SymbolicAirBuilder;
use sumcheck::ProductComputation;
use tracing::{Level, info_span, instrument, span};
use utils::{
    ConstraintFolder, ConstraintFolderPackedBase, Evaluation, FSProver, PFPacking,
    add_multilinears, multilinears_linear_combination, packed_multilinear,
};
use utils::{ConstraintFolderPackedExtension, PF};
use whir_p3::fiat_shamir::FSChallenger;
use whir_p3::poly::evals::{eval_eq, fold_multilinear, scale_poly};
use whir_p3::{
    dft::EvalsDft,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
};

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
    pub fn prove(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        witness: Vec<Vec<PF<EF>>>,
        pcs: &impl PCS<PF<EF>, EF>,
        dft: &EvalsDft<PF<EF>>,
    ) {
        assert!(
            self.univariate_skips < self.log_length,
            "TODO handle the case UNIVARIATE_SKIPS >= log_length"
        );
        let log_length = self.log_length;
        assert!(witness.iter().all(|w| w.num_variables() == log_length));

        // 1) Commit to the witness columns

        // TODO avoid cloning (use a row major matrix for the witness)

        let packed_pol = packed_multilinear(&witness);

        let packed_witness = pcs.commit(&dft, prover_state, &packed_pol);

        let constraints_batching_scalar = prover_state.sample();

        let constraints_batching_scalars =
            cyclic_subgroup_known_order(constraints_batching_scalar, self.n_constraints)
                .collect::<Vec<_>>();

        let mut zerocheck_challenges = vec![EF::ZERO; log_length + 1 - self.univariate_skips];
        for challenge in &mut zerocheck_challenges {
            *challenge = prover_state.sample();
        }

        let preprocessed_and_witness = self
            .preprocessed_columns
            .iter()
            .chain(&witness)
            .collect::<Vec<_>>();

        let columns_up_and_down_opt = if self.air.structured() {
            Some(columns_up_and_down(&preprocessed_and_witness))
        } else {
            None
        };

        let columns_for_zero_check = if self.air.structured() {
            columns_up_and_down_opt
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        } else {
            preprocessed_and_witness
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

        if self.air.structured() {
            self.open_structured_columns(
                prover_state,
                dft,
                packed_witness,
                &packed_pol,
                &witness,
                &all_inner_sums,
                &outer_sumcheck_challenge,
                pcs,
            );
        } else {
            self.open_unstructured_columns(
                prover_state,
                dft,
                packed_witness,
                &packed_pol,
                &witness,
                &all_inner_sums,
                &outer_sumcheck_challenge,
                pcs,
            );
        }
    }

    fn open_unstructured_columns<Pcs: PCS<PF<EF>, EF>>(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        dft: &EvalsDft<PF<EF>>,
        packed_witness: Pcs::Witness,
        packed_pol: &[PF<EF>],
        witness: &[Vec<PF<EF>>],
        all_inner_sums: &[EF],
        zerocheck_challenges: &[EF],
        pcs: &Pcs,
    ) {
        let span = span!(Level::INFO, "proving column MLEs for unstructured AIR").entered();
        prover_state.add_extension_scalars(&all_inner_sums[self.n_preprocessed_columns()..]);

        let mut columns_batching_scalars = vec![EF::ZERO; self.log_n_witness_columns()];
        for challenge in &mut columns_batching_scalars {
            *challenge = prover_state.sample();
        }

        let batched_column = multilinears_linear_combination(
            &witness,
            &eval_eq(&columns_batching_scalars)[..self.n_witness_columns()],
        );

        // TODO opti
        let sub_evals = fold_multilinear(
            &batched_column,
            &MultilinearPoint(zerocheck_challenges[1..].to_vec()),
        );

        prover_state.add_extension_scalars(&sub_evals);

        let mut epsilons = vec![EF::ZERO; self.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = prover_state.sample();
        }

        let point =
            MultilinearPoint([epsilons.clone(), zerocheck_challenges[1..].to_vec()].concat());

        let final_value = sub_evals.evaluate(&MultilinearPoint(epsilons));

        prover_state.add_extension_scalar(final_value);

        let statement = vec![Evaluation {
            point: MultilinearPoint([columns_batching_scalars, point.0].concat()),
            value: final_value,
        }];

        span.exit();

        pcs.open(&dft, prover_state, &statement, packed_witness, &packed_pol);
    }

    fn open_structured_columns<Pcs: PCS<PF<EF>, EF>>(
        &self,
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
        dft: &EvalsDft<PF<EF>>,
        packed_witness: Pcs::Witness,
        packed_pol: &[PF<EF>],
        witness: &[Vec<PF<EF>>],
        all_inner_sums: &[EF],
        zerocheck_challenges: &[EF],
        pcs: &Pcs,
    ) {
        let span = span!(
            Level::INFO,
            "proving column UP/DOWN MLEs for structured AIR"
        )
        .entered();

        let inner_sums_up = &all_inner_sums[self.n_preprocessed_columns()..self.n_columns()];
        let inner_sums_down = &all_inner_sums[self.n_columns() + self.n_preprocessed_columns()..];

        prover_state.add_extension_scalars(&inner_sums_up);
        prover_state.add_extension_scalars(&inner_sums_down);

        let mut columns_batching_scalars = vec![EF::ZERO; self.log_n_witness_columns()];
        for challenge in &mut columns_batching_scalars {
            *challenge = prover_state.sample();
        }

        let batched_column = multilinears_linear_combination(
            &witness,
            &eval_eq(&columns_batching_scalars)[..witness.len()],
        );

        let alpha = prover_state.sample();

        let batched_column_mixed = add_multilinears(
            &column_up(&batched_column),
            &scale_poly(&column_down(&batched_column), alpha),
        );

        // TODO opti
        let sub_evals = fold_multilinear(
            &batched_column_mixed,
            &MultilinearPoint(zerocheck_challenges[1..].to_vec()),
        );

        prover_state.add_extension_scalars(&sub_evals);

        let mut epsilons = vec![EF::ZERO; self.univariate_skips];
        for challenge in &mut epsilons {
            *challenge = prover_state.sample();
        }

        let point = [epsilons.clone(), zerocheck_challenges[1..].to_vec()].concat();
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

        span.exit();

        let statement = vec![Evaluation {
            point: MultilinearPoint(final_point),
            value: packed_value,
        }];
        pcs.open(&dft, prover_state, &statement, packed_witness, &packed_pol);
    }
}
