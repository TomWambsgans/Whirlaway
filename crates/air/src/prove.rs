use p3_air::Air;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, TwoAdicField, cyclic_subgroup_known_order,
};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use sumcheck::{SumcheckComputation, SumcheckComputationPacked, SumcheckGrinding};
use tracing::{Level, info_span, instrument, span};
use utils::{
    ConstraintFolder, ConstraintFolderPacked, add_multilinears, multilinears_linear_combination,
    packed_multilinear,
};
use utils::{PF, PFPacking};
use whir_p3::fiat_shamir::verifier::ChallengerState;
use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::prover::ProverState,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::writer::CommitmentWriter,
        prover::Prover,
        statement::{Statement, weights::Weights},
    },
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
    EF: TwoAdicField + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    A: for<'a> Air<ConstraintFolder<'a, PF<EF>, PF<EF>, EF>>
        + for<'a> Air<ConstraintFolder<'a, PF<EF>, EF, EF>>
        + for<'a> Air<ConstraintFolderPacked<'a, PF<EF>, EF>>,
{
    #[instrument(name = "air: prove", skip_all)]
    pub fn prove<H, C, Challenger, const DIGEST_ELEMS: usize>(
        &self,
        settings: &AirSettings,
        merkle_hash: H,
        merkle_compress: C,
        prover_state: &mut ProverState<PF<PF<EF>>, EF, Challenger>,
        witness: Vec<EvaluationsList<PF<EF>>>,
    ) where
        Challenger: FieldChallenger<PF<PF<EF>>> + GrindingChallenger<Witness = PF<PF<EF>>> + ChallengerState,
        H: CryptographicHasher<PFPacking<PF<EF>>, [PFPacking<PF<EF>>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<PF<EF>>, [PF<PF<EF>>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PFPacking<PF<EF>>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<PF<EF>>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<PF<EF>>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        PF<EF>: TwoAdicField,
        PF<EF>: ExtensionField<PF<PF<EF>>>,
        PF<PF<EF>>: TwoAdicField,
    {
        assert!(
            settings.univariate_skips < self.log_length,
            "TODO handle the case UNIVARIATE_SKIPS >= log_length"
        );
        let log_length = self.log_length;
        assert!(witness.iter().all(|w| w.num_variables() == log_length));

        let whir_params = self.build_whir_params(settings, merkle_hash, merkle_compress);

        // 1) Commit to the witness columns

        // TODO avoid cloning (use a row major matrix for the witness)

        let packed_pol = packed_multilinear(&witness);

        let committer = CommitmentWriter::new(&whir_params);

        let ext_dim = <EF as BasedVectorSpace<PF<EF>>>::DIMENSION;
        assert!(ext_dim.is_power_of_two());
        let dft = EvalsDft::new(
            1 << (self.log_n_witness_columns() + self.log_length + settings.whir_log_inv_rate
                - log2_strict_usize(ext_dim)),
        );

        let packed_witness = committer.commit(&dft, prover_state, packed_pol).unwrap();

        self.constraints_batching_pow(prover_state, settings)
            .unwrap();

        let constraints_batching_scalar = prover_state.sample();

        let constraints_batching_scalars =
            cyclic_subgroup_known_order(constraints_batching_scalar, self.n_constraints)
                .collect::<Vec<_>>();

        self.zerocheck_pow(prover_state, settings).unwrap();

        let mut zerocheck_challenges = vec![EF::ZERO; log_length + 1 - settings.univariate_skips];
        for challenge in &mut zerocheck_challenges {
            *challenge = prover_state.sample();
        }

        let preprocessed_and_witness = self
            .preprocessed_columns
            .iter()
            .chain(&witness)
            .collect::<Vec<_>>();
        let (zerocheck_challenges, all_inner_sums, _) = info_span!("zerocheck").in_scope(|| {
            sumcheck::prove::<PF<EF>, PF<EF>, EF, _, _, _>(
                settings.univariate_skips,
                &columns_up_and_down(&preprocessed_and_witness),
                &self.air,
                self.constraint_degree,
                &constraints_batching_scalars,
                Some(&zerocheck_challenges),
                true,
                prover_state,
                EF::ZERO,
                None,
                SumcheckGrinding::Auto {
                    security_bits: settings.security_bits,
                },
                None,
            )
        });

        let _span = span!(Level::INFO, "inner sumchecks").entered();

        let inner_sums_up = all_inner_sums[self.n_preprocessed_columns()..self.n_columns]
            .iter()
            .map(|s| s.evaluate::<EF>(&MultilinearPoint(vec![])))
            .collect::<Vec<_>>();
        let inner_sums_down = all_inner_sums[self.n_columns + self.n_preprocessed_columns()..]
            .iter()
            .map(|s| s.evaluate::<EF>(&MultilinearPoint(vec![])))
            .collect::<Vec<_>>();

        prover_state.add_extension_scalars(&inner_sums_up);
        prover_state.add_extension_scalars(&inner_sums_down);

        info_span!("pow grinding").in_scope(|| {
            self.secondary_sumchecks_batching_pow(prover_state, settings)
                .unwrap();
        });

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
            &column_down(&batched_column).scale(alpha),
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

        let (inner_challenges, inner_evals, _) = sumcheck::prove::<EF, EF, EF, _, _, _>(
            1,
            &mles_for_inner_sumcheck,
            &InnerSumcheckCircuit,
            2,
            &[EF::ONE],
            None,
            false,
            prover_state,
            inner_sum,
            None,
            SumcheckGrinding::Auto {
                security_bits: settings.security_bits,
            },
            None,
        );

        let final_point = [columns_batching_scalars.clone(), inner_challenges].concat();

        let packed_value = inner_evals[1].evaluate(&MultilinearPoint(vec![]));

        std::mem::drop(_span);

        let prover = Prover(&whir_params);

        let mut statement = Statement::new(final_point.len());
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(final_point)),
            packed_value,
        );
        prover
            .prove(&dft, prover_state, statement, packed_witness)
            .unwrap();
    }
}

pub struct InnerSumcheckCircuit;

impl<F: Field, EF: ExtensionField<F>> SumcheckComputation<F, EF, EF> for InnerSumcheckCircuit {
    fn eval(&self, point: &[EF], _: &[EF]) -> EF {
        point[0] * point[1]
    }
}

impl<F: Field, EF: ExtensionField<F>> SumcheckComputationPacked<F, EF> for InnerSumcheckCircuit {
    fn eval_packed(
        &self,
        _: &[<F as Field>::Packing],
        _: &[EF],
        _: &[Vec<F>],
    ) -> impl Iterator<Item = EF> + Send + Sync {
        // Unreachable
        std::iter::once(EF::ZERO)
    }
}
