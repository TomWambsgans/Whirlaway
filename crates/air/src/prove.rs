use p3_air::Air;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField64, TwoAdicField, dot_product};
use p3_util::log2_strict_usize;
use rand::distr::{Distribution, StandardUniform};
use sumcheck::{SumcheckComputation, SumcheckGrinding};
use tracing::{Level, info_span, instrument, span};
use utils::{
    ConstraintFolder, add_dummy_ending_variables, add_dummy_starting_variables,
    multilinear_batch_evaluate, multilinears_linear_combination, packed_multilinear, powers,
};
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
    AirSettings, MyChallenger,
    uni_skip_utils::{
        matrix_down_folded_with_univariate_skips, matrix_up_folded_with_univariate_skips,
    },
    utils::columns_up_and_down,
};

use super::table::AirTable;

/* Multi Column CCS (SuperSpartan)

cf https://eprint.iacr.org/2023/552.pdf and https://solvable.group/posts/super-air/#fnref:1

*/

impl<'a, F, EF, A> AirTable<F, EF, A>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    A: Air<ConstraintFolder<'a, F, F, EF>> + Air<ConstraintFolder<'a, F, EF, EF>>,
{
    #[instrument(name = "air: prove", skip_all)]
    pub fn prove(
        &self,
        settings: &AirSettings,
        prover_state: &mut ProverState<EF, F, MyChallenger, u8>,
        witness: Vec<EvaluationsList<F>>,
    ) where
        StandardUniform: Distribution<EF> + Distribution<F>,
    {
        assert!(
            settings.univariate_skips < self.log_length,
            "TODO handle the case UNIVARIATE_SKIPS >= log_length"
        );
        let log_length = self.log_length;
        assert!(witness.iter().all(|w| w.num_variables() == log_length));

        let whir_params = self.build_whir_params(settings);

        // 1) Commit to the witness columns

        // TODO avoid cloning (use a row major matrix for the witness)

        let packed_pol = packed_multilinear(&witness);

        let committer = CommitmentWriter::new(&whir_params);

        let ext_dim = <EF as BasedVectorSpace<F>>::DIMENSION;
        assert!(ext_dim.is_power_of_two());
        let dft = EvalsDft::new(
            1 << (self.log_n_witness_columns() + self.log_length + settings.whir_log_inv_rate
                - log2_strict_usize(ext_dim)),
        );

        let packed_witness = committer.commit(&dft, prover_state, packed_pol).unwrap();

        self.constraints_batching_pow(prover_state, settings)
            .unwrap();

        let constraints_batching_scalar = prover_state.challenge_scalars_array::<1>().unwrap()[0];

        let constraints_batching_scalars = powers(constraints_batching_scalar, self.n_constraints);

        self.zerocheck_pow(prover_state, settings).unwrap();

        let zerocheck_challenges = prover_state
            .challenge_scalars_vec(log_length + 1 - settings.univariate_skips)
            .unwrap();

        let preprocessed_and_witness = self
            .preprocessed_columns
            .iter()
            .chain(&witness)
            .collect::<Vec<_>>();
        let (zerocheck_challenges, all_inner_sums, _) = info_span!("zerocheck").in_scope(|| {
            sumcheck::prove(
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

        let (inner_sums_up, inner_sums_down) =
            all_inner_sums.split_at(self.n_columns + self.n_preprocessed_columns());
        let inner_sums = info_span!("column evaluations").in_scope(|| {
            inner_sums_up
                .iter()
                .chain(inner_sums_down)
                .map(|s| s.evaluate::<EF>(&MultilinearPoint(vec![])))
                .collect::<Vec<_>>()
        });
        prover_state.add_scalars(&inner_sums).unwrap();

        info_span!("pow grinding").in_scope(|| {
            self.secondary_sumchecks_batching_pow(prover_state, settings)
                .unwrap();
        });
        let secondary_sumcheck_batching_scalar =
            prover_state.challenge_scalars_array::<1>().unwrap()[0];

        let mles_for_inner_sumcheck = {
            let _span_mles = span!(Level::INFO, "constructing MLEs").entered();
            let mut nodes = Vec::new();
            let _span_linear_comb = span!(Level::INFO, "linear combination of columns").entered();
            let expanded_scalars = powers(
                secondary_sumcheck_batching_scalar,
                2 * self.n_witness_columns(),
            );
            for i in 0..2 {
                // up and down
                let sum = add_dummy_starting_variables(
                    &multilinears_linear_combination(
                        &witness,
                        &expanded_scalars
                            [i * self.n_witness_columns()..(i + 1) * self.n_witness_columns()],
                    ),
                    settings.univariate_skips,
                ); // TODO this is not efficient
                nodes.push(sum);
            }

            std::mem::drop(_span_linear_comb);
            nodes.push(matrix_up_folded_with_univariate_skips(
                &zerocheck_challenges,
                settings.univariate_skips,
            ));
            nodes.push(matrix_down_folded_with_univariate_skips(
                &zerocheck_challenges,
                settings.univariate_skips,
            ));

            // TODO remove
            let expanded = EvaluationsList::new(
                self.univariate_selectors
                    .iter()
                    .map(|s| s.evaluate(zerocheck_challenges[0]))
                    .collect(),
            );
            let expanded = add_dummy_ending_variables(&expanded, log_length);
            nodes.push(expanded);

            nodes
        };

        let inner_sum = info_span!("dot product").in_scope(|| {
            dot_product(
                inner_sums.into_iter(),
                powers(
                    secondary_sumcheck_batching_scalar,
                    self.n_witness_columns() * 2,
                )
                .into_iter(),
            )
        });

        let (inner_challenges, _, _) = sumcheck::prove(
            1,
            &mles_for_inner_sumcheck,
            &InnerSumcheckCircuit,
            3,
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

        let values = info_span!("evaluating witness").in_scope(|| {
            multilinear_batch_evaluate(
                &witness,
                &MultilinearPoint(inner_challenges[settings.univariate_skips..].to_vec()),
            )
        });

        prover_state.add_scalars(&values).unwrap();

        let final_random_scalars = prover_state
            .challenge_scalars_vec(self.log_n_witness_columns())
            .unwrap(); // PoW grinding required ?
        let final_point = [
            final_random_scalars.clone(),
            inner_challenges[settings.univariate_skips..].to_vec(),
        ]
        .concat();

        let packed_value = info_span!("final point").in_scope(|| {
            EvaluationsList::new(
                [
                    values,
                    EF::zero_vec((1 << self.log_n_witness_columns()) - self.n_witness_columns()),
                ]
                .concat(),
            )
            .evaluate(&MultilinearPoint(final_random_scalars))
        });

        std::mem::drop(_span);

        //pcs.open(packed_pol_witness, vec![packed_eval], fs_prover);
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
        point[4] * ((point[0] * point[2]) + (point[1] * point[3]))
    }
}
