use algebra::Multilinear;
use fiat_shamir::FsProver;
use p3_air::Air;
use p3_dft::Radix2DitParallel;
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField, dot_product,
};
use p3_keccak::KeccakF;
use rand::distr::{Distribution, StandardUniform};
use sumcheck::{SumcheckComputation, SumcheckGrinding};
use tracing::{Level, info_span, instrument, span};
use utils::{ConstraintFolder, powers};
use whir_p3::{
    fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::writer::CommitmentWriter,
        prover::Prover,
        statement::{Statement, weights::Weights},
    },
};

use crate::{
    AirSettings,
    uni_skip_utils::{
        matrix_down_folded_with_univariate_skips, matrix_up_folded_with_univariate_skips,
    },
    utils::columns_up_and_down,
};

use super::table::AirTable;

/* Multi Column CCS (SuperSpartan)

cf https://eprint.iacr.org/2023/552.pdf and https://solvable.group/posts/super-air/#fnref:1

*/

impl<
    'a,
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    A: Air<ConstraintFolder<'a, F, F, EF>> + Air<ConstraintFolder<'a, F, EF, EF>>,
> AirTable<F, EF, A>
{
    #[instrument(name = "air: prove", skip_all)]
    pub fn prove(
        &self,
        settings: &AirSettings,
        fs_prover: &mut FsProver,
        witness: Vec<Multilinear<F>>,
    ) -> ProverState<EF, F>
    where
        F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
        EF: ExtensionField<<EF as PrimeCharacteristicRing>::PrimeSubfield> + TwoAdicField + Ord,
        F::PrimeSubfield: TwoAdicField,
        StandardUniform: Distribution<EF>,
        StandardUniform: Distribution<F>,
    {
        assert!(
            settings.univariate_skips < self.log_length,
            "TODO handle the case UNIVARIATE_SKIPS >= log_length"
        );
        let log_length = self.log_length;
        assert!(witness.iter().all(|w| w.n_vars == log_length));

        let whir_params = self.build_whir_params(settings);
        let mut domainsep = DomainSeparator::new("🐎", KeccakF);
        domainsep.commit_statement(&whir_params);
        domainsep.add_whir_proof(&whir_params);
        let mut prover_state = domainsep.to_prover_state();

        // 1) Commit to the witness columns

        // TODO avoid cloning (use a row major matrix for the witness)

        let packed_pol = Multilinear::packed(&witness);

        let committer = CommitmentWriter::new(&whir_params);

        let dft_committer = Radix2DitParallel::default();

        let packed_witness = committer
            .commit(
                &dft_committer,
                &mut prover_state,
                EvaluationsList::new(packed_pol.evals),
            )
            .unwrap();

        self.constraints_batching_pow(fs_prover, settings).unwrap();

        let constraints_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];

        let constraints_batching_scalars = powers(constraints_batching_scalar, self.n_constraints);

        self.zerocheck_pow(fs_prover, settings).unwrap();

        let zerocheck_challenges =
            fs_prover.challenge_scalars::<EF>(log_length + 1 - settings.univariate_skips);

        let preprocessed_and_witness = self
            .preprocessed_columns
            .iter()
            .chain(&witness)
            .collect::<Vec<_>>();
        let (zerocheck_challenges, all_inner_sums, _) = info_span!("zerocheck").in_scope(|| {
            sumcheck::prove::<_, _, _, _, A>(
                settings.univariate_skips,
                &columns_up_and_down(&preprocessed_and_witness),
                &self.air,
                self.constraint_degree,
                &constraints_batching_scalars,
                Some(&zerocheck_challenges),
                true,
                fs_prover,
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
                .map(|s| s.evaluate::<EF>(&[]))
                .collect::<Vec<_>>()
        });
        fs_prover.add_scalars(&inner_sums);

        info_span!("pow grinding").in_scope(|| {
            self.secondary_sumchecks_batching_pow(fs_prover, settings)
                .unwrap();
        });
        let secondary_sumcheck_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];

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
                let sum = Multilinear::linear_combination(
                    &witness,
                    &expanded_scalars
                        [i * self.n_witness_columns()..(i + 1) * self.n_witness_columns()],
                )
                .add_dummy_starting_variables(settings.univariate_skips); // TODO this is not efficient
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
            let expanded = Multilinear::new(
                self.univariate_selectors
                    .iter()
                    .map(|s| s.eval(&zerocheck_challenges[0]))
                    .collect(),
            );
            let expanded = expanded.add_dummy_ending_variables(log_length);
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

        let (inner_challenges, _, _) = sumcheck::prove::<F, _, _, _, _>(
            1,
            &mles_for_inner_sumcheck,
            &InnerSumcheckCircuit,
            3,
            &[EF::ONE],
            None,
            false,
            fs_prover,
            inner_sum,
            None,
            SumcheckGrinding::Auto {
                security_bits: settings.security_bits,
            },
            None,
        );

        let values = info_span!("evaluating witness").in_scope(|| {
            Multilinear::batch_evaluate_in_large_field(
                &witness,
                &inner_challenges[settings.univariate_skips..],
            )
        });

        fs_prover.add_scalars(&values);

        let final_random_scalars = fs_prover.challenge_scalars::<EF>(self.log_n_witness_columns()); // PoW grinding required ?
        let final_point = [
            final_random_scalars.clone(),
            inner_challenges[settings.univariate_skips..].to_vec(),
        ]
        .concat();

        let packed_value = info_span!("final point").in_scope(|| {
            Multilinear::new(
                [
                    values,
                    EF::zero_vec((1 << self.log_n_witness_columns()) - self.n_witness_columns()),
                ]
                .concat(),
            )
            .evaluate(&final_random_scalars)
        });

        std::mem::drop(_span);

        //pcs.open(packed_pol_witness, vec![packed_eval], fs_prover);
        let prover = Prover(&whir_params);

        let dft_prover = Radix2DitParallel::default();

        let mut statement = Statement::new(final_point.len());
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(final_point)),
            packed_value,
        );
        prover
            .prove(&dft_prover, &mut prover_state, statement, packed_witness)
            .unwrap();

        prover_state
    }
}

pub struct InnerSumcheckCircuit;

impl<F: Field, EF: ExtensionField<F>> SumcheckComputation<F, EF, EF> for InnerSumcheckCircuit {
    fn eval(&self, point: &[EF], _: &[EF]) -> EF {
        point[4] * ((point[0] * point[2]) + (point[1] * point[3]))
    }
}
