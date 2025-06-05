use fiat_shamir::{FsError, FsVerifier};
use p3_air::Air;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField, dot_product};
use rand::distr::{Distribution, StandardUniform};
use sumcheck::{SumcheckComputation, SumcheckError, SumcheckGrinding};
use tracing::instrument;
use utils::{ConstraintFolder, eq_extension, fold_multilinear_in_large_field, powers};
use whir_p3::{
    fiat_shamir::{domain_separator::DomainSeparator, prover::ProverState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::reader::CommitmentReader,
        statement::{Statement, weights::Weights},
        verifier::Verifier,
    },
};

use crate::{
    AirSettings, MY_PERM_WIDTH, MyPerm, MySponge, MyU,
    utils::{column_down, column_up, matrix_down_lde, matrix_up_lde},
};

use super::table::AirTable;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AirVerifError {
    InvalidPcsCommitment,
    InvalidPcsOpening,
    Fs(FsError),
    Sumcheck(SumcheckError),
    InvalidBoundaryCondition,
    SumMismatch,
}

impl From<FsError> for AirVerifError {
    fn from(e: FsError) -> Self {
        Self::Fs(e)
    }
}

impl From<SumcheckError> for AirVerifError {
    fn from(e: SumcheckError) -> Self {
        Self::Sumcheck(e)
    }
}

impl<
    'a,
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    A: Air<ConstraintFolder<'a, F, EF, EF>>,
> AirTable<F, EF, A>
{
    #[instrument(name = "air table: verify", skip_all)]
    pub fn verify(
        &self,
        settings: &AirSettings,
        fs_verifier: &mut FsVerifier,
        log_length: usize,
        prover_state: ProverState<EF, F, MyPerm, MySponge, MyU, MY_PERM_WIDTH>,
    ) -> Result<(), AirVerifError>
    where
        StandardUniform: Distribution<EF> + Distribution<F>,
    {
        let whir_params = self.build_whir_params(settings);

        let commitment_reader = CommitmentReader::new(&whir_params);
        let whir_verifier = Verifier::new(&whir_params);
        let mut domainsep = DomainSeparator::new("🐎", MyPerm {});
        domainsep.commit_statement(&whir_params);
        domainsep.add_whir_proof(&whir_params);
        let mut verifier_state = domainsep.to_verifier_state::<_, 32>(prover_state.narg_string());
        let parsed_commitment = commitment_reader
            .parse_commitment::<32>(&mut verifier_state)
            .map_err(|_| AirVerifError::InvalidPcsCommitment)?;

        self.constraints_batching_pow(fs_verifier, settings)?;

        let constraints_batching_scalar = fs_verifier.challenge_scalars::<EF>(1)[0];

        self.zerocheck_pow(fs_verifier, settings).unwrap();

        let zerocheck_challenges =
            fs_verifier.challenge_scalars::<EF>(log_length - settings.univariate_skips + 1);

        let (sc_sum, outer_sumcheck_challenge) = sumcheck::verify_with_univariate_skip::<EF>(
            fs_verifier,
            self.constraint_degree + 1,
            log_length,
            settings.univariate_skips,
            SumcheckGrinding::Auto {
                security_bits: settings.security_bits,
            },
        )?;
        if sc_sum != EF::ZERO {
            return Err(AirVerifError::SumMismatch);
        }

        let witness_shifted_evals = fs_verifier.next_scalars::<EF>(2 * self.n_witness_columns())?;
        let (witness_up, witness_down) = witness_shifted_evals.split_at(self.n_witness_columns());
        let outer_selector_evals = self
            .univariate_selectors
            .iter()
            .map(|s| s.evaluate(outer_sumcheck_challenge.point[0]))
            .collect::<Vec<_>>();
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
            witness_up.to_vec(),
            preprocessed_down,
            witness_down.to_vec(),
        ]
        .concat();

        let global_constraint_eval = SumcheckComputation::eval(
            &self.air,
            &global_point,
            &powers(constraints_batching_scalar, self.n_constraints),
        );

        let zerocheck_selector_evals = self
            .univariate_selectors
            .iter()
            .map(|s| s.evaluate(zerocheck_challenges[0]));
        if dot_product::<EF, _, _>(
            zerocheck_selector_evals,
            outer_selector_evals.iter().copied(),
        ) * eq_extension(
            &zerocheck_challenges[1..],
            &outer_sumcheck_challenge.point[1..],
        ) * global_constraint_eval
            != outer_sumcheck_challenge.value
        {
            return Err(AirVerifError::SumMismatch);
        }

        self.secondary_sumchecks_batching_pow(fs_verifier, settings)?;
        let secondary_sumcheck_batching_scalar = fs_verifier.challenge_scalars::<EF>(1)[0];

        let (batched_inner_sum, inner_sumcheck_challenge) = sumcheck::verify::<EF>(
            fs_verifier,
            log_length + settings.univariate_skips,
            3,
            SumcheckGrinding::Auto {
                security_bits: settings.security_bits,
            },
        )?; // TODO degree 3 -> in fact it's degree 2 on some variables (sumcheck is sparse)

        if batched_inner_sum
            != dot_product(
                witness_shifted_evals.into_iter(),
                powers(
                    secondary_sumcheck_batching_scalar,
                    self.n_witness_columns() * 2,
                )
                .into_iter(),
            )
        {
            return Err(AirVerifError::SumMismatch);
        }

        let mut batched_inner_value = EF::ZERO;
        let matrix_lde_point = [
            inner_sumcheck_challenge.point[..settings.univariate_skips].to_vec(),
            outer_sumcheck_challenge.point[1..].to_vec(),
            inner_sumcheck_challenge.point[settings.univariate_skips..].to_vec(),
        ]
        .concat();
        let up = matrix_up_lde(&matrix_lde_point);
        let down = matrix_down_lde(&matrix_lde_point);

        let final_inner_claims = fs_verifier.next_scalars::<EF>(self.n_witness_columns())?;

        for (u, final_inner_claim) in final_inner_claims
            .iter()
            .enumerate()
            .take(self.n_witness_columns())
        {
            batched_inner_value += *final_inner_claim
                * (secondary_sumcheck_batching_scalar.exp_u64(u as u64) * up
                    + secondary_sumcheck_batching_scalar
                        .exp_u64((u + self.n_witness_columns()) as u64)
                        * down);
        }
        batched_inner_value *= EvaluationsList::new(outer_selector_evals).evaluate(
            &MultilinearPoint(inner_sumcheck_challenge.point[..settings.univariate_skips].to_vec()),
        );

        if batched_inner_value != inner_sumcheck_challenge.value {
            return Err(AirVerifError::SumMismatch);
        }

        let final_random_scalars =
            fs_verifier.challenge_scalars::<EF>(self.log_n_witness_columns());
        let final_point = [
            final_random_scalars.clone(),
            inner_sumcheck_challenge.point[settings.univariate_skips..].to_vec(),
        ]
        .concat();

        let packed_value = EvaluationsList::new(
            [
                final_inner_claims,
                vec![EF::ZERO; (1 << self.log_n_witness_columns()) - self.n_witness_columns()],
            ]
            .concat(),
        )
        .evaluate(&MultilinearPoint(final_random_scalars));

        let mut statement = Statement::<EF>::new(final_point.len());
        statement.add_constraint(
            Weights::evaluation(MultilinearPoint(final_point)),
            packed_value,
        );
        whir_verifier
            .verify(&mut verifier_state, &parsed_commitment, &statement)
            .map_err(|_| AirVerifError::InvalidPcsOpening)?;

        Ok(())
    }
}
