use algebra::pols::MultilinearHost;
use fiat_shamir::{FsError, FsVerifier};
use p3_field::{ExtensionField, Field};
use pcs::PCS;
use sumcheck::SumcheckError;
use tracing::instrument;
use utils::{Evaluation, dot_product, eq_extension, powers};

use crate::utils::{column_down, column_up};

use super::{
    table::AirTable,
    utils::{matrix_down_lde, matrix_up_lde},
};

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
        AirVerifError::Fs(e)
    }
}

impl From<SumcheckError> for AirVerifError {
    fn from(e: SumcheckError) -> Self {
        AirVerifError::Sumcheck(e)
    }
}

impl<F: Field> AirTable<F> {
    #[instrument(name = "air table: verify", skip_all)]
    pub fn verify<EF: ExtensionField<F>, Pcs: PCS<F, EF>>(
        &self,
        fs_verifier: &mut FsVerifier,
        pcs: &Pcs,
        log_length: usize,
    ) -> Result<(), AirVerifError> {
        let pcs_commitment = pcs
            .parse_commitment(fs_verifier)
            .map_err(|_| AirVerifError::InvalidPcsCommitment)?;

        let constraints_batching_scalar = fs_verifier.challenge_scalars::<EF>(1)[0];

        // tau_0, ..., tau_{log_m - 1}
        let zerocheck_challenges = fs_verifier.challenge_scalars::<EF>(log_length);

        let max_degree_per_vars = self
            .constraints
            .iter()
            .map(|c| c.composition_degree)
            .max_by_key(|d| *d)
            .unwrap();
        let (sc_sum, outer_sumcheck_challenge) =
            sumcheck::verify::<EF>(fs_verifier, &vec![max_degree_per_vars + 1; log_length], 0)?;
        if sc_sum != EF::ZERO {
            return Err(AirVerifError::SumMismatch);
        }

        let witness_shifted_evals = fs_verifier.next_scalars::<EF>(2 * self.n_witness_columns())?;
        let witness_up = &witness_shifted_evals[..self.n_witness_columns()];
        let witness_down = &witness_shifted_evals[self.n_witness_columns()..];
        let preprocessed_up = self
            .preprocessed_columns
            .iter()
            .map(|c| column_up(c).evaluate(&outer_sumcheck_challenge.point))
            .collect::<Vec<_>>();
        let preprocessed_down = self
            .preprocessed_columns
            .iter()
            .map(|c| column_down(c).evaluate(&outer_sumcheck_challenge.point))
            .collect::<Vec<_>>();

        let global_point = [
            preprocessed_up,
            witness_up.to_vec(),
            preprocessed_down,
            witness_down.to_vec(),
        ]
        .concat();

        let mut global_constraint_eval = EF::ZERO;
        for (scalar, constraint) in powers(constraints_batching_scalar, self.constraints.len())
            .into_iter()
            .zip(&self.constraints)
        {
            global_constraint_eval += scalar * constraint.eval(&global_point);
        }
        if eq_extension(&zerocheck_challenges, &outer_sumcheck_challenge.point)
            * global_constraint_eval
            != outer_sumcheck_challenge.value
        {
            return Err(AirVerifError::SumMismatch);
        }

        let inner_sumcheck_batching_scalar = fs_verifier.challenge_scalars::<EF>(1)[0];

        let (batched_inner_sum, inner_sumcheck_challenge) =
            sumcheck::verify::<EF>(fs_verifier, &vec![2; log_length], 0)?;

        if batched_inner_sum
            != dot_product(
                &witness_shifted_evals,
                &powers(inner_sumcheck_batching_scalar, self.n_witness_columns() * 2),
            )
        {
            return Err(AirVerifError::SumMismatch);
        }

        let lde_matrix_up = matrix_up_lde(log_length);
        let lde_matrix_down = matrix_down_lde(log_length);

        let mut batched_inner_value = EF::ZERO;
        let matrix_lde_point = [
            outer_sumcheck_challenge.point.clone(),
            inner_sumcheck_challenge.point.clone(),
        ]
        .concat();
        let up = lde_matrix_up.eval(&matrix_lde_point);
        let down = lde_matrix_down.eval(&matrix_lde_point);

        let final_inner_claims = fs_verifier.next_scalars::<EF>(self.n_witness_columns())?;

        for u in 0..self.n_witness_columns() {
            batched_inner_value += final_inner_claims[u]
                * (inner_sumcheck_batching_scalar.exp_u64(u as u64) * up
                    + inner_sumcheck_batching_scalar
                        .exp_u64((u + self.n_witness_columns()) as u64)
                        * down);
        }
        if batched_inner_value != inner_sumcheck_challenge.value {
            return Err(AirVerifError::SumMismatch);
        }

        let final_random_scalars =
            fs_verifier.challenge_scalars::<EF>(self.log_n_witness_columns());
        let final_point = [final_random_scalars.clone(), inner_sumcheck_challenge.point].concat();

        let packed_value = MultilinearHost::new(
            [
                final_inner_claims,
                vec![EF::ZERO; (1 << self.log_n_witness_columns()) - self.n_witness_columns()],
            ]
            .concat(),
        )
        .evaluate(&final_random_scalars);
        let packed_eval = Evaluation {
            point: final_point,
            value: packed_value,
        };

        pcs.verify(&pcs_commitment, &packed_eval, fs_verifier)
            .map_err(|_| AirVerifError::InvalidPcsOpening)?;

        Ok(())
    }
}
