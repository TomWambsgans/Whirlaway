use algebra::{field_utils::dot_product, utils::expand_randomness};
use fiat_shamir::{FsError, FsVerifier};
use p3_field::{ExtensionField, Field};
use pcs::{BatchSettings, PCS};
use sumcheck::SumcheckError;
use tracing::instrument;

use crate::N;

use super::{
    table::AirTable,
    utils::{global_constraint_degree, matrix_down_lde, matrix_up_lde},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AirVerifError {
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
        batching: &mut BatchSettings<F, EF, Pcs>,
        log_length: usize,
    ) -> Result<(), AirVerifError> {
        for boundary_condition in &self.boundary_conditions {
            if batching
                .receive_claim(
                    fs_verifier,
                    boundary_condition.col,
                    &boundary_condition.encode::<EF>(log_length).point,
                )
                .unwrap()
                != EF::from(boundary_condition.value)
            {
                return Err(AirVerifError::InvalidBoundaryCondition);
            }
        }

        let constraints_batching_scalar = fs_verifier.challenge_scalars::<EF>(1)[0];

        // tau_0, ..., tau_{log_m - 1}
        let zerocheck_challenges = fs_verifier.challenge_scalars::<EF>(log_length);

        let (sc_sum, outer_sumcheck_challenge, outer_sub_evals) =
            sumcheck::verify_zerocheck_with_univariate_skip::<F, EF>(
                fs_verifier,
                &zerocheck_challenges,
                global_constraint_degree(&self.constraints),
                log_length,
                &self.global_constraint(constraints_batching_scalar).into(),
                N,
            )?;
        if sc_sum != EF::ZERO {
            return Err(AirVerifError::SumMismatch);
        }

        let batching_scalar = fs_verifier.challenge_scalars::<EF>(1)[0];

        let (batched_inner_sum, inner_sumcheck_challenge) =
            sumcheck::verify::<EF>(fs_verifier, &vec![2; log_length], 0)?;

        if batched_inner_sum
            != dot_product(
                &outer_sub_evals,
                &expand_randomness(batching_scalar, self.n_columns * 2),
            )
        {
            return Err(AirVerifError::SumMismatch);
        }

        let lde_matrix_up = matrix_up_lde(log_length);
        let lde_matrix_down = matrix_down_lde(log_length);

        let mut batched_inner_value = EF::ZERO;
        let matrix_lde_point = [
            outer_sumcheck_challenge.clone(),
            inner_sumcheck_challenge.point.clone(),
        ]
        .concat();
        let up = lde_matrix_up.eval(&matrix_lde_point);
        let down = lde_matrix_down.eval(&matrix_lde_point);
        for u in 0..self.n_columns {
            let inner_value =
                batching.receive_claim(fs_verifier, u, &inner_sumcheck_challenge.point)?;
            batched_inner_value += inner_value
                * (batching_scalar.exp_u64(u as u64) * up
                    + batching_scalar.exp_u64((u + self.n_columns) as u64) * down);
        }
        if batched_inner_value != inner_sumcheck_challenge.value {
            return Err(AirVerifError::SumMismatch);
        }

        Ok(())
    }
}
