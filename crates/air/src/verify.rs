use algebra::{
    field_utils::{dot_product, eq_extension},
    utils::expand_randomness,
};
use fiat_shamir::{FsError, FsVerifier};
use p3_field::{ExtensionField, Field};
use pcs::{BatchSettings, PCS};
use sumcheck::SumcheckError;
use tracing::instrument;

use super::{
    table::AirTable,
    utils::{matrix_down_lde, matrix_up_lde, max_degree_per_vars_outer_sumcheck},
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

        let (sc_sum, outer_sumcheck_challenge) = sumcheck::verify::<EF>(
            fs_verifier,
            &max_degree_per_vars_outer_sumcheck(&self.constraints, log_length),
            0,
        )?;
        if sc_sum != EF::ZERO {
            return Err(AirVerifError::SumMismatch);
        }

        let inner_sums = fs_verifier.next_scalars::<EF>(2 * self.n_columns)?;

        let mut global_constraint_eval = EF::ZERO;
        for (scalar, constraint) in
            expand_randomness(constraints_batching_scalar, self.constraints.len())
                .into_iter()
                .zip(self.constraints.iter())
        {
            global_constraint_eval += scalar * constraint.expr.eval(&inner_sums);
        }
        if eq_extension(&zerocheck_challenges, &outer_sumcheck_challenge.point)
            * global_constraint_eval
            != outer_sumcheck_challenge.value
        {
            return Err(AirVerifError::SumMismatch);
        }

        let batching_scalar = fs_verifier.challenge_scalars::<EF>(1)[0];

        let (batched_inner_sum, inner_sumcheck_challenge) =
            sumcheck::verify::<EF>(fs_verifier, &vec![2; log_length], 0)?;

        if batched_inner_sum
            != dot_product(
                &inner_sums,
                &expand_randomness(batching_scalar, self.n_columns * 2),
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
