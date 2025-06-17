use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use rayon::prelude::*;
use utils::{eq_extension, log2_up};
use whir_p3::{
    fiat_shamir::{errors::ProofError, pow::blake3::Blake3PoW, prover::ProverState},
    poly::evals::EvaluationsList,
};

use crate::{AirSettings, MyChallenger, table::AirTable};

pub(crate) fn matrix_up_lde<F: Field>(point: &[F]) -> F {
    /*
        Matrix UP:

       (1 0 0 0 ... 0 0 0)
       (0 1 0 0 ... 0 0 0)
       (0 0 1 0 ... 0 0 0)
       (0 0 0 1 ... 0 0 0)
       ...      ...   ...
       (0 0 0 0 ... 1 0 0)
       (0 0 0 0 ... 0 1 0)
       (0 0 0 0 ... 0 1 0)

       Square matrix of size self.n_columns x sef.n_columns
       As a multilinear polynomial in 2 * log_length variables:
       - self.n_columns first variables -> encoding the row index
       - self.n_columns last variables -> encoding the column index
    */

    assert_eq!(point.len() % 2, 0);
    let n = point.len() / 2;
    eq_extension(&point[..n], &point[n..])
        + point[..point.len() - 1].iter().copied().product::<F>()
            * (F::ONE - point[point.len() - 1] * F::TWO)
}

pub(crate) fn matrix_down_lde<F: Field>(point: &[F]) -> F {
    /*
        Matrix DOWN:

       (0 1 0 0 ... 0 0 0)
       (0 0 1 0 ... 0 0 0)
       (0 0 0 1 ... 0 0 0)
       (0 0 0 0 ... 0 0 0)
       (0 0 0 0 ... 0 0 0)
       ...      ...   ...
       (0 0 0 0 ... 0 1 0)
       (0 0 0 0 ... 0 0 1)
       (0 0 0 0 ... 0 0 1)

       Square matrix of size self.n_columns x sef.n_columns
       As a multilinear polynomial in 2 * log_length variables:
       - self.n_columns first variables -> encoding the row index
       - self.n_columns last variables -> encoding the column index

       TODO OPTIMIZATIOn:
       the lde currently is in log(table_length)^2, but it could be log(table_length) using a recursive construction
       (However it is not representable as a polynomial in this case, but as a fraction instead)

    */
    next_mle(point) + point.iter().copied().product::<F>()

    // bottom right corner
}

fn next_mle<F: Field>(point: &[F]) -> F {
    // returns a polynomial P in 2n vars, where P(x, y) = 1 iif y = x + 1 in big indian (both numbers are n bits)
    assert!(point.len() % 2 == 0);
    let n = point.len() / 2;
    let factor = |l, r| point[l] * (F::ONE - point[r]);
    let g = |k| {
        let mut factors = vec![];
        for i in (n - k)..n {
            factors.push(factor(i, i + n));
        }
        factors.push(factor(2 * n - 1 - k, n - 1 - k));
        if k < n - 1 {
            factors.push(eq_extension(
                &(0..n - k - 1).map(|i| point[i]).collect::<Vec<_>>(),
                &(0..n - k - 1).map(|i| point[i + n]).collect::<Vec<_>>(),
            ));
        }
        factors.into_iter().product()
    };
    (0..n).map(g).sum()
}

pub(crate) fn columns_up_and_down<F: Field>(
    columns: &[&EvaluationsList<F>],
) -> Vec<EvaluationsList<F>> {
    columns
        .par_iter()
        .map(|c| column_up(c))
        .chain(columns.par_iter().map(|c| column_down(c)))
        .collect()
}

pub(crate) fn column_up<F: Field>(column: &EvaluationsList<F>) -> EvaluationsList<F> {
    let mut up = column.clone();
    up.evals_mut()[column.num_evals() - 1] = up.evals()[column.num_evals() - 2];
    up
}

pub(crate) fn column_down<F: Field>(column: &EvaluationsList<F>) -> EvaluationsList<F> {
    let mut down = column.evals()[1..].to_vec();
    down.push(*down.last().unwrap());
    EvaluationsList::new(down)
}

impl<F: TwoAdicField + PrimeField64, EF: ExtensionField<F> + TwoAdicField, A> AirTable<F, EF, A> {
    pub(crate) fn constraints_batching_pow(
        &self,
        fs: &mut ProverState<EF, F, MyChallenger, u8>,
        settings: &AirSettings,
    ) -> Result<(), ProofError> {
        fs.challenge_pow::<Blake3PoW>(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(log2_up(self.n_constraints)))
                as f64,
        )
    }

    pub(crate) fn zerocheck_pow(
        &self,
        fs: &mut ProverState<EF, F, MyChallenger, u8>,
        settings: &AirSettings,
    ) -> Result<(), ProofError> {
        fs.challenge_pow::<Blake3PoW>(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(self.log_length)) as f64,
        )
    }

    pub(crate) fn secondary_sumchecks_batching_pow(
        &self,
        fs: &mut ProverState<EF, F, MyChallenger, u8>,
        settings: &AirSettings,
    ) -> Result<(), ProofError> {
        fs.challenge_pow::<Blake3PoW>(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(log2_up(self.n_witness_columns() * 2)))
                as f64,
        )
    }
}
