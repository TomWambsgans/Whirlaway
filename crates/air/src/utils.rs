use algebra::pols::Multilinear;
use arithmetic_circuit::{ArithmeticCircuit, TransparentPolynomial};
use fiat_shamir::{FsError, FsParticipant};
use p3_field::{Field, PrimeField};
use rayon::prelude::*;
use utils::log2_up;

use crate::{AirSettings, table::AirTable};

pub(crate) fn matrix_up_lde<F: Field>(log_length: usize) -> TransparentPolynomial<F> {
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

    TransparentPolynomial::eq_extension_2n_vars(log_length)
        + TransparentPolynomial::eq_extension_n_scalars(&vec![F::ONE; log_length * 2 - 1])
            * (ArithmeticCircuit::Scalar(F::ONE)
                - ArithmeticCircuit::Node(log_length * 2 - 1) * F::TWO)
}

pub(crate) fn matrix_down_lde<F: Field>(log_length: usize) -> TransparentPolynomial<F> {
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

    TransparentPolynomial::next(log_length)
        + TransparentPolynomial::eq_extension_n_scalars(&vec![F::ONE; log_length * 2])
    // bottom right corner
}

pub(crate) fn columns_up_and_down<F: Field>(columns: &[&Multilinear<F>]) -> Vec<Multilinear<F>> {
    let mut res = Vec::with_capacity(columns.len() * 2);
    res.par_extend(columns.par_iter().map(|c| column_up(c)));
    res.par_extend(columns.par_iter().map(|c| column_down(c)));
    res
}

pub(crate) fn column_up<F: Field>(column: &Multilinear<F>) -> Multilinear<F> {
    let mut up = column.clone();
    up.evals[column.n_coefs() - 1] = up.evals[column.n_coefs() - 2];
    up
}

pub(crate) fn column_down<F: Field>(column: &Multilinear<F>) -> Multilinear<F> {
    let mut down = column.evals[1..].to_vec();
    down.push(*down.last().unwrap());
    Multilinear::new(down)
}

impl<F: PrimeField> AirTable<F> {
    pub(crate) fn constraints_batching_pow<EF: Field, FS: FsParticipant>(
        &self,
        fs: &mut FS,
        settings: &AirSettings,
    ) -> Result<(), FsError> {
        fs.challenge_pow(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(log2_up(self.constraints.len()))),
        )
    }

    pub(crate) fn zerocheck_pow<EF: Field, FS: FsParticipant>(
        &self,
        fs: &mut FS,
        settings: &AirSettings,
    ) -> Result<(), FsError> {
        fs.challenge_pow(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(self.log_length)),
        )
    }

    pub(crate) fn secondary_sumchecks_batching_pow<EF: Field, FS: FsParticipant>(
        &self,
        fs: &mut FS,
        settings: &AirSettings,
    ) -> Result<(), FsError> {
        fs.challenge_pow(
            settings
                .security_bits
                .saturating_sub(EF::bits().saturating_sub(log2_up(self.n_witness_columns() * 2))),
        )
    }
}
