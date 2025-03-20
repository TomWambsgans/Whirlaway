use algebra::pols::{ArithmeticCircuit, TransparentMultivariatePolynomial};
use fiat_shamir::FsParticipant;
use p3_field::Field;
use tracing::instrument;

use super::table::AirTable;

impl<F: Field> AirTable<F> {
    #[instrument(name = "get_global_constraint", skip_all)]
    pub(crate) fn get_global_constraint(
        &self,
        challenger: &mut impl FsParticipant,
    ) -> TransparentMultivariatePolynomial<F> {
        let constraints_batching_scalar = challenger.challenge_scalars::<F>(1)[0];
        TransparentMultivariatePolynomial::new(
            ArithmeticCircuit::new_sum(
                (0..self.constraints.len())
                    .map(|i| {
                        self.constraints[i].expr.coefs.clone()
                            * constraints_batching_scalar.exp_u64(i as u64)
                    })
                    .collect(),
            ),
            self.n_columns * 2,
        )
    }
}

pub(crate) fn matrix_up_lde<F: Field>(log_length: usize) -> TransparentMultivariatePolynomial<F> {
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

    TransparentMultivariatePolynomial::new(
        TransparentMultivariatePolynomial::eq_extension_2n_vars(log_length).coefs
            + TransparentMultivariatePolynomial::eq_extension_n_scalars(&vec![
                F::ONE;
                log_length * 2 - 1
            ])
            .coefs
                * (ArithmeticCircuit::Scalar(F::ONE)
                    - ArithmeticCircuit::Node(log_length * 2 - 1) * F::TWO),
        log_length * 2,
    )
}

pub(crate) fn matrix_down_lde<F: Field>(log_length: usize) -> TransparentMultivariatePolynomial<F> {
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

    TransparentMultivariatePolynomial::new(
        TransparentMultivariatePolynomial::next(log_length).coefs
            + TransparentMultivariatePolynomial::eq_extension_n_scalars(&vec![
                F::ONE;
                log_length * 2
            ])
            .coefs, // bottom right corner
        log_length * 2,
    )
}

pub(crate) fn max_degree_per_vars_outer_sumcheck<F: Field>(
    global_constraint: &TransparentMultivariatePolynomial<F>,
    log_length: usize,
) -> Vec<usize> {
    let circuit_degree = TransparentMultivariatePolynomial::new(
        global_constraint
            .coefs
            .clone()
            .map_node(&|_| ArithmeticCircuit::<F, _>::Node(0)),
        1,
    )
    .max_degree_per_vars()[0];
    vec![1 + circuit_degree; log_length]
}
