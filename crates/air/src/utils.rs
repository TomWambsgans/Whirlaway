use algebra::pols::{ArithmeticCircuit, GenericTransparentMultivariatePolynomial};

use p3_field::Field;

use crate::table::AirConstraint;

pub(crate) fn matrix_up_lde<F: Field>(
    log_length: usize,
) -> GenericTransparentMultivariatePolynomial<F> {
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

    GenericTransparentMultivariatePolynomial::new(
        GenericTransparentMultivariatePolynomial::eq_extension_2n_vars(log_length).coefs
            + GenericTransparentMultivariatePolynomial::eq_extension_n_scalars(&vec![
                F::ONE;
                log_length * 2
                    - 1
            ])
            .coefs
                * (ArithmeticCircuit::Scalar(F::ONE)
                    - ArithmeticCircuit::Node(log_length * 2 - 1) * F::TWO),
        log_length * 2,
    )
}

pub(crate) fn matrix_down_lde<F: Field>(
    log_length: usize,
) -> GenericTransparentMultivariatePolynomial<F> {
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

    GenericTransparentMultivariatePolynomial::new(
        GenericTransparentMultivariatePolynomial::next(log_length).coefs
            + GenericTransparentMultivariatePolynomial::eq_extension_n_scalars(&vec![
                F::ONE;
                log_length * 2
            ])
            .coefs, // bottom right corner
        log_length * 2,
    )
}

pub(crate) fn max_degree_per_vars_outer_sumcheck<F: Field>(
    global_constraint: &Vec<AirConstraint<F>>,
    log_length: usize,
) -> Vec<usize> {
    let circuit_degree = global_constraint
        .iter()
        .map(|cst| {
            GenericTransparentMultivariatePolynomial::new(
                cst.expr
                    .coefs
                    .map_node(&|_| ArithmeticCircuit::<F, _>::Node(0)),
                1,
            )
            .max_degree_per_vars()[0]
        })
        .max_by_key(|x| *x)
        .unwrap();
    vec![1 + circuit_degree; log_length]
}
