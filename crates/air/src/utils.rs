use algebra::pols::{ArithmeticCircuit, TransparentPolynomial};

use p3_field::Field;

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
