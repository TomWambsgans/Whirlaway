use crate::{
    cuda_eval_multilinear_in_monomial_basis, cuda_monomial_to_lagrange_basis, cuda_sync,
    cuda_whir_fold, memcpy_dtoh,
};
use algebra::pols::CoefficientList;
use cudarc::driver::CudaSlice;
use p3_field::Field;

use super::MultilinearPolynomialCuda;

/// A CoefficientList models a (multilinear) polynomial in `num_variable` variables in coefficient form.
///
/// The order of coefficients follows the following convention: coeffs[j] corresponds to the monomial
/// determined by the binary decomposition of j with an X_i-variable present if the
/// i-th highest-significant bit among the `num_variables` least significant bits is set.
///
/// e.g. is `num_variables` is 3 with variables X_0, X_1, X_2, then
///  - coeffs[0] is the coefficient of 1
///  - coeffs[1] is the coefficient of X_2
///  - coeffs[2] is the coefficient of X_1
///  - coeffs[4] is the coefficient of X_0
#[derive(Debug, Clone)]
pub struct CoefficientListCuda<F: Field> {
    coeffs: CudaSlice<F>, // list of coefficients. For multilinear polynomials, we have coeffs.len() == 1 << num_variables.
    num_variables: usize, // number of variables
}

impl<F: Field> CoefficientListCuda<F> {
    /// Evaluate the given polynomial at `point` from F^n.
    pub fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(self.num_variables, point.len());
        cuda_eval_multilinear_in_monomial_basis(&self.coeffs, point)
    }

    pub fn new(coeffs: CudaSlice<F>) -> Self {
        let len = coeffs.len();
        assert!(len.is_power_of_two());
        let num_variables = len.ilog2();

        Self {
            coeffs,
            num_variables: num_variables as usize,
        }
    }

    pub fn transfer_to_ram_sync(&self) -> CoefficientList<F> {
        let res = CoefficientList::new(memcpy_dtoh(&self.coeffs));
        cuda_sync();
        res
    }

    pub fn coeffs(&self) -> &CudaSlice<F> {
        &self.coeffs
    }

    pub fn n_vars(&self) -> usize {
        self.num_variables
    }

    pub fn n_coefs(&self) -> usize {
        self.coeffs.len()
    }

    /// Async
    pub fn reverse_vars_and_get_evals(&self) -> MultilinearPolynomialCuda<F> {
        let evals = cuda_monomial_to_lagrange_basis(&self.coeffs);
        MultilinearPolynomialCuda::new(evals)
    }

    /// fold folds the polynomial at the provided folding_randomness.
    ///
    /// Namely, when self is interpreted as a multi-linear polynomial f in X_0, ..., X_{n-1},
    /// it partially evaluates f at the provided `folding_randomness`.
    /// Our ordering convention is to evaluate at the higher indices, i.e. we return f(X_0,X_1,..., folding_randomness[0], folding_randomness[1],...)
    pub fn fold(&self, folding_randomness: &[F]) -> Self {
        cuda_whir_fold(self, folding_randomness)
    }
}
