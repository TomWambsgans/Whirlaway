use algebra::{
    ntt::wavelet_transform,
    pols::{MultilinearPolynomial, UnivariatePolynomial},
};
use p3_field::Field;
use {
    rayon::{join, prelude::*},
    std::mem::size_of,
};

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
pub struct CoefficientList<F> {
    coeffs: Vec<F>, // list of coefficients. For multilinear polynomials, we have coeffs.len() == 1 << num_variables.
    num_variables: usize, // number of variables
}

impl<F: Field> CoefficientList<F> {
    /// Evaluate the given polynomial at `point` from F^n.
    pub fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(self.num_variables, point.len());
        eval_multivariate(&self.coeffs, point)
    }

    /// Interprets self as a univariate polynomial (with coefficients of X^i in order of ascending i) and evaluates it at each point in `points`.
    /// We return the vector of evaluations.
    ///
    /// NOTE: For the `usual` mapping between univariate and multilinear polynomials, the coefficient ordering is such that
    /// for a single point x, we have (extending notation to a single point)
    /// self.evaluate_at_univariate(x) == self.evaluate([x^(2^n), x^(2^{n-1}), ..., x^2, x])
    pub fn evaluate_at_univariate(&self, points: &[F]) -> Vec<F> {
        // DensePolynomial::from_coefficients_slice converts to a dense univariate polynomial.
        // The coefficient order is "coefficient of 1 first".
        let univariate = UnivariatePolynomial::new(self.coeffs.clone()); // TODO: Avoid clone
        points.iter().map(|point| univariate.eval(point)).collect()
    }

    pub fn new(coeffs: Vec<F>) -> Self {
        let len = coeffs.len();
        assert!(len.is_power_of_two());
        let num_variables = len.ilog2();

        CoefficientList {
            coeffs,
            num_variables: num_variables as usize,
        }
    }

    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    pub fn num_coeffs(&self) -> usize {
        self.coeffs.len()
    }

    pub fn reverse_vars(&self) -> Self {
        let mut coeffs = vec![F::ZERO; self.coeffs.len()];
        coeffs.par_iter_mut().enumerate().for_each(|(i, coeff)| {
            let j = switch_endianness(i, self.num_variables);
            *coeff = self.coeffs[j];
        });
        CoefficientList {
            coeffs,
            num_variables: self.num_variables,
        }
    }
}

fn switch_endianness(mut x: usize, n: usize) -> usize {
    let mut y = 0;
    for _ in 0..n {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

/// Multivariate evaluation in coefficient form.
fn eval_multivariate<F: Field>(coeffs: &[F], point: &[F]) -> F {
    debug_assert_eq!(coeffs.len(), 1 << point.len());
    match point {
        [] => coeffs[0],
        [x] => coeffs[0] + coeffs[1] * *x,
        [x0, x1] => {
            let b0 = coeffs[0] + coeffs[1] * *x1;
            let b1 = coeffs[2] + coeffs[3] * *x1;
            b0 + b1 * *x0
        }
        [x0, x1, x2] => {
            let b00 = coeffs[0] + coeffs[1] * *x2;
            let b01 = coeffs[2] + coeffs[3] * *x2;
            let b10 = coeffs[4] + coeffs[5] * *x2;
            let b11 = coeffs[6] + coeffs[7] * *x2;
            let b0 = b00 + b01 * *x1;
            let b1 = b10 + b11 * *x1;
            b0 + b1 * *x0
        }
        [x0, x1, x2, x3] => {
            let b000 = coeffs[0] + coeffs[1] * *x3;
            let b001 = coeffs[2] + coeffs[3] * *x3;
            let b010 = coeffs[4] + coeffs[5] * *x3;
            let b011 = coeffs[6] + coeffs[7] * *x3;
            let b100 = coeffs[8] + coeffs[9] * *x3;
            let b101 = coeffs[10] + coeffs[11] * *x3;
            let b110 = coeffs[12] + coeffs[13] * *x3;
            let b111 = coeffs[14] + coeffs[15] * *x3;
            let b00 = b000 + b001 * *x2;
            let b01 = b010 + b011 * *x2;
            let b10 = b100 + b101 * *x2;
            let b11 = b110 + b111 * *x2;
            let b0 = b00 + b01 * *x1;
            let b1 = b10 + b11 * *x1;
            b0 + b1 * *x0
        }
        [x, tail @ ..] => {
            let (b0t, b1t) = coeffs.split_at(coeffs.len() / 2);
            let (b0t, b1t) = {
                let work_size: usize = (1 << 15) / size_of::<F>();
                if coeffs.len() > work_size {
                    join(
                        || eval_multivariate(b0t, tail),
                        || eval_multivariate(b1t, tail),
                    )
                } else {
                    (eval_multivariate(b0t, tail), eval_multivariate(b1t, tail))
                }
            };
            b0t + b1t * *x
        }
    }
}

impl<F: Field> CoefficientList<F> {
    /// fold folds the polynomial at the provided folding_randomness.
    ///
    /// Namely, when self is interpreted as a multi-linear polynomial f in X_0, ..., X_{n-1},
    /// it partially evaluates f at the provided `folding_randomness`.
    /// Our ordering convention is to evaluate at the higher indices, i.e. we return f(X_0,X_1,..., folding_randomness[0], folding_randomness[1],...)
    pub fn fold(&self, folding_randomness: &[F]) -> Self {
        let folding_factor = folding_randomness.len();
        let coeffs = self
            .coeffs
            .par_chunks_exact(1 << folding_factor)
            .map(|coeffs| eval_multivariate(coeffs, &folding_randomness))
            .collect();

        CoefficientList {
            coeffs,
            num_variables: self.num_variables() - folding_factor,
        }
    }

    pub fn into_evals(self) -> MultilinearPolynomial<F> {
        let mut evals = self.coeffs;
        wavelet_transform(&mut evals);
        MultilinearPolynomial::new(evals)
    }
}
