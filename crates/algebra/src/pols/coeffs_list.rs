use crate::{pols::Multilinear, wavelet::wavelet_transform};
use p3_dft::TwoAdicSubgroupDft;

use p3_dft::Radix2DitParallel;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use utils::{expanded_point_for_multilinear_monomial_evaluation, switch_endianness_vec};

use {rayon::join, std::mem::size_of};

/*

Multilinear polynomials are represented as a list of coefficients in the monomial basis.

*/

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct CoefficientList<F> {
    pub coeffs: Vec<F>, // with 3 variables, coeffs associated to: 1, X_3, X_2, X_2.X_3, X_1, X_1.X_3, X_1.X_2, X_1.X_2.X_3,
    pub n_vars: usize,  // number of variables
}

impl<F: Field> CoefficientList<F> {
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        assert_eq!(self.n_vars, point.len());
        Self::eval_multivariate(&self.coeffs, point)
    }

    pub fn new(coeffs: Vec<F>) -> Self {
        let len = coeffs.len();
        assert!(len.is_power_of_two());
        let num_variables = len.ilog2();

        CoefficientList {
            coeffs,
            n_vars: num_variables as usize,
        }
    }

    pub fn n_coefs(&self) -> usize {
        self.coeffs.len()
    }

    pub fn reverse_vars(&self) -> Self {
        CoefficientList {
            coeffs: switch_endianness_vec(&self.coeffs),
            n_vars: self.n_vars,
        }
    }

    pub fn to_lagrange_basis(self) -> Multilinear<F> {
        let mut evals = self.coeffs;
        wavelet_transform(&mut evals);
        Multilinear::new(evals)
    }

    pub fn whir_fold<EF: ExtensionField<F>>(
        &self,
        folding_randomness: &[EF],
    ) -> CoefficientList<EF> {
        CoefficientList::new(
            Multilinear::new(self.coeffs.clone())
                .fold_rectangular_in_large_field(
                    &expanded_point_for_multilinear_monomial_evaluation(folding_randomness),
                )
                .evals,
        )
    }

    /// Multivariate evaluation in coefficient form.
    fn eval_multivariate<EF: ExtensionField<F>>(coeffs: &[F], point: &[EF]) -> EF {
        debug_assert_eq!(coeffs.len(), 1 << point.len());
        match point {
            [] => EF::from(coeffs[0]),
            [x] => *x * coeffs[1] + coeffs[0],
            [x0, x1] => {
                let b0 = *x1 * coeffs[1] + coeffs[0];
                let b1 = *x1 * coeffs[3] + coeffs[2];
                *x0 * b1 + b0
            }
            [x0, x1, x2] => {
                let b00 = *x2 * coeffs[1] + coeffs[0];
                let b01 = *x2 * coeffs[3] + coeffs[2];
                let b10 = *x2 * coeffs[5] + coeffs[4];
                let b11 = *x2 * coeffs[7] + coeffs[6];
                let b0 = b00 + *x1 * b01;
                let b1 = b10 + *x1 * b11;
                *x0 * b1 + b0
            }
            [x0, x1, x2, x3] => {
                let b000 = *x3 * coeffs[1] + coeffs[0];
                let b001 = *x3 * coeffs[3] + coeffs[2];
                let b010 = *x3 * coeffs[5] + coeffs[4];
                let b011 = *x3 * coeffs[7] + coeffs[6];
                let b100 = *x3 * coeffs[9] + coeffs[8];
                let b101 = *x3 * coeffs[11] + coeffs[10];
                let b110 = *x3 * coeffs[13] + coeffs[12];
                let b111 = *x3 * coeffs[15] + coeffs[14];
                let b00 = b000 + *x2 * b001;
                let b01 = b010 + *x2 * b011;
                let b10 = b100 + *x2 * b101;
                let b11 = b110 + *x2 * b111;
                let b0 = b00 + *x1 * b01;
                let b1 = b10 + *x1 * b11;
                *x0 * b1 + b0
            }
            [x, tail @ ..] => {
                let (b0t, b1t) = coeffs.split_at(coeffs.len() / 2);
                let (b0t, b1t) = {
                    let work_size: usize = (1 << 15) / size_of::<F>();
                    if coeffs.len() > work_size {
                        join(
                            || Self::eval_multivariate(b0t, tail),
                            || Self::eval_multivariate(b1t, tail),
                        )
                    } else {
                        (
                            Self::eval_multivariate(b0t, tail),
                            Self::eval_multivariate(b1t, tail),
                        )
                    }
                };
                b0t + b1t * *x
            }
        }
    }

    pub fn expand_from_coeff_and_restructure(
        &self,
        expansion: usize,
        folding_factor: usize,
    ) -> Vec<F>
    where
        F: TwoAdicField + Ord,
    {
        let mut extended_coeffs = self.reverse_vars().coeffs;
        extended_coeffs.resize(self.n_coefs() * expansion, F::ZERO);
        // TODO preprocess twiddles
        Radix2DitParallel::<F>::default()
            .dft_batch(RowMajorMatrix::new(extended_coeffs, 1 << folding_factor))
            // Get natural order of rows.
            .to_row_major_matrix()
            .values
    }
}
