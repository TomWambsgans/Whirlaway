use crate::{pols::MultilinearHost, wavelet::wavelet_transform};
use p3_dft::TwoAdicSubgroupDft;

use cuda_bindings::{cuda_fold_rectangular_in_large_field, cuda_ntt};
use cuda_engine::{HostOrDeviceBuffer, cuda_sync, memcpy_dtoh};
use cudarc::driver::CudaSlice;
use p3_dft::Radix2DitParallel;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use utils::{default_hash, expanded_point_for_multilinear_monomial_evaluation};

use {rayon::join, std::mem::size_of};

/*

Multilinear polynomials are represented as a list of coefficients in the monomial basis.

*/

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct CoefficientListHost<F> {
    pub coeffs: Vec<F>, // with 3 variables, coeffs associated to: 1, X_3, X_2, X_2.X_3, X_1, X_1.X_3, X_1.X_2, X_1.X_2.X_3,
    pub n_vars: usize,  // number of variables
}

impl<F: Field> CoefficientListHost<F> {
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        assert_eq!(self.n_vars, point.len());
        Self::eval_multivariate(&self.coeffs, point)
    }

    pub fn new(coeffs: Vec<F>) -> Self {
        let len = coeffs.len();
        assert!(len.is_power_of_two());
        let num_variables = len.ilog2();

        CoefficientListHost {
            coeffs,
            n_vars: num_variables as usize,
        }
    }

    pub fn n_coefs(&self) -> usize {
        self.coeffs.len()
    }

    pub fn reverse_vars(&self) -> Self {
        let mut matrix = RowMajorMatrix::new_col(self.coeffs.clone());
        reverse_matrix_index_bits(&mut matrix);
        let coeffs = matrix.values;
        CoefficientListHost {
            coeffs,
            n_vars: self.n_vars,
        }
    }

    pub fn to_lagrange_basis(self) -> MultilinearHost<F> {
        let mut evals = self.coeffs;
        wavelet_transform(&mut evals);
        MultilinearHost::new(evals)
    }

    pub fn whir_fold<EF: ExtensionField<F>>(
        &self,
        folding_randomness: &[EF],
    ) -> CoefficientListHost<EF> {
        CoefficientListHost::new(
            MultilinearHost::new(self.coeffs.clone())
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

#[derive(Clone)]
pub struct CoefficientListDevice<F: Field> {
    pub coeffs: CudaSlice<F>,
    pub n_vars: usize,
}

impl<F: Field> CoefficientListDevice<F> {
    pub fn new(coeffs: CudaSlice<F>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        let n_vars = coeffs.len().ilog2() as usize;
        Self { coeffs, n_vars }
    }

    /// Sync
    pub fn transfer_to_host(&self) -> CoefficientListHost<F> {
        let res = CoefficientListHost::new(memcpy_dtoh(&self.coeffs));
        cuda_sync();
        res
    }

    pub fn coeffs(&self) -> &CudaSlice<F> {
        &self.coeffs
    }

    pub fn n_coefs(&self) -> usize {
        self.coeffs.len()
    }

    /// fold folds the polynomial at the provided folding_randomness.
    ///
    /// Namely, when self is interpreted as a multi-linear polynomial f in X_0, ..., X_{n-1},
    /// it partially evaluates f at the provided `folding_randomness`.
    /// Our ordering convention is to evaluate at the higher indices, i.e. we return f(X_0,X_1,..., folding_randomness[0], folding_randomness[1],...)
    ///
    /// Async
    pub fn whir_fold<EF: ExtensionField<F>>(
        &self,
        folding_randomness: &[EF],
    ) -> CoefficientListDevice<EF> {
        CoefficientListDevice::new(
            cuda_fold_rectangular_in_large_field(
                &[&self.coeffs],
                &expanded_point_for_multilinear_monomial_evaluation(folding_randomness),
            )
            .pop()
            .unwrap(),
        )
    }

    // Async
    pub fn expand_from_coeff_and_restructure(
        &self,
        expansion: usize,
        folding_factor: usize,
    ) -> CudaSlice<F>
    where
        F: TwoAdicField + Ord,
    {
        assert!(expansion.is_power_of_two());
        let expanded_size = self.n_coefs() * expansion;
        let log_expanded_size = expanded_size.trailing_zeros() as usize;

        let _span =
            tracing::info_span!("cuda_ntt", log_expanded_size = log_expanded_size).entered();
        let res = cuda_ntt(
            &self.coeffs,
            log_expanded_size - folding_factor,
            vec![(folding_factor, log_expanded_size - folding_factor)],
            Some(expansion.ilog2() as usize),
        );
        cuda_sync();
        res
    }
}

#[derive(Clone)]
pub enum CoefficientList<F: Field> {
    Host(CoefficientListHost<F>),
    Device(CoefficientListDevice<F>),
}

impl<F: Field> CoefficientList<F> {
    pub fn n_coefs(&self) -> usize {
        match self {
            Self::Host(pol) => pol.n_coefs(),
            Self::Device(pol) => pol.n_coefs(),
        }
    }

    pub fn n_vars(&self) -> usize {
        match self {
            Self::Host(pol) => pol.n_vars,
            Self::Device(pol) => pol.n_vars,
        }
    }

    // pub fn evaluate<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
    //     match self {
    //         Self::Host(pol) => pol.evaluate(point),
    //         Self::Device(_) => unimplemented!(),
    //     }
    // }

    pub fn is_device(&self) -> bool {
        matches!(self, Self::Device(_))
    }

    pub fn is_host(&self) -> bool {
        matches!(self, Self::Host(_))
    }

    pub fn expand_from_coeff_and_restructure(
        &self,
        expansion: usize,
        folding_factor: usize,
    ) -> HostOrDeviceBuffer<F>
    where
        F: TwoAdicField + Ord,
    {
        match self {
            Self::Device(coeffs) => HostOrDeviceBuffer::Device(
                coeffs.expand_from_coeff_and_restructure(expansion, folding_factor),
            ),
            Self::Host(coeffs) => HostOrDeviceBuffer::Host(
                coeffs.expand_from_coeff_and_restructure(expansion, folding_factor),
            ),
        }
    }

    pub fn whir_fold<EF: ExtensionField<F>>(
        &self,
        folding_randomness: &[EF],
    ) -> CoefficientList<EF> {
        match self {
            Self::Host(coeffs) => CoefficientList::Host(coeffs.whir_fold(folding_randomness)),
            Self::Device(coeffs) => CoefficientList::Device(coeffs.whir_fold(folding_randomness)),
        }
    }

    /// Sync
    pub fn transfer_to_host(self) -> CoefficientListHost<F> {
        match self {
            Self::Host(coeffs) => coeffs,
            Self::Device(coeffs) => coeffs.transfer_to_host(),
        }
    }

    /// Debug purpose
    /// Sync
    pub fn hash(&self) -> u64 {
        match self {
            Self::Host(coeffs) => default_hash(&coeffs.coeffs),
            Self::Device(coeffs) => default_hash(&coeffs.transfer_to_host().coeffs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_koala_bear::KoalaBear;

    type F = KoalaBear;

    #[test]
    fn test_reverse_vars_two_vars() {
        // Two variables → 4 coefficients: c0, c1, c2, c3
        let coeffs = vec![F::new(0), F::new(1), F::new(2), F::new(3)];
        let poly = CoefficientListHost::new(coeffs.clone());
        let reversed = poly.reverse_vars();

        // Bit reversal (00→00, 01→10, 10→01, 11→11)
        // Should reorder: 0, 2, 1, 3
        let expected = vec![F::new(0), F::new(2), F::new(1), F::new(3)];
        assert_eq!(reversed.coeffs, expected);

        // Original must remain unchanged
        assert_eq!(poly.coeffs, coeffs);
    }

    #[test]
    fn test_reverse_vars_single_element() {
        let coeffs = vec![F::new(42)];
        let poly = CoefficientListHost::new(coeffs.clone());
        let reversed = poly.reverse_vars();

        assert_eq!(reversed.coeffs, coeffs);
    }

    #[test]
    fn test_reverse_vars_three_vars() {
        // Three variables → 8 coefficients
        let coeffs = (0..8).map(F::new).collect::<Vec<_>>();
        let poly = CoefficientListHost::new(coeffs.clone());
        let reversed = poly.reverse_vars();

        // Bit reversal on 3 bits:
        // 000→000, 001→100, 010→010, 011→110, 100→001, 101→101, 110→011, 111→111
        let expected = vec![
            F::new(0), // 0
            F::new(4), // 1
            F::new(2), // 2
            F::new(6), // 3
            F::new(1), // 4
            F::new(5), // 5
            F::new(3), // 6
            F::new(7), // 7
        ];
        assert_eq!(reversed.coeffs, expected);
    }
}
