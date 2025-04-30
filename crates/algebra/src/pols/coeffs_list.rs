use super::{Multilinear, MultilinearDevice};
use crate::{pols::MultilinearHost, wavelet::wavelet_transform};
use p3_dft::TwoAdicSubgroupDft;

use cuda_bindings::cuda_transpose;
use cuda_bindings::{
    cuda_eval_multilinear_in_monomial_basis, cuda_monomial_to_lagrange_basis_rev, cuda_ntt,
    cuda_whir_fold,
};
use cuda_engine::{HostOrDeviceBuffer, cuda_alloc_zeros, cuda_sync, memcpy_dtod_to, memcpy_dtoh};
use cudarc::driver::CudaSlice;
use p3_dft::Radix2DitParallel;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use utils::{default_hash, switch_endianness};

use {
    rayon::{join, prelude::*},
    std::mem::size_of,
};

/*

Multilinear polynomials are represented as a list of coefficients in the monomial basis.

*/

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct CoefficientListHost<F> {
    pub coeffs: Vec<F>, // with 3 variables, coeffs associated to: 1, X_3, X_2, X_2.X_3, X_1, X_1.X_3, X_1.X_2, X_1.X_2.X_3,
    pub n_vars: usize,  // number of variables
}

impl<F: Field> CoefficientListHost<F> {
    /// Evaluate the given polynomial at `point` from F^n.
    pub fn evaluate(&self, point: &[F]) -> F {
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
        let mut coeffs = vec![F::ZERO; self.coeffs.len()];
        coeffs.par_iter_mut().enumerate().for_each(|(i, coeff)| {
            let j = switch_endianness(i, self.n_vars);
            *coeff = self.coeffs[j];
        });
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

    // pub fn reverse_vars_and_get_evals(self) -> MultilinearPolynomial<F> {
    //     // reverse_vars_and_get_evals() is the same as reverse_vars().into_evals()
    //     let mut evals = self.coeffs;
    //     let n_vars = self.num_variables;
    //     for step in 0..n_vars {
    //         let new_evals = vec![F::ZERO; 1 << n_vars];
    //         let half_size = 1 << (n_vars - step - 1);
    //         (0..1 << (n_vars - 1)).into_par_iter().for_each(|i| {
    //             let x = (i / half_size) * 2 * half_size;
    //             let y = i % half_size;
    //             let left = evals[x + 2 * y];
    //             let right = evals[x + 2 * y + 1];
    //             unsafe {
    //                 *(new_evals.as_ptr().add(x + y) as *mut F) = left;
    //                 *(new_evals.as_ptr().add(x + y + half_size) as *mut F) = left + right;
    //             }
    //         });
    //         evals = new_evals;
    //     }

    //     MultilinearPolynomial::new(evals)
    // }

    /// fold folds the polynomial at the provided folding_randomness.
    ///
    /// Namely, when self is interpreted as a multi-linear polynomial f in X_0, ..., X_{n-1},
    /// it partially evaluates f at the provided `folding_randomness`.
    /// Our ordering convention is to evaluate at the higher indices, i.e. we return f(X_0,X_1,..., folding_randomness[0], folding_randomness[1],...)
    pub fn whir_fold<EF: ExtensionField<F>>(
        &self,
        folding_randomness: &[EF],
    ) -> CoefficientListHost<EF> {
        let folding_factor = folding_randomness.len();
        let coeffs = self
            .coeffs
            .par_chunks_exact(1 << folding_factor)
            .map(|coeffs| Self::eval_multivariate(coeffs, &folding_randomness))
            .collect();

        CoefficientListHost {
            coeffs,
            n_vars: self.n_vars - folding_factor,
        }
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
}

#[derive(Clone)]
pub struct CoefficientListDevice<F: Field> {
    pub coeffs: CudaSlice<F>,
    pub n_vars: usize,
}

impl<F: Field> CoefficientListDevice<F> {
    /// Evaluate the given polynomial at `point` from F^n.
    pub fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(self.n_vars, point.len());
        cuda_eval_multilinear_in_monomial_basis(&self.coeffs, point)
    }

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

    /// Async
    pub fn to_lagrange_basis_rev(&self) -> MultilinearDevice<F> {
        let evals = cuda_monomial_to_lagrange_basis_rev(&self.coeffs);
        MultilinearDevice::new(evals)
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
        CoefficientListDevice::new(cuda_whir_fold(&self.coeffs, folding_randomness))
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

    pub fn evaluate(&self, point: &[F]) -> F {
        match self {
            Self::Host(pol) => pol.evaluate(point),
            Self::Device(pol) => pol.evaluate(point),
        }
    }

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
        let _info =
            tracing::info_span!("expand_from_coeff_and_restructure", cuda = self.is_device())
                .entered();
        match self {
            Self::Device(coeffs) => {
                let expanded_size = coeffs.n_coefs() * expansion;
                let log_expanded_size = expanded_size.trailing_zeros() as u32;
                let mut extended_coeffs = cuda_alloc_zeros::<F>(expanded_size);
                memcpy_dtod_to(
                    &coeffs.coeffs,
                    &mut extended_coeffs.slice_mut(..coeffs.n_coefs()),
                );

                let mut transposed = cuda_transpose(
                    &extended_coeffs,
                    log_expanded_size - folding_factor as u32,
                    folding_factor as u32,
                );

                cuda_ntt(&mut transposed, log_expanded_size as usize - folding_factor);

                let res = cuda_transpose(
                    &transposed,
                    folding_factor as u32,
                    log_expanded_size - folding_factor as u32,
                );

                cuda_sync();
                HostOrDeviceBuffer::Device(res)
            }
            Self::Host(coeffs) => {
                let mut extended_coeffs = coeffs.coeffs.clone();
                extended_coeffs.resize(coeffs.n_coefs() * expansion, F::ZERO);
                // TODO preprocess twiddles
                HostOrDeviceBuffer::Host(
                    Radix2DitParallel::<F>::default()
                        .dft_batch(RowMajorMatrix::new(extended_coeffs, 1 << folding_factor))
                        // Get natural order of rows.
                        .to_row_major_matrix()
                        .values,
                )
            }
        }
    }

    // convert to lagrange basis, and reverse vars
    pub fn to_lagrange_basis_rev(&self) -> Multilinear<F> {
        match self {
            Self::Host(coeffs) => Multilinear::Host(coeffs.reverse_vars().to_lagrange_basis()),
            Self::Device(coeffs) => Multilinear::Device(coeffs.to_lagrange_basis_rev()),
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
