use super::{Multilinear, MultilinearDevice};
use crate::{
    ntt::{expand_from_coeff, restructure_evaluations, wavelet_transform},
    pols::MultilinearHost,
};
use cuda_bindings::cuda_restructure_evaluations;
use cuda_bindings::{
    cuda_eval_multilinear_in_monomial_basis, cuda_expanded_ntt,
    cuda_monomial_to_lagrange_basis_rev, cuda_whir_fold,
};
use cuda_engine::{HostOrDeviceBuffer, cuda_sync, memcpy_dtoh};
use cudarc::driver::CudaSlice;
use p3_field::{ExtensionField, Field, TwoAdicField};
use utils::switch_endianness;

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
    pub fn fold(&self, folding_randomness: &[F]) -> Self {
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
    fn eval_multivariate(coeffs: &[F], point: &[F]) -> F {
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
    pub fn fold(&self, folding_randomness: &[F]) -> Self {
        Self::new(cuda_whir_fold(&self.coeffs, folding_randomness))
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

    pub fn expand_from_coeff_and_restructure<PrimeField: TwoAdicField>(
        &self,
        expansion: usize,
        domain_gen_inv: PrimeField,
        folding_factor: usize,
    ) -> HostOrDeviceBuffer<F>
    where
        F: ExtensionField<PrimeField>,
    {
        let _info =
            tracing::info_span!("expand_from_coeff_and_restructure", cuda = self.is_device())
                .entered();
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        match self {
            Self::Device(coeffs) => {
                let evals = cuda_expanded_ntt(coeffs.coeffs(), expansion);
                let folded_evals = cuda_restructure_evaluations(&evals, folding_factor);
                cuda_sync();
                HostOrDeviceBuffer::Device(folded_evals)
            }
            Self::Host(coeffs) => {
                let evals = expand_from_coeff::<PrimeField, F>(&coeffs.coeffs, expansion);
                let folded_evals = restructure_evaluations(evals, domain_gen_inv, folding_factor);
                HostOrDeviceBuffer::Host(folded_evals)
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

    pub fn fold(&self, folding_randomness: &[F]) -> Self {
        match self {
            Self::Host(coeffs) => Self::Host(coeffs.fold(folding_randomness)),
            Self::Device(coeffs) => Self::Device(coeffs.fold(folding_randomness)),
        }
    }

    /// Sync
    pub fn transfer_to_host(self) -> CoefficientListHost<F> {
        match self {
            Self::Host(coeffs) => coeffs,
            Self::Device(coeffs) => coeffs.transfer_to_host(),
        }
    }
}
