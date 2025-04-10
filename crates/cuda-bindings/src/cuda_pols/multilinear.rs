use std::ops::AddAssign;

use crate::{cuda_eval_multilinear_in_lagrange_basis, memcpy_dtoh};
use algebra::{pols::MultilinearPolynomial, tensor_algebra::TensorAlgebra};
use cudarc::driver::CudaSlice;
use p3_field::{BasedVectorSpace, ExtensionField, Field};

use crate::{
    cuda_add_assign_slices, cuda_add_slices, cuda_eq_mle, cuda_info,
    cuda_lagrange_to_monomial_basis, cuda_scale_slice_in_place, cuda_sync, memcpy_htod,
};

use super::CoefficientListCuda;

#[derive(Clone, Debug)]
pub struct MultilinearPolynomialCuda<F: Field> {
    pub n_vars: usize,
    pub evals: CudaSlice<F>, // [f(0, 0, ..., 0), f(0, 0, ..., 0, 1), f(0, 0, ..., 0, 1, 0), f(0, 0, ..., 0, 1, 1), ...]
}

impl<F: Field> MultilinearPolynomialCuda<F> {
    pub fn n_coefs(&self) -> usize {
        1 << self.n_vars
    }

    pub fn zero(n_vars: usize) -> Self {
        Self {
            n_vars,
            evals: memcpy_htod(&vec![F::ZERO; 1 << n_vars]), // TODO improve (not efficient)
        }
    }

    pub fn new(evals: CudaSlice<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        let n_vars = (evals.len() as f64).log2() as usize;
        Self { n_vars, evals }
    }

    // Async
    pub fn eq_mle(point: &[F]) -> Self {
        Self::new(cuda_eq_mle(point))
    }

    // Async
    pub fn scale_in_place(&mut self, scalar: F) {
        cuda_scale_slice_in_place(&mut self.evals, scalar);
    }

    // Async
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n_vars, other.n_vars);
        let res = cuda_add_slices(&self.evals, &other.evals);
        Self::new(res)
    }

    // Async
    pub fn as_coeffs(&self) -> CoefficientListCuda<F> {
        CoefficientListCuda::new(cuda_lagrange_to_monomial_basis(&self.evals))
    }

    pub fn packed<EF: ExtensionField<F>>(&self) -> MultilinearPolynomialCuda<EF> {
        let ext_degree = <EF as BasedVectorSpace<F>>::DIMENSION;
        assert!(ext_degree.is_power_of_two());
        assert!(self.n_coefs() >= ext_degree);
        let packed_evals = cuda_info()
            .stream
            .clone_dtod(&unsafe {
                self.evals
                    .transmute::<EF>(self.n_coefs() / ext_degree)
                    .unwrap()
            })
            .unwrap();
        cuda_sync();
        MultilinearPolynomialCuda::new(packed_evals)
    }

    pub fn transfer_to_ram_sync(&self) -> MultilinearPolynomial<F> {
        let res = MultilinearPolynomial::new(memcpy_dtoh(&self.evals));
        cuda_sync();
        res
    }

    pub fn eval_mixed_tensor<SubF: Field>(&self, point: &[F]) -> TensorAlgebra<SubF, F>
    where
        F: ExtensionField<SubF>,
    {
        // TODO implement in cuda to avoid gpu -> ram transfer
        self.transfer_to_ram_sync().eval_mixed_tensor(point)
    }

    // Async
    pub fn eval(&self, point: &[F]) -> F {
        assert_eq!(self.n_vars, point.len());
        cuda_eval_multilinear_in_lagrange_basis(&self.evals, point)
    }
}

impl<F: Field> AddAssign<MultilinearPolynomialCuda<F>> for MultilinearPolynomialCuda<F> {
    // Async
    fn add_assign(&mut self, other: Self) {
        assert_eq!(self.n_vars, other.n_vars);
        cuda_add_assign_slices(&mut self.evals, &other.evals);
    }
}
