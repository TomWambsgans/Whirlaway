use std::ops::AddAssign;

use crate::{
    CoefficientListCuda, MultilinearPolynomialCuda, VecOrCudaSlice, cuda_expanded_ntt,
    cuda_restructure_evaluations, cuda_sync,
};
use algebra::{
    ntt::{expand_from_coeff, restructure_evaluations},
    pols::{CoefficientList, MultilinearPolynomial},
    tensor_algebra::TensorAlgebra,
};
use p3_field::{ExtensionField, Field, TwoAdicField};
use tracing::instrument;

#[derive(Clone, Debug)]
pub enum MultilinearPolynomialMaybeCuda<F: Field> {
    Cpu(MultilinearPolynomial<F>),
    Cuda(MultilinearPolynomialCuda<F>),
}

impl<F: Field> From<MultilinearPolynomial<F>> for MultilinearPolynomialMaybeCuda<F> {
    fn from(pol: MultilinearPolynomial<F>) -> Self {
        Self::Cpu(pol)
    }
}

impl<F: Field> From<MultilinearPolynomialCuda<F>> for MultilinearPolynomialMaybeCuda<F> {
    fn from(pol: MultilinearPolynomialCuda<F>) -> Self {
        Self::Cuda(pol)
    }
}

impl<F: Field> MultilinearPolynomialMaybeCuda<F> {
    pub fn n_coefs(&self) -> usize {
        match self {
            Self::Cpu(pol) => pol.n_coefs(),
            Self::Cuda(pol) => pol.n_coefs(),
        }
    }

    pub fn n_vars(&self) -> usize {
        match self {
            Self::Cpu(pol) => pol.n_vars,
            Self::Cuda(pol) => pol.n_vars,
        }
    }

    pub fn zero(n_vars: usize, cuda: bool) -> Self {
        if cuda {
            Self::Cuda(MultilinearPolynomialCuda::zero(n_vars))
        } else {
            Self::Cpu(MultilinearPolynomial::zero(n_vars))
        }
    }

    pub fn eq_mle(point: &[F], cuda: bool) -> Self {
        if cuda {
            Self::Cuda(MultilinearPolynomialCuda::eq_mle(point))
        } else {
            Self::Cpu(MultilinearPolynomial::eq_mle(point))
        }
    }

    pub fn scale_in_place(&mut self, scalar: F) {
        match self {
            Self::Cpu(pol) => *pol = pol.scale(scalar),
            Self::Cuda(pol) => pol.scale_in_place(scalar),
        }
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    pub fn as_cuda(&self) -> &MultilinearPolynomialCuda<F> {
        match self {
            Self::Cuda(pol) => pol,
            Self::Cpu(_) => panic!(""),
        }
    }

    pub fn as_cpu(&self) -> &MultilinearPolynomial<F> {
        match self {
            Self::Cpu(pol) => pol,
            Self::Cuda(_) => panic!(""),
        }
    }

    // Async if cuda
    pub fn as_coefs(&self) -> CoefficientListMaybeCuda<F> {
        match self {
            Self::Cpu(pol) => CoefficientListMaybeCuda::Cpu(pol.clone().as_coefs()),
            Self::Cuda(pol) => CoefficientListMaybeCuda::Cuda(pol.as_coeffs()),
        }
    }

    // Sync
    pub fn packed<EF: ExtensionField<F>>(&self) -> MultilinearPolynomialMaybeCuda<EF> {
        match self {
            Self::Cpu(pol) => MultilinearPolynomialMaybeCuda::Cpu(pol.packed()),
            Self::Cuda(pol) => MultilinearPolynomialMaybeCuda::Cuda(pol.packed()),
        }
    }

    pub fn eval_mixed_tensor<SubF: Field>(&self, point: &[F]) -> TensorAlgebra<SubF, F>
    where
        F: ExtensionField<SubF>,
    {
        match self {
            Self::Cpu(pol) => pol.eval_mixed_tensor(point),
            Self::Cuda(pol) => pol.eval_mixed_tensor(point),
        }
    }

    pub fn eval(&self, point: &[F]) -> F {
        match self {
            Self::Cpu(pol) => pol.eval(point),
            Self::Cuda(pol) => pol.eval(point),
        }
    }
}

impl<F: Field> AddAssign<MultilinearPolynomialMaybeCuda<F>> for MultilinearPolynomialMaybeCuda<F> {
    fn add_assign(&mut self, other: Self) {
        match (self, other) {
            (Self::Cpu(pol), Self::Cpu(other)) => pol.add_assign(other),
            (Self::Cuda(pol), Self::Cuda(other)) => pol.add_assign(other),
            _ => unreachable!("Mixing CPU and CUDA polynomials is not supported"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum CoefficientListMaybeCuda<F: Field> {
    Cpu(CoefficientList<F>),
    Cuda(CoefficientListCuda<F>),
}

impl<F: Field> CoefficientListMaybeCuda<F> {
    pub fn n_coefs(&self) -> usize {
        match self {
            Self::Cpu(pol) => pol.n_coefs(),
            Self::Cuda(pol) => pol.n_coefs(),
        }
    }

    pub fn n_vars(&self) -> usize {
        match self {
            Self::Cpu(pol) => pol.n_vars(),
            Self::Cuda(pol) => pol.n_vars(),
        }
    }

    pub fn evaluate(&self, point: &[F]) -> F {
        match self {
            Self::Cpu(pol) => pol.evaluate(point),
            Self::Cuda(pol) => pol.evaluate(point),
        }
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    #[instrument(name = "expand_from_coeff_and_restructure", skip_all)]
    pub fn expand_from_coeff_and_restructure<PrimeField: TwoAdicField>(
        &self,
        expansion: usize,
        domain_gen_inv: PrimeField,
        folding_factor: usize,
    ) -> VecOrCudaSlice<F>
    where
        F: ExtensionField<PrimeField>,
    {
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        match self {
            Self::Cuda(coeffs) => {
                let evals = cuda_expanded_ntt(coeffs.coeffs(), expansion);
                let folded_evals = cuda_restructure_evaluations(&evals, folding_factor);
                cuda_sync();
                VecOrCudaSlice::Cuda(folded_evals)
            }
            Self::Cpu(coeffs) => {
                let evals = expand_from_coeff::<PrimeField, F>(coeffs.coeffs(), expansion);
                let folded_evals = restructure_evaluations(evals, domain_gen_inv, folding_factor);
                VecOrCudaSlice::Vec(folded_evals)
            }
        }
    }

    // convert to lagrange basis
    pub fn reverse_vars_and_get_evals(&self) -> MultilinearPolynomialMaybeCuda<F> {
        match self {
            Self::Cpu(coeffs) => {
                MultilinearPolynomialMaybeCuda::Cpu(coeffs.reverse_vars().into_evals())
            }
            Self::Cuda(coeffs) => {
                MultilinearPolynomialMaybeCuda::Cuda(coeffs.reverse_vars_and_get_evals())
            }
        }
    }

    pub fn fold(&self, folding_randomness: &[F]) -> Self {
        match self {
            Self::Cpu(coeffs) => Self::Cpu(coeffs.fold(folding_randomness)),
            Self::Cuda(coeffs) => Self::Cuda(coeffs.fold(folding_randomness)),
        }
    }

    pub fn convert_to_cpu_sync(self) -> CoefficientList<F> {
        match self {
            Self::Cpu(coeffs) => coeffs,
            Self::Cuda(coeffs) => coeffs.transfer_to_ram_sync(),
        }
    }
}
