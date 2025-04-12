use super::{CoefficientList, CoefficientListDevice};
use super::{CoefficientListHost, UnivariatePolynomial};
use crate::tensor_algebra::TensorAlgebra;
use cuda_bindings::{
    cuda_add_assign_slices, cuda_add_slices, cuda_eq_mle, cuda_eval_multilinear_in_lagrange_basis,
    cuda_fix_variable_in_big_field, cuda_lagrange_to_monomial_basis, cuda_scale_slice,
    cuda_scale_slice_in_place,
};
use cuda_bindings::{cuda_fix_variable_in_small_field, cuda_sum_over_hypercube_of_computation};
use cuda_engine::{
    SumcheckComputation, clone_dtod, cuda_alloc_zeros, cuda_sync, memcpy_dtod, memcpy_dtoh,
    memcpy_htod,
};
use cudarc::driver::CudaSlice;
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use rayon::prelude::*;
use std::borrow::Borrow;
use std::ops::AddAssign;
use utils::{HypercubePoint, PartialHypercubePoint};

/*

Multlinear Polynomials represented as a vector of evaluations in the Lagrange basis.

*/

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct MultilinearHost<F: Field> {
    pub n_vars: usize,
    pub evals: Vec<F>, // [f(0, 0, ..., 0), f(0, 0, ..., 0, 1), f(0, 0, ..., 0, 1, 0), f(0, 0, ..., 0, 1, 1), ...]
}

impl<F: Field> MultilinearHost<F> {
    pub fn n_coefs(&self) -> usize {
        1 << self.n_vars
    }

    pub fn zero(n_vars: usize) -> Self {
        Self {
            n_vars,
            evals: vec![F::ZERO; 1 << n_vars],
        }
    }

    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.is_empty() || evals.len().is_power_of_two());
        let n_vars = (evals.len() as f64).log2() as usize;
        Self { n_vars, evals }
    }

    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        assert_eq!(self.n_vars, point.len());
        if self.n_vars == 0 {
            return EF::from(self.evals[0]);
        }
        let mut n = self.n_coefs();
        let mut buff = self.evals[..n / 2]
            .par_iter()
            .zip(&self.evals[n / 2..])
            .map(|(l, r)| point[0] * (*r - *l) + *l)
            .collect::<Vec<_>>();
        for p in &point[1..] {
            n /= 2;
            buff = buff[..n / 2]
                .par_iter()
                .zip(&buff[n / 2..])
                .map(|(l, r)| *p * (*r - *l) + *l)
                .collect();
        }
        buff[0]
    }

    pub fn packed<EF: ExtensionField<F>>(&self) -> MultilinearHost<EF> {
        let ext_degree = <EF as BasedVectorSpace<F>>::DIMENSION;
        assert!(ext_degree.is_power_of_two());
        assert!(self.n_coefs() >= ext_degree);
        let packed_evals = self
            .evals
            .chunks(ext_degree)
            .map(|chunk| EF::from_basis_coefficients_slice(chunk))
            .collect();
        MultilinearHost::new(packed_evals)
    }

    /// fix first variables
    pub fn fix_variable_in_big_field<EF: ExtensionField<F>>(&self, z: EF) -> MultilinearHost<EF> {
        let half = self.evals.len() / 2;
        let mut new_evals = vec![EF::ZERO; half];
        new_evals
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result)| {
                *result = z * (self.evals[i + half] - self.evals[i]) + self.evals[i];
            });
        MultilinearHost::new(new_evals)
    }

    /// fix first variables
    pub fn fix_variable_in_small_field<S: Field>(&self, z: S) -> MultilinearHost<F>
    where
        F: ExtensionField<S>,
    {
        let half = self.evals.len() / 2;
        let mut new_evals = vec![F::ZERO; half];
        new_evals
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result)| {
                *result = (self.evals[i + half] - self.evals[i]) * z + self.evals[i];
            });
        MultilinearHost::new(new_evals)
    }

    pub fn eval_mixed_tensor<SubF: Field>(&self, point: &[F]) -> TensorAlgebra<SubF, F>
    where
        F: ExtensionField<SubF>,
    {
        // returns φ1(self)(φ0(point[0]), φ0(point[1]), ...)
        assert_eq!(point.len(), self.n_vars);
        MultilinearHost::eq_mle(point)
            .evals
            .par_iter()
            .zip(&self.evals)
            .map(|(l, e)| TensorAlgebra::phi_0_times_phi_1(l, e))
            .sum()
    }

    pub fn to_monomial_basis(self) -> CoefficientListHost<F> {
        let mut coeffs = self.evals;
        let n = self.n_vars;

        // TODO parallelize
        for i in 0..n {
            let step = 1 << i;
            for j in 0..(1 << n) {
                if (j & step) == 0 {
                    let temp = coeffs[j];
                    coeffs[j | step] -= temp;
                }
            }
        }

        CoefficientListHost::new(coeffs)
    }

    pub fn as_univariate(self) -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new(self.to_monomial_basis().coeffs)
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
        let univariate = self.clone().as_univariate(); // TODO avoid clone
        points
            .iter()
            .map(|point| univariate.eval_parallel(point))
            .collect()
    }

    pub fn eval_hypercube(&self, point: &HypercubePoint) -> F {
        assert_eq!(self.n_vars, point.n_vars);
        self.evals[point.val]
    }

    pub fn eval_partial_hypercube(&self, point: &PartialHypercubePoint) -> F {
        assert_eq!(self.n_vars, point.n_vars());
        F::from_u32(point.left) * self.evals[point.right.val + (1 << (self.n_vars - 1))]
            + (F::ONE - F::from_u32(point.left)) * self.evals[point.right.val]
    }

    pub fn random<R: Rng>(rng: &mut R, n_vars: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self {
            n_vars,
            evals: (0..1 << n_vars).map(|_| rng.random()).collect(),
        }
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        vec![1; self.n_vars]
    }

    pub fn embed<EF: ExtensionField<F>>(&self) -> MultilinearHost<EF> {
        // TODO avoid
        let evals = self.evals.iter().map(|&e| EF::from(e)).collect();
        MultilinearHost::new(evals)
    }

    pub fn scale<EF: ExtensionField<F>>(&self, scalar: EF) -> MultilinearHost<EF> {
        let evals = self.evals.par_iter().map(|&e| scalar * e).collect();
        MultilinearHost::new(evals)
    }

    pub fn eq_mle(scalars: &[F]) -> Self {
        if scalars.len() <= 8 {
            Self::eq_mle_single_threaded(scalars)
        } else {
            Self::eq_mle_parallel(scalars)
        }
    }

    fn eq_mle_single_threaded(scalars: &[F]) -> Self {
        let mut evals = vec![F::ZERO; 1 << scalars.len()];
        evals[0] = F::ONE;
        for (i, &s) in scalars.iter().rev().enumerate() {
            let one_minus_s = F::ONE - s;
            for j in 0..1 << i {
                evals[(1 << i) + j] = evals[j] * s;
                evals[j] *= one_minus_s;
            }
        }
        Self::new(evals)
    }

    fn eq_mle_parallel(scalars: &[F]) -> Self {
        let mut evals = vec![F::ZERO; 1 << scalars.len()];
        evals[0] = F::ONE;
        for (i, &s) in scalars.iter().rev().enumerate() {
            let one_minus_s = F::ONE - s;
            let chunk_size = 1 << i;
            let (left, rest) = evals.split_at_mut(chunk_size);
            let right = &mut rest[..chunk_size];
            left.par_iter_mut()
                .zip(right.par_iter_mut())
                .for_each(|(l, r)| {
                    let tmp = *l * s;
                    *l = *l * one_minus_s;
                    *r = tmp;
                });
        }
        Self::new(evals)
    }

    // Async
    pub fn transfer_to_device(&self) -> MultilinearDevice<F> {
        MultilinearDevice::new(memcpy_htod(&self.evals))
    }
}

impl<F: Field> AddAssign<MultilinearHost<F>> for MultilinearHost<F> {
    fn add_assign(&mut self, other: MultilinearHost<F>) {
        assert_eq!(self.n_vars, other.n_vars);
        self.evals
            .par_iter_mut()
            .zip(other.evals.par_iter())
            .for_each(|(a, b)| *a += *b);
    }
}

#[derive(Clone)]
pub struct MultilinearDevice<F: Field> {
    pub n_vars: usize,
    pub evals: CudaSlice<F>, // [f(0, 0, ..., 0), f(0, 0, ..., 0, 1), f(0, 0, ..., 0, 1, 0), f(0, 0, ..., 0, 1, 1), ...]
}

impl<F: Field> MultilinearDevice<F> {
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
    pub fn to_monomial_basis(&self) -> CoefficientListDevice<F> {
        CoefficientListDevice::new(cuda_lagrange_to_monomial_basis(&self.evals))
    }

    // Sync
    pub fn packed<EF: ExtensionField<F>>(&self) -> MultilinearDevice<EF> {
        let ext_degree = <EF as BasedVectorSpace<F>>::DIMENSION;
        assert!(ext_degree.is_power_of_two());
        assert!(self.n_coefs() >= ext_degree);
        let packed_evals = clone_dtod(&unsafe {
            self.evals
                .transmute::<EF>(self.n_coefs() / ext_degree)
                .unwrap()
        });
        cuda_sync();
        MultilinearDevice::new(packed_evals)
    }

    /// Sync
    pub fn transfer_to_host(&self) -> MultilinearHost<F> {
        let res = MultilinearHost::new(memcpy_dtoh(&self.evals));
        cuda_sync();
        res
    }

    /// Async
    pub fn eval_mixed_tensor<SubF: Field>(&self, point: &[F]) -> TensorAlgebra<SubF, F>
    where
        F: ExtensionField<SubF>,
    {
        // TODO implement in cuda to avoid device -> host transfer
        self.transfer_to_host().eval_mixed_tensor(point)
    }

    // Async
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        assert_eq!(self.n_vars, point.len());
        cuda_eval_multilinear_in_lagrange_basis(&self.evals, point)
    }

    // TODO remove
    pub fn embed<EF: ExtensionField<F>>(&self) -> MultilinearDevice<EF> {
        let host_pol = self.transfer_to_host();
        cuda_sync();
        let embedded = host_pol.embed();
        MultilinearDevice::new(memcpy_htod(&embedded.evals))
    }
}

impl<F: Field> AddAssign<MultilinearDevice<F>> for MultilinearDevice<F> {
    // Async
    fn add_assign(&mut self, other: Self) {
        assert_eq!(self.n_vars, other.n_vars);
        cuda_add_assign_slices(&mut self.evals, &other.evals);
    }
}

impl<F: Field> Borrow<CudaSlice<F>> for &MultilinearDevice<F> {
    fn borrow(&self) -> &CudaSlice<F> {
        &self.evals
    }
}

impl<F: Field> Borrow<CudaSlice<F>> for MultilinearDevice<F> {
    fn borrow(&self) -> &CudaSlice<F> {
        &self.evals
    }
}

#[derive(Clone)]
pub enum Multilinear<F: Field> {
    Host(MultilinearHost<F>),
    Device(MultilinearDevice<F>),
}

impl<F: Field> From<MultilinearHost<F>> for Multilinear<F> {
    fn from(pol: MultilinearHost<F>) -> Self {
        Self::Host(pol)
    }
}

impl<F: Field> From<MultilinearDevice<F>> for Multilinear<F> {
    fn from(pol: MultilinearDevice<F>) -> Self {
        Self::Device(pol)
    }
}

impl<F: Field> Multilinear<F> {
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

    pub fn zero(n_vars: usize, on_device: bool) -> Self {
        if on_device {
            Self::Device(MultilinearDevice::zero(n_vars))
        } else {
            Self::Host(MultilinearHost::zero(n_vars))
        }
    }

    pub fn eq_mle(point: &[F], on_device: bool) -> Self {
        if on_device {
            Self::Device(MultilinearDevice::eq_mle(point))
        } else {
            Self::Host(MultilinearHost::eq_mle(point))
        }
    }

    pub fn scale_in_place(&mut self, scalar: F) {
        match self {
            Self::Host(pol) => *pol = pol.scale(scalar),
            Self::Device(pol) => pol.scale_in_place(scalar),
        }
    }

    pub fn scale<EF: ExtensionField<F>>(&self, scalar: EF) -> Multilinear<EF> {
        match self {
            Self::Host(pol) => Multilinear::Host(pol.scale(scalar)),
            Self::Device(pol) => {
                Multilinear::Device(MultilinearDevice::new(cuda_scale_slice(&pol.evals, scalar)))
            }
        }
    }

    pub fn is_device(&self) -> bool {
        matches!(self, Self::Device(_))
    }

    pub fn is_host(&self) -> bool {
        matches!(self, Self::Host(_))
    }

    pub fn as_device_ref(&self) -> &MultilinearDevice<F> {
        match self {
            Self::Device(pol) => pol,
            Self::Host(_) => panic!(""),
        }
    }

    pub fn as_host_ref(&self) -> &MultilinearHost<F> {
        match self {
            Self::Host(pol) => pol,
            Self::Device(_) => panic!(""),
        }
    }

    pub fn as_device(self) -> MultilinearDevice<F> {
        match self {
            Self::Device(pol) => pol,
            Self::Host(_) => panic!(),
        }
    }

    pub fn as_host(self) -> MultilinearHost<F> {
        match self {
            Self::Host(pol) => pol,
            Self::Device(_) => panic!(),
        }
    }

    // Async
    pub fn to_monomial_basis(&self) -> CoefficientList<F> {
        match self {
            Self::Host(pol) => CoefficientList::Host(pol.clone().to_monomial_basis()),
            Self::Device(pol) => CoefficientList::Device(pol.to_monomial_basis()),
        }
    }

    // Sync
    pub fn packed<EF: ExtensionField<F>>(&self) -> Multilinear<EF> {
        match self {
            Self::Host(pol) => Multilinear::Host(pol.packed()),
            Self::Device(pol) => Multilinear::Device(pol.packed()),
        }
    }

    /// Async
    pub fn eval_mixed_tensor<SubF: Field>(&self, point: &[F]) -> TensorAlgebra<SubF, F>
    where
        F: ExtensionField<SubF>,
    {
        match self {
            Self::Host(pol) => pol.eval_mixed_tensor(point),
            Self::Device(pol) => pol.eval_mixed_tensor(point),
        }
    }

    /// Async
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        match self {
            Self::Host(pol) => pol.evaluate(point),
            Self::Device(pol) => pol.evaluate(point),
        }
    }

    // TODO remove
    pub fn embed<EF: ExtensionField<F>>(&self) -> Multilinear<EF> {
        match self {
            Self::Host(pol) => Multilinear::Host(pol.embed()),
            Self::Device(pol) => Multilinear::Device(pol.embed()),
        }
    }
}

impl<F: Field> AddAssign<Multilinear<F>> for Multilinear<F> {
    fn add_assign(&mut self, other: Self) {
        match (self, other) {
            (Self::Host(pol), Self::Host(other)) => pol.add_assign(other),
            (Self::Device(pol), Self::Device(other)) => pol.add_assign(other),
            _ => unreachable!("Mixing CPU and GPU polynomials is not supported"),
        }
    }
}

#[derive(Clone)]
pub enum MultilinearsSlice<'a, F: Field> {
    Host(Vec<&'a MultilinearHost<F>>),
    Device(Vec<&'a MultilinearDevice<F>>),
}

impl<'a, F: Field, M: Borrow<Multilinear<F>>> From<&'a [M]> for MultilinearsSlice<'a, F> {
    /// Panics is there are device and host polynomials at the same time
    fn from(pols: &'a [M]) -> Self {
        let pols: Vec<_> = pols.iter().map(|p| p.borrow()).collect();
        assert!(pols.iter().all(|p| p.n_vars() == pols[0].n_vars()));
        let on_device = pols[0].is_device();
        if on_device {
            Self::Device(pols.iter().map(|p| p.as_device_ref()).collect())
        } else {
            Self::Host(pols.iter().map(|p| p.as_host_ref()).collect())
        }
    }
}

impl<'a, F: Field, M: Borrow<Multilinear<F>>> From<&'a Vec<M>> for MultilinearsSlice<'a, F> {
    /// Panics if there are device and host polynomials at the same time
    fn from(pols: &'a Vec<M>) -> Self {
        let first = pols[0].borrow();
        let n_vars = first.n_vars();
        let on_device = first.is_device();
        assert!(pols.iter().all(|p| p.borrow().n_vars() == n_vars));
        if on_device {
            Self::Device(pols.iter().map(|p| p.borrow().as_device_ref()).collect())
        } else {
            Self::Host(pols.iter().map(|p| p.borrow().as_host_ref()).collect())
        }
    }
}

impl<'a, F: Field> MultilinearsSlice<'a, F> {
    pub fn n_vars(&self) -> usize {
        match self {
            Self::Host(pol) => pol[0].n_vars,
            Self::Device(pol) => pol[0].n_vars,
        }
    }

    pub fn n_coefs(&self) -> usize {
        match self {
            Self::Host(pol) => pol[0].n_coefs(),
            Self::Device(pol) => pol[0].n_coefs(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Host(pol) => pol.len(),
            Self::Device(pol) => pol.len(),
        }
    }

    pub fn is_device(&self) -> bool {
        matches!(self, Self::Device(_))
    }

    pub fn is_host(&self) -> bool {
        matches!(self, Self::Host(_))
    }

    /// Async
    pub fn sum_over_hypercube_of_computation<
        SmallField: Field,
        BigField: ExtensionField<F> + ExtensionField<SmallField>,
    >(
        &self,
        comp: &SumcheckComputation<SmallField>,
        batching_scalars: &[BigField],
        eq_mle: Option<&Multilinear<BigField>>,
    ) -> BigField
    where
        F: ExtensionField<SmallField>,
    {
        let n_vars = self.n_vars();
        match self {
            Self::Host(multilinears) => {
                let eq_mle = eq_mle.map(|pol| pol.as_host_ref());
                HypercubePoint::par_iter(n_vars)
                    .map(|x| {
                        let point = multilinears
                            .iter()
                            .map(|pol| pol.eval_hypercube(&x))
                            .collect::<Vec<_>>();
                        let eq_mle_eval = eq_mle.map(|p| p.eval_hypercube(&x));
                        eval_sumcheck_computation(comp, batching_scalars, &point, eq_mle_eval)
                    })
                    .sum::<BigField>()
            }
            Self::Device(multilinears) => {
                let eq_mle = eq_mle.map(|pol| &pol.as_device_ref().evals);
                cuda_sum_over_hypercube_of_computation(
                    comp,
                    &multilinears,
                    &batching_scalars,
                    eq_mle,
                )
            }
        }
    }

    /// Async
    pub fn fix_variable_in_small_field<SmallField: Field>(
        &self,
        scalar: SmallField,
    ) -> MultilinearsVec<F>
    where
        F: ExtensionField<SmallField>,
    {
        match self {
            Self::Host(pol) => MultilinearsVec::Host(
                pol.iter()
                    .map(|p| p.fix_variable_in_small_field(scalar))
                    .collect(),
            ),
            Self::Device(pols) => MultilinearsVec::Device(
                cuda_fix_variable_in_small_field(pols, scalar)
                    .into_iter()
                    .map(|p| MultilinearDevice::new(p))
                    .collect(),
            ),
        }
    }

    /// Async
    pub fn fix_variable_in_big_field<EF: ExtensionField<F>>(
        &self,
        scalar: EF,
    ) -> MultilinearsVec<EF> {
        match self {
            Self::Host(pol) => MultilinearsVec::Host(
                pol.iter()
                    .map(|p| p.fix_variable_in_big_field(scalar))
                    .collect(),
            ),
            Self::Device(pols) => MultilinearsVec::Device(
                cuda_fix_variable_in_big_field(pols, scalar)
                    .into_iter()
                    .map(|p: CudaSlice<EF>| MultilinearDevice::new(p))
                    .collect(),
            ),
        }
    }

    /// Async
    pub fn packed(&self) -> Multilinear<F> {
        let packed_len = (self.len() << self.n_vars()).next_power_of_two();
        match self {
            Self::Device(pols) => {
                let mut dst = cuda_alloc_zeros(packed_len);
                let mut offset = 0;
                for pol in pols {
                    memcpy_dtod(
                        &pol.evals,
                        &mut dst.slice_mut(offset..offset + pol.n_coefs()),
                    );
                    offset += pol.n_coefs();
                }
                MultilinearDevice::new(dst).into()
            }
            Self::Host(pols) => {
                let mut dst = vec![F::ZERO; packed_len];
                let mut offset = 0;
                for pol in pols {
                    dst[offset..offset + pol.n_coefs()].copy_from_slice(&pol.evals);
                    offset += pol.n_coefs();
                }
                MultilinearHost::new(dst).into()
            }
        }
    }

    pub fn chain(&self, other: &Self) -> Self {
        assert!(self.is_empty() || other.is_empty() || self.n_vars() == other.n_vars());
        match (self, other) {
            (Self::Host(me), Self::Host(other)) => {
                Self::Host(me.iter().chain(other).copied().collect())
            }
            (Self::Device(me), Self::Device(other)) => {
                Self::Device(me.iter().chain(other).copied().collect())
            }
            _ => {
                panic!("Mixing CPU and GPU polynomials is not supported")
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Host(pol) => pol.is_empty(),
            Self::Device(pol) => pol.is_empty(),
        }
    }

    /// TODO REMOVE
    pub fn embed<EF: ExtensionField<F>>(&self) -> MultilinearsVec<EF> {
        match self {
            Self::Host(pol) => MultilinearsVec::Host(pol.iter().map(|p| p.embed()).collect()),
            Self::Device(pol) => MultilinearsVec::Device(pol.iter().map(|p| p.embed()).collect()),
        }
    }
}

pub fn eval_sumcheck_computation<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    comp: &SumcheckComputation<F>,
    batching_scalars: &[EF],
    point: &[NF],
    eq_mle_eval: Option<EF>,
) -> EF {
    let mut res = if comp.exprs.len() == 1 {
        EF::from(comp.exprs[0].eval(point))
    } else {
        comp.exprs
            .iter()
            .zip(batching_scalars)
            .skip(1)
            .map(|(expr, scalar)| *scalar * expr.eval(point))
            .sum::<EF>()
            + comp.exprs[0].eval(point)
    };
    if comp.eq_mle_multiplier {
        res *= eq_mle_eval.unwrap();
    }
    res
}

#[derive(Clone)]
pub enum MultilinearsVec<F: Field> {
    Host(Vec<MultilinearHost<F>>),
    Device(Vec<MultilinearDevice<F>>),
}

impl<F: Field> MultilinearsVec<F> {
    pub fn n_vars(&self) -> usize {
        match self {
            Self::Host(pol) => pol[0].n_vars,
            Self::Device(pol) => pol[0].n_vars,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Host(pol) => pol.len(),
            Self::Device(pol) => pol.len(),
        }
    }

    pub fn n_coefs(&self) -> usize {
        match self {
            Self::Host(pol) => pol[0].n_coefs(),
            Self::Device(pol) => pol[0].n_coefs(),
        }
    }

    pub fn push(&mut self, pol: Multilinear<F>) {
        match self {
            Self::Host(pols) => pols.push(pol.as_host()),
            Self::Device(pols) => pols.push(pol.as_device()),
        }
    }

    pub fn as_ref(&self) -> MultilinearsSlice<F> {
        match self {
            Self::Host(pols) => MultilinearsSlice::Host(pols.iter().collect()),
            Self::Device(pols) => MultilinearsSlice::Device(pols.iter().collect()),
        }
    }

    pub fn decompose(self) -> Vec<Multilinear<F>> {
        match self {
            Self::Host(pols) => pols.into_iter().map(Multilinear::from).collect(),
            Self::Device(pols) => pols.into_iter().map(Multilinear::from).collect(),
        }
    }

    // Sync
    pub fn transfer_to_host(self) -> MultilinearsVec<F> {
        match self {
            Self::Host(pols) => Self::Host(pols),
            Self::Device(pols) => {
                let res = pols.into_iter().map(|pol| pol.transfer_to_host()).collect();
                cuda_sync();
                Self::Host(res)
            }
        }
    }

    // Sync
    pub fn transfer_to_device(self) -> MultilinearsVec<F> {
        match self {
            Self::Host(pols) => Self::Device(
                pols.into_iter()
                    .map(|pol| pol.transfer_to_device())
                    .collect(),
            ),
            Self::Device(pols) => Self::Device(pols),
        }
    }
}
