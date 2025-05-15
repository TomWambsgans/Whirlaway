use super::CoefficientListHost;
use super::{CoefficientList, CoefficientListDevice};
use crate::tensor_algebra::TensorAlgebra;
use cuda_bindings::{
    cuda_add_assign_slices, cuda_add_slices, cuda_dot_product, cuda_eq_mle, cuda_eval_mixed_tensor,
    cuda_eval_multilinear_in_lagrange_basis, cuda_fold_rectangular_in_small_field,
    cuda_lagrange_to_monomial_basis, cuda_linear_combination_at_row_level,
    cuda_linear_combination_large_field, cuda_linear_combination_small_field,
    cuda_repeat_slice_from_inside, cuda_repeat_slice_from_outside, cuda_scale_slice_in_place,
};
use cuda_bindings::{cuda_compute_over_hypercube, cuda_fold_rectangular_in_large_field};
use cuda_engine::{
    SumcheckComputation, clone_dtod, cuda_alloc_zeros, cuda_get_at_index, cuda_sync, memcpy_dtod,
    memcpy_dtoh, memcpy_htod,
};
use cudarc::driver::CudaSlice;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use rayon::prelude::*;
use std::borrow::Borrow;
use tracing::instrument;
use utils::{HypercubePoint, MyExtensionField, PartialHypercubePoint};
use utils::{default_hash, dot_product};

/*

Multilinear Polynomials represented as a vector of its evaluations on the boolean hypercube (Lagrange basis).

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

    /// Debug purpose
    pub fn hash(&self) -> u64 {
        default_hash(&self.evals)
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

    pub fn evaluate_in_large_field<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
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

    fn evaluate_in_small_field<S: Field>(&self, point: &[S]) -> F
    where
        F: ExtensionField<S>,
    {
        assert_eq!(self.n_vars, point.len());
        if self.n_vars == 0 {
            return F::from(self.evals[0]);
        }
        let mut n = self.n_coefs();
        let mut buff = self.evals[..n / 2]
            .par_iter()
            .zip(&self.evals[n / 2..])
            .map(|(l, r)| (*r - *l) * point[0] + *l)
            .collect::<Vec<_>>();
        for p in &point[1..] {
            n /= 2;
            buff = buff[..n / 2]
                .par_iter()
                .zip(&buff[n / 2..])
                .map(|(l, r)| (*r - *l) * *p + *l)
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
            .map(|chunk| EF::from_basis_coefficients_slice(chunk).unwrap())
            .collect();
        MultilinearHost::new(packed_evals)
    }

    pub fn fold_rectangular_in_small_field<S: Field>(&self, scalars: &[S]) -> Self
    where
        F: ExtensionField<S>,
    {
        assert!(scalars.len().is_power_of_two());
        assert!(scalars.len() <= self.n_coefs());
        let new_size = self.evals.len() / scalars.len();
        let mut new_evals = vec![F::ZERO; new_size];
        new_evals
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result)| {
                *result = scalars
                    .iter()
                    .enumerate()
                    .map(|(j, scalar)| self.evals[i + j * new_size] * *scalar)
                    .sum();
            });
        MultilinearHost::new(new_evals)
    }

    pub fn fold_rectangular_in_large_field<EF: ExtensionField<F>>(
        &self,
        scalars: &[EF],
    ) -> MultilinearHost<EF> {
        assert!(scalars.len().is_power_of_two());
        assert!(scalars.len() <= self.n_coefs());
        let new_size = self.evals.len() / scalars.len();
        let mut new_evals = vec![EF::ZERO; new_size];
        new_evals
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result)| {
                *result = scalars
                    .iter()
                    .enumerate()
                    .map(|(j, scalar)| *scalar * self.evals[i + j * new_size])
                    .sum();
            });
        MultilinearHost::new(new_evals)
    }

    pub fn linear_comb_in_large_field<EF: ExtensionField<F>>(
        pols: &[&Self],
        scalars: &[EF],
    ) -> MultilinearHost<EF> {
        assert_eq!(pols.len(), scalars.len());
        let mut sum = MultilinearHost::<EF>::zero(pols[0].n_vars);
        for i in 0..scalars.len() {
            sum.add_assign(&pols[i].scale_large_field::<EF>(scalars[i]));
        }
        sum
    }

    pub fn linear_comb_in_small_field<SF: Field>(
        pols: &[&Self],
        scalars: &[SF],
    ) -> MultilinearHost<F>
    where
        F: MyExtensionField<SF>,
    {
        assert_eq!(pols.len(), scalars.len());
        let mut sum = MultilinearHost::<F>::zero(pols[0].n_vars);
        for i in 0..scalars.len() {
            sum.add_assign::<F>(&pols[i].scale_small_field::<SF>(scalars[i]));
        }
        sum
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

    pub fn to_monomial_basis_rev(self) -> CoefficientListHost<F> {
        let mut coeffs = self.evals;
        let n = self.n_vars;

        for i in 0..n {
            coeffs.par_chunks_mut(1 << (n - i)).for_each(|chunk| {
                let n = chunk.len();
                let left = (0..n / 2).map(|j| chunk[2 * j]).collect::<Vec<_>>();
                let right = (0..n / 2)
                    .map(|j| chunk[2 * j + 1] - chunk[2 * j])
                    .collect::<Vec<_>>();
                chunk[..n / 2].copy_from_slice(&left);
                chunk[n / 2..].copy_from_slice(&right);
            });
        }

        CoefficientListHost::new(coeffs)
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

    pub fn scale_large_field<EF: ExtensionField<F>>(&self, scalar: EF) -> MultilinearHost<EF> {
        MultilinearHost::new(self.evals.par_iter().map(|&e| scalar * e).collect())
    }

    pub fn scale_small_field<SF: Field>(&self, scalar: SF) -> MultilinearHost<F>
    where
        F: MyExtensionField<SF>,
    {
        MultilinearHost::new(
            self.evals
                .par_iter()
                .map(|&e| e.my_multiply(&scalar))
                .collect(),
        )
    }

    pub fn eq_mle(scalars: &[F]) -> Self {
        if scalars.len() <= 8 {
            Self::eq_mle_single_threaded(scalars)
        } else {
            Self::eq_mle_parallel(scalars)
        }
    }

    pub fn add_dummy_starting_variables(&self, n: usize) -> Self {
        // TODO remove
        Self::new(self.evals.repeat(1 << n))
    }

    pub fn add_dummy_ending_variables(&self, n: usize) -> Self {
        // TODO remove
        let evals = self
            .evals
            .iter()
            .flat_map(|item| std::iter::repeat(*item).take(1 << n))
            .collect();
        Self::new(evals)
    }

    pub fn piecewise_dot_product_at_field_level(&self, scalars: &[F]) -> Self
    where
        F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    {
        assert_eq!(
            <F as BasedVectorSpace<F::PrimeSubfield>>::DIMENSION,
            scalars.len()
        );
        let prime_composed = self
            .evals
            .par_iter()
            .map(|e| {
                <F as BasedVectorSpace<F::PrimeSubfield>>::as_basis_coefficients_slice(e).to_vec()
            })
            .collect::<Vec<_>>();

        let mut res = Vec::new();
        for w in 0..self.n_coefs() {
            res.push(dot_product(&prime_composed[w], scalars));
        }
        MultilinearHost::new(res)
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
                    *l *= one_minus_s;
                    *r = tmp;
                });
        }
        Self::new(evals)
    }

    // Async
    pub fn transfer_to_device(&self) -> MultilinearDevice<F> {
        MultilinearDevice::new(memcpy_htod(&self.evals))
    }

    // TODO remove
    pub fn embed<EF: ExtensionField<F>>(&self) -> MultilinearHost<EF> {
        MultilinearHost::new(self.evals.par_iter().map(|e| EF::from(*e)).collect())
    }

    fn add_assign<S: Field>(&mut self, other: &MultilinearHost<S>)
    where
        F: MyExtensionField<S>,
    {
        assert_eq!(self.n_vars, other.n_vars);
        self.evals
            .par_iter_mut()
            .zip(other.evals.par_iter())
            .for_each(|(a, b)| a.my_add_assign(b));
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

    /// Debug purpose
    /// Sync
    pub fn hash(&self) -> u64 {
        default_hash(&self.transfer_to_host().evals)
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

    // Async
    pub fn piecewise_dot_product_at_field_level(&self, scalars: &[F]) -> Self
    where
        F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    {
        let evals_prime = unsafe {
            self.evals
                .transmute::<F::PrimeSubfield>(self.n_coefs() * scalars.len())
                .unwrap()
        };
        Self::new(cuda_linear_combination_at_row_level::<F::PrimeSubfield, F>(
            &evals_prime,
            scalars,
        ))
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
        TensorAlgebra::new(cuda_eval_mixed_tensor::<SubF, F>(&self.evals, point))
    }

    // Async
    pub fn add_dummy_starting_variables(&self, n: usize) -> Self {
        // TODO remove
        Self::new(cuda_repeat_slice_from_outside(&self.evals, 1 << n))
    }

    // Async
    pub fn add_dummy_ending_variables(&self, n: usize) -> Self {
        // TODO remove
        Self::new(cuda_repeat_slice_from_inside(&self.evals, 1 << n))
    }

    // Async
    pub fn evaluate_in_large_field<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        assert_eq!(self.n_vars, point.len());
        if self.n_vars == 0 {
            return EF::from(cuda_get_at_index(&self.evals, 0));
        }
        let _span = tracing::info_span!("cuda evaluate in multilinear basis").entered();
        let res = cuda_eval_multilinear_in_lagrange_basis(&self.evals, point);
        cuda_sync();
        res
    }

    // Async
    pub fn evaluate_in_small_field<S: Field>(&self, point: &[S]) -> F
    where
        F: ExtensionField<S>,
    {
        assert_eq!(self.n_vars, point.len());
        if self.n_vars == 0 {
            return F::from(cuda_get_at_index(&self.evals, 0));
        }
        let _span = tracing::info_span!("cuda evaluate in multilinear basis").entered();
        let res = cuda_eval_multilinear_in_lagrange_basis(&self.evals, point);
        cuda_sync();
        res
    }

    // Async
    fn add_assign<S: Field>(&mut self, other: &MultilinearDevice<S>)
    where
        F: MyExtensionField<S>,
    {
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

    // Async
    pub fn add_dummy_starting_variables(&self, n: usize) -> Self {
        // TODO remove
        match self {
            Self::Host(pol) => Self::Host(pol.add_dummy_starting_variables(n)),
            Self::Device(pol) => Self::Device(pol.add_dummy_starting_variables(n)),
        }
    }

    // Async
    pub fn add_dummy_ending_variables(&self, n: usize) -> Self {
        // TODO remove
        match self {
            Self::Host(pol) => Self::Host(pol.add_dummy_ending_variables(n)),
            Self::Device(pol) => Self::Device(pol.add_dummy_ending_variables(n)),
        }
    }

    pub fn scale_in_place(&mut self, scalar: F) {
        match self {
            Self::Host(pol) => *pol = pol.scale_large_field(scalar),
            Self::Device(pol) => pol.scale_in_place(scalar),
        }
    }

    // Async
    pub fn piecewise_dot_product_at_field_level(&self, scalars: &[F]) -> Self
    where
        F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    {
        match self {
            Self::Host(pol) => Multilinear::Host(pol.piecewise_dot_product_at_field_level(scalars)),
            Self::Device(pol) => {
                Multilinear::Device(pol.piecewise_dot_product_at_field_level(scalars))
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
            Self::Host(_) => panic!(""),
        }
    }

    pub fn as_host(self) -> MultilinearHost<F> {
        match self {
            Self::Host(pol) => pol,
            Self::Device(_) => panic!(""),
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
    #[instrument(name = "eval_mixed_tensor", skip_all)]
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
    pub fn evaluate_in_large_field<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        match self {
            Self::Host(pol) => pol.evaluate_in_large_field(point),
            Self::Device(pol) => pol.evaluate_in_large_field(point),
        }
    }

    /// Async
    pub fn evaluate_in_small_field<S: Field>(&self, point: &[S]) -> F
    where
        F: ExtensionField<S>,
    {
        match self {
            Self::Host(pol) => pol.evaluate_in_small_field(point),
            Self::Device(pol) => pol.evaluate_in_small_field(point),
        }
    }

    /// Debug purpose
    /// Sync
    pub fn hash(&self) -> u64 {
        match self {
            Self::Host(pol) => pol.hash(),
            Self::Device(pol) => pol.hash(),
        }
    }

    /// TODO remove
    /// Async
    pub fn embed<EF: ExtensionField<F>>(&self) -> Multilinear<EF> {
        match self {
            Self::Host(pol) => Multilinear::Host(pol.embed()),
            Self::Device(pol) => {
                Multilinear::Device(pol.transfer_to_host().embed::<EF>().transfer_to_device())
            }
        }
    }

    pub fn fold_rectangular_in_large_field<EF: ExtensionField<F>>(
        &self,
        scalars: &[EF],
    ) -> Multilinear<EF> {
        match self {
            Self::Host(pol) => Multilinear::Host(pol.fold_rectangular_in_large_field(scalars)),
            Self::Device(pol) => Multilinear::Device(MultilinearDevice::new(
                cuda_fold_rectangular_in_large_field(&[&pol.evals], scalars)
                    .pop()
                    .unwrap(),
            )),
        }
    }

    pub fn add_assign<S: Field>(&mut self, other: &Multilinear<S>)
    where
        F: MyExtensionField<S>,
    {
        match (self, other) {
            (Multilinear::Host(pol), Multilinear::Host(other)) => pol.add_assign(other),
            (Multilinear::Device(pol), Multilinear::Device(other)) => pol.add_assign(other),
            _ => unreachable!("Mixing CPU and GPU polynomials is not supported"),
        }
    }
}

#[derive(Clone)]
pub enum MultilinearsSlice<'a, F: Field> {
    Host(Vec<&'a MultilinearHost<F>>),
    Device(Vec<&'a MultilinearDevice<F>>),
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

impl<F: Field> MultilinearsSlice<'_, F> {
    pub fn n_vars(&self) -> usize {
        match self {
            Self::Host(pol) => pol[0].n_vars,
            Self::Device(pol) => pol[0].n_vars,
        }
    }
    pub fn as_host(&self) -> Vec<&MultilinearHost<F>> {
        match self {
            Self::Host(host) => host.clone(),
            _ => unreachable!(),
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
    pub fn compute_over_hypercube<S: Field, BigField: ExtensionField<F> + ExtensionField<S>>(
        &self,
        comp: &SumcheckComputation<S>,
        batching_scalars: &[BigField],
        eq_mle: Option<&Multilinear<BigField>>,
    ) -> BigField
    where
        F: ExtensionField<S>,
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
                cuda_compute_over_hypercube(comp, multilinears, batching_scalars, eq_mle)
            }
        }
    }

    /// Async
    pub fn fold_rectangular_in_small_field<S: Field>(&self, scalars: &[S]) -> MultilinearsVec<F>
    where
        F: ExtensionField<S>,
    {
        match self {
            Self::Host(pol) => MultilinearsVec::Host(
                pol.iter()
                    .map(|p| p.fold_rectangular_in_small_field(scalars))
                    .collect(),
            ),
            Self::Device(pols) => MultilinearsVec::Device(
                cuda_fold_rectangular_in_small_field(
                    &pols.iter().map(|pol| &pol.evals).collect::<Vec<_>>(),
                    scalars,
                )
                .into_iter()
                .map(|pol| MultilinearDevice::new(pol))
                .collect(),
            ),
        }
    }

    /// Async
    pub fn fold_rectangular_in_large_field<EF: ExtensionField<F>>(
        &self,
        scalars: &[EF],
    ) -> MultilinearsVec<EF> {
        match self {
            Self::Host(pol) => MultilinearsVec::Host(
                pol.iter()
                    .map(|p| p.fold_rectangular_in_large_field(scalars))
                    .collect(),
            ),
            Self::Device(pols) => MultilinearsVec::Device(
                cuda_fold_rectangular_in_large_field(
                    &pols.iter().map(|pol| &pol.evals).collect::<Vec<_>>(),
                    scalars,
                )
                .into_iter()
                .map(|pol| MultilinearDevice::new(pol))
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

    /// Async
    pub fn linear_comb_in_large_field<EF: ExtensionField<F>>(
        &self,
        scalars: &[EF],
    ) -> Multilinear<EF> {
        assert_eq!(self.len(), scalars.len());
        match self {
            Self::Host(pols) => {
                Multilinear::Host(MultilinearHost::linear_comb_in_large_field(pols, scalars))
            }
            Self::Device(pols) => Multilinear::Device(MultilinearDevice::new(
                cuda_linear_combination_large_field(pols, scalars),
            )),
        }
    }

    /// Async
    pub fn linear_comb_in_small_field<SF: Field>(&self, scalars: &[SF]) -> Multilinear<F>
    where
        F: MyExtensionField<SF>,
    {
        assert_eq!(self.len(), scalars.len());
        match self {
            Self::Host(pols) => {
                Multilinear::Host(MultilinearHost::linear_comb_in_small_field(pols, scalars))
            }
            Self::Device(pols) => Multilinear::Device(MultilinearDevice::new(
                cuda_linear_combination_small_field(pols, scalars),
            )),
        }
    }

    /// Async
    pub fn batch_evaluate_in_large_field<EF: ExtensionField<F>>(&self, point: &[EF]) -> Vec<EF> {
        assert_eq!(self.n_vars(), point.len());
        match self {
            Self::Host(pols) => pols
                .par_iter()
                .map(|pol| pol.evaluate_in_large_field(point))
                .collect(),
            Self::Device(pols) => {
                let eq_mle = cuda_eq_mle(point);
                let mut res = Vec::with_capacity(self.len());
                for pol in pols {
                    res.push(cuda_dot_product(&pol.evals, &eq_mle));
                }
                res
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

    /// Async
    pub fn sum(&self) -> Multilinear<F> {
        match self {
            Self::Host(pols) => {
                let mut res = MultilinearHost::zero(pols[0].n_vars);
                for pol in pols {
                    res.add_assign(*pol);
                }
                Multilinear::Host(res)
            }
            Self::Device(pols) => {
                Multilinear::Device(MultilinearDevice::new(cuda_add_slices(pols)))
            }
        }
    }

    /// Sync
    pub fn transfer_to_host(&self) -> MultilinearsVec<F> {
        match self {
            Self::Host(_) => panic!("Already on host"),
            Self::Device(pols) => {
                let res = pols.iter().map(|pol| pol.transfer_to_host()).collect();
                cuda_sync();
                MultilinearsVec::Host(res)
            }
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

impl<F: Field> From<Vec<Multilinear<F>>> for MultilinearsVec<F> {
    fn from(value: Vec<Multilinear<F>>) -> Self {
        assert!(value.iter().all(|p| p.n_vars() == value[0].n_vars()));
        if value[0].is_device() {
            Self::Device(value.into_iter().map(|p| p.as_device()).collect())
        } else {
            Self::Host(value.into_iter().map(|p| p.as_host()).collect())
        }
    }
}

impl<F: Field> MultilinearsVec<F> {
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

    pub fn len(&self) -> usize {
        match self {
            Self::Host(pols) => pols.len(),
            Self::Device(pols) => pols.len(),
        }
    }

    pub fn n_vars(&self) -> usize {
        match self {
            Self::Host(pol) => pol[0].n_vars,
            Self::Device(pol) => pol[0].n_vars,
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

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn test_to_monomial_basis_rev() {
        let rng = &mut StdRng::seed_from_u64(0);
        let n_vars = 7;
        type F = KoalaBear;
        let mut point = (0..n_vars).map(|_| rng.random()).collect::<Vec<F>>();
        let multilinear = MultilinearHost::<F>::random(rng, n_vars);
        let eval_1 = multilinear.clone().to_monomial_basis_rev().evaluate(&point);
        point.reverse();
        let eval_2 = multilinear.evaluate_in_large_field(&point);
        assert_eq!(eval_1, eval_2);
    }
}
