use std::borrow::Borrow;

use p3_field::{ExtensionField, Field};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use rayon::prelude::*;

/*

Multilinear Polynomials represented as a vector of its evaluations on the boolean hypercube (Lagrange basis).

*/

#[derive(Clone, Eq, PartialEq, PartialOrd, Debug)]
pub struct Multilinear<F: Field> {
    pub n_vars: usize,
    pub evals: Vec<F>, // [f(0, 0, ..., 0), f(0, 0, ..., 0, 1), f(0, 0, ..., 0, 1, 0), f(0, 0, ..., 0, 1, 1), ...]
}

impl<F: Field> Multilinear<F> {
    pub const fn n_coeffs(&self) -> usize {
        1 << self.n_vars
    }

    pub fn zero(n_vars: usize) -> Self {
        Self {
            n_vars,
            evals: F::zero_vec(1 << n_vars),
        }
    }

    pub fn new(evals: Vec<F>) -> Self {
        assert!(evals.is_empty() || evals.len().is_power_of_two());
        let n_vars = evals.len().ilog2() as usize;
        Self { n_vars, evals }
    }

    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        assert_eq!(self.n_vars, point.len());
        if self.n_vars == 0 {
            return EF::from(self.evals[0]);
        }
        let mut n = self.n_coeffs();
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

    pub fn fold_rectangular_in_small_field<S>(&self, scalars: &[S]) -> Self
    where
        S: Field,
        F: ExtensionField<S>,
    {
        assert!(scalars.len().is_power_of_two() && scalars.len() <= self.n_coeffs());
        let new_size = self.evals.len() / scalars.len();
        Self::new(
            (0..new_size)
                .into_par_iter()
                .map(|i| {
                    scalars
                        .iter()
                        .enumerate()
                        .map(|(j, s)| self.evals[i + j * new_size] * *s)
                        .sum()
                })
                .collect(),
        )
    }

    pub fn fold_rectangular_in_large_field<EF: ExtensionField<F>>(
        &self,
        scalars: &[EF],
    ) -> Multilinear<EF> {
        assert!(scalars.len().is_power_of_two() && scalars.len() <= self.n_coeffs());
        let new_size = self.evals.len() / scalars.len();
        Multilinear::new(
            (0..new_size)
                .into_par_iter()
                .map(|i| {
                    scalars
                        .iter()
                        .enumerate()
                        .map(|(j, s)| *s * self.evals[i + j * new_size])
                        .sum()
                })
                .collect(),
        )
    }

    pub fn linear_combination<EF: ExtensionField<F>, P: Borrow<Self>>(
        pols: &[P],
        scalars: &[EF],
    ) -> Multilinear<EF> {
        assert_eq!(pols.len(), scalars.len());
        let mut sum = Multilinear::<EF>::zero(pols[0].borrow().n_vars);
        for i in 0..scalars.len() {
            sum.add_assign::<EF>(&pols[i].borrow().scale_large_field(scalars[i]));
        }
        sum
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

    pub fn scale_large_field<EF: ExtensionField<F>>(&self, scalar: EF) -> Multilinear<EF> {
        Multilinear::new(self.evals.par_iter().map(|&e| scalar * e).collect())
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
            .flat_map(|item| std::iter::repeat_n(*item, 1 << n))
            .collect();
        Self::new(evals)
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

    pub fn add_assign<S: Field>(&mut self, other: &Multilinear<S>)
    where
        F: ExtensionField<S>,
    {
        assert_eq!(self.n_vars, other.n_vars);
        self.evals
            .par_iter_mut()
            .zip(other.evals.par_iter())
            .for_each(|(a, b)| *a += *b);
    }

    pub fn batch_evaluate_in_large_field<EF: ExtensionField<F>>(
        pols: &[Self],
        point: &[EF],
    ) -> Vec<EF> {
        pols.par_iter().map(|pol| pol.evaluate(point)).collect()
    }

    pub fn batch_fold_rectangular_in_small_field<S>(pols: &[&Self], scalars: &[S]) -> Vec<Self>
    where
        S: Field,
        F: ExtensionField<S>,
    {
        pols.par_iter()
            .map(|p| p.fold_rectangular_in_small_field(scalars))
            .collect()
    }

    pub fn batch_fold_rectangular_in_large_field<EF: ExtensionField<F>>(
        pols: &[&Self],
        scalars: &[EF],
    ) -> Vec<Multilinear<EF>> {
        pols.par_iter()
            .map(|p| p.fold_rectangular_in_large_field(scalars))
            .collect()
    }

    pub fn packed(pols: &[Self]) -> Self {
        let n_vars = pols[0].n_vars;
        assert!(pols.iter().all(|p| p.n_vars == n_vars));
        let packed_len = (pols.len() << n_vars).next_power_of_two();
        let mut dst = vec![F::ZERO; packed_len];
        let mut offset = 0;
        for pol in pols {
            dst[offset..offset + pol.n_coeffs()].copy_from_slice(&pol.evals);
            offset += pol.n_coeffs();
        }
        Self::new(dst)
    }
}
