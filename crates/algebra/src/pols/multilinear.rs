use std::ops::AddAssign;

use crate::tensor_algebra::TensorAlgebra;
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use rayon::prelude::*;

use super::{HypercubePoint, PartialHypercubePoint, UnivariatePolynomial};

#[derive(Clone, Debug)]
pub enum MultilinearPolynomial<F: Field> {
    Dense(DenseMultilinearPolynomial<F>),
    Sparse(SparseMultilinearPolynomial<F>),
}

impl<F: Field> From<DenseMultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn from(p: DenseMultilinearPolynomial<F>) -> Self {
        MultilinearPolynomial::Dense(p)
    }
}

impl<F: Field> From<SparseMultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn from(p: SparseMultilinearPolynomial<F>) -> Self {
        MultilinearPolynomial::Sparse(p)
    }
}

impl<F: Field> MultilinearPolynomial<F> {
    pub fn n_vars(&self) -> usize {
        match self {
            Self::Dense(p) => p.n_vars,
            Self::Sparse(p) => p.n_vars,
        }
    }

    pub fn eval_hypercube(&self, point: &HypercubePoint) -> F {
        match self {
            Self::Dense(p) => p.eval_hypercube(point),
            Self::Sparse(p) => p.eval_hypercube(point),
        }
    }

    pub fn eval<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        match self {
            Self::Dense(p) => p.eval(point),
            Self::Sparse(p) => p.eval(point),
        }
    }

    pub fn eval_partial_hypercube(&self, point: &PartialHypercubePoint) -> F {
        match self {
            Self::Dense(p) => p.eval_partial_hypercube(point),
            Self::Sparse(p) => p.eval_partial_hypercube(point),
        }
    }

    pub fn fix_variable<EF: ExtensionField<F>>(self, z: EF) -> MultilinearPolynomial<EF> {
        match self {
            Self::Dense(p) => MultilinearPolynomial::Dense(p.fix_variable(z)),
            Self::Sparse(p) => {
                if p.n_cols - p.n_final_bits_per_row >= 1 {
                    let new_evals = p
                        .evals
                        .into_par_iter()
                        .map(|(row_pol, point)| (row_pol.fix_variable(z), point))
                        .collect();
                    MultilinearPolynomial::Sparse(SparseMultilinearPolynomial::new(new_evals))
                } else {
                    MultilinearPolynomial::Dense(p.densify().fix_variable(z))
                }
            }
        }
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        match self {
            Self::Dense(p) => p.max_degree_per_vars(),
            Self::Sparse(p) => p.max_degree_per_vars(),
        }
    }

    pub fn random_dense<R: Rng>(rng: &mut R, n_vars: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        DenseMultilinearPolynomial::random(rng, n_vars).into()
    }

    pub fn embed<EF: ExtensionField<F>>(&self) -> MultilinearPolynomial<EF> {
        match self {
            Self::Dense(p) => MultilinearPolynomial::Dense(p.embed()),
            Self::Sparse(_) => todo!(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SparseMultilinearPolynomial<F: Field> {
    pub n_vars: usize,
    pub n_rows: usize,                                               // in log
    pub n_cols: usize,                                               // in log
    pub n_final_bits_per_row: usize,                                 // in log
    pub evals: Vec<(DenseMultilinearPolynomial<F>, HypercubePoint)>, // last vars = eq extension
}

impl<F: Field> SparseMultilinearPolynomial<F> {
    pub fn n_coefs(&self) -> usize {
        1 << self.n_vars
    }

    pub fn new(evals: Vec<(DenseMultilinearPolynomial<F>, HypercubePoint)>) -> Self {
        let n_rows = evals.len().next_power_of_two().trailing_zeros() as usize;
        let n_final_bits_per_row = evals[0].1.n_vars;
        for i in 0..evals.len() {
            assert_eq!(
                (evals[0].0.n_vars, evals[0].1.n_vars),
                (evals[i].0.n_vars, evals[i].1.n_vars)
            );
        }
        let n_vars = n_rows + evals[0].0.n_vars + evals[0].1.n_vars;
        let n_cols = n_vars - n_rows;
        Self {
            n_vars,
            n_rows,
            n_cols,
            n_final_bits_per_row,
            evals,
        }
    }

    pub fn eval<EF: ExtensionField<F>>(&self, _point: &[EF]) -> EF {
        todo!()
    }

    pub fn eval_hypercube(&self, point: &HypercubePoint) -> F {
        assert_eq!(self.n_vars, point.n_vars);
        let row = point.val & ((1 << self.n_rows) - 1);
        if row >= self.evals.len() {
            return F::ZERO;
        }
        let (row_pol, final_bits) = &self.evals[row];
        let col = point.val >> self.n_rows;
        let col_left = col >> self.n_final_bits_per_row;
        let col_right = col & ((1 << self.n_final_bits_per_row) - 1);
        if col_right == final_bits.val {
            row_pol.eval_hypercube(&HypercubePoint {
                n_vars: self.n_cols - self.n_final_bits_per_row,
                val: col_left,
            })
        } else {
            F::ZERO
        }
    }

    pub fn eval_partial_hypercube(&self, point: &PartialHypercubePoint) -> F {
        self.eval_hypercube(&HypercubePoint {
            n_vars: point.n_vars(),
            val: point.right.val + (1 << point.right.n_vars),
        }) * F::from_u32(point.left)
            + self.eval_hypercube(&HypercubePoint {
                n_vars: point.n_vars(),
                val: point.right.val,
            }) * (F::ONE - F::from_u32(point.left))
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        vec![1; self.n_vars]
    }

    pub fn densify(&self) -> DenseMultilinearPolynomial<F> {
        let mut evals = vec![F::ZERO; 1 << self.n_vars];
        for (row_index, (row_pol, final_bits)) in self.evals.iter().enumerate() {
            for (j, v) in row_pol.evals.iter().enumerate() {
                let mut index = j << (self.n_final_bits_per_row + self.n_rows); // first vars
                index += final_bits.val << self.n_rows;
                index += row_index;
                evals[index] = *v;
            }
        }
        DenseMultilinearPolynomial::new(evals)
    }
}

#[derive(Clone, Debug)]
pub struct DenseMultilinearPolynomial<F: Field> {
    pub n_vars: usize,
    pub evals: Vec<F>, // [f(0, 0, ..., 0), f(0, 0, ..., 0, 1), f(0, 0, ..., 0, 1, 0), f(0, 0, ..., 0, 1, 1), ...]
}

impl<F: Field> DenseMultilinearPolynomial<F> {
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

    pub fn eval<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
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

    pub fn packed<EF: ExtensionField<F>>(self) -> DenseMultilinearPolynomial<EF> {
        assert!(<EF as BasedVectorSpace<F>>::DIMENSION.is_power_of_two());
        assert!(1 << self.n_vars > <EF as BasedVectorSpace<F>>::DIMENSION);
        let evals = self
            .evals
            .chunks(<EF as BasedVectorSpace<F>>::DIMENSION)
            .map(|chunk| EF::from_basis_coefficients_slice(chunk))
            .collect();
        DenseMultilinearPolynomial::new(evals)
    }

    /// fix first variables
    pub fn fix_variable<EF: ExtensionField<F>>(&self, z: EF) -> DenseMultilinearPolynomial<EF> {
        let half = self.evals.len() / 2;
        let mut new_evals = vec![EF::ZERO; half];
        new_evals
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, result)| {
                *result = z * (self.evals[i + half] - self.evals[i]) + self.evals[i];
            });
        DenseMultilinearPolynomial::new(new_evals)
    }

    pub fn eval_mixed_tensor<SubF: Field>(&self, point: &[F]) -> TensorAlgebra<SubF, F>
    where
        F: ExtensionField<SubF>,
    {
        // returns φ1(self)(φ0(point[0]), φ0(point[1]), ...)
        assert_eq!(point.len(), self.n_vars);
        let lagrange_evals = DenseMultilinearPolynomial::eq_mle(point).evals;
        lagrange_evals
            .par_iter()
            .zip(&self.evals)
            .map(|(l, e)| {
                TensorAlgebra::phi_0_times_phi_1(l, e)
            })
            .sum()
    }

    pub fn as_coefs(self) -> Vec<F> {
        let mut coeffs = self.evals;
        let n = self.n_vars;

        // Apply Möbius transform
        for i in 0..n {
            let step = 1 << i;
            for j in 0..(1 << n) {
                if (j & step) == 0 {
                    let k = j | step;
                    let temp = coeffs[j];
                    coeffs[k] -= temp;
                }
            }
        }

        coeffs
    }

    pub fn as_univariate(self) -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new(self.as_coefs())
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
        points.iter().map(|point| univariate.eval(point)).collect()
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

    pub fn embed<EF: ExtensionField<F>>(&self) -> DenseMultilinearPolynomial<EF> {
        // TODO bas for PERFORMANCE
        let evals = self.evals.iter().map(|&e| EF::from(e)).collect();
        DenseMultilinearPolynomial::new(evals)
    }

    pub fn scale<EF: ExtensionField<F>>(&self, scalar: EF) -> DenseMultilinearPolynomial<EF> {
        let evals = self.evals.par_iter().map(|&e| scalar * e).collect();
        DenseMultilinearPolynomial::new(evals)
    }

    pub fn eq_mle(scalars: &[F]) -> Self {
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

    // parallel version:

    // pub fn eq_mle(scalars: &[F]) -> Self {
    //     let mut evals = vec![F::ZERO; 1 << scalars.len()];
    //     evals[0] = F::ONE;
    //     for (i, &s) in scalars.iter().rev().enumerate() {
    //         let one_minus_s = F::ONE - s;
    //         let chunk_size = 1 << i;
    //         let (left, rest) = evals.split_at_mut(chunk_size);
    //         let right = &mut rest[..chunk_size];
    //         left.par_iter_mut()
    //             .zip(right.par_iter_mut())
    //             .for_each(|(l, r)| {
    //                 let tmp = *l * s;
    //                 *l = *l * one_minus_s;
    //                 *r = tmp;
    //             });
    //     }
    //     Self::new(evals)
    // }
}

impl<F: Field> AddAssign<MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn add_assign(&mut self, other: MultilinearPolynomial<F>) {
        match (self, other) {
            (Self::Dense(a), Self::Dense(b)) => *a += b,
            (Self::Sparse(_), Self::Sparse(_)) => todo!(),
            _ => todo!(),
        }
    }
}

impl<F: Field> AddAssign<DenseMultilinearPolynomial<F>> for DenseMultilinearPolynomial<F> {
    fn add_assign(&mut self, other: DenseMultilinearPolynomial<F>) {
        assert_eq!(self.n_vars, other.n_vars);
        self.evals
            .par_iter_mut()
            .zip(other.evals.par_iter())
            .for_each(|(a, b)| *a += *b);
    }
}
