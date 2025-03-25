use std::ops::AddAssign;

use p3_field::{ExtensionField, Field};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use rayon::prelude::*;

use super::{HypercubePoint, PartialHypercubePoint, UnivariatePolynomial};

mod dense;
pub use dense::*;

mod sparse;
pub use sparse::*;

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

impl<F: Field> AddAssign<MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn add_assign(&mut self, other: MultilinearPolynomial<F>) {
        match (self, other) {
            (Self::Dense(a), Self::Dense(b)) => *a += b,
            (Self::Sparse(_), Self::Sparse(_)) => todo!(),
            _ => todo!(),
        }
    }
}
