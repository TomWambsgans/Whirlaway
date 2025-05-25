use p3_field::Field;
use rayon::prelude::*;
use std::fmt::{self, Debug, Formatter};

#[derive(Clone, PartialEq, Eq)]
pub struct HypercubePoint {
    pub n_vars: usize,
    pub val: usize, // 0 -> [0, 0, ..., 0], 1 -> [0, 0, ..., 0, 1], 2 -> [0, 0, ..., 0, 1, 0], 3 -> [0, 0, ..., 0, 1, 1], ...
}

impl Debug for HypercubePoint {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:0width$b}", self.val, width = self.n_vars)
    }
}

impl HypercubePoint {
    pub fn to_vec<F: Field>(&self) -> Vec<F> {
        let mut point = vec![F::ZERO; self.n_vars];
        for b in 0..self.n_vars {
            if (self.val >> b) & 1 == 1 {
                point[self.n_vars - 1 - b] = F::ONE;
            }
        }
        point
    }

    pub fn iter(n_vars: usize) -> impl Iterator<Item = Self> {
        (0..(1 << n_vars)).map(move |val| Self { val, n_vars })
    }

    pub fn par_iter(n_vars: usize) -> impl ParallelIterator<Item = Self> {
        (0..(1 << n_vars))
            .into_par_iter()
            .map(move |val| Self { val, n_vars })
    }

    pub fn new(n_vars: usize, val: usize) -> Self {
        Self { n_vars, val }
    }

    pub fn zero(n_vars: usize) -> Self {
        Self { n_vars, val: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Evaluation<F> {
    pub point: Vec<F>,
    pub value: F,
}
