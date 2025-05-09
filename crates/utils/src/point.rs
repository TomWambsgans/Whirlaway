use p3_field::Field;
use rand::Rng;
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

    pub fn random<F: Field, R: Rng>(rng: &mut R, n_vars: usize) -> Self {
        Self {
            val: rng.random_range(0..(1 << n_vars)),
            n_vars,
        }
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

pub fn concat_hypercube_points(points: &[HypercubePoint]) -> HypercubePoint {
    let mut val = 0;
    let mut n_vars = 0;
    for point in points {
        val = (val << point.n_vars) | point.val;
        n_vars += point.n_vars;
    }
    HypercubePoint { val, n_vars }
}

#[derive(Clone, Debug)]
pub struct PartialHypercubePoint {
    pub left: u32,
    pub right: HypercubePoint,
}

impl PartialHypercubePoint {
    pub fn n_vars(&self) -> usize {
        1 + self.right.n_vars
    }

    pub fn new(left: u32, right_n_vars: usize, right: usize) -> Self {
        Self {
            left,
            right: HypercubePoint::new(right_n_vars, right),
        }
    }
}

// [0x45188, 0xfc787, ..., 0x78f8d5, 0, 1, 0, 0, ..., 1]
#[derive(Clone, Debug)]
pub struct MixedPoint<F: Field> {
    pub left: Vec<F>,
    pub right: HypercubePoint,
}

impl<F: Field> MixedPoint<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut point = self.left.clone();
        point.extend(self.right.to_vec::<F>());
        point
    }
}

#[derive(Clone, Debug)]
pub struct MixedEvaluation<F: Field> {
    pub point: MixedPoint<F>,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Evaluation<F> {
    pub point: Vec<F>,
    pub value: F,
}

pub fn expanded_point_for_multilinear_monomial_evaluation<F: Field>(point: &[F]) -> Vec<F> {
    if point.is_empty() {
        return vec![F::ONE];
    }
    let sub = expanded_point_for_multilinear_monomial_evaluation(&point[1..]);
    let mut res = vec![F::ZERO; 1 << point.len()];
    res[..sub.len()].copy_from_slice(&sub);
    res[sub.len()..]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)| {
            *x = sub[i] * point[0];
        });
    res
}

#[derive(Debug, Clone, Default)]
pub struct Statement<EF> {
    pub points: Vec<Vec<EF>>,
    pub evaluations: Vec<EF>,
}
