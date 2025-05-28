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
        (0..self.n_vars)
            .map(|b| {
                if (self.val >> (self.n_vars - 1 - b)) & 1 == 1 {
                    F::ONE
                } else {
                    F::ZERO
                }
            })
            .collect()
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

    pub const fn new(n_vars: usize, val: usize) -> Self {
        Self { n_vars, val }
    }

    pub const fn zero(n_vars: usize) -> Self {
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
    pub const fn n_vars(&self) -> usize {
        1 + self.right.n_vars
    }

    pub const fn new(left: u32, right_n_vars: usize, right: usize) -> Self {
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
        [self.left.as_slice(), &self.right.to_vec()].concat()
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
    // We start with a vector containing a single 1 (base case for the recursion)
    let mut res = vec![F::ONE];
    for &var in point.iter().rev() {
        let len = res.len();
        // Reserve space for doubling the vector (avoid reallocations)
        res.reserve(len);
        // SAFETY: we will immediately fill the new slots, so it's okay to set the new length
        unsafe { res.set_len(len * 2) };

        // Split the vector into two non-overlapping halves:
        // - low: the first half (existing values)
        // - high: the second half (to be filled with scaled values)
        let (low, high) = res.split_at_mut(len);

        // Copy the first half into the second half
        high.copy_from_slice(low);

        // Multiply each element in the second half by the current variable value
        high.par_iter_mut().for_each(|x| *x *= var);
    }
    res
}

#[derive(Debug, Clone, Default)]
pub struct Statement<EF> {
    pub points: Vec<Vec<EF>>,
    pub evaluations: Vec<EF>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;

    type F = KoalaBear;

    #[test]
    fn test_to_vec_all_zero() {
        let point = HypercubePoint { n_vars: 4, val: 0 };
        let vec = point.to_vec::<F>();
        assert_eq!(vec, vec![F::ZERO; 4]);
    }

    #[test]
    fn test_to_vec_all_one() {
        let point = HypercubePoint {
            n_vars: 3,
            val: 0b111,
        };
        let vec = point.to_vec::<F>();
        assert_eq!(vec, vec![F::ONE; 3]);
    }

    #[test]
    fn test_to_vec_mixed_bits() {
        // binary 101 → [1, 0, 1]
        let point = HypercubePoint {
            n_vars: 3,
            val: 0b101,
        };
        let vec = point.to_vec::<F>();
        assert_eq!(vec, vec![F::ONE, F::ZERO, F::ONE]);
    }

    #[test]
    fn test_to_vec_single_bit() {
        let point = HypercubePoint { n_vars: 1, val: 1 };
        let vec = point.to_vec::<F>();
        assert_eq!(vec, vec![F::ONE]);
    }

    #[test]
    fn test_to_vec_large() {
        let val = 0b110101; // binary 110101 → [1,1,0,1,0,1]
        let point = HypercubePoint { n_vars: 6, val };
        let vec = point.to_vec::<F>();
        let expected = vec![F::ONE, F::ONE, F::ZERO, F::ONE, F::ZERO, F::ONE];
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_empty_point() {
        let point: Vec<F> = vec![];
        let expanded = expanded_point_for_multilinear_monomial_evaluation(&point);
        assert_eq!(expanded, vec![F::ONE]);
    }

    #[test]
    fn test_single_zero() {
        let point = vec![F::ZERO];
        let expanded = expanded_point_for_multilinear_monomial_evaluation(&point);
        assert_eq!(expanded, vec![F::ONE, F::ZERO]);
    }

    #[test]
    fn test_single_one() {
        let point = vec![F::ONE];
        let expanded = expanded_point_for_multilinear_monomial_evaluation(&point);
        assert_eq!(expanded, vec![F::ONE, F::ONE]);
    }

    #[test]
    fn test_two_variables() {
        let a = F::ONE; // 1
        let b = F::ZERO; // 0
        let point = vec![a, b];
        let expanded = expanded_point_for_multilinear_monomial_evaluation(&point);
        // Expected:
        // [1, 0, 1, 0]
        assert_eq!(expanded, vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
    }

    #[test]
    fn test_two_variables_both_one() {
        let point = vec![F::ONE, F::ONE];
        let expanded = expanded_point_for_multilinear_monomial_evaluation(&point);
        // Expected:
        // [1, 1, 1, 1]
        assert_eq!(expanded, vec![F::ONE, F::ONE, F::ONE, F::ONE]);
    }

    #[test]
    fn test_three_variables_mixed() {
        let one = F::ONE;
        let zero = F::ZERO;
        let point = vec![one, zero, one];
        let expanded = expanded_point_for_multilinear_monomial_evaluation(&point);
        // Expected:
        // [1, 1, 0, 0, 1, 1, 0, 0]
        assert_eq!(
            expanded,
            vec![
                F::ONE,
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ONE,
                F::ONE,
                F::ZERO,
                F::ZERO
            ]
        );
    }
}
