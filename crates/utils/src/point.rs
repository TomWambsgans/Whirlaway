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

    pub fn iter(n_vars: usize) -> impl Iterator<Item = Self> {
        (0..(1 << n_vars)).map(move |val| Self { val, n_vars })
    }

    pub fn par_iter(n_vars: usize) -> impl ParallelIterator<Item = Self> {
        (0..(1 << n_vars))
            .into_par_iter()
            .map(move |val| Self { n_vars, val })
    }

    pub const fn new(n_vars: usize, val: usize) -> Self {
        Self { n_vars, val }
    }

    pub const fn zero(n_vars: usize) -> Self {
        Self { n_vars, val: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Evaluation<F> {
    pub point: Vec<F>,
    pub value: F,
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
}
