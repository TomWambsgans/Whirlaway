use p3_field::Field;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use std::fmt::{self, Debug, Formatter};
use std::ops::Range;

// pub struct Point<F: Field>(pub Vec<F>);

// impl <F: Field> Point<F> {
//     pub fn random<R: Rng>(rng: &mut R, n_vars: usize) -> Self {
//         Self((0..n_vars).map(|_| F::rand(rng)).collect())
//     }
// }

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
            point[self.n_vars - 1 - b] = if (self.val >> b) & 1 == 1 {
                F::ONE
            } else {
                F::ZERO
            };
        }
        point
    }

    pub fn from_vec<F: Field>(point: &[F]) -> Self {
        let mut val = 0;
        for b in 0..point.len() {
            assert!(point[b].is_zero() || point[b].is_one());
            val |= (point[point.len() - 1 - b].is_one() as usize) << b;
        }
        Self {
            val,
            n_vars: point.len(),
        }
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

    pub fn crop(&self, range: Range<usize>) -> Self {
        assert!(range.end <= self.n_vars);
        let val = self.val >> (self.n_vars - range.end);
        let mask = (1 << range.len()) - 1;
        Self {
            val: val & mask,
            n_vars: range.len(),
        }
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
pub struct PartialHypercubePoint<F: Field> {
    pub left: F,
    pub right: HypercubePoint,
}

impl<F: Field> PartialHypercubePoint<F> {
    pub fn n_vars(&self) -> usize {
        1 + self.right.n_vars
    }

    pub fn to_vec(&self) -> Vec<F> {
        let mut point = vec![self.left];
        point.extend(self.right.to_vec::<F>());
        point
    }

    pub fn random<R: Rng>(rng: &mut R, n_vars: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self {
            left: rng.random(),
            right: HypercubePoint::random::<F, _>(rng, n_vars - 1),
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

#[cfg(test)]
mod tests {

    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;

    type F = KoalaBear;

    #[test]
    fn test_hypercube_point() {
        assert_eq!(
            HypercubePoint { val: 0, n_vars: 3 }.to_vec::<F>(),
            vec![F::ZERO, F::ZERO, F::ZERO]
        );
        assert_eq!(
            HypercubePoint { val: 1, n_vars: 3 }.to_vec::<F>(),
            vec![F::ZERO, F::ZERO, F::ONE]
        );
        assert_eq!(
            HypercubePoint { val: 2, n_vars: 3 }.to_vec::<F>(),
            vec![F::ZERO, F::ONE, F::ZERO]
        );
        assert_eq!(
            HypercubePoint { val: 3, n_vars: 3 }.to_vec::<F>(),
            vec![F::ZERO, F::ONE, F::ONE]
        );
        assert_eq!(
            HypercubePoint { val: 4, n_vars: 3 }.to_vec::<F>(),
            vec![F::ONE, F::ZERO, F::ZERO]
        );
        assert_eq!(
            HypercubePoint {
                val: (1 << 21) - 1,
                n_vars: 20
            }
            .to_vec::<F>(),
            vec![F::ONE; 20]
        );

        assert_eq!(
            HypercubePoint {
                val: 0b1011111111,
                n_vars: 10
            }
            .crop(0..5),
            HypercubePoint {
                val: 0b10111,
                n_vars: 5
            }
        );

        assert_eq!(
            HypercubePoint {
                val: 0b1011111111,
                n_vars: 10
            }
            .crop(1..5),
            HypercubePoint {
                val: 0b0111,
                n_vars: 4
            }
        );
    }
}
