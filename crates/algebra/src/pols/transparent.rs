use std::fmt::Debug;

use p3_field::{ExtensionField, Field};

use crate::pols::utils::{max_degree_per_vars_prod, max_degree_per_vars_sum};

use super::ArithmeticCircuit;

#[derive(Clone, Debug)]
pub struct TransparentMultivariatePolynomial<F: Field> {
    pub n_vars: usize,
    pub coefs: ArithmeticCircuit<F, usize>,
}

impl<F: Field> TransparentMultivariatePolynomial<F> {
    pub fn new(coefs: ArithmeticCircuit<F, usize>, n_vars: usize) -> Self {
        Self { coefs, n_vars }
    }

    pub fn single_var() -> Self {
        Self::new(ArithmeticCircuit::Node(0), 1)
    }

    pub fn eval<EF: ExtensionField<F>>(&self, point: &[EF]) -> EF {
        assert_eq!(point.len(), self.n_vars);
        self.coefs.eval_field(&|i| point[*i])
    }

    pub fn eq_extension_n_scalars(scalars: &[F]) -> Self {
        // eq(scalars, Xs) = ((scalars[0] X0 + (scalars[0] - 1) (X0 - 1)) * ((scalars[1] X1 + (scalars[1] - 1) (X1 - 1)) ...
        let left = (0..scalars.len())
            .map(|i| ArithmeticCircuit::Node(i))
            .collect::<Vec<_>>();
        let right = (0..scalars.len())
            .map(|i| ArithmeticCircuit::Scalar(scalars[i]))
            .collect::<Vec<_>>();
        Self::_eq_extension(scalars.len(), &left, &right)
    }

    pub fn eq_extension_2n_vars(n: usize) -> Self {
        // eq(Xs, Ys) = ((Y0 X0 + (Y0 - 1) (X0 - 1)) * ((Y1 X1 + (Y1 - 1) (X1 - 1)) ...
        let left = (0..n)
            .map(|i| ArithmeticCircuit::Node(i))
            .collect::<Vec<_>>();
        let right = (0..n)
            .map(|i| ArithmeticCircuit::Node(i + n))
            .collect::<Vec<_>>();
        Self::_eq_extension(2 * n, &left, &right)
    }

    pub fn next(n: usize) -> Self {
        // returns a polynomial P in 2n vars, where P(x, y) = 1 iif y = x + 1 in big indian (both numbers are n bits)
        let factor = |l, r| ArithmeticCircuit::Node(l) * (-ArithmeticCircuit::Node(r) + F::ONE);
        let g = |k| {
            let mut factors = vec![];
            for i in (n - k)..n {
                factors.push(factor(i, i + n));
            }
            factors.push(factor(2 * n - 1 - k, n - 1 - k));
            factors.push(
                Self::_eq_extension(
                    n * 2,
                    &(0..n - k - 1)
                        .map(|i| ArithmeticCircuit::Node(i))
                        .collect::<Vec<_>>(),
                    &(0..n - k - 1)
                        .map(|i| ArithmeticCircuit::Node(i + n))
                        .collect::<Vec<_>>(),
                )
                .coefs,
            );
            ArithmeticCircuit::new_product(factors)
        };
        Self::new(ArithmeticCircuit::new_sum((0..n).map(g).collect()), n * 2)
    }

    fn _eq_extension(
        n_vars: usize,
        left: &[ArithmeticCircuit<F, usize>],
        right: &[ArithmeticCircuit<F, usize>],
    ) -> Self {
        assert_eq!(left.len(), right.len());
        Self {
            n_vars,
            coefs: ArithmeticCircuit::new_product(
                left.iter()
                    .zip(right)
                    .map(|(l, r)| {
                        (l.clone() * r.clone()) + ((-l.clone() + F::ONE) * (-r.clone() + F::ONE))
                    })
                    .collect(),
            ),
        }
    }

    pub fn max_degree_per_vars(&self) -> Vec<usize> {
        self.coefs.parse(
            &|_| vec![0; self.n_vars],
            &|i| {
                let mut res = vec![0; self.n_vars];
                res[*i] = 1;
                res
            },
            &|subs| max_degree_per_vars_prod(&subs),
            &|subs| max_degree_per_vars_sum(&subs),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use p3_koala_bear::KoalaBear;

    use crate::pols::HypercubePoint;

    use super::*;
    use p3_field::PrimeCharacteristicRing;
    type F = KoalaBear;

    #[test]
    fn test_next() {
        let n = 5;
        let next = TransparentMultivariatePolynomial::<F>::next(n);
        let mut one_points = HashSet::new();
        for x in 0..(1 << n) - 1 {
            let y = x + 1;
            one_points.insert(
                [
                    HypercubePoint { n_vars: n, val: x }.to_vec::<F>(),
                    HypercubePoint { n_vars: n, val: y }.to_vec::<F>(),
                ]
                .concat(),
            );
        }
        for x in 0..1 << (n * 2) {
            let point = HypercubePoint {
                n_vars: n * 2,
                val: x,
            }
            .to_vec();
            if one_points.contains(&point) {
                assert_eq!(next.eval(&point), F::ONE);
            } else {
                assert_eq!(next.eval(&point), F::ZERO);
            }
        }
    }
}
