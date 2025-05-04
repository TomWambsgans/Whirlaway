use std::{
    cmp::max,
    ops::{Mul, MulAssign},
};

use p3_field::{ExtensionField, Field};
use rand::distr::{Distribution, StandardUniform};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
pub struct UnivariatePolynomial<F: Field> {
    pub coeffs: Vec<F>,
}

// Set some minimum number of field elements to be worked on per thread
// to avoid per-thread costs dominating parallel execution time.
const MIN_ELEMENTS_PER_THREAD: usize = 16;

// Dense
impl<F: Field> UnivariatePolynomial<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    #[inline]
    // Horner's method for polynomial evaluation
    pub fn horner_evaluate<EF: ExtensionField<F>>(poly_coeffs: &[F], point: &EF) -> EF {
        poly_coeffs
            .iter()
            .rfold(EF::ZERO, move |result, coeff| result * *point + *coeff)
    }

    pub fn eval<EF: ExtensionField<F>>(&self, x: &EF) -> EF {
        if self.coeffs.is_empty() {
            return EF::ZERO;
        } else if x.is_zero() {
            return EF::from(self.coeffs[0]);
        }
        // Horner's method
        Self::horner_evaluate(&self.coeffs, x)
    }

    pub fn sum_evals<EF: ExtensionField<F>>(&self, xs: &[EF]) -> EF {
        xs.iter().map(|x| self.eval(x)).sum()
    }

    pub fn eval_parallel<EF: ExtensionField<F>>(&self, x: &EF) -> EF {
        if self.coeffs.is_empty() {
            return EF::ZERO;
        } else if x.is_zero() {
            return EF::from(self.coeffs[0]);
        }
        // Horners method - parallel method
        // compute the number of threads we will be using.
        let num_cpus_available = rayon::current_num_threads();
        let num_coeffs = self.coeffs.len();
        let num_elem_per_thread = max(num_coeffs / num_cpus_available, MIN_ELEMENTS_PER_THREAD);

        // run Horners method on each thread as follows:
        // 1) Split up the coefficients across each thread evenly.
        // 2) Do polynomial evaluation via horner's method for the thread's coefficients
        // 3) Scale the result point^{thread coefficient start index}
        // Then obtain the final polynomial evaluation by summing each threads result.
        let result = self
            .coeffs
            .par_chunks(num_elem_per_thread)
            .enumerate()
            .map(|(i, chunk)| {
                let mut thread_result = Self::horner_evaluate(chunk, x);
                thread_result *= x.exp_u64((i * num_elem_per_thread) as u64);
                thread_result
            })
            .sum();
        result
    }

    pub fn lagrange_interpolation<S: Field>(values: &[(S, F)]) -> Option<Self>
    where
        F: ExtensionField<S>,
    {
        let n = values.len();
        let mut result = vec![F::ZERO; n];

        for i in 0..n {
            let (x_i, y_i) = values[i];
            let mut term = vec![F::ZERO; n];
            let mut product = F::ONE;

            for j in 0..n {
                if i != j {
                    let (x_j, _) = values[j];
                    product *= (x_i - x_j).try_inverse()?;
                }
            }

            term[0] = product * y_i;
            for j in 0..n {
                if i != j {
                    let (x_j, _) = values[j];
                    let mut new_term = term.clone();
                    for k in (1..n).rev() {
                        new_term[k] = new_term[k - 1];
                    }
                    new_term[0] = F::ZERO;

                    for k in 0..n {
                        term[k] = term[k] * (-x_j) + new_term[k];
                    }
                }
            }

            for j in 0..n {
                result[j] += term[j];
            }
        }

        // while result.len() > 1 && result.last() == Some(&F::zero()) {
        //     result.pop();
        // }

        Some(Self::new(result))
    }

    pub fn sum_over_hypercube(&self) -> F {
        self.eval(&F::ZERO) + self.eval(&F::ONE)
    }

    pub fn random<R: rand::Rng>(rng: &mut R, degree: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        let coefs = (0..=degree).map(|_| rng.random()).collect();
        Self::new(coefs)
    }
}

impl<F: Field> Mul for UnivariatePolynomial<F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = vec![F::ZERO; self.degree() + other.degree() + 1];

        // No need to optimize because we only multiply low degree polynomials (during sumcheck)
        for i in 0..=self.degree() {
            for j in 0..=other.degree() {
                result[i + j] += self.coeffs[i] * other.coeffs[j];
            }
        }

        Self::new(result)
    }
}

impl<F: Field> MulAssign for UnivariatePolynomial<F> {
    fn mul_assign(&mut self, other: Self) {
        *self = std::mem::take(self) * other;
    }
}

pub fn univariate_selectors<F: Field>(n: usize) -> Vec<UnivariatePolynomial<F>> {
    (0..1 << n)
        .into_par_iter()
        .map(|i| {
            let values = (0..1 << n)
                .map(|j| (F::from_u64(j), if i == j { F::ONE } else { F::ZERO }))
                .collect::<Vec<_>>();
            UnivariatePolynomial::lagrange_interpolation(&values).unwrap()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    type F = KoalaBear;

    #[test]
    fn test_lagrange_interpolation() {
        let rng = &mut StdRng::seed_from_u64(0);
        let pol = UnivariatePolynomial::random(rng, 5);
        let points = (0..pol.degree() + 1)
            .map(|_| {
                let point = rng.random::<F>();
                (point, pol.eval(&point))
            })
            .collect::<Vec<_>>();
        let pol2 = UnivariatePolynomial::lagrange_interpolation(&points).unwrap();
        assert_eq!(pol, pol2);
    }

    #[test]
    fn test_mul() {
        let rng = &mut StdRng::seed_from_u64(0);
        let pol1 = UnivariatePolynomial::<KoalaBear>::random(rng, 5);
        let pol2 = UnivariatePolynomial::<KoalaBear>::random(rng, 7);

        assert_eq!(
            (pol1.clone() * pol2.clone()).eval(&F::from_u32(7854)),
            pol1.eval(&F::from_u32(7854)) * pol2.eval(&F::from_u32(7854))
        );
    }
}
