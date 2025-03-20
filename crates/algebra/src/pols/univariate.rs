use std::cmp::max;

use p3_field::Field;
use rand::distr::{Distribution, StandardUniform};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
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
    fn horner_evaluate(poly_coeffs: &[F], point: &F) -> F {
        poly_coeffs
            .iter()
            .rfold(F::ZERO, move |result, coeff| result * *point + *coeff)
    }

    fn internal_evaluate(&self, point: &F) -> F {
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
                let mut thread_result = Self::horner_evaluate(chunk, point);
                thread_result *= point.exp_u64((i * num_elem_per_thread) as u64);
                thread_result
            })
            .sum();
        result
    }

    pub fn eval(&self, x: &F) -> F {
        if self.coeffs.is_empty() {
            return F::ZERO;
        } else if x.is_zero() {
            return self.coeffs[0];
        }
        self.internal_evaluate(x)
    }

    pub fn lagrange_interpolation(values: &[(F, F)]) -> Option<Self> {
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

#[cfg(test)]
mod tests {
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
}
