use p3_field::Field;
use rayon::prelude::*;
use tracing::instrument;
use utils::HypercubePoint;
use whir_p3::poly::evals::EvaluationsList;

#[instrument(name = "matrix_up_folded_with_univariate_skips", skip_all)]
pub(crate) fn matrix_up_folded_with_univariate_skips<F: Field>(
    zerocheck_challenges: &[F],
    univariate_skips: usize,
) -> EvaluationsList<F> {
    // TODO: It's sparse => bad performance
    let n = zerocheck_challenges.len();
    let n_vars = n + univariate_skips * 2 - 1;
    let mut folded = EvaluationsList::new(F::zero_vec(1 << n_vars));
    let point_len = univariate_skips + (n - 1);
    let inner_eq_mle = EvaluationsList::eval_eq(&zerocheck_challenges[1..]);
    folded
        .evals_mut()
        .par_chunks_mut(1 << point_len)
        .enumerate()
        .for_each(|(i, block)| {
            block[i << (n - 1)..(i + 1) << (n - 1)].copy_from_slice(inner_eq_mle.evals());
        });

    let zerocheck_challenges_prod = zerocheck_challenges[1..].iter().copied().product::<F>();
    folded.evals_mut()[(1 << n_vars) - 1] -= zerocheck_challenges_prod;
    folded.evals_mut()[(1 << n_vars) - 2] += zerocheck_challenges_prod;

    folded
}

#[instrument(name = "matrix_down_folded_with_univariate_skips", skip_all)]
pub fn matrix_down_folded_with_univariate_skips<F: Field>(
    outer_challenges: &[F],
    univariate_skips: usize,
) -> EvaluationsList<F> {
    // TODO: It's sparse => bad performance
    let n = outer_challenges.len();
    // n_vars defined as in the original function.
    let n_vars = n + univariate_skips * 2 - 1;
    let mut folded = EvaluationsList::new(F::zero_vec(1 << n_vars));
    let point_len = univariate_skips + (n - 1);
    let inner_mles = (1..n)
        .map(|i| EvaluationsList::eval_eq(&outer_challenges[1..i]))
        .collect::<Vec<_>>();
    folded
        .evals_mut()
        .par_chunks_mut(1 << point_len)
        .enumerate()
        .for_each(|(i, block)| {
            let x = HypercubePoint {
                n_vars: univariate_skips,
                val: i,
            };
            let mut point = x.to_vec();
            point.extend_from_slice(&outer_challenges[1..]);
            let m = point.len();
            for k in 0..m {
                let outer_challenges_prod =
                    (F::ONE - point[m - k - 1]) * point[m - k..].iter().copied().product::<F>();
                if outer_challenges_prod.is_zero() {
                    continue;
                }
                let eq_mle = &inner_mles[(m - k - 1).saturating_sub(x.n_vars)];
                // EvaluationsListHost::eq_mle(&outer_challenges[0..n - k - 1]);
                let eq_mle = eq_mle.scale(outer_challenges_prod);
                let n_coeffs = eq_mle.num_evals();
                for (mut i, v) in eq_mle.evals().iter().enumerate() {
                    i += (x.val >> (x.n_vars - x.n_vars.min(m - k - 1))) * n_coeffs;
                    i <<= k + 1;
                    i += 1 << k;
                    block[i] += *v;
                }
            }
        });

    folded.evals_mut()[(1 << n_vars) - 1] += outer_challenges[1..].iter().copied().product::<F>();

    folded
}
