use p3_field::Field;
use tracing::instrument;
use whir_p3::poly::evals::EvaluationsList;

#[instrument(name = "matrix_up_folded", skip_all)]
pub fn matrix_up_folded<F: Field>(outer_challenges: &[F]) -> EvaluationsList<F> {
    let n = outer_challenges.len();
    let mut folded = EvaluationsList::eval_eq(outer_challenges);
    let outer_challenges_prod: F = outer_challenges.iter().copied().product();
    folded.evals_mut()[(1 << n) - 1] -= outer_challenges_prod;
    folded.evals_mut()[(1 << n) - 2] += outer_challenges_prod;
    folded
}

#[instrument(name = "matrix_down_folded", skip_all)]
pub fn matrix_down_folded<F: Field>(outer_challenges: &[F]) -> EvaluationsList<F> {
    let n = outer_challenges.len();
    let mut folded = vec![F::ZERO; 1 << n];
    for k in 0..n {
        let outer_challenges_prod = (F::ONE - outer_challenges[n - k - 1])
            * outer_challenges[n - k..].iter().copied().product::<F>();
        let mut eq_mle = EvaluationsList::eval_eq(&outer_challenges[0..n - k - 1]);
        eq_mle = eq_mle.scale(outer_challenges_prod);
        for (mut i, v) in eq_mle.evals_mut().iter_mut().enumerate() {
            i <<= k + 1;
            i += 1 << k;
            folded[i] += *v;
        }
    }
    // bottom left corner:
    folded[(1 << n) - 1] += outer_challenges.iter().copied().product::<F>();

    EvaluationsList::new(folded)
}
