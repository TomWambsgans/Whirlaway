use algebra::pols::{
    ComposedPolynomial, Evaluation, HypercubePoint, PartialHypercubePoint, UnivariatePolynomial,
};
use fiat_shamir::{FsError, FsProver, FsVerifier};
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum SumcheckError {
    Fs(FsError),
    InvalidRound,
}

impl From<FsError> for SumcheckError {
    fn from(e: FsError) -> Self {
        SumcheckError::Fs(e)
    }
}

pub trait SumcheckSummation {
    /// The list of the evaluation points where the polynomial is non zero (a priori), that should be computed during sumcheck
    fn non_zero_points<F: Field>(&self, z: F, n_vars: usize) -> Vec<PartialHypercubePoint<F>>; // TODO return an iterator ?
}

pub struct FullSumcheckSummation;

impl SumcheckSummation for FullSumcheckSummation {
    fn non_zero_points<F: Field>(&self, z: F, n_vars: usize) -> Vec<PartialHypercubePoint<F>> {
        HypercubePoint::iter(n_vars - 1)
            .map(move |right| PartialHypercubePoint { left: z, right })
            .collect()
    }
}

pub fn prove<F: Field, EF: ExtensionField<F>>(
    pol: &mut ComposedPolynomial<F, EF>,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
) -> Vec<EF> {
    prove_with_custum_summation(
        pol,
        fs_prover,
        sum,
        n_rounds,
        pow_bits,
        &FullSumcheckSummation,
    )
}

pub fn prove_with_custum_summation<F: Field, EF: ExtensionField<F>, S: SumcheckSummation>(
    pol: &mut ComposedPolynomial<F, EF>,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
    summation: &S,
) -> Vec<EF> {
    assert!(pol.n_vars >= 1);
    let n_rounds = n_rounds.unwrap_or(pol.n_vars);
    let max_degree_per_vars = pol.max_degree_per_vars();
    let mut challenges = Vec::new();
    let mut sum = sum.unwrap_or_else(|| pol.sum_over_hypercube());
    for i in 0..n_rounds {
        let d = max_degree_per_vars[i];
        let mut p_evals = Vec::<(EF, EF)>::new();
        for z in 0..=d {
            let z = EF::from_u64(z as u64);
            let sum_z = if z.is_one() {
                sum - p_evals[0].1
            } else {
                summation
                    .non_zero_points(z, pol.n_vars)
                    .into_par_iter()
                    .map(|point| pol.eval_partial_hypercube(&point))
                    .sum::<EF>()
            };
            p_evals.push((z, sum_z));
        }
        let p = UnivariatePolynomial::lagrange_interpolation(&p_evals).unwrap();

        fs_prover.add_scalars(&p.coeffs);
        let challenge = fs_prover.challenge_scalars(1)[0];

        // Do PoW if needed
        if pow_bits > 0 {
            fs_prover.challenge_pow(pow_bits);
        }

        pol.fix_variable(challenge);
        challenges.push(challenge);
        sum = p.eval(&challenge);
    }
    challenges
}

pub fn verify<F: Field>(
    fs_verifier: &mut FsVerifier,
    max_degree_per_vars: &[usize],
    pow_bits: usize,
) -> Result<(F, Evaluation<F>), SumcheckError> {
    let mut challenges = Vec::new();
    let mut first_round = true;
    let (mut sum, mut target) = (F::ZERO, F::ZERO);

    for &d in max_degree_per_vars {
        let coefs = fs_verifier.next_scalars(d + 1)?;
        let pol = UnivariatePolynomial::new(coefs);
        if first_round {
            first_round = false;
            sum = pol.eval(&F::ZERO) + pol.eval(&F::ONE);
            target = sum;
        }

        if target != pol.eval(&F::ZERO) + pol.eval(&F::ONE) {
            return Err(SumcheckError::InvalidRound);
        }
        let challenge = fs_verifier.challenge_scalars(1)[0];

        // Do PoW if needed
        if pow_bits > 0 {
            fs_verifier.challenge_pow(pow_bits)?;
        }

        target = pol.eval(&challenge);
        challenges.push(challenge);
    }
    Ok((
        sum,
        Evaluation {
            point: challenges,
            value: target,
        },
    ))
}

#[cfg(test)]
mod tests {
    use algebra::pols::MultilinearPolynomial;
    use fiat_shamir::FsProver;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    type F = p3_koala_bear::KoalaBear;

    #[test]
    fn test_sumcheck() {
        let n_vars = 10;
        let rng = &mut StdRng::seed_from_u64(0);
        let pol = ComposedPolynomial::new_product(
            n_vars,
            (0..5)
                .map(|_| MultilinearPolynomial::<F>::random_dense(rng, n_vars))
                .collect::<Vec<_>>(),
        );
        let mut fs_prover = FsProver::new();
        let sum = pol.sum_over_hypercube();
        prove(&mut pol.clone(), &mut fs_prover, None, None, 0);

        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let (claimed_sum, postponed_verification) =
            verify::<F>(&mut fs_verifier, &pol.max_degree_per_vars(), 0).unwrap();
        assert_eq!(sum, claimed_sum);
        assert_eq!(
            pol.eval(&postponed_verification.point),
            postponed_verification.value
        );
    }
}
