use algebra::pols::UnivariatePolynomial;
use fiat_shamir::{FsError, FsVerifier};
use p3_field::Field;
use utils::Evaluation;

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
