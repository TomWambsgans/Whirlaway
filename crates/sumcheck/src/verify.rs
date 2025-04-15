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
    n_vars: usize,
    degree: usize,
    pow_bits: usize,
) -> Result<(F, Evaluation<F>), SumcheckError> {
    let sumation_sets = vec![(0..2).map(|i| F::from_usize(i)).collect::<Vec<_>>(); n_vars];
    let max_degree_per_vars = vec![degree; n_vars];
    verify_core(fs_verifier, &max_degree_per_vars, sumation_sets, pow_bits)
}

pub fn verify_with_univariate_skip<F: Field>(
    fs_verifier: &mut FsVerifier,
    degree: usize,
    n_vars: usize,
    skips: usize,
    pow_bits: usize,
) -> Result<(F, Evaluation<F>), SumcheckError> {
    let mut max_degree_per_vars = vec![degree * ((1 << skips) - 1)];
    max_degree_per_vars.extend(vec![degree; n_vars - skips]);
    let mut sumation_sets = vec![
        (0..1 << skips)
            .map(|i| F::from_usize(i))
            .collect::<Vec<_>>(),
    ];
    sumation_sets.extend(vec![
        (0..2).map(|i| F::from_usize(i)).collect::<Vec<_>>();
        n_vars - skips
    ]);
    verify_core(fs_verifier, &max_degree_per_vars, sumation_sets, pow_bits)
}

fn verify_core<F: Field>(
    fs_verifier: &mut FsVerifier,
    max_degree_per_vars: &[usize],
    sumation_sets: Vec<Vec<F>>,
    pow_bits: usize,
) -> Result<(F, Evaluation<F>), SumcheckError> {
    assert_eq!(max_degree_per_vars.len(), sumation_sets.len(),);
    let mut challenges = Vec::new();
    let mut first_round = true;
    let (mut sum, mut target) = (F::ZERO, F::ZERO);

    for (&deg, sumation_set) in max_degree_per_vars.iter().zip(sumation_sets) {
        let coefs = fs_verifier.next_scalars(deg + 1)?;
        let pol = UnivariatePolynomial::new(coefs);

        if first_round {
            first_round = false;
            sum = pol.sum_evals(&sumation_set);
        } else if target != pol.sum_evals(&sumation_set) {
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
