use algebra::pols::UnivariatePolynomial;
use fiat_shamir::{FsError, FsVerifier};
use p3_field::Field;
use rand::distr::{Distribution, StandardUniform};
use utils::Evaluation;

use crate::SumcheckGrinding;

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

pub fn verify<EF: Field>(
    fs_verifier: &mut FsVerifier,
    n_vars: usize,
    degree: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    StandardUniform: Distribution<EF>,
{
    let sumation_sets = vec![(0..2).map(|i| EF::from_usize(i)).collect::<Vec<_>>(); n_vars];
    let max_degree_per_vars = vec![degree; n_vars];
    verify_core(fs_verifier, &max_degree_per_vars, sumation_sets, grinding)
}

pub fn verify_with_univariate_skip<EF: Field>(
    fs_verifier: &mut FsVerifier,
    degree: usize,
    n_vars: usize,
    skips: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    StandardUniform: Distribution<EF>,
{
    let mut max_degree_per_vars = vec![degree * ((1 << skips) - 1)];
    max_degree_per_vars.extend(vec![degree; n_vars - skips]);
    let mut sumation_sets = vec![
        (0..1 << skips)
            .map(|i| EF::from_usize(i))
            .collect::<Vec<_>>(),
    ];
    sumation_sets.extend(vec![
        (0..2).map(|i| EF::from_usize(i)).collect::<Vec<_>>();
        n_vars - skips
    ]);
    verify_core(fs_verifier, &max_degree_per_vars, sumation_sets, grinding)
}

fn verify_core<EF: Field>(
    fs_verifier: &mut FsVerifier,
    max_degree_per_vars: &[usize],
    sumation_sets: Vec<Vec<EF>>,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    StandardUniform: Distribution<EF>,
{
    assert_eq!(max_degree_per_vars.len(), sumation_sets.len(),);
    let mut challenges = Vec::new();
    let mut first_round = true;
    let (mut sum, mut target) = (EF::ZERO, EF::ZERO);

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

        let pow_bits = grinding.pow_bits::<EF>(deg);
        fs_verifier.challenge_pow(pow_bits)?;

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
