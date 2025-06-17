use p3_challenger::HashChallenger;
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_keccak::Keccak256Hash;
use rand::distr::{Distribution, StandardUniform};
use utils::Evaluation;
use whir_p3::{
    fiat_shamir::{errors::ProofError, pow::blake3::Blake3PoW, verifier::VerifierState},
    poly::dense::WhirDensePolynomial,
};

use crate::SumcheckGrinding;

#[derive(Debug, Clone)]
pub enum SumcheckError {
    Fs(ProofError),
    InvalidRound,
}

impl From<ProofError> for SumcheckError {
    fn from(e: ProofError) -> Self {
        Self::Fs(e)
    }
}

pub fn verify<EF, F>(
    verifier_state: &mut VerifierState<'_, EF, F, HashChallenger<u8, Keccak256Hash, 32>, u8>,
    n_vars: usize,
    degree: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: Field + ExtensionField<F> + TwoAdicField,
    StandardUniform: Distribution<EF>,
{
    let sumation_sets = vec![(0..2).map(|i| EF::from_usize(i)).collect::<Vec<_>>(); n_vars];
    let max_degree_per_vars = vec![degree; n_vars];
    verify_core(
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
    )
}

pub fn verify_with_univariate_skip<EF, F>(
    verifier_state: &mut VerifierState<'_, EF, F, HashChallenger<u8, Keccak256Hash, 32>, u8>,
    degree: usize,
    n_vars: usize,
    skips: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: Field + ExtensionField<F> + TwoAdicField,
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
    verify_core(
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
    )
}

fn verify_core<EF, F>(
    verifier_state: &mut VerifierState<'_, EF, F, HashChallenger<u8, Keccak256Hash, 32>, u8>,
    max_degree_per_vars: &[usize],
    sumation_sets: Vec<Vec<EF>>,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: Field + ExtensionField<F> + TwoAdicField,
    StandardUniform: Distribution<EF>,
{
    assert_eq!(max_degree_per_vars.len(), sumation_sets.len(),);
    let mut challenges = Vec::new();
    let mut first_round = true;
    let (mut sum, mut target) = (EF::ZERO, EF::ZERO);

    for (&deg, sumation_set) in max_degree_per_vars.iter().zip(sumation_sets) {
        let coefs = verifier_state.next_scalars_vec(deg + 1)?;
        let pol = WhirDensePolynomial::from_coefficients_vec(coefs);

        let computed_sum = sumation_set.iter().map(|&s| pol.evaluate(s)).sum();
        if first_round {
            first_round = false;
            sum = computed_sum;
        } else if target != computed_sum {
            return Err(SumcheckError::InvalidRound);
        }
        let challenge = verifier_state.challenge_scalars_array::<1>().unwrap()[0];

        let pow_bits = grinding.pow_bits::<EF>(deg);
        verifier_state.challenge_pow::<Blake3PoW>(pow_bits as f64)?;

        target = pol.evaluate(challenge);
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
