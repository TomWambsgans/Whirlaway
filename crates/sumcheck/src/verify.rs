use std::fmt::Debug;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use utils::{Evaluation, PF};
use whir_p3::{
    fiat_shamir::{errors::ProofError, verifier::VerifierState},
    poly::{dense::WhirDensePolynomial, multilinear::MultilinearPoint},
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

pub fn verify<EF, Challenger>(
    verifier_state: &mut VerifierState<PF<PF<EF>>, EF, Challenger>,
    n_vars: usize,
    degree: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    Challenger: FieldChallenger<PF<PF<EF>>> + GrindingChallenger<Witness = PF<PF<EF>>>,
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

pub fn verify_with_univariate_skip<EF, Challenger>(
    verifier_state: &mut VerifierState<PF<PF<EF>>, EF, Challenger>,
    degree: usize,
    n_vars: usize,
    skips: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    Challenger: FieldChallenger<PF<PF<EF>>> + GrindingChallenger<Witness = PF<PF<EF>>>,
{
    let mut max_degree_per_vars = vec![degree * ((1 << skips) - 1)];
    max_degree_per_vars.extend(vec![degree; n_vars - skips]);
    let mut sumation_sets = vec![(0..1 << skips).map(EF::from_usize).collect::<Vec<_>>()];
    sumation_sets.extend(vec![
        (0..2).map(EF::from_usize).collect::<Vec<_>>();
        n_vars - skips
    ]);
    verify_core(
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
    )
}

fn verify_core<EF, Challenger>(
    verifier_state: &mut VerifierState<PF<PF<EF>>, EF, Challenger>,
    max_degree_per_vars: &[usize],
    sumation_sets: Vec<Vec<EF>>,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    EF: Field + ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    Challenger: FieldChallenger<PF<PF<EF>>> + GrindingChallenger<Witness = PF<PF<EF>>>,
{
    assert_eq!(max_degree_per_vars.len(), sumation_sets.len(),);
    let mut challenges = Vec::new();
    let mut first_round = true;
    let (mut sum, mut target) = (EF::ZERO, EF::ZERO);

    for (&deg, sumation_set) in max_degree_per_vars.iter().zip(sumation_sets) {
        let coeffs = verifier_state.next_extension_scalars_vec(deg + 1)?;
        let pol = WhirDensePolynomial::from_coefficients_vec(coeffs);

        let computed_sum = sumation_set.iter().map(|&s| pol.evaluate(s)).sum();
        if first_round {
            first_round = false;
            sum = computed_sum;
        } else if target != computed_sum {
            return Err(SumcheckError::InvalidRound);
        }
        let challenge = verifier_state.sample();

        let pow_bits = grinding.pow_bits::<EF>(deg);
        verifier_state.check_pow_grinding(pow_bits)?;

        target = pol.evaluate(challenge);
        challenges.push(challenge);
    }
    Ok((
        sum,
        Evaluation {
            point: MultilinearPoint(challenges),
            value: target,
        },
    ))
}
