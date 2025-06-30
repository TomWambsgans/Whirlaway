use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use rand::distr::{Distribution, StandardUniform};
use utils::Evaluation;
use whir_p3::{
    fiat_shamir::{errors::ProofError, verifier::VerifierState},
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

pub fn verify<EF, F, Challenger, const DIGEST_ELEMS: usize>(
    verifier_state: &mut VerifierState<EF, F, Challenger, DIGEST_ELEMS>,
    n_vars: usize,
    degree: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: Field + TwoAdicField,
    EF: Field + ExtensionField<F> + TwoAdicField,
    StandardUniform: Distribution<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
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

pub fn verify_with_univariate_skip<EF, F, Challenger, const DIGEST_ELEMS: usize>(
    verifier_state: &mut VerifierState<EF, F, Challenger, DIGEST_ELEMS>,
    degree: usize,
    n_vars: usize,
    skips: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: Field + TwoAdicField,
    EF: Field + ExtensionField<F> + TwoAdicField,
    StandardUniform: Distribution<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
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

fn verify_core<EF, F, Challenger, const DIGEST_ELEMS: usize>(
    verifier_state: &mut VerifierState<EF, F, Challenger, DIGEST_ELEMS>,
    max_degree_per_vars: &[usize],
    sumation_sets: Vec<Vec<EF>>,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    StandardUniform: Distribution<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    assert_eq!(max_degree_per_vars.len(), sumation_sets.len(),);
    let mut challenges = Vec::new();
    let mut first_round = true;
    let (mut sum, mut target) = (EF::ZERO, EF::ZERO);

    for (&deg, sumation_set) in max_degree_per_vars.iter().zip(sumation_sets) {
        let coeffs = verifier_state.proof_data.piop.remove(0);
        let pol = WhirDensePolynomial::from_coefficients_vec(coeffs);

        let computed_sum = sumation_set.iter().map(|&s| pol.evaluate(s)).sum();
        if first_round {
            first_round = false;
            sum = computed_sum;
        } else if target != computed_sum {
            return Err(SumcheckError::InvalidRound);
        }
        let challenge = verifier_state.challenger.sample_algebra_element();

        let pow_bits = grinding.pow_bits::<EF>(deg);
        let pow_witness = verifier_state.proof_data.pow_witnesses.remove(0);
        assert!(
            verifier_state
                .challenger
                .check_witness(pow_bits, pow_witness),
            "Witness does not satisfy the PoW condition"
        );

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
