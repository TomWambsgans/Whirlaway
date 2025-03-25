use algebra::pols::{
    ComposedPolynomial, DenseMultilinearPolynomial, Evaluation, HypercubePoint,
    PartialHypercubePoint, UnivariatePolynomial,
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
    fn non_zero_points(&self, z: u32, n_vars: usize) -> Vec<PartialHypercubePoint>; // TODO return an iterator ?
}

pub struct FullSumcheckSummation;

impl SumcheckSummation for FullSumcheckSummation {
    fn non_zero_points(&self, z: u32, n_vars: usize) -> Vec<PartialHypercubePoint> {
        HypercubePoint::iter(n_vars - 1)
            .map(move |right| PartialHypercubePoint { left: z, right })
            .collect()
    }
}

pub fn prove<F: Field, NF: ExtensionField<F>, EF: ExtensionField<NF> + ExtensionField<F>>(
    pol: ComposedPolynomial<F, NF, EF>,
    eq_factor: Option<DenseMultilinearPolynomial<EF>>,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
) -> (Vec<EF>, ComposedPolynomial<F, EF, EF>) {
    prove_with_custom_summation(
        pol,
        eq_factor,
        fs_prover,
        sum,
        n_rounds,
        pow_bits,
        &FullSumcheckSummation,
    )
}

pub fn prove_with_custom_summation<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    S: SumcheckSummation,
>(
    pol: ComposedPolynomial<F, NF, EF>,
    mut eq_factor: Option<DenseMultilinearPolynomial<EF>>,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
    summation: &S,
) -> (Vec<EF>, ComposedPolynomial<F, EF, EF>) {
    assert!(pol.n_vars >= 1);
    let n_rounds = n_rounds.unwrap_or(pol.n_vars);
    let mut max_degree_per_vars = pol.max_degree_per_vars();
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.n_vars, pol.n_vars);
        for deg in &mut max_degree_per_vars {
            *deg += 1;
        }
    }
    let mut challenges = Vec::new();
    let mut sum = sum.unwrap_or_else(|| pol.sum_over_hypercube());
    let mut folded_pol;

    (folded_pol, eq_factor) = sc_round(
        pol,
        eq_factor,
        fs_prover,
        max_degree_per_vars[0],
        &mut sum,
        pow_bits,
        &mut challenges,
        summation,
    );
    for i in 1..n_rounds {
        (folded_pol, eq_factor) = sc_round(
            folded_pol,
            eq_factor,
            fs_prover,
            max_degree_per_vars[i],
            &mut sum,
            pow_bits,
            &mut challenges,
            summation,
        );
    }
    (challenges, folded_pol)
}

fn sc_round<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    S: SumcheckSummation,
>(
    pol: ComposedPolynomial<F, NF, EF>,
    mut eq_factor: Option<DenseMultilinearPolynomial<EF>>,
    fs_prover: &mut FsProver,
    degree: usize,
    sum: &mut EF,
    pow_bits: usize,
    challenges: &mut Vec<EF>,
    summation: &S,
) -> (
    ComposedPolynomial<F, EF, EF>,
    Option<DenseMultilinearPolynomial<EF>>,
) {
    let mut p_evals = Vec::<(EF, EF)>::new();
    for z in 0..=degree as u32 {
        let sum_z = if z == 1 {
            *sum - p_evals[0].1
        } else {
            if let Some(eq_factor) = &eq_factor {
                summation
                    .non_zero_points(z, pol.n_vars)
                    .into_par_iter()
                    .map(|point| {
                        pol.eval_partial_hypercube(&point)
                            * eq_factor.eval_partial_hypercube(&point)
                    })
                    .sum::<EF>()
            } else {
                summation
                    .non_zero_points(z, pol.n_vars)
                    .into_par_iter()
                    .map(|point| pol.eval_partial_hypercube(&point))
                    .sum::<EF>()
            }
        };
        p_evals.push((EF::from_u32(z), sum_z));
    }
    let p = UnivariatePolynomial::lagrange_interpolation(&p_evals).unwrap();

    fs_prover.add_scalars(&p.coeffs);
    let challenge = fs_prover.challenge_scalars(1)[0];

    // Do PoW if needed
    if pow_bits > 0 {
        fs_prover.challenge_pow(pow_bits);
    }

    let pol = pol.fix_variable(challenge);
    eq_factor = eq_factor.map(|eq_factor| eq_factor.fix_variable(challenge));
    challenges.push(challenge);
    *sum = p.eval(&challenge);
    (pol, eq_factor)
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
        prove(pol.clone(), None, &mut fs_prover, None, None, 0);

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
