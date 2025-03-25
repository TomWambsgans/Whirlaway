#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use algebra::{
    field_utils::eq_extension,
    pols::{
        ComposedPolynomial, DenseMultilinearPolynomial, Evaluation, HypercubePoint,
        PartialHypercubePoint, UnivariatePolynomial,
    },
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
    eq_factor: Option<&[EF]>,
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
    eq_factor: Option<&[EF]>,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
    summation: &S,
) -> (Vec<EF>, ComposedPolynomial<F, EF, EF>) {
    assert!(pol.n_vars >= 1);
    let n_rounds = n_rounds.unwrap_or(pol.n_vars);
    let max_degree_per_vars = pol.max_degree_per_vars();
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.len(), pol.n_vars);
    }
    let mut challenges = Vec::new();
    let mut sum = sum.unwrap_or_else(|| pol.sum_over_hypercube());
    let mut folded_pol;

    folded_pol = sc_round(
        pol,
        eq_factor,
        fs_prover,
        max_degree_per_vars[0],
        &mut sum,
        pow_bits,
        &mut challenges,
        summation,
        0,
    );
    for i in 1..n_rounds {
        folded_pol = sc_round(
            folded_pol,
            eq_factor,
            fs_prover,
            max_degree_per_vars[i],
            &mut sum,
            pow_bits,
            &mut challenges,
            summation,
            i,
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
    eq_factor: Option<&[EF]>,
    fs_prover: &mut FsProver,
    degree: usize,
    sum: &mut EF,
    pow_bits: usize,
    challenges: &mut Vec<EF>,
    summation: &S,
    round: usize,
) -> ComposedPolynomial<F, EF, EF> {
    let _span = if round <= 2 {
        Some(tracing::span!(tracing::Level::INFO, "Sumcheck round").entered())
    } else {
        None
    };
    let mut p_evals = Vec::<(EF, EF)>::new();
    let eq_mle = if let Some(eq_factor) = &eq_factor {
        DenseMultilinearPolynomial::eq_mle(&eq_factor[1 + round..])
    } else {
        DenseMultilinearPolynomial::zero(0)
    };
    for z in 0..=degree as u32 {
        let sum_z = if z == 1 {
            if let Some(eq_factor) =eq_factor {
                let f = eq_extension(&eq_factor[..round], &challenges);
                (*sum - p_evals[0].1 * f * (EF::ONE - eq_factor[round])) / (f * eq_factor[round])
            } else {
                *sum - p_evals[0].1
            }
        } else {
            if eq_factor.is_some() {
                summation
                    .non_zero_points(z, pol.n_vars)
                    .into_par_iter()
                    .map(|point| {
                        pol.eval_partial_hypercube(&point) * eq_mle.eval_hypercube(&point.right)
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
    let mut p = UnivariatePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some(eq_factor) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        let f = eq_extension(&eq_factor[..round], &challenges);
        p *= UnivariatePolynomial::new(vec![
            f * (EF::ONE - eq_factor[round]),
            f * ((eq_factor[round] * EF::TWO) - EF::ONE),
        ]);
    }

    fs_prover.add_scalars(&p.coeffs);
    let challenge = fs_prover.challenge_scalars(1)[0];

    // Do PoW if needed
    if pow_bits > 0 {
        fs_prover.challenge_pow(pow_bits);
    }

    let pol = pol.fix_variable(challenge);
    challenges.push(challenge);
    *sum = p.eval(&challenge);
    pol
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
