use algebra::{
    field_utils::eq_extension,
    pols::{
        ComposedPolynomial, HypercubePoint, MultilinearPolynomial, PartialHypercubePoint,
        UnivariatePolynomial,
    },
};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

pub fn prove<F: Field, NF: ExtensionField<F>, EF: ExtensionField<NF> + ExtensionField<F>>(
    pol: ComposedPolynomial<F, NF, EF>,
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
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
        is_zerofier,
        fs_prover,
        max_degree_per_vars[0],
        &mut sum,
        pow_bits,
        &mut challenges,
        0,
    );
    for i in 1..n_rounds {
        folded_pol = sc_round(
            folded_pol,
            eq_factor,
            is_zerofier,
            fs_prover,
            max_degree_per_vars[i],
            &mut sum,
            pow_bits,
            &mut challenges,
            i,
        );
    }
    (challenges, folded_pol)
}

fn sc_round<F: Field, NF: ExtensionField<F>, EF: ExtensionField<NF> + ExtensionField<F>>(
    pol: ComposedPolynomial<F, NF, EF>,
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    degree: usize,
    sum: &mut EF,
    pow_bits: usize,
    challenges: &mut Vec<EF>,
    round: usize,
) -> ComposedPolynomial<F, EF, EF> {
    let _span = if round <= 2 {
        Some(tracing::span!(tracing::Level::INFO, "Sumcheck round").entered())
    } else {
        None
    };
    let mut p_evals = Vec::<(EF, EF)>::new();
    let eq_mle = if let Some(eq_factor) = &eq_factor {
        MultilinearPolynomial::eq_mle(&eq_factor[1 + round..])
    } else {
        MultilinearPolynomial::zero(0)
    };

    let start = if is_zerofier && round == 0 {
        p_evals.push((EF::ZERO, EF::ZERO));
        p_evals.push((EF::ONE, EF::ZERO));
        2
    } else {
        0
    };
    for z in start..=degree as u32 {
        let sum_z = if z == 1 {
            if let Some(eq_factor) = eq_factor {
                let f = eq_extension(&eq_factor[..round], &challenges);
                (*sum - p_evals[0].1 * f * (EF::ONE - eq_factor[round])) / (f * eq_factor[round])
            } else {
                *sum - p_evals[0].1
            }
        } else {
            if eq_factor.is_some() {
                (0..1 << (pol.n_vars - 1))
                    .into_par_iter()
                    .map(|x| {
                        pol.eval_partial_hypercube(&PartialHypercubePoint::new(
                            z,
                            pol.n_vars - 1,
                            x,
                        )) * eq_mle.eval_hypercube(&HypercubePoint::new(eq_mle.n_vars, x))
                    })
                    .sum::<EF>()
            } else {
                (0..1 << (pol.n_vars - 1))
                    .into_par_iter()
                    .map(|x| {
                        pol.eval_partial_hypercube(&PartialHypercubePoint::new(
                            z,
                            pol.n_vars - 1,
                            x,
                        ))
                    })
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
