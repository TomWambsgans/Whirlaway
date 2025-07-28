/*
Logup* (Lev Soukhanov)

https://eprint.iacr.org/2025/946.pdf

with custom GKR

*/

use p3_field::{ExtensionField, Field, PrimeField64, dot_product};
use rayon::prelude::*;

use sumcheck::{SumcheckComputation, SumcheckComputationPacked, SumcheckGrinding};
use tracing::{info_span, instrument};
use utils::{Evaluation, FSChallenger, FSProver, FSVerifier, PF};
use whir_p3::fiat_shamir::errors::ProofError;
use whir_p3::poly::dense::WhirDensePolynomial;
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::poly::multilinear::MultilinearPoint;

/*
Custom GKR to compute sum of fractions.

A: [a0, a1, a2, a3, a4, a5, a6, a7]
B: [b0, b1, b2, b3, b4, b5, b6, b7]
AB: [a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7]
AB' = [a0.b4 + a4.b0, a1.b5 +a5.b1, a2.b6 + a6.b2, a3.b7 + a7.b3, b0.b4, b1.b5, b2.b6, b3.b7] (sum of quotients 2 by 2)

For i = (i1, i2, ..., i_{n-1}) on the hypercube:
AB'(i1, i2, ..., i_{n-1}) = i1.AB(1, 0, i2, i3, ..., i_{n-1}).AB(1, 1, i2, i3, ..., i_{n-1})
                            + (1 - i1).[AB(0, 0, i2, i3, ..., i_{n-1}).AB(1, 1, i2, i3, ..., i_{n-1}) + AB(0, 1, i2, i3, ..., i_{n-1}).AB(1, 0, i2, i3, ..., i_{n-1})]
                          = i1.AB(1 0 --- ).AB(1 1 --- ) + (1 - i1).[AB(0 0 --- ).AB(1 1 --- ) + AB(0 1 --- ).AB(1 0 --- )]
                          = U4.U2.U3 + U5.[U0.U3 + U1.U2]
with: U0 = AB(0 0 --- )
      U1 = AB(0  1 ---)
      U2 = AB(1 0 --- )
      U3 = AB(1 1 --- )
      U4 = i1
      U5 = (1 - i1)

*/

#[instrument(skip_all)]
pub fn prove_gkr<EF: Field>(
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    final_layer: EvaluationsList<EF>,
) -> Evaluation<EF>
where
    EF: ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    PF<EF>: PrimeField64,
{
    let n = final_layer.num_variables();
    assert!(n >= 2);
    let mut layers = Vec::new();
    layers.push(final_layer);
    for i in 0..n - 1 {
        layers.push(sum_quotients_2_by_2(&layers[i]));
    }

    assert_eq!(layers[n - 1].num_variables(), 1);
    prover_state.add_extension_scalars(&layers[n - 1].evals());

    let mut point = MultilinearPoint(vec![prover_state.sample()]);
    let mut claim = layers[n - 1].evaluate(&point);

    for layer in layers.iter().rev().skip(1) {
        let (next_point, next_claim) = prove_gkr_step(prover_state, layer, &point, claim).into();
        point = next_point;
        claim = next_claim;
    }

    (point, claim).into()
}

pub fn verify_gkr<EF: Field>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    n_vars: usize,
) -> Result<(EF, Evaluation<EF>), ProofError>
where
    EF: ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    PF<EF>: PrimeField64,
{
    let [a, b] = verifier_state.next_extension_scalars_const()?;
    if b == EF::ZERO {
        return Err(ProofError::InvalidProof);
    }
    let quotient = a / b;

    let mut point = MultilinearPoint(vec![verifier_state.sample()]);
    let mut claim = EvaluationsList::new(vec![a, b]).evaluate(&point);

    for i in 1..n_vars {
        let (next_point, next_claim) = verify_gkr_step(verifier_state, i, &point, claim)?.into();
        point = next_point;
        claim = next_claim;
    }

    Ok((quotient, (point, claim).into()))
}

#[instrument(skip_all)]
fn prove_gkr_step<EF: Field>(
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    up_layer: &EvaluationsList<EF>,
    point: &MultilinearPoint<EF>,
    eval: EF,
) -> Evaluation<EF>
where
    EF: ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    PF<EF>: PrimeField64,
{
    assert_eq!(up_layer.num_variables() - 1, point.0.len());
    let len = up_layer.num_evals();
    let mid_len = len / 2;
    let quarter_len = mid_len / 2;

    let up_layer = up_layer.evals();

    let _first_sumcheck_round_span = info_span!("first_sumcheck_round").entered();

    let eq_poly = info_span!("eq_poly").in_scope(|| EvaluationsList::eval_eq(&point.0[1..]));

    let mut sums_x = EF::zero_vec(up_layer.len() / 4);
    let mut sums_one_minus_x = EF::zero_vec(up_layer.len() / 4);

    sums_x
        .par_iter_mut()
        .zip(sums_one_minus_x.par_iter_mut())
        .enumerate()
        .for_each(|(i, (x, one_minus_x))| {
            let eq_eval = eq_poly.evals()[i];
            let u0 = up_layer[i];
            let u1 = up_layer[quarter_len + i];
            let u2 = up_layer[mid_len + i];
            let u3 = up_layer[mid_len + quarter_len + i];
            *x = eq_eval * u2 * u3;
            *one_minus_x = eq_eval * (u0 * u3 + u1 * u2);
        });

    let sum_x = sums_x.into_par_iter().sum::<EF>();
    let sum_one_minus_x = sums_one_minus_x.into_par_iter().sum::<EF>();

    let first_sumcheck_polynomial = &WhirDensePolynomial::from_coefficients_vec(vec![
        EF::ONE - point[0],
        point[0].double() - EF::ONE,
    ]) * &WhirDensePolynomial::from_coefficients_vec(vec![
        sum_one_minus_x,
        sum_x - sum_one_minus_x,
    ]);

    // sanity check
    assert_eq!(
        first_sumcheck_polynomial.evaluate(EF::ZERO) + first_sumcheck_polynomial.evaluate(EF::ONE),
        eval
    );

    prover_state.add_extension_scalars(&first_sumcheck_polynomial.coeffs);

    let first_sumcheck_challenge = prover_state.sample();

    let next_sum = first_sumcheck_polynomial.evaluate(first_sumcheck_challenge);

    let (u0_folded, u1_folded, u2_folded, u3_folded) = (
        &up_layer[..quarter_len],
        &up_layer[quarter_len..mid_len],
        &up_layer[mid_len..mid_len + quarter_len],
        &up_layer[mid_len + quarter_len..],
    );

    let u4_const = first_sumcheck_challenge;
    let u5_const = EF::ONE - first_sumcheck_challenge;
    let missing_mul_factor = first_sumcheck_challenge * point[0]
        + (EF::ONE - first_sumcheck_challenge) * (EF::ONE - point[0]);

    std::mem::drop(_first_sumcheck_round_span);

    let (sc_point, inner_evals) = if up_layer.len() == 4 {
        (
            MultilinearPoint(vec![first_sumcheck_challenge]),
            vec![
                EvaluationsList::new(u0_folded.to_vec()),
                EvaluationsList::new(u1_folded.to_vec()),
                EvaluationsList::new(u2_folded.to_vec()),
                EvaluationsList::new(u3_folded.to_vec()),
            ],
        )
    } else {
        let (mut sc_point, inner_evals, _) =
            info_span!("remaining sumcheck rounds").in_scope(|| {
                sumcheck::prove::<PF<EF>, EF, EF, _, _>(
                    1,
                    &[u0_folded, u1_folded, u2_folded, u3_folded],
                    &GKRQuotientComputation { u4_const, u5_const },
                    2,
                    &[EF::ONE],
                    Some(&point.0[1..]),
                    false,
                    prover_state,
                    next_sum,
                    None,
                    SumcheckGrinding::None,
                    Some(missing_mul_factor),
                    false,
                )
            });
        sc_point.insert(0, first_sumcheck_challenge);
        (sc_point, inner_evals)
    };

    let quarter_evals = inner_evals[..4]
        .iter()
        .map(|e| e.as_constant())
        .collect::<Vec<_>>();

    prover_state.add_extension_scalars(&quarter_evals);

    let mixing_challenge_a = prover_state.sample();
    let mixing_challenge_b = prover_state.sample();

    let mut next_point = sc_point.clone();
    next_point.0.insert(0, mixing_challenge_a);
    next_point.0[1] = mixing_challenge_b;

    let next_claim = dot_product::<EF, _, _>(
        quarter_evals.into_iter(),
        EvaluationsList::eval_eq(&[mixing_challenge_a, mixing_challenge_b])
            .evals()
            .iter()
            .cloned(),
    );

    (next_point, next_claim).into()
}

fn verify_gkr_step<EF: Field>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    current_layer_log_len: usize,
    point: &MultilinearPoint<EF>,
    eval: EF,
) -> Result<Evaluation<EF>, ProofError>
where
    EF: ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    PF<EF>: PrimeField64,
{
    let (sc_eval, postponed) = sumcheck::verify_with_custom_degree_at_first_round(
        verifier_state,
        current_layer_log_len,
        2,
        3,
        SumcheckGrinding::None,
    )
    .map_err(|_| ProofError::InvalidProof)?;

    if sc_eval != eval {
        panic!()
    }

    let [q0, q1, q2, q3] = verifier_state.next_extension_scalars_const()?;

    let postponed_target = point.eq_poly_outside(&postponed.point)
        * (postponed.point.0[0] * q2 * q3 + (EF::ONE - postponed.point.0[0]) * (q0 * q3 + q1 * q2));
    if postponed_target != postponed.value {
        return Err(ProofError::InvalidProof);
    }

    let mixing_challenge_a = verifier_state.sample();
    let mixing_challenge_b = verifier_state.sample();

    let mut next_point = postponed.point.clone();
    next_point.0.insert(0, mixing_challenge_a);
    next_point.0[1] = mixing_challenge_b;

    let next_claim = dot_product::<EF, _, _>(
        [q0, q1, q2, q3].into_iter(),
        EvaluationsList::eval_eq(&[mixing_challenge_a, mixing_challenge_b])
            .evals()
            .iter()
            .cloned(),
    );

    Ok((next_point, next_claim).into())
}

pub struct GKRQuotientComputation<EF> {
    u4_const: EF,
    u5_const: EF,
}

impl<F: Field, EF: ExtensionField<F>> SumcheckComputation<F, EF, EF>
    for GKRQuotientComputation<EF>
{
    fn eval(&self, point: &[EF], _: &[EF]) -> EF {
        // U4.U2.U3 + U5.[U0.U3 + U1.U2]
        self.u4_const * point[2] * point[3]
            + self.u5_const * (point[0] * point[3] + point[1] * point[2])
    }
}

impl<F: Field, EF: ExtensionField<F>> SumcheckComputationPacked<F, EF>
    for GKRQuotientComputation<EF>
{
    fn eval_packed(
        &self,
        _: &[<F as Field>::Packing],
        _: &[EF],
        _: &[Vec<F>],
    ) -> impl Iterator<Item = EF> + Send + Sync {
        // Unreachable
        std::iter::once(EF::ZERO)
    }
}

fn sum_quotients_2_by_2<EF: Field>(layer: &EvaluationsList<EF>) -> EvaluationsList<EF> {
    let n = layer.num_evals();
    EvaluationsList::new(
        (0..n / 2)
            .into_par_iter()
            .map(|i| {
                if i < n / 4 {
                    layer[i] * layer[n * 3 / 4 + i] + layer[n / 4 + i] * layer[n / 2 + i]
                } else {
                    layer[n / 4 + i] * layer[n / 2 + i]
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_challenger::DuplexChallenger;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 8>;

    type Poseidon16 = Poseidon2KoalaBear<16>;

    type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

    fn sum_all_quotients(layer: &EvaluationsList<EF>) -> EF {
        (0..layer.num_evals() / 2)
            .into_par_iter()
            .map(|i| layer[i] / layer[layer.num_evals() / 2 + i])
            .sum()
    }

    #[test]
    fn test_gkr_step() {
        let log_n = 10;
        let n = 1 << log_n;

        let mut rng = StdRng::seed_from_u64(0);

        let big = EvaluationsList::new((0..n).map(|_| rng.random()).collect::<Vec<EF>>());
        let small = sum_quotients_2_by_2(&big);

        // sanity check
        assert_eq!(
            (0..n / 2).map(|i| big[i] / big[n / 2 + i]).sum::<EF>(),
            (0..n / 4).map(|i| small[i] / small[n / 4 + i]).sum::<EF>()
        );

        let point = MultilinearPoint((0..log_n - 1).map(|_| rng.random()).collect::<Vec<EF>>());
        let eval = small.evaluate(&point);

        let poseidon16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
        let challenger = MyChallenger::new(poseidon16);
        let mut prover_state = FSProver::new(challenger.clone());

        prove_gkr_step(&mut prover_state, &big, &point, eval);

        let mut verifier_state = FSVerifier::new(prover_state.proof_data().to_vec(), challenger);

        let postponed = verify_gkr_step(&mut verifier_state, log_n - 1, &point, eval).unwrap();
        assert_eq!(big.evaluate(&postponed.point), postponed.value);
    }

    #[test]
    fn test_gkr() {
        let log_n = 10;
        let n = 1 << log_n;

        let mut rng = StdRng::seed_from_u64(0);

        let layer = EvaluationsList::new((0..n).map(|_| rng.random()).collect::<Vec<EF>>());
        let real_quotient = sum_all_quotients(&layer);

        let poseidon16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
        let challenger = MyChallenger::new(poseidon16);
        let mut prover_state = FSProver::new(challenger.clone());

        prove_gkr(&mut prover_state, layer.clone());

        let mut verifier_state = FSVerifier::new(prover_state.proof_data().to_vec(), challenger);

        let (retrieved_quotient, postponed) = verify_gkr::<EF>(&mut verifier_state, log_n).unwrap();
        assert_eq!(layer.evaluate(&postponed.point), postponed.value);
        assert_eq!(retrieved_quotient, real_quotient);
    }
}
