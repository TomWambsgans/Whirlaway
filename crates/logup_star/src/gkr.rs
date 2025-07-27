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
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::poly::multilinear::MultilinearPoint;

/*
Custom GKR to compute sum of fractions.

A: [a0, a1, a2, a3]
B: [b0, b1, b2, b3]
AB: [a0, a1, a2, a3, b0, b1, b2, b3]
AB' = [a0.b1, + a1.b0, a2.b3 + a3.b2, b1.b2, b2.b3]

For i = (i1, i2, ..., i_{n-1}) on the hypercube:
AB'(i1, i2, ..., i_{n-1}) = i1.AB(1, i2, i3, ..., i_{n-1}, 0).AB(1, i2, i3, ..., i_{n-1}, 1)
                            + (1 - i1).[AB(0, i2, i3, ..., i_{n-1}, 0).AB(1, i2, i3, ..., i_{n-1}, 1) + AB(0, i2, i3, ..., i_{n-1}, 1).AB(1, i2, i3, ..., i_{n-1}, 0)]
                          = i1.AB(1 --- 0).AB(1 --- 1) + (1 - i1).[AB(0 --- 0).AB(1 --- 1) + AB(0 --- 1).AB(1 --- 0)]
                          = U4.U2.U3 + U5.[U0.U3 + U1.U2]
with: U0 = AB(0 --- 0)
      U1 = AB(0 --- 1)
      U2 = AB(1 --- 0)
      U3 = AB(1 --- 1)
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

fn verify_gkr<EF: Field>(
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
    assert!(len >= 4);
    let mid_len = len / 2;
    let quarter_len = mid_len / 2;
    let (u0, u1, u2, u3) = (
        EF::zero_vec(mid_len),
        EF::zero_vec(mid_len),
        EF::zero_vec(mid_len),
        EF::zero_vec(mid_len),
    );
    up_layer.evals()[..mid_len]
        .par_chunks_exact(2)
        .enumerate()
        .for_each(|(i, chunk)| unsafe {
            *(u0.as_ptr() as *mut EF).add(i) = chunk[0];
            *(u0.as_ptr() as *mut EF).add(i + quarter_len) = chunk[0];
            *(u1.as_ptr() as *mut EF).add(i) = chunk[1];
            *(u1.as_ptr() as *mut EF).add(i + quarter_len) = chunk[1];
        });

    up_layer.evals()[mid_len..]
        .par_chunks_exact(2)
        .enumerate()
        .for_each(|(i, chunk)| unsafe {
            *(u2.as_ptr() as *mut EF).add(i) = chunk[0];
            *(u2.as_ptr() as *mut EF).add(i + quarter_len) = chunk[0];
            *(u3.as_ptr() as *mut EF).add(i) = chunk[1];
            *(u3.as_ptr() as *mut EF).add(i + quarter_len) = chunk[1];
        });

    let mut u4 = EF::zero_vec(quarter_len); // i1 ...
    u4.extend(vec![EF::ONE; quarter_len]);
    let mut u5 = vec![EF::ONE; quarter_len]; // (1 - i1) ...
    u5.extend(EF::zero_vec(quarter_len));

    let (u0, u1, u2, u3, u4, u5) = (
        EvaluationsList::new(u0),
        EvaluationsList::new(u1),
        EvaluationsList::new(u2),
        EvaluationsList::new(u3),
        EvaluationsList::new(u4),
        EvaluationsList::new(u5),
    );

    let (sc_point, inner_evals, _) = info_span!("sumcheck").in_scope(|| {
        sumcheck::prove::<PF<EF>, EF, EF, _, _>(
            1,
            &[&u0, &u1, &u2, &u3, &u4, &u5],
            &GKRQuotientComputation,
            3,
            &[EF::ONE],
            Some(&point.0),
            false,
            prover_state,
            eval,
            None,
            SumcheckGrinding::None,
            None,
            false,
        )
    });

    let quarter_evals = inner_evals[..4]
        .iter()
        .map(|e| e.as_constant())
        .collect::<Vec<_>>();

    prover_state.add_extension_scalars(&quarter_evals);

    let mixing_challenge_a = prover_state.sample();
    let mixing_challenge_b = prover_state.sample();

    let mut next_point = sc_point.clone();
    next_point.0[0] = mixing_challenge_a;
    next_point.0.push(mixing_challenge_b);

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
    let (sc_eval, postponed) = sumcheck::verify(
        verifier_state,
        current_layer_log_len,
        4,
        SumcheckGrinding::None,
    )
    .map_err(|_| ProofError::InvalidProof)?;

    if sc_eval != eval {
        return Err(ProofError::InvalidProof);
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
    next_point.0[0] = mixing_challenge_a;
    next_point.0.push(mixing_challenge_b);

    let next_claim = dot_product::<EF, _, _>(
        [q0, q1, q2, q3].into_iter(),
        EvaluationsList::eval_eq(&[mixing_challenge_a, mixing_challenge_b])
            .evals()
            .iter()
            .cloned(),
    );

    Ok((next_point, next_claim).into())
}

pub struct GKRQuotientComputation;

impl<F: Field, EF: ExtensionField<F>> SumcheckComputation<F, EF, EF> for GKRQuotientComputation {
    fn eval(&self, point: &[EF], _: &[EF]) -> EF {
        // U4.U2.U3 + U5.[U0.U3 + U1.U2]
        point[4] * point[2] * point[3] + point[5] * (point[0] * point[3] + point[1] * point[2])
    }
}

impl<F: Field, EF: ExtensionField<F>> SumcheckComputationPacked<F, EF> for GKRQuotientComputation {
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
                    layer[2 * i] * layer[n / 2 + 2 * i + 1]
                        + layer[2 * i + 1] * layer[n / 2 + 2 * i]
                } else {
                    layer[2 * i] * layer[2 * i + 1]
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
