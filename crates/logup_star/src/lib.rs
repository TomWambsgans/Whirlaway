/*
Logup* (Lev Soukhanov)

https://eprint.iacr.org/2025/946.pdf

*/

use p3_field::{ExtensionField, Field, PrimeField64};
use rayon::prelude::*;

use p3_field::PrimeCharacteristicRing;
use sumcheck::{ProductComputation, SumcheckGrinding};
use tracing::{info_span, instrument};
use utils::{FSChallenger, FSProver, FSVerifier, PF};
use whir_p3::fiat_shamir::errors::ProofError;
use whir_p3::poly::evals::EvaluationsList;
use whir_p3::poly::multilinear::MultilinearPoint;
use whir_p3::utils::parallel_clone;

use crate::gkr::{prove_gkr, verify_gkr};

pub mod gkr;

#[instrument(skip_all)]
pub fn prove_logup_star<EF: Field>(
    prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    table: &EvaluationsList<PF<EF>>,
    indexes: &EvaluationsList<PF<EF>>,
    values: &EvaluationsList<PF<EF>>,
    point: &MultilinearPoint<EF>, // "r" in the paper
) where
    EF: ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    PF<EF>: PrimeField64,
{
    let table_length = table.len();
    let indexes_length = indexes.len();
    let eval = values.evaluate(&point);
    prover_state.add_extension_scalar(eval);

    let poly_eq_point = EvaluationsList::eval_eq(&point.0);
    let pushforward = compute_pushforward(indexes.evals(), table_length, poly_eq_point.evals());

    // commit to pushforward
    // TODO
    let table_embedded =
        EvaluationsList::new(table.evals().par_iter().map(|&x| EF::from(x)).collect()); // TODO avoid embedding

    let (_sc_point, inner_evals, prod) =
        info_span!("logup_star sumcheck", table_length, indexes_length).in_scope(|| {
            sumcheck::prove::<PF<EF>, EF, EF,  _>(
                1,
                [table_embedded.evals(), pushforward.evals()].to_vec(),
                &ProductComputation,
                2,
                &[EF::ONE],
                None,
                false,
                prover_state,
                eval,
                None,
                SumcheckGrinding::None,
                None,
                false,
            )
        });

    // open table at sc_point
    let table_eval = inner_evals[0].evaluate(&Default::default());
    prover_state.add_extension_scalar(table_eval); // phony opening for now

    // open pushforward at sc_point
    let pushforwardt_eval = inner_evals[1].evaluate(&Default::default());
    prover_state.add_extension_scalar(pushforwardt_eval); // phony opening for now

    // sanity check
    assert_eq!(prod, table_eval * pushforwardt_eval);

    // "c" in the paper
    let random_challenge = prover_state.sample();

    let mut gkr_layer_left = EF::zero_vec(indexes_length * 2);
    parallel_clone(&poly_eq_point, &mut gkr_layer_left[..indexes_length]);
    gkr_layer_left[indexes_length..]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)| {
            *x = random_challenge - indexes.evals()[i];
        });

    let claim_left = prove_gkr(prover_state, EvaluationsList::new(gkr_layer_left));

    let mut gkr_layer_right = EF::zero_vec(table_length * 2);
    parallel_clone(&pushforward, &mut gkr_layer_right[..table_length]);
    gkr_layer_right[table_length..]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)| {
            *x = random_challenge - PF::<EF>::from_usize(i);
        });
    let claim_right = prove_gkr(prover_state, EvaluationsList::new(gkr_layer_right));

    // open Indexes at claim_left.point[1..]
    // phony opening for now
    prover_state
        .add_extension_scalar(indexes.evaluate(&MultilinearPoint(claim_left.point[1..].to_vec())));

    // open pushforward at claim_right.point[1..]
    // phony opening for now
    prover_state.add_extension_scalar(
        pushforward.evaluate(&MultilinearPoint(claim_right.point[1..].to_vec())),
    );
}

pub fn verify_logup_star<EF: Field>(
    verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    log_table_len: usize,
    log_indexes_len: usize,
    point: &MultilinearPoint<EF>, // "r" in the paper
) -> Result<EF, ProofError>
where
    EF: ExtensionField<PF<EF>> + ExtensionField<PF<PF<EF>>>,
    PF<EF>: PrimeField64,
{
    let claimed_eval = verifier_state.next_extension_scalar()?;

    // receive commitment of pushforward
    // TODO

    let (sum, postponed) =
        sumcheck::verify(verifier_state, log_table_len, 2, SumcheckGrinding::None)
            .map_err(|_| ProofError::InvalidProof)?;

    if sum != claimed_eval {
        return Err(ProofError::InvalidProof);
    }

    let table_eval = verifier_state.next_extension_scalar()?; // should be done by opening a commitment
    let pushforward_eval = verifier_state.next_extension_scalar()?; // should be done by opening a commitment

    if table_eval * pushforward_eval != postponed.value {
        return Err(ProofError::InvalidProof);
    }

    let random_challenge = verifier_state.sample(); // "c" in the paper

    let (quotient_left, claim_left) = verify_gkr(verifier_state, log_indexes_len + 1)?;
    let (quotient_right, claim_right) = verify_gkr(verifier_state, log_table_len + 1)?;

    if quotient_left != quotient_right {
        return Err(ProofError::InvalidProof);
    }

    let index_openined_value = verifier_state.next_extension_scalar()?; // Phony opening for now (at claim_left.point[1..])

    if claim_left.value
        != MultilinearPoint(claim_left.point[1..].to_vec()).eq_poly_outside(point)
            * (EF::ONE - claim_left.point[0])
            + (random_challenge - index_openined_value) * claim_left.point[0]
    {
        return Err(ProofError::InvalidProof);
    }

    let pushforward_openined_value = verifier_state.next_extension_scalar()?; // Phony opening for now (at claim_right.point[1..])

    let big_endian_mle = claim_right.point[1..]
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &p)| p * EF::TWO.exp_u64(i as u64))
        .sum::<EF>();

    if claim_right.value
        != pushforward_openined_value * (EF::ONE - claim_right.point[0])
            + (random_challenge - big_endian_mle) * claim_right.point[0]
    {
        return Err(ProofError::InvalidProof);
    }

    Ok(claimed_eval)
}

fn compute_pushforward<F: PrimeField64, EF: ExtensionField<EF>>(
    indexes: &[F],
    table_length: usize,
    i: &[EF],
) -> EvaluationsList<EF> {
    assert_eq!(indexes.len(), i.len());
    // TODO there are a lot of fun optimizations here
    let mut pushforward = EF::zero_vec(table_length);
    for (index, value) in indexes.iter().zip(i) {
        let index_usize = index.as_canonical_u64() as usize;
        pushforward[index_usize] += *value; // TODO unchecked arithmetic for performance
    }
    return EvaluationsList::new(pushforward);
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use tracing::level_filters::LevelFilter;
    use tracing_forest::ForestLayer;
    use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 8>;

    type Poseidon16 = Poseidon2KoalaBear<16>;

    type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;
    #[test]
    fn test_logup_star() {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();

        let log_table_len = 19;
        let table_len = 1 << log_table_len;

        let log_indexes_len = log_table_len + 2;
        let indexes_len = 1 << log_indexes_len;

        let mut rng = StdRng::seed_from_u64(0);

        let table = EvaluationsList::new((0..table_len).map(|_| rng.random()).collect::<Vec<F>>());

        let mut indexes = vec![];
        let mut values = vec![];
        for _ in 0..indexes_len {
            let index = rng.random_range(0..table_len);
            indexes.push(F::from_usize(index));
            values.push(table[index]);
        }
        let indexes = EvaluationsList::new(indexes);
        let values = EvaluationsList::new(values);

        // Commit to the table
        let commited_table = table.clone(); // Phony commitment for the example
        // commit to the indexes
        let commited_indexes = indexes.clone(); // Phony commitment for the example

        let poseidon16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));

        let challenger = MyChallenger::new(poseidon16);

        let point = MultilinearPoint(
            (0..log_indexes_len)
                .map(|_| rng.random())
                .collect::<Vec<EF>>(),
        );

        let mut prover_state = FSProver::new(challenger.clone());

        let time = std::time::Instant::now();
        prove_logup_star(
            &mut prover_state,
            &commited_table,
            &commited_indexes,
            &values,
            &point,
        );
        println!("Proving logup_star took {} ms", time.elapsed().as_millis());

        let mut verifier_state = FSVerifier::new(prover_state.proof_data().to_vec(), challenger);
        let result =
            verify_logup_star(&mut verifier_state, log_table_len, log_indexes_len, &point).unwrap();

        assert_eq!(result, values.evaluate(&point));
    }
}
