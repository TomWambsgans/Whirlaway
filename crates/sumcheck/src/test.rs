use algebra::pols::{Multilinear, MultilinearDevice, MultilinearHost, MultilinearsSlice};
use arithmetic_circuit::{CircuitComputation, TransparentPolynomial};
use cuda_engine::{SumcheckComputation, memcpy_htod};
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{ExtensionField, Field, extension::BinomialExtensionField};
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::powers;

use super::*;

// TODO make it work with F = KoalaBear
type F = BinomialExtensionField<KoalaBear, 8>;
type EF = BinomialExtensionField<KoalaBear, 8>;

#[test]
fn test_sumcheck() {
    let n_vars = 15;
    let n_exprs = 10;
    let n_multilinears = 20;
    let rng = &mut StdRng::seed_from_u64(0);
    let exprs = (0..n_exprs)
        .map(|_| TransparentPolynomial::random(rng, n_multilinears, 1).fix_computation(true))
        .collect::<Vec<_>>();

    cuda_engine::init::<KoalaBear>(
        &[SumcheckComputation {
            exprs: &exprs,
            n_multilinears,
            eq_mle_multiplier: false,
        }],
        0,
    );

    for gpu in [true, false] {
        let multilinears_host = (0..n_multilinears)
            .map(|_| MultilinearHost::<F>::random(rng, n_vars))
            .collect::<Vec<_>>();
        let batching_scalar: EF = rng.random();
        let batching_scalars = powers(batching_scalar, n_exprs);

        let sum = MultilinearsSlice::Host(multilinears_host.iter().collect())
            .sum_over_hypercube_of_computation(
                &SumcheckComputation {
                    exprs: &exprs,
                    n_multilinears,
                    eq_mle_multiplier: false,
                },
                &batching_scalars,
            );

        let multilinears = if gpu {
            multilinears_host
                .iter()
                .map(|m| Multilinear::Device(MultilinearDevice::new(memcpy_htod(&m.evals))))
                .collect::<Vec<_>>()
        } else {
            multilinears_host
                .iter()
                .map(|m| Multilinear::Host(m.clone()))
                .collect::<Vec<_>>()
        };

        let mut fs_prover = FsProver::new();

        let time = std::time::Instant::now();
        prove(
            &multilinears,
            &exprs,
            &batching_scalars,
            None,
            false,
            &mut fs_prover,
            Some(sum),
            None,
            0,
        );
        println!(
            "{} sumcheck: {} ms",
            if gpu { "GPU" } else { "CPU" },
            time.elapsed().as_millis()
        );

        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let max_degree_per_vars = exprs
            .iter()
            .map(|expr| expr.composition_degree)
            .max()
            .unwrap();
        let (claimed_sum, postponed_verification) =
            verify::<EF>(&mut fs_verifier, &vec![max_degree_per_vars; n_vars], 0).unwrap();
        assert_eq!(sum, claimed_sum);
        assert_eq!(
            eval_batched_exprs_of_multilinears(
                &multilinears_host,
                &exprs,
                &batching_scalars,
                &postponed_verification.point
            ),
            postponed_verification.value
        );
    }
}

pub fn eval_batched_exprs<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    point: &[NF],
) -> EF {
    if exprs.len() == 1 {
        EF::from(exprs[0].eval(point))
    } else {
        exprs
            .iter()
            .zip(batching_scalars)
            .skip(1)
            .map(|(expr, scalar)| *scalar * expr.eval(point))
            .sum::<EF>()
            + exprs[0].eval(point)
    }
}

pub fn eval_batched_exprs_of_multilinears<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    multilinears: &[MultilinearHost<NF>],
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    point: &[EF],
) -> EF {
    let inner_evals = multilinears
        .iter()
        .map(|pol| pol.evaluate(point))
        .collect::<Vec<_>>();
    eval_batched_exprs(exprs, batching_scalars, &inner_evals)
}
