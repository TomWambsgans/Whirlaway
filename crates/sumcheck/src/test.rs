use algebra::pols::{Multilinear, MultilinearDevice, MultilinearHost, MultilinearsVec};
use arithmetic_circuit::{CircuitComputation, TransparentPolynomial};
use cuda_engine::{
    SumcheckComputation, cuda_init, cuda_preprocess_sumcheck_computation, memcpy_htod,
};
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{ExtensionField, Field, extension::BinomialExtensionField};
use p3_koala_bear::KoalaBear;
use rand::{Rng, SeedableRng, rngs::StdRng};
use utils::{eq_extension, powers};

use super::*;

// type F = KoalaBear;
type EF = BinomialExtensionField<KoalaBear, 8>;

// TODO make it work with multilinears in the prime field

#[test]
fn test_sumcheck() {
    let n_vars = 11;
    let n_exprs = 10;
    let n_multilinears = 20;
    let rng = &mut StdRng::seed_from_u64(0);
    let exprs = (0..n_exprs)
        .map(|_| {
            TransparentPolynomial::<KoalaBear>::random(rng, n_multilinears, 1).fix_computation(true)
        })
        .collect::<Vec<_>>();
    let eq_factor = (0..n_vars).map(|_| EF::random(rng)).collect::<Vec<_>>();

    cuda_init();
    let sumcheck_computation = SumcheckComputation {
        exprs: &exprs,
        n_multilinears: n_multilinears + 1,
        eq_mle_multiplier: true,
    };
    cuda_preprocess_sumcheck_computation(&sumcheck_computation);

    for gpu in [true, false] {
        let multilinears_host = (0..n_multilinears)
            .map(|_| MultilinearHost::<EF>::random(rng, n_vars))
            .collect::<Vec<_>>();
        let multilinears = if gpu {
            MultilinearsVec::Device(
                multilinears_host
                    .iter()
                    .map(|m| MultilinearDevice::new(memcpy_htod(&m.evals)))
                    .collect::<Vec<_>>(),
            )
        } else {
            MultilinearsVec::Host(
                multilinears_host
                    .iter()
                    .map(|m| m.clone())
                    .collect::<Vec<_>>(),
            )
        };

        let batching_scalar: EF = rng.random();
        let batching_scalars = powers(batching_scalar, n_exprs);
        let eq_mle = Multilinear::eq_mle(&eq_factor, gpu);
        let sum = multilinears.as_ref().sum_over_hypercube_of_computation(
            &sumcheck_computation,
            &batching_scalars,
            Some(&eq_mle),
        );

        let mut fs_prover = FsProver::new();

        let time = std::time::Instant::now();
        prove(
            multilinears.as_ref(),
            &exprs,
            &batching_scalars,
            Some(&eq_factor),
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
            verify::<EF>(&mut fs_verifier, &vec![1 + max_degree_per_vars; n_vars], 0).unwrap();
        assert_eq!(sum, claimed_sum);

        assert_eq!(
            eval_batched_exprs_of_multilinears(
                &multilinears_host,
                &exprs,
                &batching_scalars,
                &postponed_verification.point
            ) * eq_extension(&postponed_verification.point, &eq_factor),
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
