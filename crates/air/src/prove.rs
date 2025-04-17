use algebra::pols::{Multilinear, MultilinearDevice, MultilinearHost, MultilinearsVec};
use arithmetic_circuit::ArithmeticCircuit;
use cuda_engine::{cuda_sync, memcpy_htod};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use pcs::PCS;
use rayon::prelude::*;
use tracing::{Level, instrument, span};
use utils::{Evaluation, HypercubePoint, dot_product, powers};

use crate::{UNIVARIATE_SKIPS, utils::columns_up_and_down};

use super::table::AirTable;

/* Multi Column CCS (SuperSpartan)

cf https://eprint.iacr.org/2023/552.pdf and https://solvable.group/posts/super-air/#fnref:1

*/

impl<F: Field> AirTable<F> {
    #[instrument(name = "air: prove", skip_all)]
    pub fn prove<EF: ExtensionField<F>, Pcs: PCS<F, EF>>(
        &self,
        fs_prover: &mut FsProver,
        pcs: &Pcs,
        witness_: Vec<MultilinearHost<F>>,
        cuda: bool,
    ) {
        assert!(
            UNIVARIATE_SKIPS < self.log_length,
            "TODO handle the case UNIVARIATE_SKIPS == log_length"
        );
        let log_length = self.log_length;
        assert!(witness_.iter().all(|w| w.n_vars == log_length));

        // 1) Commit to the witness columns

        let witness = if cuda {
            MultilinearsVec::Device(
                witness_
                    .iter()
                    .map(|w| MultilinearDevice::new(memcpy_htod(&w.evals)))
                    .collect(),
            )
        } else {
            MultilinearsVec::Host(witness_)
        };

        // TODO avoid cloning (use a row major matrix for the witness)

        let packed_pol = witness.as_ref().packed();
        cuda_sync();
        let packed_pol_witness = pcs.commit(packed_pol, fs_prover);

        let constraints_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];
        let constraints_batching_scalars =
            powers(constraints_batching_scalar, self.constraints.len());

        let zerocheck_challenges =
            fs_prover.challenge_scalars::<EF>(log_length + 1 - UNIVARIATE_SKIPS);

        let preprocessed_columns = if cuda {
            MultilinearsVec::Device(
                self.preprocessed_columns
                    .iter()
                    .map(|w| MultilinearDevice::new(memcpy_htod(&w.evals)))
                    .collect(),
            )
        } else {
            MultilinearsVec::Host(self.preprocessed_columns.clone())
        };
        let preprocessed_and_witness = preprocessed_columns.as_ref().chain(&witness.as_ref());
        let (outer_challenges, all_inner_sums, _) = {
            let _span = span!(Level::INFO, "outer sumcheck").entered();
            sumcheck::prove(
                UNIVARIATE_SKIPS,
                columns_up_and_down(&preprocessed_and_witness).as_ref(),
                &self.constraints,
                &constraints_batching_scalars,
                Some(&zerocheck_challenges),
                true,
                fs_prover,
                EF::ZERO,
                None,
                0,
                None,
            )
        };

        let _span = span!(Level::INFO, "inner sumchecks").entered();

        let inner_sums_up = &all_inner_sums[self.n_preprocessed_columns()..self.n_columns];
        let inner_sums_down = &all_inner_sums[self.n_columns + self.n_preprocessed_columns()..];
        let _span_evals = span!(Level::INFO, "transfering column evalutations from cuda").entered();
        let inner_sums = inner_sums_up
            .into_iter()
            .chain(inner_sums_down)
            .map(|s| s.evaluate::<EF>(&[]))
            .collect::<Vec<_>>();
        cuda_sync();
        std::mem::drop(_span_evals);
        fs_prover.add_scalars(&inner_sums);

        let inner_sumcheck_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];

        let mles_for_inner_sumcheck = {
            let _span_mles = span!(Level::INFO, "constructing MLEs").entered();
            let mut nodes = Vec::<Multilinear<EF>>::new();
            let _span_linear_comb = span!(Level::INFO, "linear combination of columns").entered();
            let expanded_scalars =
                powers(inner_sumcheck_batching_scalar, 2 * self.n_witness_columns());
            for i in 0..2 {
                // up and down
                let sum = witness
                    .as_ref()
                    .linear_comb(
                        &expanded_scalars
                            [i * self.n_witness_columns()..(i + 1) * self.n_witness_columns()],
                    )
                    .add_dummy_starting_variables(UNIVARIATE_SKIPS); // TODO this is not efficient
                nodes.push(sum);
            }
            cuda_sync();
            std::mem::drop(_span_linear_comb);
            nodes.push(matrix_up_folded_with_univariate_skips(
                &outer_challenges,
                cuda,
                UNIVARIATE_SKIPS,
            ));
            nodes.push(matrix_down_folded_with_univariate_skips(
                &outer_challenges,
                cuda,
                UNIVARIATE_SKIPS,
            ));

            // TODO remove
            let expanded = MultilinearHost::new(
                self.univariate_selectors
                    .iter()
                    .map(|s| s.eval(&outer_challenges[0]))
                    .collect(),
            )
            .add_dummy_ending_variables(log_length);
            nodes.push(match cuda {
                false => expanded.into(),
                true => MultilinearDevice::new(memcpy_htod(&expanded.evals)).into(), // bad
            });

            nodes
        };

        let inner_sumcheck_circuit = ArithmeticCircuit::Node(4)
            * ((ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(2))
                + (ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(3)));

        let inner_sum = dot_product(
            &inner_sums,
            &powers(inner_sumcheck_batching_scalar, self.n_witness_columns() * 2),
        );

        let (inner_challenges, _, _) = sumcheck::prove(
            1,
            &mles_for_inner_sumcheck,
            &[inner_sumcheck_circuit.fix_computation(false)],
            &[EF::ONE],
            None,
            false,
            fs_prover,
            inner_sum,
            None,
            0,
            None,
        );

        let _span_evals = span!(Level::INFO, "evaluating witness").entered();
        let values = witness
            .as_ref()
            .batch_evaluate(&inner_challenges[UNIVARIATE_SKIPS..]);

        cuda_sync();
        std::mem::drop(_span_evals);
        fs_prover.add_scalars(&values);

        let final_random_scalars = fs_prover.challenge_scalars::<EF>(self.log_n_witness_columns());
        let final_point = [
            final_random_scalars.clone(),
            inner_challenges[UNIVARIATE_SKIPS..].to_vec(),
        ]
        .concat();
        let packed_value = MultilinearHost::new(
            [
                values,
                vec![EF::ZERO; (1 << self.log_n_witness_columns()) - self.n_witness_columns()],
            ]
            .concat(),
        )
        .evaluate(&final_random_scalars);
        let packed_eval = Evaluation {
            point: final_point,
            value: packed_value,
        };

        std::mem::drop(_span);

        pcs.open(packed_pol_witness, &packed_eval, fs_prover);
    }
}

/// Async
#[instrument(name = "matrix_up_folded_with_univariate_skips", skip_all)]
fn matrix_up_folded_with_univariate_skips<F: Field>(
    outer_challenges: &[F],
    on_device: bool,
    univariate_skips: usize,
) -> Multilinear<F> {
    // TODO: It's sparse => bad performance
    let n = outer_challenges.len();
    let n_vars = n + univariate_skips * 2 - 1;
    let mut folded = MultilinearHost::zero(n_vars);
    let point_len = univariate_skips + (n - 1);
    let inner_mle = MultilinearHost::eq_mle(&outer_challenges[1..]);
    folded
        .evals
        .par_chunks_mut(1 << point_len)
        .enumerate()
        .for_each(|(i, block)| {
            block[i << (n - 1)..(i + 1) << (n - 1)].copy_from_slice(&inner_mle.evals);
        });

    let outer_challenges_prod = outer_challenges[1..].iter().copied().product::<F>();
    folded.evals[(1 << n_vars) - 1] -= outer_challenges_prod;
    folded.evals[(1 << n_vars) - 2] += outer_challenges_prod;

    // TODO do it on the device directly
    if on_device {
        MultilinearDevice::new(memcpy_htod(&folded.evals)).into()
    } else {
        folded.into()
    }
}

/// Async
#[instrument(name = "matrix_down_folded_with_univariate_skips", skip_all)]
fn matrix_down_folded_with_univariate_skips<F: Field>(
    outer_challenges: &[F],
    on_device: bool,
    univariate_skips: usize,
) -> Multilinear<F> {
    // TODO: It's sparse => bad performance
    let n = outer_challenges.len();
    // n_vars defined as in the original function.
    let n_vars = n + univariate_skips * 2 - 1;
    let mut folded = MultilinearHost::zero(n_vars);
    let point_len = univariate_skips + (n - 1);
    let inner_mles = (1..outer_challenges.len())
        .map(|i| MultilinearHost::eq_mle(&outer_challenges[1..i]))
        .collect::<Vec<_>>();
    folded
        .evals
        .par_chunks_mut(1 << point_len)
        .enumerate()
        .for_each(|(i, block)| {
            let x = HypercubePoint {
                n_vars: univariate_skips,
                val: i,
            };
            let mut point = x.to_vec();
            point.extend_from_slice(&outer_challenges[1..]);
            let m = point.len();
            for k in 0..m {
                let outer_challenges_prod =
                    (F::ONE - point[m - k - 1]) * point[m - k..].iter().copied().product::<F>();
                if outer_challenges_prod.is_zero() {
                    continue;
                }
                let eq_mle = &inner_mles[(m - k - 1).saturating_sub(x.n_vars)];
                // MultilinearHost::eq_mle(&outer_challenges[0..n - k - 1]);
                let eq_mle = eq_mle.scale(outer_challenges_prod);
                let n_coefs = eq_mle.n_coefs();
                for (mut i, v) in eq_mle.evals.into_iter().enumerate() {
                    i += (x.val >> (x.n_vars - x.n_vars.min(m - k - 1))) * n_coefs;
                    i <<= k + 1;
                    i += 1 << k;
                    block[i] += v;
                }
            }
        });

    folded.evals[(1 << n_vars) - 1] += outer_challenges[1..].iter().copied().product::<F>();

    // TODO do it on the device directly
    if on_device {
        MultilinearDevice::new(memcpy_htod(&folded.evals)).into()
    } else {
        folded.into()
    }
}
