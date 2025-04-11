use algebra::pols::{Multilinear, MultilinearDevice, MultilinearHost, MultilinearsVec};
use arithmetic_circuit::ArithmeticCircuit;
use cuda_engine::{cuda_sync, memcpy_htod};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use pcs::PCS;
use rayon::prelude::*;
use tracing::{Level, instrument, span};
use utils::{Evaluation, dot_product, powers};

use crate::utils::columns_up_and_down;

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

        // let rng = &mut StdRng::seed_from_u64(0);
        // dbg!(packed_pol.embed::<EF>().evaluate(&(0..packed_pol.n_vars()).map(|_| EF::random( rng)).collect::<Vec<EF>>()));

        // dbg!(fs_prover.state_hex());

        let packed_pol_witness = pcs.commit(packed_pol, fs_prover);
        // dbg!(fs_prover.state_hex());

        let constraints_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];
        let constraints_batching_scalars =
            powers(constraints_batching_scalar, self.constraints.len());

        let zerocheck_challenges = fs_prover.challenge_scalars::<EF>(log_length);

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

        let (outer_challenges, all_inner_sums) = {
            let _span = span!(Level::INFO, "outer sumcheck").entered();
            sumcheck::prove(
                columns_up_and_down(&preprocessed_and_witness)
                    .as_ref()
                    .embed::<EF>()
                    .as_ref(), // TODO avoid this embedding
                &self.constraints,
                &constraints_batching_scalars,
                Some(&zerocheck_challenges),
                true,
                fs_prover,
                Some(EF::ZERO),
                None,
                0,
            )
        };

        let _span = span!(Level::INFO, "inner sumchecks").entered();

        let inner_sums_up = &all_inner_sums[self.n_preprocessed_columns()..self.n_columns];
        let inner_sums_down = &all_inner_sums[self.n_columns + self.n_preprocessed_columns()..];
        let inner_sums = inner_sums_up
            .into_iter()
            .chain(inner_sums_down)
            .map(|s| s.evaluate::<EF>(&[]))
            .collect::<Vec<_>>();
        cuda_sync();
        fs_prover.add_scalars(&inner_sums);

        let inner_sumcheck_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];

        let witness = witness.decompose();
        let mles_for_inner_sumcheck = {
            let mut nodes = Vec::<Multilinear<EF>>::with_capacity(self.n_witness_columns() * 2 + 2);
            let mut scalar = EF::ONE;
            for _ in 0..2 {
                // up and down
                let mut sum = Multilinear::<EF>::zero(log_length, cuda);
                for w in &witness {
                    sum += w.scale(scalar);
                    scalar *= inner_sumcheck_batching_scalar;
                }
                nodes.push(sum);
            }
            nodes.push(matrix_up_folded(&outer_challenges, cuda));
            nodes.push(matrix_down_folded(&outer_challenges, cuda));

            nodes
        };

        let inner_sumcheck_circuit = (ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(2))
            + (ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(3));

        let inner_sum = dot_product(
            &inner_sums,
            &powers(inner_sumcheck_batching_scalar, self.n_witness_columns() * 2),
        );

        let (inner_challenges, _) = sumcheck::prove(
            &mles_for_inner_sumcheck,
            &[inner_sumcheck_circuit.fix_computation(false)],
            &[EF::ONE],
            None,
            false,
            fs_prover,
            Some(inner_sum),
            None,
            0,
        );

        let values = witness
            .par_iter()
            .map(|w| w.embed::<EF>().evaluate(&inner_challenges)) // TODO avoid this embedding
            .collect::<Vec<_>>();
        cuda_sync();
        fs_prover.add_scalars(&values);

        let final_random_scalars = fs_prover.challenge_scalars::<EF>(self.log_n_witness_columns());
        let final_point = [final_random_scalars.clone(), inner_challenges].concat();
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
fn matrix_up_folded<F: Field>(outer_challenges: &[F], on_device: bool) -> Multilinear<F> {
    let n = outer_challenges.len();
    let mut folded = MultilinearHost::eq_mle(&outer_challenges);
    let outer_challenges_prod: F = outer_challenges.iter().copied().product();
    folded.evals[(1 << n) - 1] -= outer_challenges_prod;
    folded.evals[(1 << n) - 2] += outer_challenges_prod;

    // TODO do it on the device directly
    if on_device {
        MultilinearDevice::new(memcpy_htod(&folded.evals)).into()
    } else {
        folded.into()
    }
}

fn matrix_down_folded<F: Field>(outer_challenges: &[F], on_device: bool) -> Multilinear<F> {
    let n = outer_challenges.len();
    let mut folded = vec![F::ZERO; 1 << n];
    for k in 0..n {
        let outer_challenges_prod = (F::ONE - outer_challenges[n - k - 1])
            * outer_challenges[n - k..].iter().copied().product::<F>();
        let mut eq_mle = MultilinearHost::eq_mle(&outer_challenges[0..n - k - 1]);
        eq_mle = eq_mle.scale(outer_challenges_prod);
        for (mut i, v) in eq_mle.evals.into_iter().enumerate() {
            i <<= k + 1;
            i += 1 << k;
            folded[i] += v;
        }
    }
    // bottom left corner:
    folded[(1 << n) - 1] += outer_challenges.iter().copied().product::<F>();

    // TODO do it on the device directly
    if on_device {
        MultilinearDevice::new(memcpy_htod(&folded)).into()
    } else {
        MultilinearHost::new(folded).into()
    }
}
