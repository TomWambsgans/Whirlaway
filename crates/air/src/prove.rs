use algebra::{
    field_utils::dot_product,
    pols::{ArithmeticCircuit, Evaluation, MultilinearPolynomial},
    utils::expand_randomness,
};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use pcs::PCS;
use rayon::prelude::*;
use tracing::{Level, instrument, span};

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
        witness: &[MultilinearPolynomial<F>],
        cuda: bool,
    ) {
        let log_length = self.log_length;
        assert!(witness.iter().all(|w| w.n_vars == log_length));

        // 1) Commit to the witness columns

        // TODO avoid cloning (use a row major matrix for the witness)
        let mut batch_evals = vec![F::ZERO; 1 << (log_length + self.log_n_witness_columns())];
        for (i, poly) in witness.iter().enumerate() {
            batch_evals[i << log_length..(i + 1) << log_length].copy_from_slice(&poly.evals);
        }
        let packed_pol = MultilinearPolynomial::new(batch_evals);

        let packed_pol_witness = pcs.commit(packed_pol, fs_prover);

        let constraints_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];
        let constraints_batching_scalars =
            expand_randomness(constraints_batching_scalar, self.constraints.len());

        let zerocheck_challenges = fs_prover.challenge_scalars::<EF>(log_length);

        let preprocessed_and_witness = self
            .preprocessed_columns
            .iter()
            .chain(witness)
            .collect::<Vec<_>>();

        let (outer_challenges, all_inner_sums) = {
            let _span = span!(Level::INFO, "outer sumcheck").entered();
            if cuda {
                sumcheck::prove_with_cuda(
                    columns_up_and_down(&preprocessed_and_witness),
                    &self.constraints,
                    &constraints_batching_scalars,
                    Some(&zerocheck_challenges),
                    true,
                    fs_prover,
                    Some(EF::ZERO),
                    None,
                    0,
                )
            } else {
                sumcheck::prove::<F, F, EF>(
                    columns_up_and_down(&preprocessed_and_witness),
                    &self.constraints,
                    &constraints_batching_scalars,
                    Some(&zerocheck_challenges),
                    true,
                    fs_prover,
                    Some(EF::ZERO),
                    None,
                    0,
                )
            }
        };

        let _span = span!(Level::INFO, "inner sumchecks").entered();

        let inner_sums_up = &all_inner_sums[self.n_preprocessed_columns()..self.n_columns];
        let inner_sums_down = &all_inner_sums[self.n_columns + self.n_preprocessed_columns()..];
        let inner_sums = inner_sums_up
            .into_iter()
            .chain(inner_sums_down)
            .map(|s| s.eval::<EF>(&[]))
            .collect::<Vec<_>>();
        fs_prover.add_scalars(&inner_sums);

        let inner_sumcheck_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];

        let mles_for_inner_sumcheck = {
            let mut nodes =
                Vec::<MultilinearPolynomial<EF>>::with_capacity(self.n_witness_columns() * 2 + 2);
            let mut scalar = EF::ONE;
            for _ in 0..2 {
                // up and down
                let mut sum = MultilinearPolynomial::<EF>::zero(log_length);
                for w in witness {
                    sum += w.scale(scalar);
                    scalar *= inner_sumcheck_batching_scalar;
                }
                nodes.push(sum);
            }
            nodes.push(matrix_up_folded(&outer_challenges));
            nodes.push(matrix_down_folded(&outer_challenges));

            nodes
        };

        let inner_sumcheck_circuit = (ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(2))
            + (ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(3));

        let inner_sum = dot_product(
            &inner_sums,
            &expand_randomness(inner_sumcheck_batching_scalar, self.n_witness_columns() * 2),
        );
        let (inner_challenges, _) = sumcheck::prove(
            mles_for_inner_sumcheck,
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
            .map(|w| w.eval(&inner_challenges))
            .collect::<Vec<_>>();
        fs_prover.add_scalars(&values);

        let final_random_scalars = fs_prover.challenge_scalars::<EF>(self.log_n_witness_columns());
        let final_point = [final_random_scalars.clone(), inner_challenges].concat();
        let packed_value = MultilinearPolynomial::new(
            [
                values,
                vec![EF::ZERO; (1 << self.log_n_witness_columns()) - self.n_witness_columns()],
            ]
            .concat(),
        )
        .eval(&final_random_scalars);
        let packed_eval = Evaluation {
            point: final_point,
            value: packed_value,
        };

        std::mem::drop(_span);

        pcs.open(packed_pol_witness, &packed_eval, fs_prover);
    }
}

fn matrix_up_folded<F: Field>(outer_challenges: &[F]) -> MultilinearPolynomial<F> {
    let n = outer_challenges.len();
    let mut folded = MultilinearPolynomial::eq_mle(&outer_challenges);
    let outer_challenges_prod: F = outer_challenges.iter().copied().product();
    folded.evals[(1 << n) - 1] -= outer_challenges_prod;
    folded.evals[(1 << n) - 2] += outer_challenges_prod;
    folded
}

fn matrix_down_folded<F: Field>(outer_challenges: &[F]) -> MultilinearPolynomial<F> {
    let n = outer_challenges.len();
    let mut folded = vec![F::ZERO; 1 << n];
    for k in 0..n {
        let outer_challenges_prod = (F::ONE - outer_challenges[n - k - 1])
            * outer_challenges[n - k..].iter().copied().product::<F>();
        let mut eq_mle = MultilinearPolynomial::eq_mle(&outer_challenges[0..n - k - 1]);
        eq_mle = eq_mle.scale(outer_challenges_prod);
        for (mut i, v) in eq_mle.evals.into_iter().enumerate() {
            i <<= k + 1;
            i += 1 << k;
            folded[i] += v;
        }
    }
    // bottom left corner:
    folded[(1 << n) - 1] += outer_challenges.iter().copied().product::<F>();

    MultilinearPolynomial::new(folded)
}
