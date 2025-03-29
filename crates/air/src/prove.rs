use algebra::{
    field_utils::dot_product,
    pols::{
        ArithmeticCircuit, ComposedPolynomial, CustomTransparentMultivariatePolynomial, Evaluation,
        GenericTransparentMultivariatePolynomial, MultilinearPolynomial,
    },
    utils::expand_randomness,
};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use pcs::{BatchSettings, PCS};
use rayon::prelude::*;
use tracing::{Level, instrument, span};

use super::table::AirTable;

/* Multi Column CCS (SuperSpartan)

cf https://eprint.iacr.org/2023/552.pdf and https://solvable.group/posts/super-air/#fnref:1

*/

impl<F: Field> AirTable<F> {
    #[instrument(name = "air: prove", skip_all)]
    pub fn prove<EF: ExtensionField<F>, Pcs: PCS<F, EF>>(
        &self,
        fs_prover: &mut FsProver,
        batching: &mut BatchSettings<F, EF, Pcs>,
        witness: &[MultilinearPolynomial<F>],
    ) {
        let log_length = witness[0].n_vars;
        assert!(witness.iter().all(|w| w.n_vars == log_length));

        for boundary_condition in &self.boundary_conditions {
            // TODO: no need to send scalar to transcript, verifier already knows it
            batching.register_claim(
                boundary_condition.col,
                boundary_condition.encode::<EF>(log_length),
                fs_prover,
            );
        }

        let constraints_batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];

        let zerocheck_challenges = fs_prover.challenge_scalars::<EF>(log_length);

        let zeroed_pol = {
            let mut nodes = Vec::<MultilinearPolynomial<F>>::with_capacity(self.n_columns * 2);

            for n in witnesses_up_and_down(witness) {
                nodes.push(n);
            }

            let mut batched_constraints = Vec::new();
            for (scalar, constraint) in
                expand_randomness(constraints_batching_scalar, self.constraints.len())
                    .into_iter()
                    .zip(self.constraints.iter())
            {
                batched_constraints.push((scalar, constraint.expr.coefs.clone()));
            }

            ComposedPolynomial::new(
                log_length,
                nodes,
                CustomTransparentMultivariatePolynomial::new(
                    self.n_columns * 2,
                    batched_constraints,
                ),
            )
        };

        let (outer_challenges, _) = {
            let _span = span!(Level::INFO, "outer sumcheck").entered();
            sumcheck::prove::<F, F, EF>(
                zeroed_pol,
                Some(&zerocheck_challenges),
                true,
                fs_prover,
                Some(EF::ZERO),
                None,
                0,
            )
        };

        let _span = span!(Level::INFO, "inner sumchecks").entered();
        let initial_transcript_len = fs_prover.transcript_len();

        let inner_sums = witnesses_up_and_down(witness)
            .par_iter()
            .map(|n| n.eval(&outer_challenges))
            .collect::<Vec<_>>();
        fs_prover.add_scalars(&inner_sums);

        let batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];

        let batch_pol = {
            let mut nodes = Vec::<MultilinearPolynomial<EF>>::with_capacity(self.n_columns * 2 + 2);
            let mut scalar = EF::ONE;
            for _ in 0..2 {
                // up and down
                let mut sum = MultilinearPolynomial::<EF>::zero(log_length);
                for w in witness {
                    sum += w.scale(scalar);
                    scalar *= batching_scalar;
                }
                nodes.push(sum);
            }
            nodes.push(matrix_up_folded(&outer_challenges));
            nodes.push(matrix_down_folded(&outer_challenges));

            let circuit = (ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(2))
                + (ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(3));

            let structure = GenericTransparentMultivariatePolynomial::new(circuit, 4);

            ComposedPolynomial::new(log_length, nodes, structure)
        };

        let inner_sum = dot_product(
            &inner_sums,
            &expand_randomness(batching_scalar, self.n_columns * 2),
        );
        let (inner_challenges, _) =
            sumcheck::prove(batch_pol, None, false, fs_prover, Some(inner_sum), None, 0);

        let values = witness
            .par_iter()
            .map(|w| w.eval(&inner_challenges))
            .collect::<Vec<_>>();
        for u in 0..self.n_columns {
            batching.register_claim(
                u,
                Evaluation {
                    point: inner_challenges.clone(),
                    value: values[u],
                },
                fs_prover,
            );
        }

        tracing::info!(
            "inner sumchecks transcript length: {:.1}Kib",
            (fs_prover.transcript_len() - initial_transcript_len) as f64 / 1024.
        );
    }
}

fn witnesses_up_and_down<F: Field>(
    witnesses: &[MultilinearPolynomial<F>],
) -> Vec<MultilinearPolynomial<F>> {
    let mut res = Vec::with_capacity(witnesses.len() * 2);
    res.extend(witnesses_up(witnesses));
    res.extend(witnesses_down(witnesses));
    res
}

fn witnesses_up<F: Field>(witnesses: &[MultilinearPolynomial<F>]) -> Vec<MultilinearPolynomial<F>> {
    let mut res = Vec::with_capacity(witnesses.len());
    for w in witnesses {
        let mut up = w.clone();
        up.evals[w.n_coefs() - 1] = up.evals[w.n_coefs() - 2];
        res.push(up);
    }
    res
}

fn witnesses_down<F: Field>(
    witnesses: &[MultilinearPolynomial<F>],
) -> Vec<MultilinearPolynomial<F>> {
    let mut res = Vec::with_capacity(witnesses.len());
    for w in witnesses {
        let mut down = w.evals[1..].to_vec();
        down.push(*down.last().unwrap());
        res.push(MultilinearPolynomial::new(down));
    }
    res
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
