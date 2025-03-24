use algebra::{
    field_utils::dot_product,
    pols::{
        ArithmeticCircuit, ComposedPolynomial, DenseMultilinearPolynomial, Evaluation,
        MultilinearPolynomial, TransparentMultivariatePolynomial,
    },
    utils::expand_randomness,
};
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};
use pcs::{BatchSettings, PCS};
use tracing::{Level, instrument, span};

use super::table::AirTable;

/* Multi Column CCS (SuperSpartan)

cf https://eprint.iacr.org/2023/552.pdf and https://solvable.group/posts/super-air/#fnref:1

The witness is split into n columns: z1, ..., z_(c-1), each represented by a multivariate polynomial with log(n) variables.
M0, ..., M_(t-1) are t matrices of dimension log(m) x log(n).
C is an arithmic circuit with c entries.
col: {0, ..., t-1} -> {0, ..., c-1} assigns to each matrix u a witness column z_{col_u}.
The statement to prove is: for all i in {0, 1}^log(m), C(M0.z_{col_0}(i), ..., M_(c-1).z_{col_(c-1)}(i)) = 0.
which is equivalent (with negligible probability) to: sum_{i in {0, 1}^log(m)} eq(i0, ..., i_{log_m - 1}, tau0, ..., tau_{log_m - 1}) C(M0.z_{col_0}(i), ..., M_(c-1).z_{col_(c-1)}(i))) = 0
*/

impl<F: Field> AirTable<F> {
    #[instrument(name = "air: prove", skip_all)]
    pub fn prove<EF: ExtensionField<F>, Pcs: PCS<F, EF>>(
        &self,
        fs_prover: &mut FsProver,
        batching: &mut BatchSettings<F, EF, Pcs>,
        witness: &[DenseMultilinearPolynomial<F>],
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

        let global_constraint = self.get_global_constraint::<EF>(fs_prover);

        let zerocheck_challenges = fs_prover.challenge_scalars::<EF>(log_length);

        let mut zerofier_pol =
            self.compute_zerofier_pol(witness, &zerocheck_challenges, &global_constraint);

        let outer_challenges = {
            let _span = span!(Level::INFO, "outer sumcheck").entered();
            sumcheck::prove(&mut zerofier_pol, fs_prover, Some(EF::ZERO), None, 0) // sum should equal 0
        };

        let _span = span!(Level::INFO, "inner sumchecks").entered();
        let initial_transcript_len = fs_prover.transcript_len();

        let inner_sums = witnesses_up_and_down(witness)
            .iter()
            .map(|n| n.eval(&outer_challenges))
            .collect::<Vec<_>>();
        fs_prover.add_scalars(&inner_sums);

        let batching_scalar = fs_prover.challenge_scalars::<EF>(1)[0];

        let mut batch_pol = {
            let mut nodes = Vec::<MultilinearPolynomial<EF>>::with_capacity(self.n_columns * 2 + 2);
            let mut scalar = EF::ONE;
            for _ in 0..2 {
                // up and down
                for w in witness {
                    let mut scaled = w.embed::<EF>();
                    scaled.scale(scalar);
                    nodes.push(scaled.into());
                    scalar *= batching_scalar;
                }
            }
            nodes.push(matrix_up_folded(&outer_challenges).into());
            nodes.push(matrix_down_folded(&outer_challenges).into());

            let circuit = (ArithmeticCircuit::Node(self.n_columns * 2) // matrix_up_folded
                    * ArithmeticCircuit::new_sum(
                        (0..self.n_columns)
                            .map(|c| ArithmeticCircuit::Node(c))
                            .collect::<Vec<_>>(),
                    ))
                + (ArithmeticCircuit::Node(self.n_columns * 2 + 1) // matrix_down_folded
                    * ArithmeticCircuit::new_sum(
                        (self.n_columns..2 * self.n_columns)
                            .map(|c| ArithmeticCircuit::Node(c))
                            .collect::<Vec<_>>(),
                    ));

            let structure = TransparentMultivariatePolynomial::new(circuit, self.n_columns * 2 + 2);

            ComposedPolynomial::new_without_shift(log_length, nodes, structure)
        };

        let inner_sum = dot_product(
            &inner_sums,
            &expand_randomness(batching_scalar, self.n_columns * 2),
        );
        let inner_challenges = sumcheck::prove(&mut batch_pol, fs_prover, Some(inner_sum), None, 0);

        for u in 0..self.n_columns {
            let value = witness[u].eval(&inner_challenges);
            batching.register_claim(
                u,
                Evaluation {
                    point: inner_challenges.clone(),
                    value,
                },
                fs_prover,
            );
        }

        tracing::info!(
            "inner sumchecks transcript length: {:.1}Kib",
            (fs_prover.transcript_len() - initial_transcript_len) as f64 / 1024.
        );
    }

    #[instrument(name = "compute_zerofier_pol", skip_all)]
    fn compute_zerofier_pol<EF: ExtensionField<F>>(
        &self,
        witness: &[DenseMultilinearPolynomial<F>],
        zerocheck_challenges: &[EF],
        global_constraint: &TransparentMultivariatePolynomial<EF>,
    ) -> ComposedPolynomial<EF> {
        let log_length = witness[0].n_vars;

        let mut nodes = Vec::with_capacity(self.n_columns * 2 + 1);

        for n in witnesses_up_and_down(witness) {
            nodes.push(n.embed::<EF>().into()); // TODO avoid embed
        }

        nodes.push(DenseMultilinearPolynomial::eq_mle(&zerocheck_challenges).into());

        let circuit = ArithmeticCircuit::Node(self.n_columns * 2)
            * global_constraint.coefs.clone().embed::<EF>(); // TODO avoid embed

        ComposedPolynomial::new_without_shift(
            log_length,
            nodes,
            TransparentMultivariatePolynomial::new(circuit, self.n_columns * 2 + 1),
        )
    }
}

fn witnesses_up_and_down<F: Field>(
    witnesses: &[DenseMultilinearPolynomial<F>],
) -> Vec<DenseMultilinearPolynomial<F>> {
    let mut res = Vec::with_capacity(witnesses.len() * 2);
    res.extend(witnesses_up(witnesses));
    res.extend(witnesses_down(witnesses));
    res
}

fn witnesses_up<F: Field>(
    witnesses: &[DenseMultilinearPolynomial<F>],
) -> Vec<DenseMultilinearPolynomial<F>> {
    let mut res = Vec::with_capacity(witnesses.len());
    for w in witnesses {
        let mut up = w.clone();
        up.evals[w.n_coefs() - 1] = up.evals[w.n_coefs() - 2];
        res.push(up);
    }
    res
}

fn witnesses_down<F: Field>(
    witnesses: &[DenseMultilinearPolynomial<F>],
) -> Vec<DenseMultilinearPolynomial<F>> {
    let mut res = Vec::with_capacity(witnesses.len());
    for w in witnesses {
        // TODO opti
        let mut down = w.clone();
        down.evals.remove(0);
        down.evals.push(*down.evals.last().unwrap());
        res.push(down);
    }
    res
}

#[instrument(name = "matrix_up_folded", skip_all)]
fn matrix_up_folded<F: Field>(outer_challenges: &[F]) -> DenseMultilinearPolynomial<F> {
    let n = outer_challenges.len();
    let mut folded = DenseMultilinearPolynomial::eq_mle(&outer_challenges);
    let outer_challenges_prod: F = outer_challenges.iter().copied().product();
    folded.evals[(1 << n) - 1] -= outer_challenges_prod;
    folded.evals[(1 << n) - 2] += outer_challenges_prod;
    folded
}

#[instrument(name = "matrix_down_folded", skip_all)]
fn matrix_down_folded<F: Field>(outer_challenges: &[F]) -> DenseMultilinearPolynomial<F> {
    let n = outer_challenges.len();
    let mut folded = vec![F::ZERO; 1 << n];
    for k in 0..n {
        let outer_challenges_prod = (F::ONE - outer_challenges[n - k - 1])
            * outer_challenges[n - k..].iter().copied().product::<F>();
        let mut eq_mle = DenseMultilinearPolynomial::eq_mle(&outer_challenges[0..n - k - 1]);
        eq_mle.scale(outer_challenges_prod);
        for (mut i, v) in eq_mle.evals.into_iter().enumerate() {
            i <<= k + 1;
            i += 1 << k;
            folded[i] += v;
        }
    }
    // bottom left corner:
    folded[(1 << n) - 1] += outer_challenges.iter().copied().product::<F>();

    DenseMultilinearPolynomial::new(folded)
}
