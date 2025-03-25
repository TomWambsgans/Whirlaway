/*
To commit to 2^β multilinear polynomials: f_0, f_1, ... (all in μ variables), we commit to F, where:
F(0, ..., 0, 0, 0, x_1, ..., x_k) = f_0(x_1, ..., x_k)
F(0, ..., 0, 0, 1, x_1, ..., x_k) = f_1(x_1, ..., x_k)
F(0, ..., 0, 1, 0, x_1, ..., x_k) = f_2(x_1, ..., x_k)
F(0, ..., 0, 1, 1, x_1, ..., x_k) = f_3(x_1, ..., x_k)
...
F have β + μ = κ variables

To prove the evalutions of f_0, f_1, ... on a a set of points, we simply need to be able to prove the evaluation of F on a corresponding
set of points (where we basically "encode" the polynomial index by ones and zero at the beginning of the point).

Suppose we have a list of 2^k claims of the form F(zi) = yi
This is equivalent to the cancellation of the following polynomial in the variables (X1, ..., Xk):

Sum_{ (i_1, ..., i_k) in {0, 1}^k } [
                                      eq((i_1, ..., i_k), (X1, ..., Xk))
                                      * [
                                          Sum_{ (b_1, ..., b_κ) in {0, 1}^κ } [ eq((b_1,  ..., b_κ), (zi)) * F(b_1, ..., b_κ) ]
                                          - y_i
                                        ]
                                    ]

The verifier will check this by chosing random challenges t_1, ..., t_k  in place of X1, ..., Xk, and by running a sumcheck.
*/

use fiat_shamir::{FsError, FsProver, FsVerifier};
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use sumcheck::{FullSumcheckSummation, SumcheckError, SumcheckSummation};
use tracing::instrument;

use algebra::{
    field_utils::eq_extension,
    pols::{
        ArithmeticCircuit, ComposedPolynomial, DenseMultilinearPolynomial, Evaluation,
        GenericTransparentMultivariatePolynomial, HypercubePoint, MixedEvaluation, MixedPoint,
        PartialHypercubePoint, SparseMultilinearPolynomial, concat_hypercube_points,
    },
};

use super::PCS;

#[derive(Clone)]
pub struct BatchSettings<F: Field, EF: ExtensionField<F>, Pcs: PCS<F, EF>> {
    pcs: Pcs,
    n_polys: usize,
    n_vars: usize, // all polynomials have the same number of variables "n_vars"
    pub claims: Vec<(usize, Evaluation<EF>)>, // (pol_index, eval)
    _small_field: std::marker::PhantomData<F>,
}

// there is a lot of duplication TODO
pub struct BatchWitness<F: Field, EF: ExtensionField<F>, Pcs: PCS<F, EF>> {
    pub packed: DenseMultilinearPolynomial<F>,
    pub polys: Vec<DenseMultilinearPolynomial<F>>,
    pub witness: Pcs::Witness,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchPcsError<InnerError> {
    Inner(InnerError),
    Sumcheck(SumcheckError),
    InvalidSum,
}

impl<F: Field, EF: ExtensionField<F>, Pcs: PCS<F, EF>> BatchSettings<F, EF, Pcs> {
    pub fn new(n_polys: usize, n_vars: usize, params: &Pcs::Params) -> Self {
        let log_n_polys = n_polys.next_power_of_two().trailing_zeros() as usize;
        BatchSettings {
            pcs: Pcs::new(log_n_polys + n_vars, &params),
            n_polys,
            n_vars,
            claims: vec![],
            _small_field: std::marker::PhantomData,
        }
    }

    pub fn k(&self) -> usize {
        self.claims.len().next_power_of_two().trailing_zeros() as usize
    }

    pub fn log_n_polys(&self) -> usize {
        self.n_polys.next_power_of_two().trailing_zeros() as usize
    }

    fn max_degree_per_vars_for_batching_sumcheck(&self) -> Vec<usize> {
        vec![2; self.k() + self.n_vars + self.log_n_polys()]
    }

    pub fn packed_claims(&self) -> Vec<MixedEvaluation<EF>> {
        self.claims
            .iter()
            .map(|(pol_index, eval)| MixedEvaluation {
                point: MixedPoint {
                    left: eval.point.clone(),
                    right: HypercubePoint {
                        n_vars: self.log_n_polys(),
                        val: *pol_index,
                    },
                },
                value: eval.value,
            })
            .collect()
    }

    #[instrument(name = "batch commit", skip_all)]
    pub fn commit(
        &self,
        fs_prover: &mut FsProver,
        polys: Vec<DenseMultilinearPolynomial<F>>,
    ) -> BatchWitness<F, EF, Pcs> {
        // Commit to a batch of polynomials, and later evaluate them individually at a an arbitrary number of points
        // For simplicity for now, we impose that all polynomials have the same number of variables
        let n_vars = polys[0].n_vars;
        assert!(polys.iter().all(|p| p.n_vars == n_vars));
        assert_eq!(polys.len(), self.n_polys);

        let mut batch_evals = vec![F::ZERO; 1 << (n_vars + self.log_n_polys())];
        for (i, poly) in polys.iter().enumerate() {
            for (j, eval) in poly.evals.iter().enumerate() {
                batch_evals[(j << self.log_n_polys()) + i] = *eval;
            }
        }
        let packed = DenseMultilinearPolynomial::new(batch_evals);

        let witness = self.pcs.commit(packed.clone(), fs_prover);

        BatchWitness {
            polys,
            packed,
            witness,
        }
    }

    pub fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<Pcs::ParsedCommitment, Pcs::VerifError> {
        self.pcs.parse_commitment(fs_verifier)
    }

    pub fn register_claim(
        &mut self,
        pol_index: usize,
        eval: Evaluation<EF>,
        fs_prover: &mut FsProver,
    ) {
        fs_prover.add_scalars(&[eval.value]);
        self.claims.push((pol_index, eval));
    }

    #[instrument(name = "batch: prove", skip_all)]
    pub fn prove(self, witness: BatchWitness<F, EF, Pcs>, fs_prover: &mut FsProver) {
        let kappa = witness.packed.n_vars;
        let k = self.k();
        let packed_claims = self.packed_claims();

        let t = fs_prover.challenge_scalars::<EF>(k);

        let mut nodes = vec![witness.packed.embed::<EF>().into()]; // TODO avoid
        let mut vars_shift = vec![0..kappa];

        nodes.push(DenseMultilinearPolynomial::eq_mle(&t).into());
        vars_shift.push(kappa..kappa + k);

        let eq_zi_b_evals = (0..packed_claims.len())
            .into_par_iter()
            .map(|u| {
                (
                    DenseMultilinearPolynomial::eq_mle(&packed_claims[u].point.left),
                    packed_claims[u].point.right.clone(),
                )
            })
            .collect::<Vec<_>>();

        let map = (0..packed_claims.len())
            .map(|u| HypercubePoint {
                n_vars: self.log_n_polys(),
                val: self.claims[u].0,
            })
            .collect::<Vec<_>>();

        let eq_zi_b = SparseMultilinearPolynomial::new(eq_zi_b_evals);
        nodes.push(eq_zi_b.into());
        vars_shift.push(0..k + kappa);

        let structure = GenericTransparentMultivariatePolynomial::new(
            ArithmeticCircuit::Node(0) * ArithmeticCircuit::Node(1) * ArithmeticCircuit::Node(2),
            3,
        );

        let g_star = ComposedPolynomial::<EF, EF>::new(k + kappa, nodes, vars_shift, structure);

        // TODO pass `t` as a zeriofier (arg `eq_factor`), but this require to improve the var shift mechnanism
        let (challenges, _) = sumcheck::prove_with_custom_summation(
            g_star,
            None,
            fs_prover,
            Some(self.s(&t, &packed_claims)),
            None,
            0,
            &SparseSumcheckSummation {
                k,
                n: self.log_n_polys(),
                map,
            },
        );

        let value = witness.packed.eval(&challenges[k..k + kappa]);
        self.pcs.open(
            witness.witness,
            &Evaluation {
                point: challenges[0..kappa].to_vec(),
                value,
            },
            fs_prover,
        );
    }

    pub fn receive_claim(
        &mut self,
        fs_verifier: &mut FsVerifier,
        pol_index: usize,
        point: &[EF],
    ) -> Result<EF, FsError> {
        let value = fs_verifier.next_scalars(1)?[0];
        self.claims.push((
            pol_index,
            Evaluation {
                point: point.to_vec(),
                value,
            },
        ));
        Ok(value)
    }

    #[instrument(name = "batch: verify", skip_all)]
    pub fn verify(
        self,
        fs_verifier: &mut FsVerifier,
        parsed_commitment: &Pcs::ParsedCommitment,
    ) -> Result<(), BatchPcsError<Pcs::VerifError>> {
        let kappa = self.log_n_polys() + self.n_vars;
        let k = self.claims.len().next_power_of_two().trailing_zeros() as usize;

        let packed_claims = self.packed_claims();

        let t = fs_verifier.challenge_scalars::<EF>(k);

        let (claimed_s, mut final_check) = sumcheck::verify::<EF>(
            fs_verifier,
            &self.max_degree_per_vars_for_batching_sumcheck(),
            0,
        )
        .map_err(BatchPcsError::Sumcheck)?;

        if self.s(&t, &packed_claims) != claimed_s {
            return Err(BatchPcsError::InvalidSum);
        }

        // TODO handle div by zero

        // eq((i), (t))
        final_check.value =
            final_check.value / eq_extension(&t, &final_check.point[kappa..kappa + k]);

        // MLE eq( (b), (zi) )
        final_check.value = final_check.value / {
            let mut evals = vec![EF::ZERO; packed_claims.len().next_power_of_two()];
            for i in 0..packed_claims.len() {
                evals[i] = eq_extension(
                    &packed_claims[i].point.to_vec(),
                    &final_check.point[..kappa],
                );
            }
            DenseMultilinearPolynomial::new(evals).eval(&final_check.point[kappa..kappa + k])
        };

        final_check.point = final_check.point[..kappa].to_vec();
        self.pcs
            .verify(parsed_commitment, &final_check, fs_verifier)
            .map_err(BatchPcsError::Inner)?;

        Ok(())
    }

    fn s(&self, t: &[EF], packed_claims: &[MixedEvaluation<EF>]) -> EF {
        let mut values = packed_claims.iter().map(|e| e.value).collect::<Vec<_>>();
        values.resize(1 << self.k(), EF::ZERO);
        DenseMultilinearPolynomial::new(values).eval(&t)
    }
}

// iterate first over the last k variables: for each value X = x_1, x2, ..., x_k (in big indian),
// map(X) tels us the only combination of n ending variables (the ending variables) of the (n_vars - k) first variables for
// which the sumcheck polynomial is not zero.
struct SparseSumcheckSummation {
    k: usize,
    n: usize,
    map: Vec<HypercubePoint>,
}

impl SumcheckSummation for SparseSumcheckSummation {
    fn non_zero_points(&self, z: u32, n_vars: usize) -> Vec<algebra::pols::PartialHypercubePoint> {
        if n_vars <= self.n + self.k {
            FullSumcheckSummation.non_zero_points(z, n_vars)
        } else {
            let n_first_vars = n_vars - 1 - self.k - self.n;
            HypercubePoint::iter(self.k)
                .enumerate()
                .map(move |(k_index, k_vars)| {
                    if k_index >= self.map.len() {
                        return vec![];
                    }
                    HypercubePoint::iter(n_first_vars)
                        .map(move |first_vars| {
                            concat_hypercube_points(&[
                                first_vars,
                                self.map[k_index].clone(),
                                k_vars.clone(),
                            ])
                        })
                        .collect::<Vec<_>>()
                })
                .flatten()
                .map(move |right| PartialHypercubePoint { left: z, right })
                .collect()
        }
    }
}
#[cfg(test)]
mod test {

    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use whir::parameters::WhirParameters;

    use crate::{RingSwitch, WhirPCS};

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<KoalaBear, 4>;

    #[test]
    fn test_batch_pcs() {
        let n_vars = 6;
        let n_polys: usize = 5;
        let n_claims = 21;
        let security_bits = 45;
        let log_inv_rate = 4;
        let mut batch_prover = BatchSettings::<F, EF, RingSwitch<F, EF, WhirPCS<EF>>>::new(
            n_polys,
            n_vars,
            &WhirParameters::standard(security_bits, log_inv_rate),
        );
        let rng = &mut StdRng::seed_from_u64(0);

        let mut fs_prover = FsProver::new();

        let polys = (0..n_polys)
            .map(|_| DenseMultilinearPolynomial::<F>::random(rng, n_vars))
            .collect::<Vec<_>>();
        let witness = batch_prover.commit(&mut fs_prover, polys);

        let mut pol_index = 0;
        let mut claims_to_verify = vec![];
        for _ in 0..n_claims {
            let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<_>>();
            let value = witness.polys[pol_index].eval(&point);
            let eval = Evaluation { point, value };
            claims_to_verify.push((pol_index, eval.clone()));
            batch_prover.register_claim(pol_index, eval, &mut fs_prover);
            pol_index = (pol_index + 1) % n_polys;
        }

        batch_prover.prove(witness, &mut fs_prover);

        let mut batch_verifier = BatchSettings::<F, EF, RingSwitch<F, EF, WhirPCS<EF>>>::new(
            n_polys,
            n_vars,
            &WhirParameters::standard(security_bits, log_inv_rate),
        );
        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let parsed_commitment = batch_verifier.parse_commitment(&mut fs_verifier).unwrap();

        for (i, eval) in claims_to_verify {
            assert_eq!(
                batch_verifier
                    .receive_claim(&mut fs_verifier, i, &eval.point)
                    .unwrap(),
                eval.value
            );
        }

        batch_verifier
            .verify(&mut fs_verifier, &parsed_commitment)
            .unwrap();
    }
}
