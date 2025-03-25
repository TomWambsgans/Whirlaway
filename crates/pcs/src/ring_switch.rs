use algebra::field_utils::dot_product;
use algebra::pols::{ComposedPolynomial, DenseMultilinearPolynomial, Evaluation};
use algebra::tensor_algebra::TensorAlgebra;
use fiat_shamir::{FsError, FsProver, FsVerifier};
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use sumcheck::SumcheckError;

use crate::PcsWitness;

use super::PCS;

#[derive(Clone)]
pub struct RingSwitch<F: Field, EF: ExtensionField<F>, Pcs: PCS<EF, EF>> {
    inner: Pcs,
    _small_field: std::marker::PhantomData<F>,
    _extension_field: std::marker::PhantomData<EF>,
}

pub struct RingSwitchWitness<F: Field, InnerWitness> {
    pol: DenseMultilinearPolynomial<F>,
    inner_witness: InnerWitness,
}

impl<F: Field, InnerWitness> PcsWitness<F> for RingSwitchWitness<F, InnerWitness> {
    fn pol(&self) -> &DenseMultilinearPolynomial<F> {
        &self.pol
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RingSwitchError<InnerError> {
    Inner(InnerError),
    Fs(FsError),
    Sumcheck(SumcheckError),
    Outer,
}

impl<InnerError> From<FsError> for RingSwitchError<InnerError> {
    fn from(e: FsError) -> Self {
        RingSwitchError::Fs(e)
    }
}

impl<InnerError> From<SumcheckError> for RingSwitchError<InnerError> {
    fn from(e: SumcheckError) -> Self {
        RingSwitchError::Sumcheck(e)
    }
}

impl<F: Field, EF: ExtensionField<F>, Pcs: PCS<EF, EF>> PCS<F, EF> for RingSwitch<F, EF, Pcs> {
    type ParsedCommitment = Pcs::ParsedCommitment;
    type Witness = RingSwitchWitness<F, Pcs::Witness>;
    type VerifError = RingSwitchError<Pcs::VerifError>;
    type Params = Pcs::Params;

    fn new(n_vars: usize, params: &Self::Params) -> Self {
        let two_pow_kappa = <EF as BasedVectorSpace<F>>::DIMENSION;
        assert!(two_pow_kappa.is_power_of_two());
        let kappa = two_pow_kappa.ilog2() as usize;
        assert!(n_vars > kappa);
        let inner = Pcs::new(n_vars - kappa, params);
        Self {
            inner,
            _small_field: std::marker::PhantomData,
            _extension_field: std::marker::PhantomData,
        }
    }

    fn commit(
        &self,
        pol: DenseMultilinearPolynomial<F>,
        fs_prover: &mut FsProver,
    ) -> Self::Witness {
        let pol_clone = pol.clone(); // TODO avoid
        let inner_witness = self.inner.commit(pol.packed::<EF>(), fs_prover);
        RingSwitchWitness {
            pol: pol_clone,
            inner_witness,
        }
    }

    fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<Pcs::ParsedCommitment, Self::VerifError> {
        self.inner
            .parse_commitment(fs_verifier)
            .map_err(RingSwitchError::Inner)
    }

    #[allow(non_snake_case)]
    fn open(&self, witness: Self::Witness, eval: &Evaluation<EF>, fs_prover: &mut FsProver) {
        let two_pow_kappa = <EF as BasedVectorSpace<F>>::DIMENSION;
        assert!(two_pow_kappa.is_power_of_two());
        let kappa = two_pow_kappa.ilog2() as usize;
        let packed_pol = witness.inner_witness.pol();
        let point = &eval.point;
        let packed_point = &point[..point.len() - kappa];

        let s_hat = packed_pol.eval_mixed_tensor(&packed_point);
        fs_prover.add_scalar_matrix(&s_hat.data, true);

        let r_pp = fs_prover.challenge_scalars::<EF>(kappa);
        let lagranged_r_pp = DenseMultilinearPolynomial::eq_mle(&r_pp).evals;

        let mut A_coefs = vec![vec![F::ZERO; two_pow_kappa]; packed_pol.n_coefs()];
        for (w, lagrange_eval) in DenseMultilinearPolynomial::eq_mle(&packed_point)
            .evals
            .iter()
            .enumerate()
        {
            A_coefs[w] = lagrange_eval.as_basis_coefficients_slice().to_vec();
        }

        let A_pol = {
            let mut A_basis = Vec::with_capacity(packed_pol.n_coefs());
            for w in 0..packed_pol.n_coefs() {
                A_basis.push(dot_product(&A_coefs[w], &lagranged_r_pp));
            }
            DenseMultilinearPolynomial::new(A_basis)
        };
        let h = ComposedPolynomial::<EF, EF>::new_product(
            packed_pol.n_vars,
            vec![packed_pol.clone().into(), A_pol.into()],
        );

        let s0 = dot_product(&s_hat.rows(), &lagranged_r_pp);
        let (r_p, _) = sumcheck::prove::<EF, EF, EF>(h, None, fs_prover, Some(s0), None, 0);

        let packed_value = witness.inner_witness.pol().eval(&r_p);
        let packed_eval = Evaluation {
            point: r_p.clone(),
            value: packed_value,
        };
        fs_prover.add_scalars(&[packed_value]);

        self.inner
            .open(witness.inner_witness, &packed_eval, fs_prover)
    }

    fn verify(
        &self,
        parsed_commitment: &Pcs::ParsedCommitment,
        eval: &Evaluation<EF>,
        fs_verifier: &mut FsVerifier,
    ) -> Result<(), RingSwitchError<Pcs::VerifError>> {
        let two_pow_kappa = <EF as BasedVectorSpace<F>>::DIMENSION;
        assert!(two_pow_kappa.is_power_of_two());
        let kappa = two_pow_kappa.ilog2() as usize;
        let n_packed_vars = eval.point.len() - kappa;

        let s_hat = TensorAlgebra::<F, EF>::new(
            fs_verifier.next_scalar_matrix(Some((two_pow_kappa, two_pow_kappa)))?,
        );

        let lagrange_evals = DenseMultilinearPolynomial::eq_mle(&eval.point[n_packed_vars..]).evals;
        let columns = s_hat.columns();

        if dot_product(&columns, &lagrange_evals) != eval.value {
            return Err(RingSwitchError::Outer);
        }

        let r_pp = fs_verifier.challenge_scalars::<EF>(kappa);

        let rows = s_hat.rows();
        let lagranged_r_pp = DenseMultilinearPolynomial::eq_mle(&r_pp).evals;
        let s0 = dot_product(&rows, &lagranged_r_pp);

        let (claimed_s0, sc_claim) = sumcheck::verify(fs_verifier, &vec![2; n_packed_vars], 0)?;
        if claimed_s0 != s0 {
            return Err(RingSwitchError::Outer);
        }

        let s_prime = fs_verifier.next_scalars(1)?[0];
        self.inner
            .verify(
                parsed_commitment,
                &Evaluation {
                    point: sc_claim.point.clone(),
                    value: s_prime,
                },
                fs_verifier,
            )
            .map_err(RingSwitchError::Inner)?;

        // e := eq(φ0(rκ), . . . , φ0(rℓ−1), φ1(r′0), . . . , φ1(r′ℓ′−1))
        let mut e = TensorAlgebra::one();
        for (&r, &r_prime) in eval.point[..n_packed_vars].iter().zip(&sc_claim.point) {
            e = e.scale_columns(r).scale_rows(r_prime)
                + e.scale_columns(EF::ONE - r).scale_rows(EF::ONE - r_prime);
        }

        if s_prime * dot_product(&e.rows(), &lagranged_r_pp) != sc_claim.value {
            return Err(RingSwitchError::Outer);
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use whir::parameters::WhirParameters;

    use crate::WhirPCS;

    use super::*;
    type F = KoalaBear;
    type EF = BinomialExtensionField<KoalaBear, 8>;

    #[test]
    fn test_ring_switch() {
        let n_vars = 14;
        let security_bits = 50;
        let log_inv_rate = 4;

        let rng = &mut StdRng::seed_from_u64(0);
        let ring_switch = RingSwitch::<F, EF, WhirPCS<EF>>::new(
            n_vars,
            &WhirParameters::standard(security_bits, log_inv_rate),
        );
        let pol = DenseMultilinearPolynomial::<F>::random(rng, n_vars);
        let mut fs_prover = FsProver::new();
        let commitment = ring_switch.commit(pol.clone(), &mut fs_prover);
        let point = (0..n_vars).map(|_| rng.random()).collect::<Vec<_>>();
        let value = pol.eval(&point);
        let eval = Evaluation {
            point: point.clone(),
            value,
        };
        ring_switch.open(commitment, &eval, &mut fs_prover);

        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let parsed_commitment = ring_switch.parse_commitment(&mut fs_verifier).unwrap();
        ring_switch
            .verify(&parsed_commitment, &eval, &mut fs_verifier)
            .unwrap();
    }
}
