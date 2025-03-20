use algebra::pols::{DenseMultilinearPolynomial, Evaluation};
use fiat_shamir::{FsProver, FsVerifier};
use merkle_tree::KeccakDigest;
use p3_field::TwoAdicField;
use whir::{
    parameters::MultivariateParameters,
    poly_utils::coeffs::CoefficientList,
    whir::{
        Statement,
        committer::Committer,
        parameters::WhirConfig,
        prover::Prover,
        verifier::{ParsedCommitment, Verifier, WhirError},
    },
};

use super::{PCS, PcsWitness};

pub use whir::parameters::WhirParameters;

#[derive(Clone)]
pub struct WhirPCS<F: TwoAdicField> {
    config: WhirConfig<F>,
}

pub struct WhirWitness<F: TwoAdicField> {
    // TODO avoid duplication
    pub pol: DenseMultilinearPolynomial<F>,
    pub inner: whir::whir::committer::Witness<F>,
}

impl<F: TwoAdicField> PcsWitness<F> for WhirWitness<F> {
    fn pol(&self) -> &DenseMultilinearPolynomial<F> {
        &self.pol
    }
}

impl<F: TwoAdicField> PCS<F, F> for WhirPCS<F> {
    type Witness = WhirWitness<F>;
    type ParsedCommitment = ParsedCommitment<F, KeccakDigest>;
    type VerifError = WhirError;
    type Params = WhirParameters;

    fn new(n_vars: usize, params: &Self::Params) -> Self {
        let mv_params = MultivariateParameters::<F>::new(n_vars);
        let config = WhirConfig::<_>::new(mv_params, params);
        // println!("WhirPCS config: {}", config);
        Self { config }
    }

    fn commit(
        &self,
        pol: DenseMultilinearPolynomial<F>,
        fs_prover: &mut FsProver,
    ) -> Self::Witness {
        let committer = Committer::new(self.config.clone());
        let inner = committer
            .commit(fs_prover, CoefficientList::new(pol.clone().as_coefs()))
            .unwrap();
        WhirWitness { pol, inner }
    }

    fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<Self::ParsedCommitment, Self::VerifError> {
        Verifier::new(self.config.clone()).parse_commitment(fs_verifier)
    }

    fn open(&self, witness: Self::Witness, eval: &Evaluation<F>, fs_prover: &mut FsProver) {
        let statement = Statement {
            points: vec![eval.point.clone()],
            evaluations: vec![eval.value],
        };
        Prover(self.config.clone())
            .prove(fs_prover, statement, witness.inner)
            .unwrap();
    }

    fn verify(
        &self,
        parsed_commitment: &Self::ParsedCommitment,
        eval: &Evaluation<F>,
        fs_verifier: &mut FsVerifier,
    ) -> Result<(), Self::VerifError> {
        let statement = Statement {
            points: vec![eval.point.clone()],
            evaluations: vec![eval.value],
        };
        Verifier::new(self.config.clone()).verify(fs_verifier, parsed_commitment, &statement)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;

    type F = BinomialExtensionField<KoalaBear, 8>;
    #[test]
    fn test_whir_pcs() {
        let n_vars = 10;
        let security_bits = 100;
        let log_inv_rate = 4;
        let pcs = WhirPCS::<F>::new(
            n_vars,
            &WhirParameters::standard(security_bits, log_inv_rate),
        );

        let mut fs_prover = FsProver::new();

        let evals = (0..1 << n_vars)
            .map(|x| F::from_u64(x as u64))
            .collect::<Vec<_>>();
        let pol = DenseMultilinearPolynomial::new(evals);
        let point = (0..n_vars)
            .map(|x| F::from_u64(x as u64))
            .collect::<Vec<_>>();
        let value = pol.eval(&point);
        let eval = Evaluation { point, value };
        let witness = pcs.commit(pol, &mut fs_prover);
        pcs.open(witness, &eval, &mut fs_prover);

        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let parsed_commitment = pcs.parse_commitment(&mut fs_verifier).unwrap();
        pcs.verify(&parsed_commitment, &eval, &mut fs_verifier)
            .unwrap();
    }
}
