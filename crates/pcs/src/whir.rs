use algebra::pols::Multilinear;
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{ExtensionField, Field, TwoAdicField};
use utils::{Evaluation, KeccakDigest};
use whir::{
    parameters::MultivariateParameters,
    whir::{
        Statement,
        committer::Committer,
        parameters::WhirConfig,
        prover::Prover,
        verifier::{ParsedCommitment, Verifier, WhirError},
    },
};

use whir::whir::committer::Witness as WhirWitness;

use super::{PCS, PcsWitness};

pub use whir::parameters::WhirParameters;

#[derive(Clone)]
pub struct WhirPCS<F: TwoAdicField, EF: ExtensionField<F>> {
    config: WhirConfig<F, EF>,
}

impl<'a, EF: Field> PcsWitness<EF> for WhirWitness<EF> {
    fn pol(&self) -> &Multilinear<EF> {
        &self.lagrange_polynomial
    }
}

impl<F: TwoAdicField, EF: ExtensionField<F> + TwoAdicField + Ord> PCS<EF, EF> for WhirPCS<F, EF> {
    type Witness = WhirWitness<EF>;
    type ParsedCommitment = ParsedCommitment<EF, KeccakDigest>;
    type VerifError = WhirError;
    type Params = WhirParameters;

    fn new(n_vars: usize, params: &Self::Params) -> Self {
        let mv_params = MultivariateParameters::<EF>::new(n_vars);
        let config = WhirConfig::<F, EF>::new(mv_params, params);
        // println!("WhirPCS config: {}", config);
        Self { config }
    }

    fn commit(&self, pol: Multilinear<EF>, fs_prover: &mut FsProver) -> Self::Witness {
        Committer::new(self.config.clone())
            .commit(fs_prover, pol)
            .unwrap()
    }

    fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<Self::ParsedCommitment, Self::VerifError> {
        Verifier::<F, EF>::new(self.config.clone()).parse_commitment(fs_verifier)
    }

    fn open(&self, witness: Self::Witness, eval: &Evaluation<EF>, fs_prover: &mut FsProver) {
        let statement = Statement {
            points: vec![eval.point.clone()],
            evaluations: vec![eval.value],
        };
        Prover(self.config.clone())
            .prove(fs_prover, statement, witness)
            .unwrap();
    }

    fn verify(
        &self,
        parsed_commitment: &Self::ParsedCommitment,
        eval: &Evaluation<EF>,
        fs_verifier: &mut FsVerifier,
    ) -> Result<(), Self::VerifError> {
        let statement = Statement {
            points: vec![eval.point.clone()],
            evaluations: vec![eval.value],
        };
        Verifier::new(self.config.clone()).verify(fs_verifier, parsed_commitment, statement)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use algebra::pols::MultilinearHost;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;
    use tracing_forest::{ForestLayer, util::LevelFilter};
    use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
    use whir::parameters::SoundnessType;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 8>;

    #[test]
    fn test_whir_pcs() {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();

        let n_vars = 16;
        let security_bits = 100;
        let log_inv_rate = 3;
        let pcs = WhirPCS::<F, EF>::new(
            n_vars,
            &WhirParameters::standard(
                SoundnessType::ProvableList,
                security_bits,
                log_inv_rate,
                false,
            ),
        );

        let mut fs_prover = FsProver::new();

        let evals = (0..1 << n_vars)
            .map(|x| EF::from_u64(x as u64))
            .collect::<Vec<_>>();
        let pol = Multilinear::Host(MultilinearHost::new(evals));
        let point = (0..n_vars)
            .map(|x| EF::from_u64(x as u64))
            .collect::<Vec<_>>();
        let value = pol.evaluate(&point);
        let eval = Evaluation { point, value };
        let witness = pcs.commit(pol, &mut fs_prover);
        pcs.open(witness, &eval, &mut fs_prover);

        let mut fs_verifier = FsVerifier::new(fs_prover.transcript());
        let parsed_commitment = pcs.parse_commitment(&mut fs_verifier).unwrap();
        pcs.verify(&parsed_commitment, &eval, &mut fs_verifier)
            .unwrap();
    }
}
