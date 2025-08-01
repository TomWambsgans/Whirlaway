use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use utils::{Evaluation, FSChallenger, FSProver, FSVerifier, PF, PFPacking};
use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::{WhirFS, errors::ProofError, prover::ProverState},
    poly::evals::EvaluationsList,
    whir::{
        committer::{Witness, reader::ParsedCommitment, writer::CommitmentWriter},
        config::{WhirConfig, WhirConfigBuilder},
    },
};

pub trait PCS<F: Field, EF: ExtensionField<F>> {
    type ParsedCommitment;
    type Witness;
    type VerifError: Debug;
    fn commit(
        &self,
        dft: &EvalsDft<PF<F>>,
        prover_state: &mut ProverState<PF<F>, EF, impl WhirFS<F>>,
        pol: &EvaluationsList<F>,
    ) -> Self::Witness;
    fn open(
        &self,
        witness: Self::Witness,
        statements: &[Evaluation<EF>],
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    ) -> Result<(), Self::VerifError>;
    fn parse_commitment(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    ) -> Result<Self::ParsedCommitment, Self::VerifError>;
    fn verify(
        &self,
        parsed_commitment: &Self::ParsedCommitment,
        statements: &[Evaluation<EF>],
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    ) -> Result<(), Self::VerifError>;
}

impl<F, EF, H, C, const DIGEST_ELEMS: usize> PCS<F, EF>
    for WhirConfigBuilder<F, EF, H, C, DIGEST_ELEMS>
where
    F: TwoAdicField,
    PF<F>: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + ExtensionField<PF<PF<F>>> + ExtensionField<PF<F>>,
    F: ExtensionField<PF<F>>,
    H: CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
        + CryptographicHasher<PFPacking<F>, [PFPacking<F>; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PFPacking<F>; DIGEST_ELEMS], 2>
        + Sync,
    [PF<F>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ParsedCommitment = ParsedCommitment<F, EF, DIGEST_ELEMS>;
    type Witness = Witness<F, EF, DIGEST_ELEMS>;
    type VerifError = ProofError;

    fn commit(
        &self,
        dft: &EvalsDft<PF<F>>,
        prover_state: &mut ProverState<PF<F>, EF, impl WhirFS<F>>,
        pol: &EvaluationsList<F>,
    ) -> Self::Witness {
        let config = WhirConfig::new(self.clone(), pol.num_variables());
        CommitmentWriter::new(&config)
            .commit(dft, prover_state, pol)
            .unwrap()
    }

    fn open(
        &self,
        witness: Self::Witness,
        statements: &[Evaluation<EF>],
        prover_state: &mut FSProver<EF, impl FSChallenger<EF>>,
    ) -> Result<(), Self::VerifError> {
        todo!()
    }

    fn parse_commitment(
        &self,
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    ) -> Result<Self::ParsedCommitment, Self::VerifError> {
        todo!()
    }

    fn verify(
        &self,
        parsed_commitment: &Self::ParsedCommitment,
        statements: &[Evaluation<EF>],
        verifier_state: &mut FSVerifier<EF, impl FSChallenger<EF>>,
    ) -> Result<(), Self::VerifError> {
        todo!()
    }
}
