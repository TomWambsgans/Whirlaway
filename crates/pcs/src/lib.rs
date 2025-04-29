use std::fmt::Debug;

use algebra::pols::Multilinear;
use fiat_shamir::{FsProver, FsVerifier};
use p3_field::{ExtensionField, Field};

mod whir;
use utils::Evaluation;
pub use whir::*;

mod ring_switch;
pub use ring_switch::*;

pub trait PcsWitness<F: Field> {
    fn pol(&self) -> &Multilinear<F>;
}

pub trait PcsParams {
    fn security_bits(&self) -> usize;
}

pub trait PCS<F: Field, EF: ExtensionField<F>> {
    type ParsedCommitment;
    type Witness: PcsWitness<F>;
    type VerifError: Debug;
    type Params: PcsParams;
    fn new(n_vars: usize, params: &Self::Params) -> Self;
    fn commit(&self, pol: Multilinear<F>, fs_prover: &mut FsProver) -> Self::Witness;
    fn parse_commitment(
        &self,
        fs_verifier: &mut FsVerifier,
    ) -> Result<Self::ParsedCommitment, Self::VerifError>;
    fn open<PrimeField: Field>(
        &self,
        witness: Self::Witness,
        eval: &Evaluation<EF>,
        fs_prover: &mut FsProver,
    ) where
        EF: ExtensionField<PrimeField>;
    fn verify(
        &self,
        parsed_commitment: &Self::ParsedCommitment,
        eval: &Evaluation<EF>,
        fs_verifier: &mut FsVerifier,
    ) -> Result<(), Self::VerifError>;
}
