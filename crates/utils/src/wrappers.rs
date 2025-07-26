use p3_challenger::FieldChallenger;
use p3_challenger::GrindingChallenger;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_symmetric::CryptographicHasher;
use p3_symmetric::PseudoCompressionFunction;
use whir_p3::fiat_shamir::verifier::ChallengerState;
use whir_p3::fiat_shamir::{prover::ProverState, verifier::VerifierState};

pub type PF<F> = <F as PrimeCharacteristicRing>::PrimeSubfield;
pub type PFPacking<F> = <PF<F> as Field>::Packing;

pub type FSProver<EF, Challenger> = ProverState<PF<PF<EF>>, EF, Challenger>;
pub type FSVerifier<EF, Challenger> = VerifierState<PF<PF<EF>>, EF, Challenger>;

pub trait FSChallenger<EF: Field>:
    FieldChallenger<PF<PF<EF>>> + GrindingChallenger<Witness = PF<PF<EF>>> + ChallengerState
{
}

pub trait MerkleHasher<EF: Field, const DIGEST_ELEMS: usize>:
    CryptographicHasher<PFPacking<PF<EF>>, [PFPacking<PF<EF>>; DIGEST_ELEMS]>
    + CryptographicHasher<PF<PF<EF>>, [PF<PF<EF>>; DIGEST_ELEMS]>
    + Sync
{
}

pub trait MerkleCompress<EF: Field, const DIGEST_ELEMS: usize>:
    PseudoCompressionFunction<[PFPacking<PF<EF>>; DIGEST_ELEMS], 2>
    + PseudoCompressionFunction<[PF<PF<EF>>; DIGEST_ELEMS], 2>
    + Sync
{
}

impl<
    EF: Field,
    C: FieldChallenger<PF<PF<EF>>> + GrindingChallenger<Witness = PF<PF<EF>>> + ChallengerState,
> FSChallenger<EF> for C
{
}

impl<
    EF: Field,
    MH: CryptographicHasher<PFPacking<PF<EF>>, [PFPacking<PF<EF>>; DIGEST_ELEMS]>
        + CryptographicHasher<PF<PF<EF>>, [PF<PF<EF>>; DIGEST_ELEMS]>
        + Sync,
    const DIGEST_ELEMS: usize,
> MerkleHasher<EF, DIGEST_ELEMS> for MH
{
}

impl<
    EF: Field,
    MC: PseudoCompressionFunction<[PFPacking<PF<EF>>; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PF<PF<EF>>; DIGEST_ELEMS], 2>
        + Sync,
    const DIGEST_ELEMS: usize,
> MerkleCompress<EF, DIGEST_ELEMS> for MC
{
}
