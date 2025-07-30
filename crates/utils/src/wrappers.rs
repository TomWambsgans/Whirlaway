use p3_challenger::FieldChallenger;
use p3_challenger::GrindingChallenger;
use p3_field::ExtensionField;
use p3_field::PackedFieldExtension;
use p3_field::PackedValue;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_symmetric::CryptographicHasher;
use p3_symmetric::PseudoCompressionFunction;
use rayon::prelude::*;
use whir_p3::fiat_shamir::verifier::ChallengerState;
use whir_p3::fiat_shamir::{prover::ProverState, verifier::VerifierState};

pub type PF<F> = <F as PrimeCharacteristicRing>::PrimeSubfield;
pub type PFPacking<F> = <PF<F> as Field>::Packing;

pub type FSProver<EF, Challenger> = ProverState<PF<PF<EF>>, EF, Challenger>;
pub type FSVerifier<EF, Challenger> = VerifierState<PF<PF<EF>>, EF, Challenger>;
pub type EFPacking<EF> = <EF as ExtensionField<PF<EF>>>::ExtensionPacking;

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

pub fn pack_extension<EF: Field + ExtensionField<PF<EF>>>(slice: &[EF]) -> Vec<EFPacking<EF>> {
    slice
        .par_chunks_exact(PFPacking::<EF>::WIDTH)
        .map(EFPacking::<EF>::from_ext_slice)
        .collect::<Vec<_>>()
}

pub fn transmute_vec<Before, After>(vec: Vec<Before>) -> Vec<After> {
    assert!(vec.len() * std::mem::size_of::<Before>() % std::mem::size_of::<After>() == 0);
    let new_len = vec.len() * std::mem::size_of::<Before>() / std::mem::size_of::<After>();
    let mut res: Vec<After> = unsafe { std::mem::transmute(vec) };
    unsafe {
        res.set_len(new_len);
    }
    res
}

pub fn transmute_slice<Before, After>(slice: &[Before]) -> &[After] {
    assert!(slice.len() * std::mem::size_of::<Before>() % std::mem::size_of::<After>() == 0);
    let new_len = slice.len() * std::mem::size_of::<Before>() / std::mem::size_of::<After>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const After, new_len) }
}
