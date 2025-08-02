use p3_challenger::DuplexChallenger;
use p3_field::BasedVectorSpace;
use p3_field::ExtensionField;
use p3_field::PackedFieldExtension;
use p3_field::PackedValue;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_koala_bear::KoalaBear;
use p3_koala_bear::Poseidon2KoalaBear;
use p3_symmetric::CryptographicHasher;
use p3_symmetric::PaddingFreeSponge;
use p3_symmetric::PseudoCompressionFunction;
use p3_symmetric::TruncatedPermutation;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use whir_p3::fiat_shamir::{prover::ProverState, verifier::VerifierState};

pub type PF<F> = <F as PrimeCharacteristicRing>::PrimeSubfield;
pub type PFPacking<F> = <PF<F> as Field>::Packing;
pub type EFPacking<EF> = <EF as ExtensionField<PF<EF>>>::ExtensionPacking;

pub type FSProver<EF, Challenger> = ProverState<PF<EF>, EF, Challenger>;
pub type FSVerifier<EF, Challenger> = VerifierState<PF<EF>, EF, Challenger>;

pub type Poseidon16 = Poseidon2KoalaBear<16>;
pub type Poseidon24 = Poseidon2KoalaBear<24>;

pub type MyMerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
pub type MyMerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
pub type MyChallenger = DuplexChallenger<KoalaBear, Poseidon16, 16, 8>;

pub trait MerkleHasher<EF: Field, const DIGEST_ELEMS: usize>:
    CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
    + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
    + Sync
{
}

pub trait MerkleCompress<EF: Field, const DIGEST_ELEMS: usize>:
    PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
    + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
    + Sync
{
}

impl<
    EF: Field,
    MH: CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
        + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
        + Sync,
    const DIGEST_ELEMS: usize,
> MerkleHasher<EF, DIGEST_ELEMS> for MH
{
}

impl<
    EF: Field,
    MC: PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
        + Sync,
    const DIGEST_ELEMS: usize,
> MerkleCompress<EF, DIGEST_ELEMS> for MC
{
}

pub fn pack_extension<EF: Field + ExtensionField<PF<EF>>>(slice: &[EF]) -> Vec<EFPacking<EF>> {
    slice
        .par_chunks_exact(packing_width::<EF>())
        .map(EFPacking::<EF>::from_ext_slice)
        .collect::<Vec<_>>()
}

pub fn unpack_extension<EF: Field + ExtensionField<PF<EF>>>(vec: &[EFPacking<EF>]) -> Vec<EF> {
    vec.into_iter()
        .flat_map(|x| {
            let packed_coeffs = x.as_basis_coefficients_slice();
            (0..packing_width::<EF>())
                .map(|i| EF::from_basis_coefficients_fn(|j| packed_coeffs[j].as_slice()[i]))
                .collect::<Vec<_>>()
        })
        .collect()
}

pub const fn packing_log_width<EF: Field>() -> usize {
    packing_width::<EF>().ilog2() as usize
}

pub const fn packing_width<EF: Field>() -> usize {
    PFPacking::<EF>::WIDTH
}

pub fn build_poseidon16() -> Poseidon16 {
    Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0))
}

pub fn build_poseidon24() -> Poseidon24 {
    Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(0))
}

pub fn build_challenger() -> MyChallenger {
    MyChallenger::new(build_poseidon16())
}

pub fn build_merkle_hash() -> MyMerkleHash {
    MyMerkleHash::new(build_poseidon24())
}

pub fn build_merkle_compress() -> MyMerkleCompress {
    MyMerkleCompress::new(build_poseidon16())
}

pub fn build_prover_state<EF: ExtensionField<KoalaBear>>()
-> ProverState<KoalaBear, EF, MyChallenger> {
    ProverState::new(build_challenger())
}

pub fn build_verifier_state<EF: ExtensionField<KoalaBear>>(
    prover_state: &ProverState<KoalaBear, EF, MyChallenger>,
) -> VerifierState<KoalaBear, EF, MyChallenger> {
    VerifierState::new(prover_state.proof_data().to_vec(), build_challenger())
}
