use p3_challenger::FieldChallenger;
use p3_challenger::GrindingChallenger;
use p3_field::BasedVectorSpace;
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

pub fn unpack_extension<EF: Field + ExtensionField<PF<EF>>>(vec: &[EFPacking<EF>]) -> Vec<EF> {
    vec.into_iter()
        .flat_map(|x| {
            let packed_coeffs = x.as_basis_coefficients_slice();
            (0..PFPacking::<EF>::WIDTH)
                .map(|i| EF::from_basis_coefficients_fn(|j| packed_coeffs[j].as_slice()[i]))
                .collect::<Vec<_>>()
        })
        .collect()
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

pub fn mul_extension_field_packing_by_base_scalar<EF: Field + ExtensionField<PF<EF>>>(
    ef_packing: &EFPacking<EF>,
    scalar: PF<EF>,
) -> EFPacking<EF> {
    let mut res = EFPacking::<EF>::default();
    for i in 0..EF::DIMENSION {
        unsafe {
            let ef_ptr = ef_packing as *const EFPacking<EF> as *const PFPacking<EF>;
            let res_ptr = &mut res as *mut EFPacking<EF> as *mut PFPacking<EF>;

            let base_elem = *ef_ptr.add(i);
            *res_ptr.add(i) = base_elem * scalar;
        }
    }
    res
}

pub const fn packing_log_width<EF: Field>() -> usize {
    PFPacking::<EF>::WIDTH.ilog2() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::BasedVectorSpace;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;

    type EF = BinomialExtensionField<KoalaBear, 4>;
    const D: usize = <EF as BasedVectorSpace<PF<EF>>>::DIMENSION;
    const W: usize = PFPacking::<EF>::WIDTH;

    #[test]
    fn test_mul_extension_field_packing_by_base_scalar() {
        let base: [PF<EF>; W * D] = (0..W * D)
            .map(|i| PF::<EF>::from_usize(i))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let ef_packing = unsafe { std::mem::transmute::<[PF<EF>; W * D], EFPacking<EF>>(base) };
        let mul =
            mul_extension_field_packing_by_base_scalar::<EF>(&ef_packing, PF::<EF>::from_usize(2));
        let res_transmuted: [PF<EF>; W * D] =
            unsafe { std::mem::transmute::<EFPacking<EF>, [PF<EF>; W * D]>(mul) };
        for i in 0..W * D {
            assert_eq!(res_transmuted[i], PF::<EF>::from_usize(i * 2));
        }
        dbg!(res_transmuted);
    }
}
