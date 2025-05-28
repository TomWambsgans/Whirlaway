use p3_field::{ExtensionField, Field};

/// outputs the vector [1, base, base^2, base^3, ...] of length len.
pub fn powers<F: Field>(base: F, len: usize) -> Vec<F> {
    base.powers().take(len).collect()
}

pub fn eq_extension<F: Field, EF: ExtensionField<F>>(s1: &[F], s2: &[EF]) -> EF {
    assert_eq!(s1.len(), s2.len());
    if s1.is_empty() {
        return EF::ONE;
    }
    (0..s1.len())
        .map(|i| s2[i] * s1[i] + (EF::ONE - s2[i]) * (F::ONE - s1[i]))
        .product()
}

pub fn dot_product<F: Field, EF: ExtensionField<F>>(a: &[F], b: &[EF]) -> EF {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *y * *x).sum()
}

pub fn embed_vec<F: Field, EF: ExtensionField<F>>(a: &Vec<F>) -> Vec<EF> {
    a.iter().copied().map(EF::from).collect()
}

pub fn embed_vec_vec<F: Field, EF: ExtensionField<F>>(a: &[Vec<F>]) -> Vec<Vec<EF>> {
    a.iter().map(embed_vec).collect()
}

pub fn serialize_field<F: Field>(f: &F) -> Vec<u8> {
    let size = std::mem::size_of::<F>();
    let mut bytes = Vec::with_capacity(size);
    unsafe {
        let src_ptr = f as *const F as *const u8;
        bytes.set_len(size);
        std::ptr::copy_nonoverlapping(src_ptr, bytes.as_mut_ptr(), size);
    }
    bytes
}

pub fn deserialize_field<F: Field>(bytes: &[u8]) -> Option<F> {
    // TODO check that the representation is correct
    if bytes.len() != std::mem::size_of::<F>() {
        return None;
    }

    let mut result = std::mem::MaybeUninit::<F>::uninit();

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), result.as_mut_ptr() as *mut u8, bytes.len());

        Some(result.assume_init())
    }
}
