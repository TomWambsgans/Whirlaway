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

pub fn embed_vec<F: Field, EF: ExtensionField<F>>(a: &Vec<F>) -> Vec<EF> {
    a.iter().copied().map(EF::from).collect()
}

pub fn embed_vec_vec<F: Field, EF: ExtensionField<F>>(a: &[Vec<F>]) -> Vec<Vec<EF>> {
    a.iter().map(embed_vec).collect()
}

pub fn serialize_field<F: Field>(f: &F) -> Vec<u8> {
    bincode::serde::encode_to_vec(f, bincode::config::standard().with_fixed_int_encoding()).unwrap()
}

pub fn deserialize_field<F: Field>(bytes: &[u8]) -> Option<F> {
    Some(
        bincode::serde::decode_from_slice(
            bytes,
            bincode::config::standard().with_fixed_int_encoding(),
        )
        .ok()?
        .0,
    )
}

#[cfg(test)]
mod test {
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_koala_bear::KoalaBear;

    use crate::*;

    #[test]
    fn test_serialization() {
        type F = BinomialExtensionField<KoalaBear, 4>;
        let f = F::ONE;
        let bytes = serialize_field(&f);
        let f2 = deserialize_field::<F>(&bytes).unwrap();
        assert_eq!(f, f2);
    }
}
