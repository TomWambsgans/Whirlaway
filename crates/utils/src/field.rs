use p3_field::{ExtensionField, Field};

/// outputs the vector [1, base, base^2, base^3, ...] of length len.
pub fn powers<F: Field>(base: F, len: usize) -> Vec<F> {
    base.powers().take(len).collect()
}

/// Computes `eq(s1, s2)`, where `s1` is over the base field and `s2` is over the extension field.
///
/// The **equality polynomial** for two vectors is:
/// ```ignore
/// eq(s1, s2) = ∏ (s1_i * s2_i + (1 - s1_i) * (1 - s2_i))
/// ```
/// which evaluates to `1` if `s1 == s2`, and `0` otherwise.
///
/// This uses the algebraic identity:
/// ```ignore
/// s1_i * s2_i + (1 - s1_i) * (1 - s2_i) = 1 + 2 * s1_i * s2_i - s1_i - s2_i
/// ```
/// to avoid unnecessary multiplications.
pub fn eq_extension<F: Field, EF: ExtensionField<F>>(s1: &[F], s2: &[EF]) -> EF {
    assert_eq!(s1.len(), s2.len());
    if s1.is_empty() {
        return EF::ONE;
    }
    s1.iter()
        .zip(s2.iter())
        .map(|(&l, &r)| EF::ONE + r * l.double() - r - l)
        .product()
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
