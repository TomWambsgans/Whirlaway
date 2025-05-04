use std::{any::TypeId, fmt::Display};

use p3_baby_bear::BabyBear;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PrimeField, extension::BinomialExtensionField,
};
use p3_koala_bear::KoalaBear;
use rayon::prelude::*;

use crate::log2_up;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupportedField {
    KoalaBear,
    BabyBear,
}

impl Display for SupportedField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SupportedField::KoalaBear => write!(f, "KoalaBear"),
            SupportedField::BabyBear => write!(f, "BabyBear"),
        }
    }
}

impl SupportedField {
    pub fn guess<F: Field>() -> Self {
        if TypeId::of::<F::PrimeSubfield>() == TypeId::of::<KoalaBear>() {
            SupportedField::KoalaBear
        } else if TypeId::of::<F::PrimeSubfield>() == TypeId::of::<BabyBear>() {
            SupportedField::BabyBear
        } else {
            panic!("Unsupported field type for CUDA")
        }
    }
}

// TODO this is ugly but to remove it we need BinomialExtensionField<2N, X> to implement ExtensionField<BinomialExtensionField<N, X>>
pub fn small_to_big_extension<
    F: PrimeField,
    SmallExt: ExtensionField<F>,
    BigExt: ExtensionField<F>,
>(
    x: SmallExt,
) -> BigExt {
    let small_dim = <SmallExt as BasedVectorSpace<F>>::DIMENSION;
    let big_dim = <BigExt as BasedVectorSpace<F>>::DIMENSION;
    assert!(big_dim % small_dim == 0);

    if small_dim == 1 {
        let mut coeffs = x.as_basis_coefficients_slice().to_vec();
        coeffs.resize(big_dim, F::ZERO);
        return BigExt::from_basis_coefficients_slice(&coeffs).unwrap();
    } else if TypeId::of::<SmallExt>() == TypeId::of::<BigExt>() {
        return BigExt::from_basis_coefficients_slice(x.as_basis_coefficients_slice()).unwrap();
    } else if [TypeId::of::<BabyBear>(), TypeId::of::<KoalaBear>()].contains(&TypeId::of::<F>())
        && big_dim % small_dim == 0
    {
        let small_coeffs = x.as_basis_coefficients_slice();
        let mut big_coeffs = vec![F::ZERO; big_dim];
        for i in 0..small_dim {
            big_coeffs[i * big_dim / small_dim] = small_coeffs[i];
        }
        return BigExt::from_basis_coefficients_slice(&big_coeffs).unwrap();
    } else {
        todo!()
    }
}

/// outputs the vector [1, base, base^2, base^3, ...] of length len.
pub fn powers<F: Field>(base: F, len: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(len);
    let mut acc = F::ONE;
    for _ in 0..len {
        res.push(acc);
        acc *= base;
    }

    res
}

/// outputs the vector [1, base, base^2, base^3, ...] of length len.
pub fn powers_parallel<F: Field>(base: F, len: usize) -> Vec<F> {
    let num_threads = rayon::current_num_threads().next_power_of_two();

    if len <= num_threads * log2_up(num_threads) {
        powers(base, len)
    } else {
        let chunk_size = (len + num_threads - 1) / num_threads;
        (0..num_threads)
            .into_par_iter()
            .map(|j| {
                let mut start = base.exp_u64(j as u64 * chunk_size as u64);
                let mut chunck = Vec::new();
                let chunk_size = if j == num_threads - 1 {
                    len - j * chunk_size
                } else {
                    chunk_size
                };
                for _ in 0..chunk_size {
                    chunck.push(start);
                    start = start * base;
                }
                chunck
            })
            .flatten()
            .collect()
    }
}

pub fn eq_extension<F: Field>(s1: &[F], s2: &[F]) -> F {
    assert_eq!(s1.len(), s2.len());
    if s1.len() == 0 {
        return F::ONE;
    }
    (0..s1.len())
        .map(|i| s1[i] * s2[i] + (F::ONE - s1[i]) * (F::ONE - s2[i]))
        .product()
}

pub fn dot_product<F: Field, EF: ExtensionField<F>>(a: &[F], b: &[EF]) -> EF {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *y * *x).sum()
}

// TODO find a better name
pub fn multilinear_point_from_univariate<F: Field>(point: F, num_variables: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(num_variables);
    let mut cur = point;
    for _ in 0..num_variables {
        res.push(cur);
        cur = cur * cur;
    }

    res
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

pub fn extension_degree<F: Field>() -> usize {
    // TODO there must be a simpler way
    if [TypeId::of::<KoalaBear>(), TypeId::of::<BabyBear>()].contains(&TypeId::of::<F>()) {
        1
    } else if [
        TypeId::of::<BinomialExtensionField<KoalaBear, 4>>(),
        TypeId::of::<BinomialExtensionField<BabyBear, 4>>(),
    ]
    .contains(&TypeId::of::<F>())
    {
        4
    } else if [
        TypeId::of::<BinomialExtensionField<KoalaBear, 8>>(),
        TypeId::of::<BinomialExtensionField<BabyBear, 8>>(),
    ]
    .contains(&TypeId::of::<F>())
    {
        8
    } else if [TypeId::of::<BinomialExtensionField<KoalaBear, 16>>()].contains(&TypeId::of::<F>()) {
        16
    } else {
        todo!("Add extension degree for this field")
    }
}

#[cfg(test)]
mod tests {
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn test_powers() {
        let base = p3_koala_bear::KoalaBear::new(185);
        let len = 1478;
        assert_eq!(powers(base, len), powers_parallel(base, len));
    }

    #[test]
    fn test_serialize_deserialize() {
        type F = BinomialExtensionField<KoalaBear, 8>;
        let rng = &mut StdRng::seed_from_u64(0);
        let f: F = rng.random();
        let bytes = serialize_field(&f);
        let deserialized: F = deserialize_field(&bytes).unwrap();
        assert_eq!(f, deserialized);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn test_small_to_big_extension() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;
        type NF = BinomialExtensionField<F, 8>;
        let a: EF = StdRng::seed_from_u64(0).random();
        let b = small_to_big_extension::<F, EF, NF>(a);
        assert_eq!(b * b + b, small_to_big_extension::<F, EF, NF>(a * a + a))
    }
}
