// TODO REMOVE THIS HORRIBLE FILE

use p3_field::{ExtensionField, Field, PrimeField};

fn prime_field_to_bytes<F: PrimeField>(f: F) -> Vec<u8> {
    if F::bits() <= 16 {
        unimplemented!()
    } else if F::bits() <= 32 {
        if f.is_zero() {
            return vec![0; 4];
        }
        f.as_canonical_biguint().to_u32_digits()[0]
            .to_le_bytes()
            .to_vec()
    } else if F::bits() <= 64 {
        if f.is_zero() {
            return vec![0; 8];
        }
        f.as_canonical_biguint().to_u64_digits()[0]
            .to_le_bytes()
            .to_vec()
    } else {
        unimplemented!()
    }
}

fn prime_field_from_bytes<F: PrimeField>(bytes: &[u8]) -> Option<F> {
    if F::bits() <= 16 {
        unimplemented!()
    } else if F::bits() <= 32 {
        if bytes.len() != 4 {
            return None;
        }
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        Some(F::from_u32(u32::from_le_bytes(arr)))
    } else if F::bits() <= 64 {
        if bytes.len() != 8 {
            return None;
        }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Some(F::from_u64(u64::from_le_bytes(arr)))
    } else {
        unimplemented!()
    }
}

pub fn serialize_field<F: Field>(f: F) -> Vec<u8> {
    let prime_bytes = F::PrimeSubfield::bits().div_ceil(8);
    assert!(prime_bytes == 4 || prime_bytes == 8);
    assert!(size_of::<F>() % prime_bytes == 0);
    let ext_dim = size_of::<F>() / prime_bytes;

    let subfields = unsafe {
        let ptr = &f as *const F as *const F::PrimeSubfield;
        std::slice::from_raw_parts(ptr, ext_dim)
    };

    let mut bytes = Vec::new();
    for i in 0..ext_dim {
        bytes.extend_from_slice(&prime_field_to_bytes(subfields[i]));
    }
    bytes
}

pub fn deserialize_field<F: Field>(bytes: &[u8]) -> Option<F> {
    let prime_bytes = F::PrimeSubfield::bits().div_ceil(8);
    assert!(prime_bytes == 4 || prime_bytes == 8);
    assert!(size_of::<F>() % prime_bytes == 0);
    let ext_dim = size_of::<F>() / prime_bytes;
    if bytes.len() != ext_dim * prime_bytes {
        return None;
    }

    let mut subfields = Vec::new();
    for i in 0..ext_dim {
        subfields.push(
            prime_field_from_bytes::<F::PrimeSubfield>(
                &bytes[i * prime_bytes..(i + 1) * prime_bytes],
            )
            .unwrap(),
        );
    }

    unsafe {
        let ptr = subfields.as_ptr() as *const F;
        Some(std::ptr::read(ptr))
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

pub fn hadamard_product<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect()
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

    // Reverse so higher power is first
    res.reverse();

    res
}
