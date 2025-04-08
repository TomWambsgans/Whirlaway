use std::cmp::max;

use cudarc::driver::DeviceRepr;
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use sha3::{Digest, Keccak256};

#[inline(always)]
pub const fn log2(x: usize) -> u32 {
    // rounded up
    if x == 0 {
        0
    } else if x.is_power_of_two() {
        1usize.leading_zeros() - x.leading_zeros()
    } else {
        0usize.leading_zeros() - x.leading_zeros()
    }
}

// Given a vector of field elements {v_i}, compute the vector {v_i^(-1)}
pub fn batch_inversion<F: Field>(v: &mut [F]) {
    batch_inversion_and_mul(v, &F::ONE);
}

// Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}
pub fn batch_inversion_and_mul<F: Field>(v: &mut [F], coeff: &F) {
    // Divide the vector v evenly between all available cores
    let min_elements_per_thread = 1;
    let num_cpus_available = rayon::current_num_threads();
    let num_elems = v.len();
    let num_elem_per_thread = max(num_elems / num_cpus_available, min_elements_per_thread);

    // Batch invert in parallel, without copying the vector
    v.par_chunks_mut(num_elem_per_thread).for_each(|chunk| {
        serial_batch_inversion_and_mul(chunk, coeff);
    });
}

/// Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}.
/// This method is explicitly single-threaded.
fn serial_batch_inversion_and_mul<F: Field>(v: &mut [F], coeff: &F) {
    // Montgomery’s Trick and Fast Implementation of Masked AES
    // Genelle, Prouff and Quisquater
    // Section 3.2
    // but with an optimization to multiply every element in the returned vector by
    // coeff

    // First pass: compute [a, ab, abc, ...]
    let mut prod = Vec::with_capacity(v.len());
    let mut tmp = F::ONE;
    for f in v.iter().filter(|f| !f.is_zero()) {
        tmp.mul_assign(*f);
        prod.push(tmp);
    }

    // Invert `tmp`.
    tmp = tmp.inverse(); // Guaranteed to be nonzero.

    // Multiply product by coeff, so all inverses will be scaled by coeff
    tmp *= *coeff;

    // Second pass: iterate backwards to compute inverses
    for (f, s) in v
        .iter_mut()
        // Backwards
        .rev()
        // Ignore normalized elements
        .filter(|f| !f.is_zero())
        // Backwards, skip last element, fill in one for last term.
        .zip(prod.into_iter().rev().skip(1).chain(Some(F::ONE)))
    {
        // tmp := tmp * f; f := tmp * s = 1/f
        let new_tmp = tmp * *f;
        *f = tmp * s;
        tmp = new_tmp;
    }
}

pub fn field_bytes_in_memory<F: Field>() -> usize {
    let ext_degree: usize = F::bits().div_ceil(F::PrimeSubfield::bits()); // TODO very bad
    (F::PrimeSubfield::bits().div_ceil(8)) * ext_degree
}
// checks whether the given number n is a power of two.
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1) == 0)
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

    if len <= num_threads * log2(num_threads) as usize {
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeccakDigest(pub [u8; 32]);

unsafe impl DeviceRepr for KeccakDigest {}

impl KeccakDigest {
    pub fn to_string(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

pub fn keccak256(data: &[u8]) -> KeccakDigest {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    KeccakDigest(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powers() {
        let base = p3_koala_bear::KoalaBear::new(185);
        let len = 1478;
        assert_eq!(powers(base, len), powers_parallel(base, len));
    }
}
