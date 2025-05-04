use rayon::prelude::*;
use std::collections::BTreeSet;
use std::hash::{DefaultHasher, Hash, Hasher};

#[inline(always)]
pub const fn log2_up(x: usize) -> usize {
    // rounded up
    (if x == 0 {
        0
    } else if x.is_power_of_two() {
        1usize.leading_zeros() - x.leading_zeros()
    } else {
        0usize.leading_zeros() - x.leading_zeros()
    }) as usize
}

#[inline(always)]
pub const fn log2_down(x: usize) -> usize {
    // rounded down
    if x == 0 {
        0
    } else if x.is_power_of_two() {
        (1usize.leading_zeros() - x.leading_zeros()) as usize
    } else {
        (0usize.leading_zeros() - x.leading_zeros() - 1) as usize
    }
}

pub fn switch_endianness(mut x: usize, n: usize) -> usize {
    assert!(x < 1 << n);
    let mut y = 0;
    for _ in 0..n {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

pub fn switch_endianness_vec<A: Clone + Default + Sync + Send>(v: &[A]) -> Vec<A> {
    assert!(v.len().is_power_of_two());
    let n = v.len().trailing_zeros() as usize;
    let mut res = vec![A::default(); v.len()];
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        let j = switch_endianness(i, n);
        *x = v[j].clone();
    });
    res
}

pub fn count_ending_zero_bits(buff: &[u8]) -> usize {
    let mut count = 0;
    for byte in buff.iter().rev() {
        for i in 0..8 {
            if byte & (1 << i) != 0 {
                return count;
            }
            count += 1;
        }
    }
    count
}

/// Deduplicates AND orders a vector
pub fn dedup<T: Ord>(v: impl IntoIterator<Item = T>) -> Vec<T> {
    Vec::from_iter(BTreeSet::from_iter(v))
}

pub fn default_hash<H: Hash>(h: H) -> u64 {
    let mut hasher = DefaultHasher::new();
    h.hash(&mut hasher);
    hasher.finish()
}
