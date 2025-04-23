//! NTT and related algorithms.

use matrix::MatrixMut;

mod matrix;
mod transpose;

use crate::wavelet::transpose::transpose;
use p3_field::Field;
use rayon::prelude::*;
use std::cmp::max;

/// Fast Wavelet Transform.
///
/// The input slice must have a length that is a power of two.
/// Recursively applies the kernel
///   [1 0]
///   [1 1]
pub fn wavelet_transform<F: Field>(values: &mut [F]) {
    debug_assert!(values.len().is_power_of_two());
    wavelet_transform_batch(values, values.len())
}

fn wavelet_transform_batch<F: Field>(values: &mut [F], size: usize) {
    debug_assert_eq!(values.len() % size, 0);
    debug_assert!(size.is_power_of_two());
    if values.len() > workload_size::<F>() && values.len() != size {
        // Multiple wavelet transforms, compute in parallel.
        // Work size is largest multiple of `size` smaller than `WORKLOAD_SIZE`.
        let workload_size = size * max(1, workload_size::<F>() / size);
        return values.par_chunks_mut(workload_size).for_each(|values| {
            wavelet_transform_batch(values, size);
        });
    }
    match size {
        0 | 1 => {}
        2 => {
            for v in values.chunks_exact_mut(2) {
                v[1] += v[0]
            }
        }
        4 => {
            for v in values.chunks_exact_mut(4) {
                v[1] += v[0];
                v[3] += v[2];
                v[2] += v[0];
                v[3] += v[1];
            }
        }
        8 => {
            for v in values.chunks_exact_mut(8) {
                v[1] += v[0];
                v[3] += v[2];
                v[2] += v[0];
                v[3] += v[1];
                v[5] += v[4];
                v[7] += v[6];
                v[6] += v[4];
                v[7] += v[5];
                v[4] += v[0];
                v[5] += v[1];
                v[6] += v[2];
                v[7] += v[3];
            }
        }
        16 => {
            for v in values.chunks_exact_mut(16) {
                for v in v.chunks_exact_mut(4) {
                    v[1] += v[0];
                    v[3] += v[2];
                    v[2] += v[0];
                    v[3] += v[1];
                }
                let (a, v) = v.split_at_mut(4);
                let (b, v) = v.split_at_mut(4);
                let (c, d) = v.split_at_mut(4);
                for i in 0..4 {
                    b[i] += a[i];
                    d[i] += c[i];
                    c[i] += a[i];
                    d[i] += b[i];
                }
            }
        }
        n => {
            let n1 = 1 << (n.trailing_zeros() / 2);
            let n2 = n / n1;
            wavelet_transform_batch(values, n1);
            transpose(values, n2, n1);
            wavelet_transform_batch(values, n2);
            transpose(values, n1, n2);
        }
    }
}

/// Target single-thread workload size for `T`.
/// Should ideally be a multiple of a cache line (64 bytes)
/// and close to the L1 cache size (32 KB).
const fn workload_size<T: Sized>() -> usize {
    const CACHE_SIZE: usize = 1 << 15;
    CACHE_SIZE / size_of::<T>()
}
