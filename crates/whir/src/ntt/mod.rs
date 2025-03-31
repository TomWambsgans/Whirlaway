//! NTT and related algorithms.

mod matrix;
mod ntt;
mod transpose;
mod utils;
mod wavelet;

use self::matrix::MatrixMut;
use p3_field::TwoAdicField;

use rayon::prelude::*;
use tracing::instrument;

pub use self::{
    ntt::{intt, intt_batch, ntt, ntt_batch},
    transpose::transpose,
    wavelet::wavelet_transform,
};

/// RS encode at a rate 1/`expansion`.
#[instrument(name = "ntt: expand_from_coeff", skip_all)]
pub fn expand_from_coeff<F: TwoAdicField>(coeffs: &[F], expansion: usize) -> Vec<F> {
    let engine = ntt::NttEngine::<F>::new_from_cache();
    let expanded_size = coeffs.len() * expansion;
    let mut result = Vec::with_capacity(expanded_size);
    // Note: We can also zero-extend the coefficients and do a larger NTT.
    // But this is more efficient.

    // Do coset NTT.
    let root = engine.root(expanded_size);
    result.extend_from_slice(coeffs);
    result.par_extend((1..expansion).into_par_iter().flat_map(|i| {
        let root_i = root.exp_u64(i as u64);
        coeffs
            .par_iter()
            .enumerate()
            .map_with(F::ZERO, move |root_j, (j, coeff)| {
                if root_j.is_zero() {
                    *root_j = root_i.exp_u64(j as u64);
                } else {
                    *root_j *= root_i;
                }
                *coeff * *root_j
            })
    }));

    ntt_batch(&mut result, coeffs.len());
    transpose(&mut result, expansion, coeffs.len());
    result
}
