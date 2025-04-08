use algebra::ntt::{expand_from_coeff, restructure_evaluations};
use p3_field::{ExtensionField, TwoAdicField};
use std::collections::BTreeSet;
use tracing::instrument;

/// performs big-endian binary decomposition of `value` and returns the result.
///
/// `n_bits` must be at must usize::BITS. If it is strictly smaller, the most significant bits of `value` are ignored.
/// The returned vector v ends with the least significant bit of `value` and always has exactly `n_bits` many elements.
pub fn to_binary(value: usize, n_bits: usize) -> Vec<bool> {
    // Ensure that n is within the bounds of the input integer type
    assert!(n_bits <= usize::BITS as usize);
    let mut result = vec![false; n_bits];
    for i in 0..n_bits {
        result[n_bits - 1 - i] = (value & (1 << i)) != 0;
    }
    result
}

/// Deduplicates AND orders a vector
pub fn dedup<T: Ord>(v: impl IntoIterator<Item = T>) -> Vec<T> {
    Vec::from_iter(BTreeSet::from_iter(v))
}

#[instrument(name = "whir: expand_from_coeff_and_restructure", skip_all)]
pub fn expand_from_coeff_and_restructure<F: TwoAdicField, EF: ExtensionField<F>>(
    coeffs: &[EF],
    expansion: usize,
    domain_gen_inv: F,
    folding_factor: usize,
    cuda: bool,
) -> Vec<EF> {
    if cuda && coeffs.len() >= 1024 {
        let evals = cuda_bindings::cuda_expanded_ntt(coeffs, expansion);
        let folded_evals_dev = cuda_bindings::cuda_restructure_evaluations(&evals, folding_factor);
        let folded_evals = cuda_bindings::memcpy_dtoh(&folded_evals_dev);
        cuda_bindings::cuda_sync();
        folded_evals
    } else {
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let evals = expand_from_coeff::<F, EF>(coeffs, expansion);
        restructure_evaluations(evals, domain_gen_inv, folding_factor)
    }
}
