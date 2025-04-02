use algebra::ntt::{expand_from_coeff, transpose};
use p3_field::{Field, TwoAdicField};
use std::collections::BTreeSet;

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

// FIXME(Gotti): comment does not match what function does (due to mismatch between folding_factor and folding_factor_exp)
// Also, k should be defined: k = evals.len() / 2^{folding_factor}, I guess.

/// Takes the vector of evaluations (assume that evals[i] = f(omega^i))
/// and folds them into a vector of such that folded_evals[i] = [f(omega^(i + k * j)) for j in 0..folding_factor]
pub fn stack_evaluations<F: Field>(mut evals: Vec<F>, folding_factor: usize) -> Vec<F> {
    let folding_factor_exp = 1 << folding_factor;
    assert!(evals.len() % folding_factor_exp == 0);
    let size_of_new_domain = evals.len() / folding_factor_exp;

    // interpret evals as (folding_factor_exp x size_of_new_domain)-matrix and transpose in-place
    transpose(&mut evals, folding_factor_exp, size_of_new_domain);
    evals
}

pub fn expand_from_coeff_maybe_with_cuda<F: TwoAdicField>(
    coeffs: &[F],
    expansion: usize,
    cuda: bool,
) -> Vec<F> {
    if cuda && coeffs.len() >= 1024 {
        cuda_bindings::cuda_ntt(coeffs, expansion)
    } else {
        expand_from_coeff(coeffs, expansion)
    }
}
