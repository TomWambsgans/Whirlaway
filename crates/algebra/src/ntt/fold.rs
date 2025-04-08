use crate::ntt::{intt_batch, transpose};
use p3_field::{ExtensionField, Field, TwoAdicField};
use rayon::prelude::*;

pub fn restructure_evaluations<F: TwoAdicField, EF: ExtensionField<F>>(
    mut evals: Vec<EF>,
    domain_gen_inv: F,
    folding_factor: usize,
) -> Vec<EF> {
    let folding_size = 1_u64 << folding_factor;
    assert_eq!(evals.len() % (folding_size as usize), 0);

    evals = stack_evaluations(evals, folding_factor);

    // TODO: This partially undoes the NTT transform from tne encoding.
    // Maybe there is a way to not do the full transform in the first place.

    // Batch inverse NTTs
    intt_batch::<F, EF>(&mut evals, folding_size as usize);

    // Apply coset and size correction.
    // Stacked evaluation at i is f(B_l) where B_l = w^i * <w^n/k>
    let size_inv = F::from_u64(folding_size).inverse();
    evals
        .par_chunks_exact_mut(folding_size as usize)
        .enumerate()
        .for_each_with(F::ZERO, |offset, (i, answers)| {
            if *offset == F::ZERO {
                *offset = domain_gen_inv.exp_u64(i as u64);
            } else {
                *offset *= domain_gen_inv;
            }
            let mut scale = size_inv;
            for v in answers.iter_mut() {
                *v *= scale;
                scale *= *offset;
            }
        });

    evals
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
