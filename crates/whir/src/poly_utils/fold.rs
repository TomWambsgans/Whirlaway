use algebra::ntt::intt_batch;
use p3_field::{ExtensionField, TwoAdicField};
use rayon::prelude::*;
use tracing::instrument;

use crate::utils::stack_evaluations;

#[instrument(name = "restructure_evaluations", skip_all)]
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
