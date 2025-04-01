use algebra::ntt::intt_batch;
use p3_field::TwoAdicField;

use rayon::prelude::*;

pub fn restructure_evaluations<F: TwoAdicField>(
    mut stacked_evaluations: Vec<F>,
    _domain_gen: F,
    domain_gen_inv: F,
    folding_factor: usize,
) -> Vec<F> {
    let folding_size = 1_u64 << folding_factor;
    assert_eq!(stacked_evaluations.len() % (folding_size as usize), 0);

    // TODO: This partially undoes the NTT transform from tne encoding.
    // Maybe there is a way to not do the full transform in the first place.

    // Batch inverse NTTs
    intt_batch(&mut stacked_evaluations, folding_size as usize);

    // Apply coset and size correction.
    // Stacked evaluation at i is f(B_l) where B_l = w^i * <w^n/k>
    let size_inv = F::from_u64(folding_size).inverse();
    stacked_evaluations
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

    stacked_evaluations
}
