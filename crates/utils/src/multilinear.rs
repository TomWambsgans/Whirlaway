use std::borrow::Borrow;

use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use whir_p3::poly::{evals::EvaluationsList, multilinear::MultilinearPoint};

pub fn fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
    m: &EvaluationsList<EF>,
    scalars: &[F],
) -> EvaluationsList<EF> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.num_evals());
    let new_size = m.num_evals() / scalars.len();
    EvaluationsList::new(
        (0..new_size)
            .into_par_iter()
            .map(|i| {
                scalars
                    .iter()
                    .enumerate()
                    .map(|(j, s)| m.evals()[i + j * new_size] * *s)
                    .sum()
            })
            .collect(),
    )
}

pub fn fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
    m: &EvaluationsList<F>,
    scalars: &[EF],
) -> EvaluationsList<EF> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.num_evals());
    let new_size = m.num_evals() / scalars.len();
    EvaluationsList::new(
        (0..new_size)
            .into_par_iter()
            .map(|i| {
                scalars
                    .iter()
                    .enumerate()
                    .map(|(j, s)| *s * m.evals()[i + j * new_size])
                    .sum()
            })
            .collect(),
    )
}

pub fn multilinears_linear_combination<
    F: Field,
    EF: ExtensionField<F>,
    P: Borrow<EvaluationsList<F>>,
>(
    pols: &[P],
    scalars: &[EF],
) -> EvaluationsList<EF> {
    assert_eq!(pols.len(), scalars.len());
    let mut sum = EvaluationsList::new(EF::zero_vec(pols[0].borrow().num_evals()));
    for i in 0..scalars.len() {
        add_assign(&mut sum, &pols[i].borrow().scale(scalars[i]));
    }
    sum
}

pub fn add_dummy_starting_variables<F: Field>(
    m: &EvaluationsList<F>,
    n: usize,
) -> EvaluationsList<F> {
    // TODO remove
    EvaluationsList::new(m.evals().repeat(1 << n))
}

pub fn add_dummy_ending_variables<F: Field>(
    m: &EvaluationsList<F>,
    n: usize,
) -> EvaluationsList<F> {
    // TODO remove
    let evals = m
        .evals()
        .iter()
        .flat_map(|item| std::iter::repeat_n(*item, 1 << n))
        .collect();
    EvaluationsList::new(evals)
}

fn add_assign<F: Field>(m: &mut EvaluationsList<F>, other: &EvaluationsList<F>) {
    assert_eq!(m.num_variables(), other.num_variables());
    m.evals_mut()
        .par_iter_mut()
        .zip(other.evals().par_iter())
        .for_each(|(a, b)| *a += *b);
}

pub fn multilinear_batch_evaluate<F: Field, EF: ExtensionField<F>>(
    pols: &[EvaluationsList<F>],
    point: &MultilinearPoint<EF>,
) -> Vec<EF> {
    pols.par_iter().map(|pol| pol.evaluate(point)).collect()
}

pub fn batch_fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
    polys: &[&EvaluationsList<F>],
    scalars: &[EF],
) -> Vec<EvaluationsList<EF>> {
    polys
        .par_iter()
        .map(|poly| fold_multilinear_in_large_field(poly, scalars))
        .collect()
}

pub fn batch_fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
    polys: &[&EvaluationsList<EF>],
    scalars: &[F],
) -> Vec<EvaluationsList<EF>> {
    polys
        .par_iter()
        .map(|poly| fold_multilinear_in_small_field(poly, scalars))
        .collect()
}

pub fn packed_multilinear<F: Field>(pols: &[EvaluationsList<F>]) -> EvaluationsList<F> {
    let n_vars = pols[0].num_variables();
    assert!(pols.iter().all(|p| p.num_variables() == n_vars));
    let packed_len = (pols.len() << n_vars).next_power_of_two();
    let mut dst = F::zero_vec(packed_len);
    let mut offset = 0;
    // TODO parallelize
    for pol in pols {
        dst[offset..offset + pol.num_evals()].copy_from_slice(pol.evals());
        offset += pol.num_evals();
    }
    EvaluationsList::new(dst)
}
