use std::any::TypeId;
use std::borrow::Borrow;

use p3_field::PackedValue;
use p3_field::{ExtensionField, Field, dot_product};
use rayon::prelude::*;
use tracing::instrument;
use whir_p3::poly::evals::EvaluationsList;

pub fn fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
    m: &EvaluationsList<EF>,
    scalars: &[F],
) -> EvaluationsList<EF> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.num_evals());
    let new_size = m.num_evals() / scalars.len();

    if TypeId::of::<F>() == TypeId::of::<EF>() {
        return unsafe {
            std::mem::transmute(fold_multilinear_packed::<F>(
                std::mem::transmute(m),
                scalars,
            ))
        };
    }

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

pub fn fold_multilinear_in_small_field_no_skip<F: Field, EF: ExtensionField<F>>(
    m: &EvaluationsList<EF>,
    scalars: &[F],
) -> EvaluationsList<EF> {
    assert!(m.num_evals() >= 2);
    let new_size = m.num_evals() / 2;
    let (first_half, second_half) = m.evals().split_at(new_size);

    EvaluationsList::new(
        first_half
            .iter()
            .zip(second_half.iter())
            .map(|(&a, &b)| a * scalars[0] + b * scalars[1])
            .collect(),
    )
}

// TODO packing for all the cases
pub fn fold_multilinear_packed<F: Field>(
    m: &EvaluationsList<F>,
    scalars: &[F],
) -> EvaluationsList<F> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.num_evals());
    let new_size = m.num_evals() / scalars.len();

    let inners = (0..scalars.len())
        .map(|i| &m.evals()[i * new_size..(i + 1) * new_size])
        .collect::<Vec<_>>();

    let inners_packed = inners
        .iter()
        .map(|inner| F::Packing::pack_slice(inner))
        .collect::<Vec<_>>();

    let packed_res = (0..new_size / F::Packing::WIDTH)
        .into_par_iter()
        .map(|i| {
            scalars
                .iter()
                .enumerate()
                .map(|(j, s)| inners_packed[j][i] * *s)
                .sum::<F::Packing>()
        })
        .collect::<Vec<_>>();

    let mut unpacked: Vec<F> = unsafe { std::mem::transmute(packed_res) };
    unsafe {
        unpacked.set_len(new_size);
    }

    EvaluationsList::new(unpacked)
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

pub fn fold_multilinear_in_large_field_no_skip<F: Field, EF: ExtensionField<F>>(
    m: &EvaluationsList<F>,
    scalars: &[EF],
) -> EvaluationsList<EF> {
    assert!(m.num_evals() >= 2);
    let new_size = m.num_evals() / 2;
    let (first_half, second_half) = m.evals().split_at(new_size);

    EvaluationsList::new(
        first_half
            .iter()
            .zip(second_half.iter())
            .map(|(&a, &b)| scalars[0] * a + scalars[1] * b)
            .collect(),
    )
}

#[instrument(name = "multilinears_linear_combination", skip_all)]
pub fn multilinears_linear_combination<
    F: Field,
    EF: ExtensionField<F>,
    P: Borrow<EvaluationsList<F>> + Send + Sync,
>(
    pols: &[P],
    scalars: &[EF],
) -> EvaluationsList<EF> {
    assert_eq!(pols.len(), scalars.len());
    let n_vars = pols[0].borrow().num_variables();
    assert!(pols.iter().all(|p| p.borrow().num_variables() == n_vars));
    let evals = (0..1 << n_vars)
        .into_par_iter()
        .map(|i| {
            dot_product(
                scalars.iter().copied(),
                pols.iter().map(|p| p.borrow().evals()[i]),
            )
        })
        .collect::<Vec<_>>();
    EvaluationsList::new(evals)
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

pub fn batch_fold_multilinear_in_large_field_no_skip<F: Field, EF: ExtensionField<F>>(
    polys: &[&EvaluationsList<F>],
    scalars: &[EF],
) -> Vec<EvaluationsList<EF>> {
    polys
        .par_iter()
        .map(|poly| fold_multilinear_in_large_field_no_skip(poly, scalars))
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

pub fn batch_fold_multilinear_in_small_field_no_skip<F: Field, EF: ExtensionField<F>>(
    polys: &[&EvaluationsList<EF>],
    scalars: &[F],
) -> Vec<EvaluationsList<EF>> {
    polys
        .par_iter()
        .map(|poly| fold_multilinear_in_small_field_no_skip(poly, scalars))
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

#[instrument(name = "add_multilinears", skip_all)]
pub fn add_multilinears<F: Field>(
    pol1: &EvaluationsList<F>,
    pol2: &EvaluationsList<F>,
) -> EvaluationsList<F> {
    assert_eq!(pol1.num_variables(), pol2.num_variables());
    let mut dst = pol1.evals().to_vec();
    dst.par_iter_mut()
        .zip(pol2.evals().par_iter())
        .for_each(|(a, b)| *a += *b);
    EvaluationsList::new(dst)
}
