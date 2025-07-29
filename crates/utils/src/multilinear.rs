use std::borrow::Borrow;

use p3_field::{BasedVectorSpace, PackedValue};
use p3_field::{ExtensionField, Field, dot_product};
use rayon::prelude::*;
use tracing::instrument;
use whir_p3::poly::evals::EvaluationsList;

use crate::{EFPacking, PF};

pub fn fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>, D>(
    m: &[D],
    scalars: &[F],
) -> Vec<EF> {
    // TODO ...
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.len());
    let new_size = m.len() / scalars.len();

    let dim = <EF as BasedVectorSpace<F>>::DIMENSION;

    let m_transmuted: &[F] =
        unsafe { std::slice::from_raw_parts(std::mem::transmute(m.as_ptr()), m.len() * dim) };
    let res_transmuted = {
        let new_size = m.len() * dim / scalars.len();

        if new_size < F::Packing::WIDTH {
            (0..new_size)
                .into_par_iter()
                .map(|i| {
                    scalars
                        .iter()
                        .enumerate()
                        .map(|(j, s)| *s * m_transmuted[i + j * new_size])
                        .sum()
                })
                .collect()
        } else {
            let inners = (0..scalars.len())
                .map(|i| &m_transmuted[i * new_size..(i + 1) * new_size])
                .collect::<Vec<_>>();
            let inners_packed = inners
                .iter()
                .map(|&inner| F::Packing::pack_slice(inner))
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

            unpacked
        }
    };
    let res: Vec<EF> = unsafe {
        let mut res: Vec<EF> = std::mem::transmute(res_transmuted);
        res.set_len(new_size);
        res
    };

    res
}

pub fn fold_multilinear_in_large_field<F: Field, EF: ExtensionField<F>>(
    m: &[F],
    scalars: &[EF],
) -> Vec<EF> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.len());
    let new_size = m.len() / scalars.len();
    (0..new_size)
        .into_par_iter()
        .map(|i| {
            scalars
                .iter()
                .enumerate()
                .map(|(j, s)| *s * m[i + j * new_size])
                .sum()
        })
        .collect()
}

pub fn fold_multilinear_in_large_field_packed<EF: Field + ExtensionField<PF<EF>>>(
    m: &[EFPacking<EF>],
    scalars: &[EF],
) -> Vec<EFPacking<EF>> {
    assert!(scalars.len().is_power_of_two() && scalars.len() <= m.len());
    let new_size = m.len() / scalars.len();

    (0..new_size)
        .into_par_iter()
        .map(|i| {
            scalars
                .iter()
                .enumerate()
                .map(|(j, s)| m[i + j * new_size] * *s)
                .sum()
        })
        .collect()
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
    polys: &[&[F]],
    scalars: &[EF],
) -> Vec<Vec<EF>> {
    polys
        .par_iter()
        .map(|poly| fold_multilinear_in_large_field(poly, scalars))
        .collect()
}

pub fn batch_fold_multilinear_in_large_field_packed<EF: Field + ExtensionField<PF<EF>>>(
    polys: &[&[EFPacking<EF>]],
    scalars: &[EF],
) -> Vec<Vec<EFPacking<EF>>> {
    polys
        .iter()
        .map(|poly| fold_multilinear_in_large_field_packed(poly, scalars))
        .collect()
}

pub fn batch_fold_multilinear_in_small_field<F: Field, EF: ExtensionField<F>>(
    polys: &[&[EF]],
    scalars: &[F],
) -> Vec<Vec<EF>> {
    polys
        .par_iter()
        .map(|poly| fold_multilinear_in_small_field(poly, scalars))
        .collect()
}

pub fn batch_fold_multilinear_in_small_field_packed<EF: Field + ExtensionField<PF<EF>>>(
    polys: &[&[EFPacking<EF>]],
    scalars: &[PF<EF>],
) -> Vec<Vec<EF>> {
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
