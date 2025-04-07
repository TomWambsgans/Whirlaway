#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use algebra::pols::{
    CircuitComputation, HypercubePoint, MultilinearPolynomial, PartialHypercubePoint,
};
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

mod prove;
pub use prove::*;

mod custom;
pub use custom::*;

mod verify;
pub use verify::*;

mod cuda;
pub use cuda::*;

#[cfg(test)]
mod test;

pub fn sum_batched_exprs_over_hypercube<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    multilinears: &[MultilinearPolynomial<NF>],
    n_vars: usize,
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
) -> EF {
    HypercubePoint::par_iter(n_vars)
        .map(|x| {
            let point = multilinears
                .iter()
                .map(|pol| pol.eval_hypercube(&x))
                .collect::<Vec<_>>();
            eval_batched_exprs(exprs, batching_scalars, &point)
        })
        .sum::<EF>()
}

pub fn eval_batched_exprs<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    point: &[NF],
) -> EF {
    if exprs.len() == 1 {
        EF::from(exprs[0].eval(point))
    } else {
        exprs
            .iter()
            .zip(batching_scalars)
            .skip(1)
            .map(|(expr, scalar)| *scalar * expr.eval(point))
            .sum::<EF>()
            + exprs[0].eval(point)
    }
}

pub fn eval_batched_exprs_mle<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    multilinears: &[MultilinearPolynomial<NF>],
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    point: &[EF],
) -> EF {
    let inner_evals = multilinears
        .iter()
        .map(|pol| pol.eval(point))
        .collect::<Vec<_>>();
    eval_batched_exprs(exprs, batching_scalars, &inner_evals)
}

pub fn eval_batched_exprs_on_partial_hypercube<
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
>(
    multilinears: &[MultilinearPolynomial<NF>],
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    point: &PartialHypercubePoint,
) -> EF {
    let inner_evals = multilinears
        .iter()
        .map(|pol| pol.eval_partial_hypercube(point))
        .collect::<Vec<_>>();
    eval_batched_exprs(exprs, batching_scalars, &inner_evals)
}
