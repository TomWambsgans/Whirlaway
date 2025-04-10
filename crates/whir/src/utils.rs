use std::collections::BTreeSet;

use algebra::pols::CircuitComputation;
use fiat_shamir::FsProver;
use p3_field::{ExtensionField, Field};

use cuda_bindings::{
    MultilinearPolynomialCuda, MultilinearPolynomialMaybeCuda, cuda_sync, memcpy_htod,
};

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

// Sync
pub fn sumcheck_prove_with_cuda_or_cpu<F: Field, EF: ExtensionField<F>>(
    multilinears: &[MultilinearPolynomialMaybeCuda<EF>],
    exprs: &[CircuitComputation<F>],
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut FsProver,
    sum: Option<EF>,
    n_rounds: Option<usize>,
    pow_bits: usize,
    cuda: bool,
) -> (Vec<EF>, Vec<MultilinearPolynomialMaybeCuda<EF>>) {
    let (challenges, folded_multilinears) = if cuda {
        assert!(multilinears.iter().all(|m| m.is_cuda()));
        let multilinears = multilinears
            .into_iter()
            .map(|m| m.as_cuda())
            .collect::<Vec<_>>();
        sumcheck::prove_with_cuda(
            &multilinears,
            exprs,
            batching_scalars,
            eq_factor,
            is_zerofier,
            fs_prover,
            sum,
            n_rounds,
            pow_bits,
        )
    } else {
        assert!(multilinears.iter().all(|m| m.is_cpu()));
        let multilinears = multilinears
            .into_iter()
            .map(|m| m.as_cpu())
            .collect::<Vec<_>>();
        sumcheck::prove(
            &multilinears,
            exprs,
            batching_scalars,
            eq_factor,
            is_zerofier,
            fs_prover,
            sum,
            n_rounds,
            pow_bits,
        )
    };

    let folded_multilinears = folded_multilinears
        .into_iter()
        .map(|m| {
            if cuda {
                MultilinearPolynomialMaybeCuda::Cuda(MultilinearPolynomialCuda::new(memcpy_htod(
                    &m.evals,
                ))) // TODO Avoid, the cuda sumcheck should return a cuda slice
            } else {
                MultilinearPolynomialMaybeCuda::Cpu(m)
            }
        })
        .collect::<Vec<_>>();
    if cuda {
        cuda_sync();
    }

    (challenges, folded_multilinears)
}
