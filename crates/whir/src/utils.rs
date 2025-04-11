// use std::collections::BTreeSet;

// use algebra::pols::{Multilinear, MultilinearDevice};
// use arithmetic_circuit::CircuitComputation;
// use cuda_engine::{cuda_sync, memcpy_htod};
// use fiat_shamir::FsProver;
// use p3_field::{ExtensionField, Field};

// /// performs big-endian binary decomposition of `value` and returns the result.
// ///
// /// `n_bits` must be at must usize::BITS. If it is strictly smaller, the most significant bits of `value` are ignored.
// /// The returned vector v ends with the least significant bit of `value` and always has exactly `n_bits` many elements.
// pub fn to_binary(value: usize, n_bits: usize) -> Vec<bool> {
//     // Ensure that n is within the bounds of the input integer type
//     assert!(n_bits <= usize::BITS as usize);
//     let mut result = vec![false; n_bits];
//     for i in 0..n_bits {
//         result[n_bits - 1 - i] = (value & (1 << i)) != 0;
//     }
//     result
// }

// // Sync
// pub fn sumcheck_prove_with_cuda_or_cpu<F: Field, EF: ExtensionField<F>>(
//     multilinears: &[Multilinear<EF>],
//     exprs: &[CircuitComputation<F>],
//     batching_scalars: &[EF],
//     eq_factor: Option<&[EF]>,
//     is_zerofier: bool,
//     fs_prover: &mut FsProver,
//     sum: Option<EF>,
//     n_rounds: Option<usize>,
//     pow_bits: usize,
//     cuda: bool,
// ) -> (Vec<EF>, Vec<Multilinear<EF>>) {
//     let (challenges, folded_multilinears) = if cuda {
//         assert!(multilinears.iter().all(|m| m.is_device()));
//         let multilinears = multilinears
//             .into_iter()
//             .map(|m| m.as_device_ref())
//             .collect::<Vec<_>>();
//         sumcheck::prove_with_cuda(
//             &multilinears,
//             exprs,
//             batching_scalars,
//             eq_factor,
//             is_zerofier,
//             fs_prover,
//             sum,
//             n_rounds,
//             pow_bits,
//         )
//     } else {
//         assert!(multilinears.iter().all(|m| m.is_host()));
//         let multilinears = multilinears
//             .into_iter()
//             .map(|m| m.as_host_ref())
//             .collect::<Vec<_>>();
//         sumcheck::prove(
//             &multilinears,
//             exprs,
//             batching_scalars,
//             eq_factor,
//             is_zerofier,
//             fs_prover,
//             sum,
//             n_rounds,
//             pow_bits,
//         )
//     };

//     let folded_multilinears = folded_multilinears
//         .into_iter()
//         .map(|m| {
//             if cuda {
//                 Multilinear::Device(MultilinearDevice::new(memcpy_htod(&m.evals))) // TODO Avoid, the cuda sumcheck should return a cuda slice
//             } else {
//                 Multilinear::Host(m)
//             }
//         })
//         .collect::<Vec<_>>();
//     cuda_sync();

//     (challenges, folded_multilinears)
// }
