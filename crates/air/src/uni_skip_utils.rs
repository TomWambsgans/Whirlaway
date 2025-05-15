use algebra::pols::Multilinear;
use algebra::pols::{MultilinearDevice, MultilinearHost};
use cuda_bindings::{
    cuda_matrix_down_folded_with_univariate_skips, cuda_matrix_up_folded_with_univariate_skips,
};
use cuda_engine::cuda_sync;
use p3_field::Field;
use rayon::prelude::*;
use tracing::instrument;

#[instrument(name = "matrix_up_folded_with_univariate_skips", skip_all)]
pub(crate) fn matrix_up_folded_with_univariate_skips<F: Field>(
    zerocheck_challenges: &[F],
    on_device: bool,
    univariate_skips: usize,
) -> Multilinear<F> {
    // TODO: It's sparse => bad performance
    if on_device {
        matrix_up_folded_with_univariate_skips_device(zerocheck_challenges, univariate_skips).into()
    } else {
        let res =
            matrix_up_folded_with_univariate_skips_host(zerocheck_challenges, univariate_skips)
                .into();
        cuda_sync();
        res
    }
}

pub(crate) fn matrix_up_folded_with_univariate_skips_host<F: Field>(
    zerocheck_challenges: &[F],
    univariate_skips: usize,
) -> MultilinearHost<F> {
    // TODO: It's sparse => bad performance
    let n = zerocheck_challenges.len();
    let n_vars = n + univariate_skips * 2 - 1;
    let mut folded = MultilinearHost::zero(n_vars);
    let point_len = univariate_skips + (n - 1);
    let inner_eq_mle = MultilinearHost::eq_mle(&zerocheck_challenges[1..]);
    folded
        .evals
        .par_chunks_mut(1 << point_len)
        .enumerate()
        .for_each(|(i, block)| {
            block[i << (n - 1)..(i + 1) << (n - 1)].copy_from_slice(&inner_eq_mle.evals);
        });

    let zerocheck_challenges_prod = zerocheck_challenges[1..].iter().copied().product::<F>();
    folded.evals[(1 << n_vars) - 1] -= zerocheck_challenges_prod;
    folded.evals[(1 << n_vars) - 2] += zerocheck_challenges_prod;

    folded.into()
}

/// Async
pub(crate) fn matrix_up_folded_with_univariate_skips_device<F: Field>(
    zerocheck_challenges: &[F],
    univariate_skips: usize,
) -> MultilinearDevice<F> {
    MultilinearDevice::new(cuda_matrix_up_folded_with_univariate_skips(
        zerocheck_challenges,
        univariate_skips,
    ))
}

/// Async
#[instrument(name = "matrix_down_folded_with_univariate_skips", skip_all)]
pub(crate) fn matrix_down_folded_with_univariate_skips<F: Field>(
    zerocheck_challenges: &[F],
    on_device: bool,
    univariate_skips: usize,
) -> Multilinear<F> {
    if on_device {
        let res =
            matrix_down_folded_with_univariate_skips_device(zerocheck_challenges, univariate_skips)
                .into();
        cuda_sync();
        res
    } else {
        matrix_down_folded_with_univariate_skips_host(zerocheck_challenges, univariate_skips).into()
    }
}

pub(crate) fn matrix_down_folded_with_univariate_skips_device<F: Field>(
    zerocheck_challenges: &[F],
    univariate_skips: usize,
) -> MultilinearDevice<F> {
    MultilinearDevice::new(cuda_matrix_down_folded_with_univariate_skips(
        zerocheck_challenges,
        univariate_skips,
    ))
}

pub(crate) fn matrix_down_folded_with_univariate_skips_host<F: Field>(
    zerocheck_challenges: &[F],
    univariate_skips: usize,
) -> MultilinearHost<F> {
    // TODO: It's sparse => bad performance
    let n = zerocheck_challenges.len();
    // n_vars defined as in the original function.
    let n_vars = n + univariate_skips * 2 - 1;
    let mut folded = MultilinearHost::zero(n_vars);
    let point_len = univariate_skips + (n - 1);
    let inner_eq_mles = (1..zerocheck_challenges.len())
        .map(|i| MultilinearHost::eq_mle(&zerocheck_challenges[1..i]))
        .collect::<Vec<_>>();

    let mut suffix_prods = vec![F::ZERO; n];
    suffix_prods[n - 1] = F::ONE;
    suffix_prods[n - 2] = zerocheck_challenges[n - 1];
    for i in (0..n - 2).rev() {
        suffix_prods[i] = zerocheck_challenges[i + 1] * suffix_prods[i + 1];
    }

    folded
        .evals
        .par_chunks_mut(1 << point_len)
        .enumerate()
        .for_each(|(i, block)| {
            for pos in 0..block.len() {
                let computed_value_for_pos;
                let k_inner = pos.trailing_zeros() as usize;
                if k_inner >= point_len {
                    continue;
                }
                if point_len - k_inner >= univariate_skips + n {
                    continue;
                }

                let y = point_len - k_inner;
                let mut zerocheck_challenges_prod_inner = if y >= univariate_skips {
                    suffix_prods[y - univariate_skips]
                } else {
                    // does i finish with (univariate_skips - y) consecutive ones ?
                    let mask = (1 << (univariate_skips - y)) - 1;

                    if i & mask != mask {
                        continue;
                    }
                    suffix_prods[0]
                };

                let z = point_len - k_inner - 1;
                if z < univariate_skips {
                    if (i & (1 << (univariate_skips - z - 1))) != 0 {
                        continue;
                    }
                } else {
                    zerocheck_challenges_prod_inner *=
                        F::ONE - zerocheck_challenges[1 + z - univariate_skips]
                };

                if zerocheck_challenges_prod_inner.is_zero() {
                    continue;
                }

                let y_val = pos >> (k_inner + 1);
                if k_inner + 2 < n {
                    let shift_s1 = n - (k_inner + 2);
                    let x_shifted_component = i << shift_s1;

                    if y_val < x_shifted_component {
                        continue;
                    }
                    let j_original_idx = y_val - x_shifted_component;
                    let eq_mle_idx = n - (k_inner + 2);
                    if eq_mle_idx >= inner_eq_mles.len() {
                        continue;
                    }
                    let eq_mle = &inner_eq_mles[eq_mle_idx];
                    if j_original_idx >= eq_mle.evals.len() {
                        continue;
                    }
                    computed_value_for_pos =
                        eq_mle.evals[j_original_idx] * zerocheck_challenges_prod_inner;
                } else {
                    let shift_s4 = k_inner + 2 - n;
                    if y_val != (i >> shift_s4) {
                        continue;
                    }
                    computed_value_for_pos = zerocheck_challenges_prod_inner;
                }

                block[pos] = computed_value_for_pos;
            }
        });

    folded.evals[(1 << n_vars) - 1] += zerocheck_challenges[1..].iter().copied().product::<F>();

    folded
}

#[cfg(test)]
mod test {
    use super::*;
    use cuda_engine::{CudaFunctionInfo, cuda_init, cuda_load_function};
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;

    type F = KoalaBear;

    #[test]
    fn test_matrix_folded_with_univariate_skips() {
        cuda_init();
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "univariate_skip.cu",
            "matrix_up_folded_with_univariate_skips",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "univariate_skip.cu",
            "matrix_down_folded_with_univariate_skips",
        ));
        cuda_load_function(CudaFunctionInfo::one_field::<F>(
            "multilinear.cu",
            "eq_mle_start",
        ));
        let n = 10;
        for univariate_skips in 1..5 {
            let zerocheck_challenges = (0..n).map(|i| F::from_usize(i)).collect::<Vec<_>>();

            let cpu_res = matrix_down_folded_with_univariate_skips_host(
                &zerocheck_challenges,
                univariate_skips,
            );
            let gpu_res = matrix_down_folded_with_univariate_skips_device(
                &zerocheck_challenges,
                univariate_skips,
            );
            assert_eq!(cpu_res.hash(), gpu_res.hash());

            let cpu_res = matrix_up_folded_with_univariate_skips_host(
                &zerocheck_challenges,
                univariate_skips,
            );
            let gpu_res = matrix_up_folded_with_univariate_skips_device(
                &zerocheck_challenges,
                univariate_skips,
            );
            assert_eq!(cpu_res.hash(), gpu_res.hash());
        }
    }
}
