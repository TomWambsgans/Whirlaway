use cuda_engine::{CudaCall, CudaFunctionInfo, concat_pointers, cuda_alloc_zeros, memcpy_htod};
use cudarc::driver::{CudaSlice, PushKernelArg};
use p3_field::Field;

use crate::cuda_eq_mle;

// Async
pub fn cuda_matrix_up_folded_with_univariate_skips<F: Field>(
    zerocheck_challenges: &[F],
    univariate_skips: usize,
) -> CudaSlice<F> {
    let n = zerocheck_challenges.len();
    let n_vars = n + univariate_skips * 2 - 1;

    let inner_eq_mle = cuda_eq_mle(&zerocheck_challenges[1..]);

    let mut res = cuda_alloc_zeros(1 << n_vars);

    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>(
            "univariate_skip.cu",
            "matrix_up_folded_with_univariate_skips",
        ),
        inner_eq_mle.len(),
    );
    let n_u32 = n as u32;
    let univariate_skips_u32 = univariate_skips as u32;
    let zerocheck_challenges_prod = zerocheck_challenges[1..].iter().copied().product::<F>();

    call.arg(&mut res);
    call.arg(&inner_eq_mle);
    call.arg(&n_u32);
    call.arg(&univariate_skips_u32);
    call.arg(&zerocheck_challenges_prod);
    call.launch();

    res
}

// Async
pub fn cuda_matrix_down_folded_with_univariate_skips<F: Field>(
    zerocheck_challenges: &[F],
    univariate_skips: usize,
) -> CudaSlice<F> {
    // TODO duplicated eq_mle (the last element of inner_eq_mles == inner_eq_mle in cuda_matrix_up_folded_with_univariate_skips)

    let n = zerocheck_challenges.len();
    let n_vars = n + univariate_skips * 2 - 1;

    let mut res = cuda_alloc_zeros(1 << n_vars);

    let mut suffix_prods = vec![F::ZERO; n];
    suffix_prods[n - 1] = F::ONE;
    suffix_prods[n - 2] = zerocheck_challenges[n - 1];
    for i in (0..n - 2).rev() {
        suffix_prods[i] = zerocheck_challenges[i + 1] * suffix_prods[i + 1];
    }

    let inner_eq_mles = (1..zerocheck_challenges.len())
        .map(|i| cuda_eq_mle(&zerocheck_challenges[1..i]))
        .collect::<Vec<_>>();
    let inner_eq_mles_ptrs = concat_pointers(&inner_eq_mles);

    let mut call = CudaCall::new(
        CudaFunctionInfo::one_field::<F>(
            "univariate_skip.cu",
            "matrix_down_folded_with_univariate_skips",
        ),
        1 << n_vars,
    )
    .shared_mem_bytes(2 * n * size_of::<F>());
    let n_u32 = n as u32;
    let univariate_skips_u32 = univariate_skips as u32;
    let zerocheck_challenges_dev = memcpy_htod(&zerocheck_challenges);
    let suffix_prods_dev = memcpy_htod(&suffix_prods);
    let zerocheck_challenges_prod = zerocheck_challenges[1..].iter().copied().product::<F>();

    call.arg(&mut res);
    call.arg(&inner_eq_mles_ptrs);
    call.arg(&n_u32);
    call.arg(&univariate_skips_u32);
    call.arg(&zerocheck_challenges_dev);
    call.arg(&suffix_prods_dev);
    call.arg(&zerocheck_challenges_prod);
    call.launch();

    res
}
