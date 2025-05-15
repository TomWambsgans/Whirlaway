
#include <stdio.h>
#include <stdint.h>
#include <cassert>

#include "ff_wrapper.cu"
#include "utils.cu"

extern "C" __global__ void matrix_up_folded_with_univariate_skips(Field_A __restrict__ *res,
                                                                  const Field_A __restrict__ *inner_eq_mle,
                                                                  uint32_t len_zerocheck_challenge,
                                                                  uint32_t univariate_skips,
                                                                  Field_A zerocheck_challenge_prod)
{
    int n = len_zerocheck_challenge;
    int n_vars = n + univariate_skips * 2 - 1;
    int point_len = univariate_skips + n - 1;

    int total_n_threads = blockDim.x * gridDim.x;

    int n_ops = (1 << (n - 1));
    int n_reps = (n_ops + total_n_threads - 1) / total_n_threads;
    for (int rep = 0; rep < n_reps; rep++)
    {
        int tid = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (tid >= n_ops)
        {
            break;
        }

        Field_A eval = inner_eq_mle[tid];

        for (int i = 0; i < 1 << univariate_skips; i++)
        {
            Field_A *block = &res[i << point_len];
            int offset = i << (n - 1);
            block[offset + tid] = eval;
        }

        if (tid == n_ops - 1)
        {
            SUB_AA(res[(1 << n_vars) - 1], zerocheck_challenge_prod, res[(1 << n_vars) - 1])
        }
        if (tid == n_ops - 2)
        {
            ADD_AA(res[(1 << n_vars) - 2], zerocheck_challenge_prod, res[(1 << n_vars) - 2])
        }
    }
}

extern "C" __global__ void matrix_down_folded_with_univariate_skips(Field_A __restrict__ *res,
                                                                    const Field_A __restrict__ **inner_eq_mles,
                                                                    uint32_t len_zerocheck_challenge,
                                                                    uint32_t univariate_skips,
                                                                    const Field_A __restrict__ *zerocheck_challenge,
                                                                    const Field_A __restrict__ *suffix_prods,
                                                                    Field_A zerocheck_challenge_prod)
{
    int n = len_zerocheck_challenge;
    int n_vars = n + univariate_skips * 2 - 1;
    int point_len = univariate_skips + n - 1;

    extern __shared__ Field_A cached;
    Field_A *cached_zerocheck_challenge = &cached;      // len_zerocheck_challenge elements
    Field_A *cached_suffix_prods = &cached + point_len; // len_zerocheck_challenge elements

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
    }

    assert(blockDim.x >= point_len * 2);
    if (threadIdx.x < point_len)
    {
        cached_zerocheck_challenge[threadIdx.x] = zerocheck_challenge[threadIdx.x];
    }
    else if (threadIdx.x < point_len * 2)
    {
        cached_suffix_prods[threadIdx.x - point_len] = suffix_prods[threadIdx.x - point_len];
    }
    __syncthreads();

    int total_n_threads = blockDim.x * gridDim.x;
    int n_ops = (1 << n_vars);
    int n_reps = (n_ops + total_n_threads - 1) / total_n_threads;

    for (int rep = 0; rep < n_reps; rep++)
    {
        int tid = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (tid >= n_ops)
        {
            break;
        }

        Field_A computed_value_for_pos;

        int i = tid >> point_len;
        int pos = tid % (1 << point_len);

        int k_inner = trailing_zeros(pos);
        if (k_inner >= point_len)
        {
            continue;
        }

        if (point_len - k_inner >= univariate_skips + n)
        {
            continue;
        }

        int y = point_len - k_inner;
        Field_A zerocheck_challenges_prod_inner;
        if (y >= univariate_skips)
        {
            zerocheck_challenges_prod_inner = cached_suffix_prods[y - univariate_skips];
        }
        else
        {
            // does i finish with (univariate_skips - y) consecutive ones ?
            int mask = (1 << (univariate_skips - y)) - 1;
            if ((i & mask) != mask)
            {
                continue;
            }
            zerocheck_challenges_prod_inner = cached_suffix_prods[0];
        }

        int z = point_len - k_inner - 1;
        if (z < univariate_skips)
        {
            if ((i & (1 << (univariate_skips - z - 1))) != 0)
            {
                continue;
            }
        }
        else
        {
            Field_A one_minus_zerocheck_challenge;
            Field_A one = Field_A::one();
            SUB_AA(one, cached_zerocheck_challenge[1 + z - univariate_skips], one_minus_zerocheck_challenge);
            Field_A prod;
            MUL_AA(zerocheck_challenges_prod_inner, one_minus_zerocheck_challenge, prod);
            zerocheck_challenges_prod_inner = prod;
        }

        int y_val = pos >> (k_inner + 1);
        if (k_inner + 2 < n)
        {
            int shift_s1 = n - (k_inner + 2);
            int x_shifted_component = i << shift_s1;

            if (y_val < x_shifted_component)
            {
                continue;
            }
            int j_original_idx = y_val - x_shifted_component;
            int eq_mle_idx = n - (k_inner + 2);
            if (eq_mle_idx >= n - 1)
            {
                continue;
            }
            const Field_A *eq_mle = inner_eq_mles[eq_mle_idx];
            if (j_original_idx >= 1 << eq_mle_idx)
            {
                continue;
            }
            MUL_AA(eq_mle[j_original_idx], zerocheck_challenges_prod_inner, computed_value_for_pos)
        }
        else
        {
            int shift_s4 = k_inner + 2 - n;
            if (y_val != (i >> shift_s4))
            {
                continue;
            }
            computed_value_for_pos = zerocheck_challenges_prod_inner;
        }

        if (tid == (1 << n_vars) - 1)
        {
            ADD_AA(computed_value_for_pos, zerocheck_challenge_prod, computed_value_for_pos);
        }

        res[tid] = computed_value_for_pos;
    }
}
