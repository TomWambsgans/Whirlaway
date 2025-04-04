#include <stdio.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "finite_field.cu"

// TODO avoid hardcoding
#define LOG_N_THREADS_PER_BLOCK 8
#define N_THREADS_PER_BLOCK (1 << LOG_N_THREADS_PER_BLOCK)

#define N_BATCHING_SCALARS 3
#define N_REGISTERS 4 // Should idealy be as low as possible, because otherwise the cuda compiler will be forced to use global memory (-> "local" memory) instead of thread registers

/*
fold_X_by_Y:

"prime" -> Small Prime field
"ext" -> Big Extension field

inputs has length n_slices
inputs[0], inputs[1], ... all have length 2^slice_log_len

res has length n_slices
res[0], res[1], ... all have length 2^(slice_log_len-1)

at the end of ther kernel, for all i, for all j, res[i] = (1 - scalar) * inputs[i][j] + scalar * inputs[i][j + 2^(slice_log_len-1)]

*/

__device__ int n_folding_repetitions(const uint32_t n_slices, const uint32_t slice_log_len)
{
    const int total_n_threads = blockDim.x * gridDim.x;
    const int n_folding_ops = n_slices * (1 << (slice_log_len - 1));
    return (n_folding_ops + total_n_threads - 1) / total_n_threads;
}

extern "C" __global__ void fold_prime_by_prime(const uint32_t **inputs, uint32_t **res, const uint32_t scalar, const uint32_t n_slices, const uint32_t slice_log_len)
{
    const int n_repetitions = n_folding_repetitions(n_slices, slice_log_len);
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        const int slice_index = thread_index / (1 << (slice_log_len - 1));
        if (slice_index >= n_slices)
            return;
        const int slice_offset = thread_index % (1 << (slice_log_len - 1));
        uint32_t diff = monty_field_sub(inputs[slice_index][slice_offset + (1 << (slice_log_len - 1))], inputs[slice_index][slice_offset]);
        res[slice_index][slice_offset] = monty_field_add(monty_field_mul(scalar, diff), inputs[slice_index][slice_offset]);
    }
}

extern "C" __global__ void fold_prime_by_ext(const uint32_t **inputs, ExtField **res, const ExtField *scalar, const uint32_t n_slices, const uint32_t slice_log_len)
{
    const int n_repetitions = n_folding_repetitions(n_slices, slice_log_len);
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        const int slice_index = thread_index / (1 << (slice_log_len - 1));
        if (slice_index >= n_slices)
            return;
        const int slice_offset = thread_index % (1 << (slice_log_len - 1));
        uint32_t diff = monty_field_sub(inputs[slice_index][slice_offset + (1 << (slice_log_len - 1))], inputs[slice_index][slice_offset]);
        ExtField mul;
        mul_prime_by_ext_field(scalar, diff, &mul);
        add_prime_and_ext_field(&mul, inputs[slice_index][slice_offset], &res[slice_index][slice_offset]);
    }
}

extern "C" __global__ void fold_ext_by_prime(const ExtField **inputs, ExtField **res, const uint32_t scalar, const uint32_t n_slices, const uint32_t slice_log_len)
{
    const int n_repetitions = n_folding_repetitions(n_slices, slice_log_len);
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        const int slice_index = thread_index / (1 << (slice_log_len - 1));
        if (slice_index >= n_slices)
            return;
        const int slice_offset = thread_index % (1 << (slice_log_len - 1));
        ExtField diff;
        ext_field_sub(&inputs[slice_index][slice_offset + (1 << (slice_log_len - 1))], &inputs[slice_index][slice_offset], &diff);
        ExtField mul;
        mul_prime_by_ext_field(&diff, scalar, &mul);
        ext_field_add(&mul, &inputs[slice_index][slice_offset], &res[slice_index][slice_offset]);
    }
}

extern "C" __global__ void fold_ext_by_ext(const ExtField **inputs, ExtField **res, const ExtField *scalar, const uint32_t n_slices, const uint32_t slice_log_len)
{
    const int n_repetitions = n_folding_repetitions(n_slices, slice_log_len);
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * blockDim.x;
        const int slice_index = thread_index / (1 << (slice_log_len - 1));
        if (slice_index >= n_slices)
            return;
        const int slice_offset = thread_index % (1 << (slice_log_len - 1));
        ExtField diff;
        ext_field_sub(&inputs[slice_index][slice_offset + (1 << (slice_log_len - 1))], &inputs[slice_index][slice_offset], &diff);
        ExtField mul;
        ext_field_mul(&diff, scalar, &mul);
        ext_field_add(&mul, &inputs[slice_index][slice_offset], &res[slice_index][slice_offset]);
    }
}
// TODO avoid embedding overhead in the first round
extern "C" __global__ void sum_over_hypercube_ext(const ExtField **multilinears, ExtField *sums, const ExtField *batching_scalars, const uint32_t n_vars, ExtField *res)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int n_total_threads = N_THREADS_PER_BLOCK * gridDim.x;

    __shared__ ExtField cached_batching_scalars[N_BATCHING_SCALARS];

    // 1) Copy batching_scalars to shared memory
    for (int rep = 0; rep < (N_BATCHING_SCALARS + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK; rep++)
    {
        const int thread_index = threadIdx.x + rep * N_THREADS_PER_BLOCK;
        if (thread_index < N_BATCHING_SCALARS)
            cached_batching_scalars[thread_index] = batching_scalars[thread_index];
    }

    __syncthreads();

    // 2) Compute all the sums, and stored them in `sums`

    const int n_reps = ((1 << n_vars) + n_total_threads - 1) / n_total_threads;

    for (int rep = 0; rep < n_reps; rep++)
    {
        const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * N_THREADS_PER_BLOCK;

        // we want to compute sum[thread_index]

        ExtField regs[N_REGISTERS];

        // Computation here
        {

            regs[0] = multilinears[0][thread_index];
            mul_prime_by_ext_field(&regs[0], to_monty(11), &regs[0]);
            ext_field_mul(&regs[0], &cached_batching_scalars[0], &regs[1]);
            regs[2] = regs[1];

            regs[0] = multilinears[1][thread_index];
            mul_prime_by_ext_field(&regs[0], to_monty(22), &regs[0]);
            ext_field_mul(&regs[0], &cached_batching_scalars[1], &regs[1]);

            ext_field_add(&regs[1], &regs[2], &regs[2]);

            regs[0] = multilinears[2][thread_index];
            mul_prime_by_ext_field(&regs[0], to_monty(33), &regs[0]);
            ext_field_mul(&regs[0], &cached_batching_scalars[2], &regs[1]);

            ext_field_add(&regs[1], &regs[2], &regs[2]);

            ext_field_mul(&regs[2], &multilinears[3][thread_index], &regs[3]);
        }

        sums[thread_index] = regs[N_REGISTERS - 1];
    }

    grid.sync();

    // 3) Compute the final sum

    for (int step = 0; step < n_vars; step++)
    {
        const int half_len = 1 << (n_vars - step - 1);
        const int n_reps = (half_len + n_total_threads - 1) / n_total_threads;
        for (int rep = 0; rep < n_reps; rep++)
        {
            const int thread_index = threadIdx.x + (blockIdx.x + rep * gridDim.x) * N_THREADS_PER_BLOCK;
            if (thread_index < half_len)
            {
                ext_field_add(&sums[thread_index], &sums[thread_index + half_len], &sums[thread_index]);
            }
        }
        grid.sync();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        res[0] = sums[0];
    }
}