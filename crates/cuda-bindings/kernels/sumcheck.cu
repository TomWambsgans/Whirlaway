#include <stdio.h>
#include <stdint.h>

#include "finite_field.cu"

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
