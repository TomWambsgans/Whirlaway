#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <algorithm>

#include "finite_field.cu"

// we need: N_THREADS_PER_BLOCK * 2 * (EXT_DEGREE + 1) * 4 bytes <= shared memory
// TODO avoid hardcoding
#define MAX_LOG_N_THREADS_PER_BLOCK 8
#define MAX_N_THREADS_PER_BLOCK (1 << MAX_LOG_N_THREADS_PER_BLOCK)

__device__ void reverse_bit_order(ExtField *data, int block, int bits)
{
    int idx = (block * blockDim.x + threadIdx.x) % (1 << bits);
    int rev_idx = __brev(idx) >> (32 - bits);

    // Only process when idx < rev_idx to avoid swapping twice
    if (idx < rev_idx)
    {
        ExtField temp = data[idx];
        data[idx] = data[rev_idx];
        data[rev_idx] = temp;
    }
}

__device__ void batch_reverse_bit_order(ExtField *data, int block, int bits)
{
    int idx = block * blockDim.x + threadIdx.x;
    int len = (1 << bits);
    reverse_bit_order(&data[(idx / len) * len], block, bits);
}

__device__ void ntt_at_block_level(ExtField *buff, const int block, const int log_chunck_size, const uint32_t *twiddles)
{
    // the initial steps of the NTT are done at block level, to make use of shared memory
    // *buff constains N_THREADS_PER_BLOCK * 2 ExtField elements
    // *twiddles: w^0, w^1, w^2, w^3, ..., w^(N_THREADS_PER_BLOCK * 2 - 1) where w is a "2 * N_THREADS_PER_BLOCK" root of unity
    // block is not necessarily blockIdx.x
    // we should have log_chunck_size <= LOG_N_THREADS_PER_BLOCK + 1

    const int threadId = threadIdx.x;
    const int n_threads = blockDim.x;

    __shared__ ExtField cached_buff[MAX_N_THREADS_PER_BLOCK * 2];

    cached_buff[threadId] = buff[threadId + n_threads * 2 * block];
    cached_buff[threadId + n_threads] = buff[threadId + n_threads * (2 * block + 1)];

    __shared__ uint32_t cached_twiddles[MAX_N_THREADS_PER_BLOCK * 2];

    cached_twiddles[threadId] = twiddles[threadId];
    cached_twiddles[threadId + n_threads] = twiddles[threadId + n_threads];

    __syncthreads();

    // step 0

    ExtField even = cached_buff[threadId * 2];
    ExtField odd = cached_buff[threadId * 2 + 1];

    ext_field_add(&even, &odd, &cached_buff[threadId * 2]);
    ext_field_sub(&even, &odd, &cached_buff[threadId * 2 + 1]);

    for (int step = 1; step < log_chunck_size; step++)
    {
        int packet_size = 1 << step;
        int even_index = threadId + (threadId / packet_size) * packet_size;
        int odd_index = even_index + packet_size;

        ExtField even = cached_buff[even_index];
        ExtField odd = cached_buff[odd_index];

        int i = threadId % packet_size;
        // w^i where w is a "2 * packet_size" root of unity
        uint32_t first_twiddle = cached_twiddles[i * blockDim.x / packet_size];
        // w^(i + packet_size) where w is a "2 * packet_size" root of unity
        uint32_t second_twiddle = cached_twiddles[(i + packet_size) * blockDim.x / packet_size];

        // cached_buff[even_index] = even + first_twiddle * odd
        mul_prime_and_ext_field(&odd, first_twiddle, &cached_buff[even_index]);
        ext_field_add(&even, &cached_buff[even_index], &cached_buff[even_index]);

        // cached_buff[odd_index] = even + second_twiddle * odd
        mul_prime_and_ext_field(&odd, second_twiddle, &cached_buff[odd_index]);
        ext_field_add(&even, &cached_buff[odd_index], &cached_buff[odd_index]);

        __syncthreads();
    }

    // copy back to global memory
    buff[threadId + blockDim.x * 2 * block] = cached_buff[threadId];
    buff[threadId + blockDim.x * (2 * block + 1)] = cached_buff[threadId + blockDim.x];
}

__device__ void ntt(ExtField *buff, const int log_len, const int log_chunk_size, const uint32_t *twiddles)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const uint32_t n_repetitions = (1 << log_len) / (blockDim.x * gridDim.x * 2);

    // 1) Bit reverse order

    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        batch_reverse_bit_order(buff, blockIdx.x + gridDim.x * rep, log_chunk_size);
    }
    grid.sync();

    // 2) Do the NTT at block level

    const int log_n_threads_per_block = __ffs(blockDim.x) - 1;

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        ntt_at_block_level(buff, blockIdx.x + gridDim.x * rep, min(log_n_threads_per_block + 1, log_chunk_size), &twiddles[blockDim.x * 2 - 1]);
    }

    // 3) Finish the NTT

    for (int step = log_n_threads_per_block + 1; step < log_chunk_size; step++)
    {
        grid.sync();

        // we group together pairs which each side contains 1 << step elements

        for (int rep = 0; rep < n_repetitions; rep++)
        {
            int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;

            int packet_size = 1 << step;
            int even_index = threadIndex + (threadIndex / packet_size) * packet_size;
            int odd_index = even_index + packet_size;

            ExtField even = buff[even_index];
            ExtField odd = buff[odd_index];

            int i = threadIndex % packet_size;
            // w^i where w is a "2 * packet_size" root of unity
            uint32_t first_twiddle = twiddles[packet_size * 2 - 1 + i];
            // w^(i + packet_size) where w is a "2 * packet_size" root of unity
            uint32_t second_twiddle = twiddles[packet_size * 2 - 1 + i + packet_size];

            // result[even_index] = even + first_twiddle * odd
            mul_prime_and_ext_field(&odd, first_twiddle, &buff[even_index]);
            ext_field_add(&even, &buff[even_index], &buff[even_index]);

            // result[odd_index] = even + second_twiddle * odd
            mul_prime_and_ext_field(&odd, second_twiddle, &buff[odd_index]);
            ext_field_add(&even, &buff[odd_index], &buff[odd_index]);
        }
    }

    grid.sync();
}

extern "C" __global__ void expanded_ntt(ExtField *input, ExtField *buff, ExtField *result, const uint32_t log_len, const uint32_t log_extension_factor, const uint32_t *twiddles)
{
    // twiddles = 1
    // followed by w^0, w^1 where w is a 2-root of unity
    // followed by w^0, w^1, w^2, w^3 where w is a 4-root of unity
    // followed by w^0, w^1, w^2, w^3, w^4, w^5, w^6, w^7 where w is a 8-root of unity
    // ...
    // input has size 1 << log_len (the coefs of the polynomial we want to NTT)
    // buff and result both have size 1 << (log_len + log_extension_factor)

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    // we should have N_THREADS_PER_BLOCK * NUM_BLOCKS * n_repetitions * 2 = 1 << (log_len + log_extension_factor)
    // WARNING: We assume the number of blocks is a power of 2
    const uint32_t n_repetitions = (1 << (log_len + log_extension_factor)) / (blockDim.x * gridDim.x * 2);

    // int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    const int len = 1 << log_len;
    const int expansion_factor = 1 << log_extension_factor;

    // 1) Expand input several times to fill result, multiplying by the appropriate twiddle factors
    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;

        if (threadIndex < len)
        {
            buff[threadIndex] = input[threadIndex];
        }
        else
        {
            uint32_t twidle = twiddles[(1 << (log_len + log_extension_factor)) - 1 + (threadIndex % len) * (threadIndex / len)];
            mul_prime_and_ext_field(&input[threadIndex % len], twidle, &buff[threadIndex]);
        }
    }

    grid.sync();

    // 2) Core ntt

    ntt(buff, log_len + log_extension_factor, log_len, twiddles);

    // 3) Transpose buff to result

    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;

        result[threadIndex] = buff[(threadIndex % expansion_factor) * len + (threadIndex / expansion_factor)];
    }
}

extern "C" __global__ void restructure_evaluations(ExtField *input, ExtField *result, const uint32_t log_len, const uint32_t folding_factor, const uint32_t *twiddles, const uint32_t *correction_twiddles)
{
    // size_inv = inv(folding_size) (in the prime field)
    // w_inv_i = inv(unit root selected in the ntt twiddles for the domain of size 2^(folding_factor + i))
    // correction_twiddles_i = size_inv, size_inv, ..., size_inv (folding_size times)
    //                         size_inv, size_inv.w_inv_i, size_inv.w_inv_i^2, ..., size_inv.w_inv_i^(folding_size - 1)]
    //                         size_inv, size_inv.w_inv_i^2, (size_inv.w_inv_i^2)^2, ..., (size_inv.w_inv_i^2)^(folding_size - 1)]
    //                         size_inv, size_inv.w_inv_i^3, (size_inv.w_inv_i^3)^2, ..., (size_inv.w_inv_i^3)^(folding_size - 1)]
    //                         ...
    //                         (2^i times)
    // correction_twiddles = correction_twiddles_0 | by correction_twiddles_1 | by correction_twiddles_2 ...

    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    // WARNING: We assume the number of blocks is a power of 2
    const int n_repetitions = (1 << log_len) / (blockDim.x * gridDim.x * 2);
    const int folding_size = 1 << folding_factor;
    const int len = 1 << log_len;

    // 1) Transpose
    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;

        result[threadIndex] = input[(threadIndex % folding_size) * (len / folding_size) + (threadIndex / folding_size)];
    }
    grid.sync();

    // 2) Inverse NTT

    // 2) a) For each chunck c of size folding_size in result, reverse c[1..]
    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        int chunk_index = threadIndex / (folding_size / 2);
        int chunk_offset = threadIndex % (folding_size / 2);
        if (chunk_offset != 0)
        {
            // swap result[i] and result[j]
            int i = chunk_index * folding_size + chunk_offset;
            int j = (chunk_index + 1) * folding_size - chunk_offset;
            ExtField temp = result[i];
            result[i] = result[j];
            result[j] = temp;
        }
    }
    grid.sync();

    // 2) b)
    ntt(result, log_len, folding_factor, twiddles);

    // 3) Apply coset and size correction

    const int i = log_len - folding_factor;
    const uint32_t *involded_correction_twiddles = &correction_twiddles[((1 << i) - 1) * folding_size];

    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        mul_prime_and_ext_field(&result[threadIndex], involded_correction_twiddles[threadIndex], &result[threadIndex]);
    }
}

extern "C" __global__ void ntt_global(ExtField *buff, const int log_len, const int log_chunk_size, const uint32_t *twiddles)
{
    ntt(buff, log_len, log_chunk_size, twiddles);
}
