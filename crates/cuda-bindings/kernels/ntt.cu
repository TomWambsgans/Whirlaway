#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "finite_field.cu"

// we need: N_THREADS_PER_BLOCK * 2 * (EXT_DEGREE + 1) * 4 bytes <= shared memory
// TODO avoid hardcoding
#define LOG_N_THREADS_PER_BLOCK 8
#define N_THREADS_PER_BLOCK (1 << LOG_N_THREADS_PER_BLOCK)

__device__ void ntt_at_block_level(ExtField *buff, const int block, const uint32_t *twiddles)
{
    // the initial steps of the NTT are done at block level, to make use of shared memory
    // *buff constains N_THREADS_PER_BLOCK * 2 ExtField elements
    // *twiddles: w^0, w^1, w^2, w^3, ..., w^(N_THREADS_PER_BLOCK * 2 - 1) where w is a "2 * N_THREADS_PER_BLOCK" root of unity
    // block is not necessarily blockIdx.x

    const int threadId = threadIdx.x;

    __shared__ ExtField cached_buff[N_THREADS_PER_BLOCK * 2];

    cached_buff[threadId] = buff[threadId + N_THREADS_PER_BLOCK * 2 * block];
    cached_buff[threadId + N_THREADS_PER_BLOCK] = buff[threadId + N_THREADS_PER_BLOCK * (2 * block + 1)];

    __shared__ uint32_t cached_twiddles[N_THREADS_PER_BLOCK * 2];

    cached_twiddles[threadId] = twiddles[threadId];
    cached_twiddles[threadId + N_THREADS_PER_BLOCK] = twiddles[threadId + N_THREADS_PER_BLOCK];

    __syncthreads();

    // step 0

    ExtField even = cached_buff[threadId * 2];
    ExtField odd = cached_buff[threadId * 2 + 1];

    ext_field_add(&even, &odd, &cached_buff[threadId * 2]);
    ext_field_sub(&even, &odd, &cached_buff[threadId * 2 + 1]);

    for (int step = 1; step <= LOG_N_THREADS_PER_BLOCK; step++)
    {
        int packet_size = 1 << step;
        int even_index = threadId + (threadId / packet_size) * packet_size;
        int odd_index = even_index + packet_size;

        ExtField even = cached_buff[even_index];
        ExtField odd = cached_buff[odd_index];

        int i = threadId % packet_size;
        // w^i where w is a "2 * packet_size" root of unity
        uint32_t first_twiddle = cached_twiddles[i * N_THREADS_PER_BLOCK / packet_size];
        // w^(i + packet_size) where w is a "2 * packet_size" root of unity
        uint32_t second_twiddle = cached_twiddles[(i + packet_size) * N_THREADS_PER_BLOCK / packet_size];

        // cached_buff[even_index] = even + first_twiddle * odd
        mul_prime_and_ext_field(&odd, first_twiddle, &cached_buff[even_index]);
        ext_field_add(&even, &cached_buff[even_index], &cached_buff[even_index]);

        // cached_buff[odd_index] = even + second_twiddle * odd
        mul_prime_and_ext_field(&odd, second_twiddle, &cached_buff[odd_index]);
        ext_field_add(&even, &cached_buff[odd_index], &cached_buff[odd_index]);

        __syncthreads();
    }

    // copy back to global memory
    buff[threadId + N_THREADS_PER_BLOCK * 2 * block] = cached_buff[threadId];
    buff[threadId + N_THREADS_PER_BLOCK * (2 * block + 1)] = cached_buff[threadId + N_THREADS_PER_BLOCK];
}

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

// TODO use only one buffer, but I don't know how to fill it "partially" with cudarc, since the crate asserts dest size = src size when copying data
extern "C" __global__ void ntt(ExtField *input, ExtField *buff, ExtField *result, const uint32_t log_len, const uint32_t log_extension_factor, const uint32_t *twiddles)
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
    const uint32_t n_repetitions = (1 << (log_len + log_extension_factor)) / (N_THREADS_PER_BLOCK * gridDim.x * 2);

    // int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    const int len = 1 << log_len;
    const int expansion_factor = 1 << log_extension_factor;

    // 1) Expand input several times to fill result, multiplying by the appropriate twiddle factors
    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * N_THREADS_PER_BLOCK;

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

    // 2) Bit reverse order

    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        batch_reverse_bit_order(buff, blockIdx.x + gridDim.x * rep, log_len);
    }

    grid.sync();

    // 3) Do the NTT at block level

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        ntt_at_block_level(buff, blockIdx.x + gridDim.x * rep, &twiddles[N_THREADS_PER_BLOCK * 2 - 1]);
    }

    // 4) Finish the NTT

    for (int step = LOG_N_THREADS_PER_BLOCK + 1; step < log_len; step++)
    {
        grid.sync();

        // we group together pairs which each side contains 1 << step elements

        for (int rep = 0; rep < n_repetitions; rep++)
        {
            int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * N_THREADS_PER_BLOCK;

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

    // 5) Transpose buff to result

    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * N_THREADS_PER_BLOCK;

        result[threadIndex] = buff[(threadIndex % expansion_factor) * len + (threadIndex / expansion_factor)];
    }
}
