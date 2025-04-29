#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <algorithm>

#include "../ff_wrapper.cu"

// we need: N_THREADS_PER_BLOCK * 2 * (EXT_DEGREE + 1) * 4 bytes <= shared memory
// TODO avoid hardcoding
#define MAX_LOG_N_THREADS_PER_BLOCK 8
#define MAX_N_THREADS_PER_BLOCK (1 << MAX_LOG_N_THREADS_PER_BLOCK)

__device__ void reverse_bit_order(Field_B *data, int block, int bits)
{
    int idx = (block * blockDim.x + threadIdx.x) % (1 << bits);
    int rev_idx = __brev(idx) >> (32 - bits);

    // Only process when idx < rev_idx to avoid swapping twice
    if (idx < rev_idx)
    {
        Field_B temp = data[idx];
        data[idx] = data[rev_idx];
        data[rev_idx] = temp;
    }
}

__device__ void batch_reverse_bit_order(Field_B *data, int block, int bits)
{
    int idx = block * blockDim.x + threadIdx.x;
    int len = (1 << bits);
    reverse_bit_order(&data[(idx / len) * len], block, bits);
}

__device__ void ntt_at_block_level(Field_B *buff, const int block, const int log_chunck_size, const Field_A *twiddles)
{
    // the initial steps of the NTT are done at block level, to make use of shared memory
    // *buff constains N_THREADS_PER_BLOCK * 2 Field_B elements
    // *twiddles: w^0, w^1, w^2, w^3, ..., w^(N_THREADS_PER_BLOCK * 2 - 1) where w is a "2 * N_THREADS_PER_BLOCK" root of unity
    // block is not necessarily blockIdx.x
    // we should have log_chunck_size <= LOG_N_THREADS_PER_BLOCK + 1

    const int threadId = threadIdx.x;
    const int n_threads = blockDim.x;

    __shared__ Field_B cached_buff[MAX_N_THREADS_PER_BLOCK * 2];

    cached_buff[threadId] = buff[threadId + n_threads * 2 * block];
    cached_buff[threadId + n_threads] = buff[threadId + n_threads * (2 * block + 1)];

    __shared__ Field_A cached_twiddles[MAX_N_THREADS_PER_BLOCK * 2]; // TODO use constant memory instead

    cached_twiddles[threadId] = twiddles[threadId];
    cached_twiddles[threadId + n_threads] = twiddles[threadId + n_threads];

    __syncthreads();

    // step 0

    Field_B even = cached_buff[threadId * 2];
    Field_B odd = cached_buff[threadId * 2 + 1];

    ADD_BB(even, odd, cached_buff[threadId * 2]);
    SUB_BB(even, odd, cached_buff[threadId * 2 + 1]);

    for (int step = 1; step < log_chunck_size; step++)
    {
        int packet_size = 1 << step;
        int even_index = threadId + (threadId / packet_size) * packet_size;
        int odd_index = even_index + packet_size;

        Field_B even = cached_buff[even_index];
        Field_B odd = cached_buff[odd_index];

        int i = threadId % packet_size;
        // w^i where w is a "2 * packet_size" root of unity
        Field_A first_twiddle = cached_twiddles[i * blockDim.x / packet_size];
        // w^(i + packet_size) where w is a "2 * packet_size" root of unity
        Field_A second_twiddle = cached_twiddles[(i + packet_size) * blockDim.x / packet_size];

        // cached_buff[even_index] = even + first_twiddle * odd
        MUL_BA(odd, first_twiddle, cached_buff[even_index]);
        ADD_BB(even, cached_buff[even_index], cached_buff[even_index]);

        // cached_buff[odd_index] = even + second_twiddle * odd
        MUL_BA(odd, second_twiddle, cached_buff[odd_index]);
        ADD_BB(even, cached_buff[odd_index], cached_buff[odd_index]);

        __syncthreads();
    }

    // copy back to global memory
    buff[threadId + blockDim.x * 2 * block] = cached_buff[threadId];
    buff[threadId + blockDim.x * (2 * block + 1)] = cached_buff[threadId + blockDim.x];
}

extern "C" __global__ void ntt(Field_B *buff, uint32_t log_len, uint32_t log_chunk_size, const Field_A *twiddles)
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

            Field_B even = buff[even_index];
            Field_B odd = buff[odd_index];

            int i = threadIndex % packet_size;
            // w^i where w is a "2 * packet_size" root of unity
            Field_A first_twiddle = twiddles[packet_size * 2 - 1 + i];
            // w^(i + packet_size) where w is a "2 * packet_size" root of unity
            Field_A second_twiddle = twiddles[packet_size * 2 - 1 + i + packet_size];

            // result[even_index] = even + first_twiddle * odd
            Field_B temp;
            MUL_BA(odd, first_twiddle, temp);
            ADD_BB(even, temp, buff[even_index]);

            // result[odd_index] = even + second_twiddle * odd
            MUL_BA(odd, second_twiddle, temp);
            ADD_BB(even, temp, buff[odd_index]);
        }
    }

    grid.sync();
}
