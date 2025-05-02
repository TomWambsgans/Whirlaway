#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "../ff_wrapper.cu"

// we need: MAX_NTT_SIZE_AT_BLOCK_LEVEL * (EXT_DEGREE + 1) * 4 bytes <= shared memory
// TODO avoid hardcoding
#if !defined(MAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL)
#define MAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL 0
#endif

extern "C" __global__ void ntt_at_block_level(Field_B *buff, uint32_t log_len, uint32_t log_chunck_size, Field_A *twiddles)
{
    // the initial steps of the NTT are done at block level, to make use of shared memory
    // *buff constains N_THREADS_PER_BLOCK * 2 Field_B elements
    // *twiddles: w^0, w^1, w^2, w^3, ..., w^(N_THREADS_PER_BLOCK * 2 - 1) where w is a "2 * N_THREADS_PER_BLOCK" root of unity
    // block is not necessarily blockIdx.x
    // we should have log_chunck_size <= LOG_N_THREADS_PER_BLOCK + 1

    if (log_chunck_size == 0) {
        return;
    }

    int threadId = threadIdx.x;
    int n_threads = blockDim.x;

    const int log_n_threads_per_block = __ffs(blockDim.x) - 1;

    const uint32_t n_repetitions = (1 << log_len) / (blockDim.x * gridDim.x * 2);

    __shared__ Field_B cached_buff[1 << MAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL];
    __shared__ Field_A cached_twiddles[1 << MAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL]; // TODO use constant memory instead

    cached_twiddles[threadId] = twiddles[n_threads * 2 - 1 + threadId];
    cached_twiddles[threadId + n_threads] = twiddles[n_threads * 3 - 1 + threadId];

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int block = blockIdx.x + gridDim.x * rep;

        cached_buff[threadId] = buff[threadId + n_threads * 2 * block];
        cached_buff[threadId + n_threads] = buff[threadId + n_threads * (2 * block + 1)];

        __syncthreads();

        // step 0

        Field_B even = cached_buff[threadId * 2];
        Field_B odd = cached_buff[threadId * 2 + 1];

        ADD_BB(even, odd, cached_buff[threadId * 2]);
        SUB_BB(even, odd, cached_buff[threadId * 2 + 1]);

        __syncthreads();

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

        __syncthreads();
    }
}

extern "C" __global__ void ntt_step(Field_B *buff, uint32_t log_len, uint32_t step, Field_A *twiddles)
{
    // twiddles = 1
    // followed by w^0, w^1 where w is a 2-root of unity
    // followed by w^0, w^1, w^2, w^3 where w is a 4-root of unity
    // followed by w^0, w^1, w^2, w^3, w^4, w^5, w^6, w^7 where w is a 8-root of unity
    // ...
    // buff has size 1 << log_len

    int total_threads = blockDim.x * gridDim.x;
    const uint32_t n_repetitions = ((1 << (log_len - 1)) + total_threads - 1) / total_threads;

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
