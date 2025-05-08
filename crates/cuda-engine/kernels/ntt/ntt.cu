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
    // *twiddles: w^0, w^1, w^2, w^3, ..., w^(log_chunck_size * 2 - 1) where w is a "2 * log_chunck_size" root of unity

    if (log_chunck_size == 0)
    {
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

        for (int step = 0; step < log_chunck_size; step++)
        {
            int threadIndex = threadIdx.x;
            int fft_index = threadIndex / (1 << (log_chunck_size - 1));
            threadIndex = threadIndex % (1 << (log_chunck_size - 1));

            int inner_fft_size = 1 << step;
            int left_shift = fft_index * (1 << log_chunck_size) + (threadIndex / inner_fft_size);
            int interspace = 1 << (log_chunck_size - step - 1);
            int even_src = left_shift + (threadIndex % inner_fft_size) * 2 * interspace;
            int odd_src = left_shift + ((threadIndex % inner_fft_size) * 2 + 1) * interspace;
            int even_dest = left_shift + (threadIndex % inner_fft_size) * interspace;
            int odd_dest = left_shift + ((threadIndex % inner_fft_size) + inner_fft_size) * interspace;

            Field_B even = cached_buff[even_src];
            Field_B odd = cached_buff[odd_src];

            __syncthreads();

            int i = threadId % inner_fft_size;
            // w^i where w is a "2 * packet_size" root of unity
            Field_A first_twiddle = cached_twiddles[i * blockDim.x / inner_fft_size];
            // w^(i + packet_size) where w is a "2 * packet_size" root of unity
            Field_A second_twiddle = cached_twiddles[(i + inner_fft_size) * blockDim.x / inner_fft_size];

            // cached_buff[even_index] = even + first_twiddle * odd
            Field_B temp;
            MUL_BA(odd, first_twiddle, temp);
            ADD_BB(even, temp, cached_buff[even_dest]);

            // cached_buff[odd_index] = even + second_twiddle * odd
            MUL_BA(odd, second_twiddle, temp);
            ADD_BB(even, temp, cached_buff[odd_dest]);

            __syncthreads();
        }

        // copy back to global memory
        buff[threadId + blockDim.x * 2 * block] = cached_buff[threadId];
        buff[threadId + blockDim.x * (2 * block + 1)] = cached_buff[threadId + blockDim.x];

        __syncthreads();
    }
}

extern "C" __global__ void apply_twiddles(Field_B *buff, uint32_t full_log_len, uint32_t inner_log_len, uint32_t log_chunck_size, Field_A *twiddles)
{
    int total_threads = blockDim.x * gridDim.x;
    const uint32_t n_repetitions = ((1 << full_log_len) + total_threads - 1) / total_threads;

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        int inner_matrix_index = threadIndex / (1 << inner_log_len);

        int inner_idx = threadIndex % (1 << inner_log_len);
        int i = inner_idx % (1 << log_chunck_size);
        int j = inner_idx / (1 << log_chunck_size);
        Field_A twiddle = twiddles[(1 << inner_log_len) - 1 + i * j];

        Field_B src = buff[threadIndex];
        Field_B result;
        MUL_AB(twiddle, src, result);
        buff[threadIndex] = result;
    }
}