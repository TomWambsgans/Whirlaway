#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "ff_wrapper.cu"
#include "utils.cu"

// we need: MAX_NTT_SIZE_AT_BLOCK_LEVEL * (EXT_DEGREE + 1) * 4 bytes <= shared memory
// TODO avoid hardcoding
#if !defined(MAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL)
#define MAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL 1
#endif

// __device__ int whir_flip_index(int idx, uint32_t log_len, uint32_t log_expansion_factor, uint32_t log_chunk_size)
// {
//     // 0) Interleave everything with zeros to increase the size of by 1 << log_expansion_factor_u32
//     // 1) Bit reverse (everything)
//     // 2) transpose (log_expansion_factor, log_len)

//     int a = idx * (1 << log_expansion_factor);
//     int b = __brev(a) >> (32 - log_len);
//     int c = index_transpose(b, log_len - log_chunk_size, log_len);
//     return c;
// }

__device__ int whir_flip_index_bijection(int idx, uint32_t log_len, uint32_t log_expansion_factor, uint32_t log_chunk_size)
{
    // inverse of whir_flip

    int a = index_transpose(idx, log_len - log_chunk_size, log_chunk_size);
    int b = __brev(a) >> (32 - log_len);

    if (b % (1 << log_expansion_factor) != 0)
    {
        return -1;
    }
    else
    {
        return b / (1 << log_expansion_factor);
    }
}

__device__ int block_ntt_index(int tid, uint32_t log_len, uint32_t inner_log_len, uint32_t log_chunck_size, bool on_rows,  uint32_t log_whir_expansion_factor)
{
    int res = on_rows ? tid : index_transpose(tid, inner_log_len - log_chunck_size, log_chunck_size);

    if (log_whir_expansion_factor != 0)
    {
        res = whir_flip_index_bijection(res, log_len, log_whir_expansion_factor, inner_log_len);
    }

    return res;
}

__device__ Field_A final_twiddle(int idx, uint32_t full_log_len, uint32_t inner_log_len, uint32_t log_chunck_size, Field_A **twiddles)
{

    int inner_matrix_index = idx / (1 << inner_log_len);

    int inner_idx = idx % (1 << inner_log_len);
    int i = inner_idx % (1 << log_chunck_size);
    int j = inner_idx / (1 << log_chunck_size);
    int ij = i * j;

    Field_A twiddle;
    if (ij < 1 << (inner_log_len - 1))
    {
        twiddle = twiddles[inner_log_len - 1][ij];
    }
    else
    {
        twiddle = twiddles[inner_log_len - 1][ij - (1 << (inner_log_len - 1))];
        SUB_AA({0}, twiddle, twiddle);
    }

    return twiddle;
}

extern "C" __global__ void ntt_at_block_level(Field_B *input, Field_B *output, uint32_t log_len, uint32_t inner_log_len, uint32_t log_chunck_size, bool on_rows,
                                              bool final_twiddles, Field_A **twiddles, uint32_t log_whir_expansion_factor, uint32_t n_final_transpositions,
                                              uint32_t tr_row_0, uint32_t tr_col_0, uint32_t tr_row_1, uint32_t tr_col_1, uint32_t tr_row_2, uint32_t tr_col_2)
{

    int threadId = threadIdx.x;
    int n_threads = blockDim.x;

    const int log_n_threads_per_block = __ffs(blockDim.x) - 1;

    const uint32_t n_repetitions = (1 << log_len) / (blockDim.x * gridDim.x * 2);

    __shared__ Field_B cached_buff[1 << MAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL];
    __shared__ Field_A cached_twiddles[1 << (MAX_NTT_LOG_SIZE_AT_BLOCK_LEVEL - 1)]; // TODO use constant memory instead

    if (threadId < (1 << log_chunck_size))
    {
        cached_twiddles[threadId] = twiddles[log_chunck_size - 1][threadId];
    }

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int block = blockIdx.x + gridDim.x * rep;

        int index_x = block_ntt_index(threadId + n_threads * 2 * block, log_len, inner_log_len, log_chunck_size, on_rows, log_whir_expansion_factor);
        int index_y = block_ntt_index(threadId + n_threads * (2 * block + 1), log_len, inner_log_len, log_chunck_size, on_rows, log_whir_expansion_factor);

        if (index_x == -1)
        {
            cached_buff[threadId] = {0};
        }
        else
        {
            cached_buff[threadId] = input[index_x];
        }

        if (index_y == -1)
        {
            cached_buff[threadId + n_threads] = {0};
        }
        else
        {
            cached_buff[threadId + n_threads] = input[index_y];
        }

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
            Field_A twiddle = cached_twiddles[i * (1 << (log_chunck_size - step - 1))];

            // cached_buff[even_index] = even + first_twiddle * odd
            Field_B temp;
            MUL_BA(odd, twiddle, temp);
            ADD_BB(even, temp, cached_buff[even_dest]);

            // cached_buff[odd_index] = even + second_twiddle * odd
            MUL_BA(odd, twiddle, temp);
            SUB_BB(even, temp, cached_buff[odd_dest]);

            __syncthreads();
        }

        Field_B x = cached_buff[threadId];
        Field_B y = cached_buff[threadId + n_threads];

        index_x = block_ntt_index(threadId + blockDim.x * 2 * block, log_len, inner_log_len, log_chunck_size, on_rows, 0);
        index_y = block_ntt_index(threadId + blockDim.x * (2 * block + 1), log_len, inner_log_len, log_chunck_size, on_rows, 0);

        if (final_twiddles)
        {
            Field_B temp;

            Field_A twiddle_x = final_twiddle(index_x, log_len, inner_log_len, inner_log_len - log_chunck_size, twiddles);
            MUL_BA(x, twiddle_x, temp);
            x = temp;
            Field_A twiddle_y = final_twiddle(index_y, log_len, inner_log_len, inner_log_len - log_chunck_size, twiddles);
            MUL_BA(y, twiddle_y, temp);
            y = temp;
        }

        if (n_final_transpositions >= 1)
        {
            index_x = index_transpose(index_x, tr_row_0, tr_col_0);
            index_y = index_transpose(index_y, tr_row_0, tr_col_0);
        }
        if (n_final_transpositions >= 2)
        {
            index_x = index_transpose(index_x, tr_row_1, tr_col_1);
            index_y = index_transpose(index_y, tr_row_1, tr_col_1);
        }
        if (n_final_transpositions >= 3)
        {
            index_x = index_transpose(index_x, tr_row_2, tr_col_2);
            index_y = index_transpose(index_y, tr_row_2, tr_col_2);
        }

        // copy back to global memory
        output[index_x] = x;
        output[index_y] = y;

        __syncthreads();
    }
}
