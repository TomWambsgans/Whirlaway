#include <stdio.h>
#include <stdint.h>

#include "../ff_wrapper.cu"

// we need: MAX_NTT_SIZE_AT_BLOCK_LEVEL * (EXT_DEGREE + 1) * 4 bytes <= shared memory
#if !defined(MAX_TRANSPOSE_LOG_SIZE_AT_BLOCK_LEVEL)
#define MAX_TRANSPOSE_LOG_SIZE_AT_BLOCK_LEVEL 0
#endif

__device__ int compute_row_major_index(int blockIndex, int n_blocks_cols, int inner_cols, int cols)
{
    int above = (blockIndex / n_blocks_cols) * n_blocks_cols * blockDim.x;
    int remaining = (blockIndex % n_blocks_cols) * inner_cols + (threadIdx.x / inner_cols) * cols + (threadIdx.x % inner_cols);
    return above + remaining;
}

extern "C" __global__ void transpose(Field_A *input, Field_A *result, uint32_t log_rows, uint32_t log_cols)
{
    // block size should be a power of 2, and should be <= 2^MAX_TRANSPOSE_LOG_SIZE_AT_BLOCK_LEVEL
    int n_total_n_threads = blockDim.x * gridDim.x;
    int n_ops = 1 << (log_cols + log_rows);
    int n_repetitions = (n_ops + n_total_n_threads - 1) / n_total_n_threads;
    int rows = 1 << log_rows;
    int cols = 1 << log_cols;

    int inner_rows = min(rows, max(blockDim.x / cols, 1 << ((31 - __builtin_clz(blockDim.x)) / 2)));
    int inner_cols = min(cols, blockDim.x / inner_rows);
    int n_blocks_rows = rows / inner_rows;
    int n_blocks_cols = cols / inner_cols;

    __shared__ Field_A cached_buff[1 << MAX_TRANSPOSE_LOG_SIZE_AT_BLOCK_LEVEL];

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int blockIndex = blockIdx.x + gridDim.x * rep;
        int idx = threadIdx.x + blockIndex * blockDim.x;
        if (idx >= n_ops)
        {
            return;
        }

        if (log_rows == 0 || log_cols == 0)
        {
            result[idx] = input[idx];
            continue;
        }

        int src_index = compute_row_major_index(blockIndex, n_blocks_cols, inner_cols, cols);
        int transposedBlockIndex = (blockIndex % n_blocks_cols) * n_blocks_rows + (blockIndex / n_blocks_cols);
        int dest_index = compute_row_major_index(transposedBlockIndex, n_blocks_rows, inner_rows, rows);
        int transposed_inner_index = (threadIdx.x % inner_rows) * inner_cols + (threadIdx.x / inner_rows);

        cached_buff[threadIdx.x] = input[src_index];
        __syncthreads();
        result[dest_index] = cached_buff[transposed_inner_index];
        __syncthreads();
    }
}
