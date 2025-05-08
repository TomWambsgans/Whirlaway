#include <stdio.h>
#include <stdint.h>

#include "../ff_wrapper.cu"

extern "C" __global__ void transpose(Field_A *input, Field_A *result, uint32_t log_len, uint32_t log_rows, uint32_t log_cols)
{
    // in the simple transpose (only 1 matrix), we have: log_len = log_rows + log_cols
    // we should have: log_rows + log_cols <= log_len
    int n_total_n_threads = blockDim.x * gridDim.x;
    int n_ops = 1 << log_len;
    int n_repetitions = (n_ops + n_total_n_threads - 1) / n_total_n_threads;
    int rows = 1 << log_rows;
    int cols = 1 << log_cols;
    int matrix_size = rows * cols;

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex >= n_ops)
        {
            return;
        }
        int matrix_index = threadIndex / matrix_size;
        result[threadIndex] = input[matrix_index * matrix_size + (threadIndex % rows) * cols + ((threadIndex % matrix_size) / rows)];
    }
}
