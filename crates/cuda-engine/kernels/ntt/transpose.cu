#include <stdio.h>
#include <stdint.h>

#include "../ff_wrapper.cu"

extern "C" __global__ void transpose(Field_A *input, Field_A *result, const uint32_t log_rows, const uint32_t log_cols)
{
    const int n_total_n_threads = blockDim.x * gridDim.x;
    const int n_ops = 1 << (log_cols + log_rows);
    const int n_repetitions = (n_ops + n_total_n_threads - 1) / n_total_n_threads;
    const int rows = 1 << log_rows;
    const int cols = 1 << log_cols;

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        if (threadIndex >= n_ops)
        {
            return;
        }
        result[threadIndex] = input[(threadIndex % rows) * cols + (threadIndex / rows)];
    }
}
