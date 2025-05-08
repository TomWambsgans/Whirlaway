#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "../ff_wrapper.cu"

__device__ int index_transpose(int i, int log_width, int log_len)
{
    int col = i % (1 << log_width);
    int row = i / (1 << log_width);
    return col * (1 << (log_len - log_width)) + row;
}

extern "C" __global__ void reverse_bit_order_for_ntt(Field_A *input, Field_A *output, uint32_t log_len, uint32_t log_expansion_factor, uint32_t log_chunk_size, uint32_t inner_transposition_log_rows)
{
    // 0) Interleave everything with zeros to increase the size of by 1 << log_expansion_factor_u32
    // 1) Bit reverse (everything)
    // 2) transpose (log_expansion_factor, log_len)
    // 3) on each consecutive chunk of size Z^log_chunk_size, transpose (inner_transposition_log_rows, log_chunk_size - inner_transposition_log_rows)

    int total_threads = blockDim.x * gridDim.x;
    const uint32_t n_repetitions = ((1 << log_len) + total_threads - 1) / total_threads;
    int log_expanded_len = log_len + log_expansion_factor;

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        int i = threadIdx.x + (blockIdx.x + gridDim.x * rep) * blockDim.x;
        int log_width = log_expanded_len - log_chunk_size;
        int k = index_transpose(__brev(i * (1 << log_expansion_factor)) >> (32 - log_expanded_len), log_width, log_expanded_len);

        int l = (k >> log_chunk_size) << log_chunk_size;
        int m = k % (1 << log_chunk_size);


        int n = m % (1 << (log_chunk_size - inner_transposition_log_rows));
        int o = m / (1 << (log_chunk_size - inner_transposition_log_rows));
        output[l + (1 << inner_transposition_log_rows) * n + o] = input[i];
    }
}
