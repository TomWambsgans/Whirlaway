#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "../ff_wrapper.cu"

__device__ void reverse_bit_order(Field_A *data, int block, int bits)
{
    int idx = (block * blockDim.x + threadIdx.x) % (1 << bits);
    int rev_idx = __brev(idx) >> (32 - bits);

    // Only process when idx < rev_idx to avoid swapping twice
    if (idx < rev_idx)
    {
        Field_A temp = data[idx];
        data[idx] = data[rev_idx];
        data[rev_idx] = temp;
    }
}

__device__ void batch_reverse_bit_order(Field_A *data, int block, int bits)
{
    int idx = block * blockDim.x + threadIdx.x;
    int len = (1 << bits);
    reverse_bit_order(&data[(idx / len) * len], block, bits);
}

extern "C" __global__ void reverse_bit_order_global(Field_A *buff, uint32_t log_len, uint32_t log_chunk_size)
{
    int total_threads = blockDim.x * gridDim.x;
    const uint32_t n_repetitions = ((1 << log_len) + total_threads - 1) / total_threads;

    // 1) Bit reverse order

    for (int rep = 0; rep < n_repetitions; rep++)
    {
        batch_reverse_bit_order(buff, blockIdx.x + gridDim.x * rep, log_chunk_size);
    }
}
