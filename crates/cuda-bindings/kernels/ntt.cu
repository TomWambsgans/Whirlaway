#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

// Koala bear field
#define MONTY_PRIME 0x7f000001
#define MONTY_BITS 32
#define MONTY_MASK ((1ULL << MONTY_BITS) - 1U)
#define MONTY_MU 0x81000001

#define EXT_DEGREE 8
#define W 100663290U // montgomery representation of 3, X^8 - 3 is irreducible

// we need: thread_per_block * 2 * (EXT_DEGREE + 1) * 4 bytes <= shared memory
// TODO avoid hardcoding
#define LOG_THREAD_PER_BLOCK 8
#define THREAD_PER_BLOCK (1 << LOG_THREAD_PER_BLOCK)

__device__ uint32_t monty_reduce(uint64_t x)
{
    uint64_t t = x * MONTY_MU & MONTY_MASK;
    uint64_t u = t * MONTY_PRIME;

    uint64_t x_sub_u = x - u;
    bool over = x < u;
    uint32_t x_sub_u_hi = (x_sub_u >> MONTY_BITS);
    uint32_t corr = over ? MONTY_PRIME : 0;
    return x_sub_u_hi + corr;
}

// CUDA kernel for field addition
__device__ void monty_field_add(const uint32_t a, const uint32_t b, uint32_t *result)
{
    uint32_t sum = a + b;
    if (sum >= MONTY_PRIME)
    {
        sum -= MONTY_PRIME;
    }
    *result = sum;
}

// CUDA kernel for field multiplication
__device__ void monty_field_mul(const uint32_t a, const uint32_t b, uint32_t *result)
{
    uint64_t long_prod = (uint64_t)a * (uint64_t)b;
    *result = monty_reduce(long_prod);
}

// CUDA kernel for field subtraction
__device__ void monty_field_sub(const uint32_t a, const uint32_t b, uint32_t *result)
{
    uint32_t diff = a - b;
    bool over = a < b; // Detect underflow
    uint32_t corr = over ? MONTY_PRIME : 0;
    *result = diff + corr;
}

// Extension field implementation
typedef struct
{
    uint32_t coeffs[EXT_DEGREE]; // Polynomial coefficients
} ExtField;

__device__ void print_ext_field(const ExtField *a)
{
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        printf("%u ", a->coeffs[i]);
    }
    printf("\n");
}

// Add two extension field elements
__device__ void ext_field_add(const ExtField *a, const ExtField *b, ExtField *result)
{
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        monty_field_add(a->coeffs[i], b->coeffs[i], &result->coeffs[i]);
    }
}

// Subtract two extension field elements
__device__ void ext_field_sub(const ExtField *a, const ExtField *b, ExtField *result)
{
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        monty_field_sub(a->coeffs[i], b->coeffs[i], &result->coeffs[i]);
    }
}

__device__ void mul_prime_by_ext_field(const ExtField *a, uint32_t b, ExtField *result)
{
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        monty_field_mul(a->coeffs[i], b, &result->coeffs[i]);
    }
}

// TODO Karatsuba ?
__device__ void ext_field_mul(const ExtField *a, const ExtField *b, ExtField *result)
{
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        result->coeffs[i] = 0;
    }

    // Schoolbook multiplication
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        for (int j = 0; j < EXT_DEGREE; j++)
        {
            uint32_t prod;
            monty_field_mul(a->coeffs[i], b->coeffs[j], &prod);

            if (i + j < EXT_DEGREE)
            {
                uint32_t temp;
                monty_field_add(result->coeffs[i + j], prod, &temp);
                result->coeffs[i + j] = temp;
            }
            else
            {
                uint32_t temp;
                monty_field_mul(prod, W, &temp);
                monty_field_add(result->coeffs[i + j - EXT_DEGREE], temp, &temp);
                result->coeffs[i + j - EXT_DEGREE] = temp;
            }
        }
    }
}

__device__ void ntt_at_block_level(ExtField *buff, const int block, const uint32_t *twiddles)
{
    // the initial steps of the NTT are done at block level, to make use of shared memory
    // *buff constains THREAD_PER_BLOCK * 2 ExtField elements
    // *twiddles: w^0, w^1, w^2, w^3, ..., w^(THREAD_PER_BLOCK * 2 - 1) where w is a "2 * THREAD_PER_BLOCK" root of unity
    // block is not necessarily blockIdx.x

    const int threadId = threadIdx.x;

    __shared__ ExtField cached_buff[THREAD_PER_BLOCK * 2];

    cached_buff[threadId] = buff[threadId + THREAD_PER_BLOCK * 2 * block];
    cached_buff[threadId + THREAD_PER_BLOCK] = buff[threadId + THREAD_PER_BLOCK * (2 * block + 1)];

    __shared__ uint32_t cached_twiddles[THREAD_PER_BLOCK * 2];

    cached_twiddles[threadId] = twiddles[threadId];
    cached_twiddles[threadId + THREAD_PER_BLOCK] = twiddles[threadId + THREAD_PER_BLOCK];

    __syncthreads();

    // step 0

    ExtField even = cached_buff[threadId * 2];
    ExtField odd = cached_buff[threadId * 2 + 1];

    ext_field_add(&even, &odd, &cached_buff[threadId * 2]);
    ext_field_sub(&even, &odd, &cached_buff[threadId * 2 + 1]);

    for (int step = 1; step <= LOG_THREAD_PER_BLOCK; step++)
    {
        int packet_size = 1 << step;
        int even_index = threadId + (threadId / packet_size) * packet_size;
        int odd_index = even_index + packet_size;

        ExtField even = cached_buff[even_index];
        ExtField odd = cached_buff[odd_index];

        int i = threadId % packet_size;
        // w^i where w is a "2 * packet_size" root of unity
        uint32_t first_twiddle = cached_twiddles[i * THREAD_PER_BLOCK / packet_size];
        // w^(i + packet_size) where w is a "2 * packet_size" root of unity
        uint32_t second_twiddle = cached_twiddles[(i + packet_size) * THREAD_PER_BLOCK / packet_size];

        // cached_buff[even_index] = even + first_twiddle * odd
        mul_prime_by_ext_field(&odd, first_twiddle, &cached_buff[even_index]);
        ext_field_add(&even, &cached_buff[even_index], &cached_buff[even_index]);

        // cached_buff[odd_index] = even + second_twiddle * odd
        mul_prime_by_ext_field(&odd, second_twiddle, &cached_buff[odd_index]);
        ext_field_add(&even, &cached_buff[odd_index], &cached_buff[odd_index]);

        __syncthreads();
    }

    // copy back to global memory
    buff[threadId + THREAD_PER_BLOCK * 2 * block] = cached_buff[threadId];
    buff[threadId + THREAD_PER_BLOCK * (2 * block + 1)] = cached_buff[threadId + THREAD_PER_BLOCK];
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

    // we should have THREAD_PER_BLOCK * NUM_BLOCKS * n_repetitions * 2 = 1 << (log_len + log_extension_factor)
    // WARNING: We assume the number of blocks is a power of 2
    const uint32_t n_repetitions = (1 << (log_len + log_extension_factor)) / (THREAD_PER_BLOCK * gridDim.x * 2);

    // int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    const int len = 1 << log_len;
    const int expansion_factor = 1 << log_extension_factor;

    // 1) Expand input several times to fill result, multiplying by the appropriate twiddle factors
    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * THREAD_PER_BLOCK;

        if (threadIndex < len)
        {
            buff[threadIndex] = input[threadIndex];
        }
        else
        {
            uint32_t twidle = twiddles[(1 << (log_len + log_extension_factor)) - 1 + (threadIndex % len) * (threadIndex / len)];
            mul_prime_by_ext_field(&input[threadIndex % len], twidle, &buff[threadIndex]);
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
        ntt_at_block_level(buff, blockIdx.x + gridDim.x * rep, &twiddles[THREAD_PER_BLOCK * 2 - 1]);
    }

    // 4) Finish the NTT

    for (int step = LOG_THREAD_PER_BLOCK + 1; step < log_len; step++)
    {
        grid.sync();

        // we group together pairs which each side contains 1 << step elements

        for (int rep = 0; rep < n_repetitions; rep++)
        {
            int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * THREAD_PER_BLOCK;

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
            mul_prime_by_ext_field(&odd, first_twiddle, &buff[even_index]);
            ext_field_add(&even, &buff[even_index], &buff[even_index]);

            // result[odd_index] = even + second_twiddle * odd
            mul_prime_by_ext_field(&odd, second_twiddle, &buff[odd_index]);
            ext_field_add(&even, &buff[odd_index], &buff[odd_index]);
        }
    }

    grid.sync();

    // 5) Transpose buff to result

    for (int rep = 0; rep < n_repetitions * 2; rep++)
    {
        int threadIndex = threadIdx.x + (blockIdx.x + gridDim.x * rep) * THREAD_PER_BLOCK;

        result[threadIndex] = buff[(threadIndex % expansion_factor) * len + (threadIndex / expansion_factor)];
    }
}
