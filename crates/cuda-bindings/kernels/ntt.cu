#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// // Constants to be defined at compile time
// #define EXT_DEGREE 1  // Example value, replace as needed
// #define P 2130706433U // Koala-Bear prime
// #define W 3U          // Such that X^EXT_DEGREE - W is irreducible
// #define MAX_SIZE 2 * EXT_DEGREE   // Maximum size for arrays, adjust based on your needs

// Montgomery parameters defined as macros
#define MONTY_PRIME 0x7f000001
#define MONTY_BITS 32
#define MONTY_MASK ((1ULL << MONTY_BITS) - 1U)
#define MONTY_MU 0x81000001

// Montgomery reduction for CUDA
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

extern "C" __global__ void test_add(uint32_t *a, uint32_t *b, uint32_t *result)
{
    monty_field_add(*a, *b, result);
}

extern "C" __global__ void test_mul(uint32_t *a, uint32_t *b, uint32_t *result)
{
    monty_field_mul(*a, *b, result);
}

extern "C" __global__ void test_sub(uint32_t *a, uint32_t *b, uint32_t *result)
{
    monty_field_sub(*a, *b, result);
}