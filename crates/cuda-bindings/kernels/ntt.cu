#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Koala bear field
#define MONTY_PRIME 0x7f000001
#define MONTY_BITS 32
#define MONTY_MASK ((1ULL << MONTY_BITS) - 1U)
#define MONTY_MU 0x81000001

#define EXT_DEGREE 8
#define W 100663290U // montgomery representation of 3, X^8 - 3 is irreducible

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

// Example kernel for testing extension field operations
extern "C" __global__ void test_add(ExtField *a, ExtField *b, ExtField *result)
{
    ext_field_add(a, b, result);
}

extern "C" __global__ void test_sub(ExtField *a, ExtField *b, ExtField *result)
{
    ext_field_sub(a, b, result);
}

extern "C" __global__ void test_mul(ExtField *a, ExtField *b, ExtField *result)
{
    ext_field_mul(a, b, result);
}