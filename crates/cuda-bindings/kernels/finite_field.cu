#include <stdio.h>
#include <stdint.h>

#ifdef USE_NOINLINE
#define MAYBE_NOINLINE __noinline__
#else
#define MAYBE_NOINLINE
#endif

// Koala bear field
#define MONTY_PRIME 0x7f000001
#define MONTY_BITS 32
#define MONTY_MASK ((1ULL << MONTY_BITS) - 1U)
#define MONTY_MU 0x81000001

#define EXT_DEGREE 8
#define W 100663290U // montgomery representation of 3, X^8 - 3 is irreducible

__device__ constexpr uint32_t to_monty(uint32_t x)
{
    return (uint32_t)(((uint64_t)x << MONTY_BITS) % MONTY_PRIME);
}

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

__device__ uint32_t monty_field_add(const uint32_t a, const uint32_t b)
{
    uint32_t sum = a + b;
    if (sum >= MONTY_PRIME)
    {
        sum -= MONTY_PRIME;
    }
    return sum;
}

__device__ uint32_t monty_field_mul(const uint32_t a, const uint32_t b)
{
    uint64_t long_prod = (uint64_t)a * (uint64_t)b;
    return monty_reduce(long_prod);
}

__device__ uint32_t monty_field_sub(const uint32_t a, const uint32_t b)
{
    uint32_t diff = a - b;
    bool over = a < b; // Detect underflow
    uint32_t corr = over ? MONTY_PRIME : 0;
    return diff + corr;
}

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
__device__ MAYBE_NOINLINE void ext_field_add(const ExtField *a, const ExtField *b, ExtField *result)
{
    // Works even if result is the same as a or b
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        result->coeffs[i] = monty_field_add(a->coeffs[i], b->coeffs[i]);
    }
}

// Subtract two extension field elements
__device__ MAYBE_NOINLINE void ext_field_sub(const ExtField *a, const ExtField *b, ExtField *result)
{
    // Works even if result is the same as a or b
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        result->coeffs[i] = monty_field_sub(a->coeffs[i], b->coeffs[i]);
    }
}

__device__ MAYBE_NOINLINE void mul_prime_and_ext_field(const ExtField *a, uint32_t b, ExtField *result)
{
    // Works even if result is the same as a
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        result->coeffs[i] = monty_field_mul(a->coeffs[i], b);
    }
}

__device__ MAYBE_NOINLINE void add_prime_and_ext_field(const ExtField *a, uint32_t b, ExtField *result)
{
    // TODO this would be more efficient in place (to avoid the copy loop)

    result->coeffs[0] = monty_field_add(a->coeffs[0], b);
    for (int i = 1; i < EXT_DEGREE; i++)
    {
        result->coeffs[i] = a->coeffs[i];
    }
}

// TODO Karatsuba ?
__device__ MAYBE_NOINLINE void ext_field_mul(const ExtField *a, const ExtField *b, ExtField *result)
{
    // Does not work if result is the same as a or b
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        result->coeffs[i] = 0;
    }

    // Schoolbook multiplication
    for (int i = 0; i < EXT_DEGREE; i++)
    {
        for (int j = 0; j < EXT_DEGREE; j++)
        {

            uint32_t prod = monty_field_mul(a->coeffs[i], b->coeffs[j]);

            if (i + j < EXT_DEGREE)
            {
                uint32_t temp = monty_field_add(result->coeffs[i + j], prod);
                result->coeffs[i + j] = temp;
            }
            else
            {
                uint32_t temp = monty_field_mul(prod, W);
                result->coeffs[i + j - EXT_DEGREE] = monty_field_add(result->coeffs[i + j - EXT_DEGREE], temp);
            }
        }
    }
}