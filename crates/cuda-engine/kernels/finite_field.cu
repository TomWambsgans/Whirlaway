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
#define W SmallField::from_canonical(3)

typedef struct SmallField
{
    uint32_t x; // montgomery representation

    __device__ static constexpr SmallField from(uint64_t x)
    {
        SmallField field = {};
        field.x = x;
        return field;
    }

    __device__ static SmallField monty_reduce(uint64_t x)
    {
        uint64_t t = x * MONTY_MU & MONTY_MASK;
        uint64_t u = t * MONTY_PRIME;

        uint64_t x_sub_u = x - u;
        bool over = x < u;
        uint32_t x_sub_u_hi = (x_sub_u >> MONTY_BITS);
        uint32_t corr = over ? MONTY_PRIME : 0;
        return SmallField::from(x_sub_u_hi + corr);
    }

public:
    __device__ static constexpr SmallField from_canonical(uint32_t x)
    {
        return SmallField::from((uint32_t)(((uint64_t)x << MONTY_BITS) % MONTY_PRIME));
    }

    __device__ static SmallField add(const SmallField a, const SmallField b)
    {
        uint32_t sum = a.x + b.x;
        if (sum >= MONTY_PRIME)
        {
            sum -= MONTY_PRIME;
        }
        return SmallField::from(sum);
    }

    __device__ static SmallField mul(const SmallField a, const SmallField b)
    {
        uint64_t long_prod = (uint64_t)a.x * (uint64_t)b.x;
        return monty_reduce(long_prod);
    }

    __device__ static SmallField sub(const SmallField a, const SmallField b)
    {
        uint32_t diff = a.x - b.x;
        bool over = a.x < b.x; // Detect underflow
        uint32_t corr = over ? MONTY_PRIME : 0;
        return SmallField::from(diff + corr);
    }
} SmallField;

typedef struct BigField
{
    SmallField coeffs[EXT_DEGREE]; // Polynomial coefficients

public:
    __device__ static MAYBE_NOINLINE void add(const BigField *a, const BigField *b, BigField *result)
    {
        // Works even if result is the same as a or b
        for (int i = 0; i < EXT_DEGREE; i++)
        {
            result->coeffs[i] = SmallField::add(a->coeffs[i], b->coeffs[i]);
        }
    }

    __device__ static MAYBE_NOINLINE void sub(const BigField *a, const BigField *b, BigField *result)
    {
        // Works even if result is the same as a or b
        for (int i = 0; i < EXT_DEGREE; i++)
        {
            result->coeffs[i] = SmallField::sub(a->coeffs[i], b->coeffs[i]);
        }
    }

    // Extension field multiplication (with 1 "karatsuba step")
    __device__ static MAYBE_NOINLINE void mul(const BigField *a, const BigField *b, BigField *result)
    {
        // a = a0 + a1.X + a2.X^2 + a3.X^3 + X^4.(a4 + a5.X + a6.X^2 + a7.X^3) = A0 + A1.X^4
        // b = b0 + b1.X + b2.X^2 + b3.X^3 + X^4.(b4 + b5.X + b6.X^2 + b7.X^3) = B0 + B1.X^4
        // a * b = A0.B0 + W.A1.B1 + X^4.[(A0 + A1).(B0 + B1) - A0.A1 - A1.B1]

        BigField A0_B0 = {0};
        for (int i = 0; i < EXT_DEGREE / 2; i++)
        {
            for (int j = 0; j < EXT_DEGREE / 2; j++)
            {
                SmallField prod = SmallField::mul(a->coeffs[i], b->coeffs[j]);
                A0_B0.coeffs[i + j] = SmallField::add(A0_B0.coeffs[i + j], prod);
            }
        }

        BigField A1_B1 = {0};
        for (int i = 0; i < EXT_DEGREE / 2; i++)
        {
            for (int j = 0; j < EXT_DEGREE / 2; j++)
            {
                SmallField prod = SmallField::mul(a->coeffs[i + EXT_DEGREE / 2], b->coeffs[j + EXT_DEGREE / 2]);
                A1_B1.coeffs[i + j] = SmallField::add(A1_B1.coeffs[i + j], prod);
            }
        }

        SmallField A0_PLUS_A1[EXT_DEGREE / 2];
        for (int i = 0; i < EXT_DEGREE / 2; i++)
        {
            A0_PLUS_A1[i] = SmallField::add(a->coeffs[i], a->coeffs[i + EXT_DEGREE / 2]);
        }

        SmallField B0_PLUS_B1[EXT_DEGREE / 2];
        for (int i = 0; i < EXT_DEGREE / 2; i++)
        {
            B0_PLUS_B1[i] = SmallField::add(b->coeffs[i], b->coeffs[i + EXT_DEGREE / 2]);
        }

        BigField A0_PLUS_A1_MULT_B0_PLUS_B1 = {0};
        for (int i = 0; i < EXT_DEGREE / 2; i++)
        {
            for (int j = 0; j < EXT_DEGREE / 2; j++)
            {
                SmallField prod = SmallField::mul(A0_PLUS_A1[i], B0_PLUS_B1[j]);
                A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + j] = SmallField::add(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + j], prod);
            }
        }

        for (int i = 0; i < EXT_DEGREE / 2; i++)
        {
            result->coeffs[i] = SmallField::sub(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + EXT_DEGREE / 2], A0_B0.coeffs[i + EXT_DEGREE / 2]);
            result->coeffs[i] = SmallField::sub(result->coeffs[i], A1_B1.coeffs[i + EXT_DEGREE / 2]);
            result->coeffs[i] = SmallField::mul(result->coeffs[i], W);
        }

        for (int i = 0; i < EXT_DEGREE / 2; i++)
        {
            result->coeffs[i + EXT_DEGREE / 2] = SmallField::sub(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i], A0_B0.coeffs[i]);
            result->coeffs[i + EXT_DEGREE / 2] = SmallField::sub(result->coeffs[i + EXT_DEGREE / 2], A1_B1.coeffs[i]);
        }

        BigField::add(&A0_B0, result, result);
        BigField::mul_small_field(&A1_B1, W, &A1_B1);
        BigField::add(result, &A1_B1, result);
    }

    __device__ static MAYBE_NOINLINE void mul_small_field(const BigField *a, const SmallField b, BigField *result)
    {
        // Works even if result is the same as a
        for (int i = 0; i < EXT_DEGREE; i++)
        {
            result->coeffs[i] = SmallField::mul(a->coeffs[i], b);
        }
    }

    __device__ static MAYBE_NOINLINE void add_small_field(const BigField *a, const SmallField b, BigField *result)
    {
        // Works even if result is the same as a
        result->coeffs[0] = SmallField::add(a->coeffs[0], b);
        for (int i = 1; i < EXT_DEGREE; i++)
        {
            result->coeffs[i] = a->coeffs[i];
        }
    }

    __device__ static MAYBE_NOINLINE void sub_to_small_field(const SmallField a, const BigField *b, BigField *result)
    {
        // Works even if result is the same as b
        result->coeffs[0] = SmallField::sub(a, b->coeffs[0]);
        for (int i = 1; i < EXT_DEGREE; i++)
        {
            result->coeffs[i] = SmallField::sub(SmallField::from_canonical(0), b->coeffs[i]);
        }
    }

    __device__ static MAYBE_NOINLINE void sub_from_small_field(const BigField *a, const SmallField b, BigField *result)
    {
        // Works even if result is the same as a
        result->coeffs[0] = SmallField::sub(a->coeffs[0], b);
        for (int i = 1; i < EXT_DEGREE; i++)
        {
            result->coeffs[i] = a->coeffs[i];
        }
    }

} BigField;

typedef struct TensorAlgebra
{
    SmallField coeffs[EXT_DEGREE][EXT_DEGREE];

public:
    __device__ static void add(const TensorAlgebra *a, const TensorAlgebra *b, TensorAlgebra *result)
    {
        // Works even if result is the same as a or b
        for (int i = 0; i < EXT_DEGREE; i++)
        {
            for (int j = 0; j < EXT_DEGREE; j++)
            {
                result->coeffs[i][j] = SmallField::add(a->coeffs[i][j], b->coeffs[i][j]);
            }
        }
    }

    __device__ static void phi_0_times_phi_1(const BigField *a, const BigField *b, TensorAlgebra *result)
    {
        for (int i = 0; i < EXT_DEGREE; i++)
        {
            for (int j = 0; j < EXT_DEGREE; j++)
            {
                result->coeffs[i][j] = SmallField::mul(a->coeffs[i], b->coeffs[j]);
            }
        }
    }

} TensorAlgebra;

// Schoolbook multiplication for extension fields
// __device__ MAYBE_NOINLINE void BigField::mul(const BigField *a, const BigField *b, BigField *result)
// {
//     // Does not work if result is the same as a or b
//     for (int i = 0; i < EXT_DEGREE; i++)
//     {
//         result->coeffs[i] = 0;
//     }

//     // Schoolbook multiplication
//     for (int i = 0; i < EXT_DEGREE; i++)
//     {
//         for (int j = 0; j < EXT_DEGREE; j++)
//         {

//             uint32_t prod = SmallField::mul(a->coeffs[i], b->coeffs[j]);

//             if (i + j < EXT_DEGREE)
//             {
//                 uint32_t temp = SmallField::add(result->coeffs[i + j], prod);
//                 result->coeffs[i + j] = temp;
//             }
//             else
//             {
//                 uint32_t temp = SmallField::mul(prod, W);
//                 result->coeffs[i + j - EXT_DEGREE] = SmallField::add(result->coeffs[i + j - EXT_DEGREE], temp);
//             }
//         }
//     }
// }
