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

#define W_EXT4 KoalaBear::from_canonical(3)
#define W_EXT8 KoalaBear::from_canonical(3)

typedef struct KoalaBear
{
    uint32_t x; // montgomery representation

    const static int EXTENSION_DEGREE = 1;

    __device__ static constexpr KoalaBear one() {
        return KoalaBear::from_canonical(1);
    }


    __device__ static constexpr KoalaBear from(uint64_t x)
    {
        KoalaBear field = {};
        field.x = x;
        return field;
    }

    __device__ static KoalaBear monty_reduce(uint64_t x)
    {
        uint64_t t = x * MONTY_MU & MONTY_MASK;
        uint64_t u = t * MONTY_PRIME;

        uint64_t x_sub_u = x - u;
        bool over = x < u;
        uint32_t x_sub_u_hi = (x_sub_u >> MONTY_BITS);
        uint32_t corr = over ? MONTY_PRIME : 0;
        return KoalaBear::from(x_sub_u_hi + corr);
    }

public:
    __device__ static constexpr KoalaBear from_canonical(uint32_t x)
    {
        return KoalaBear::from((uint32_t)(((uint64_t)x << MONTY_BITS) % MONTY_PRIME));
    }

    __device__ static KoalaBear add(const KoalaBear a, const KoalaBear b)
    {
        uint32_t sum = a.x + b.x;
        if (sum >= MONTY_PRIME)
        {
            sum -= MONTY_PRIME;
        }
        return KoalaBear::from(sum);
    }

    __device__ static KoalaBear mul(const KoalaBear a, const KoalaBear b)
    {
        uint64_t long_prod = (uint64_t)a.x * (uint64_t)b.x;
        return monty_reduce(long_prod);
    }

    __device__ static KoalaBear sub(const KoalaBear a, const KoalaBear b)
    {
        uint32_t diff = a.x - b.x;
        bool over = a.x < b.x; // Detect underflow
        uint32_t corr = over ? MONTY_PRIME : 0;
        return KoalaBear::from(diff + corr);
    }
} KoalaBear;

template <int N>
struct KoalaBearExtension
{
    const static int EXTENSION_DEGREE = N;
    KoalaBear coeffs[N]; // Polynomial coefficients

    __device__ static constexpr KoalaBearExtension<N> one() {
        KoalaBearExtension<N> res = {0};
        res.coeffs[0] = KoalaBear::one();
        return res;
    }

    __device__ static MAYBE_NOINLINE void add(KoalaBearExtension<N> *a, KoalaBearExtension<N> *b, KoalaBearExtension<N> *result)
    {
        // Works even if result is the same as a or b
        for (int i = 0; i < N; i++)
        {
            result->coeffs[i] = KoalaBear::add(a->coeffs[i], b->coeffs[i]);
        }
    }

    __device__ static MAYBE_NOINLINE void sub(const KoalaBearExtension<N> *a, const KoalaBearExtension<N> *b, KoalaBearExtension<N> *result)
    {
        // Works even if result is the same as a or b
        for (int i = 0; i < N; i++)
        {
            result->coeffs[i] = KoalaBear::sub(a->coeffs[i], b->coeffs[i]);
        }
    }

    __device__ static MAYBE_NOINLINE void mul(const KoalaBearExtension<N> *a, const KoalaBearExtension<N> *b, KoalaBearExtension<N> *result);

    __device__ static MAYBE_NOINLINE void mul_prime_field(const KoalaBearExtension<N> *a, const KoalaBear b, KoalaBearExtension<N> *result)
    {
        // Works even if result is the same as a
        for (int i = 0; i < N; i++)
        {
            result->coeffs[i] = KoalaBear::mul(a->coeffs[i], b);
        }
    }

    __device__ static MAYBE_NOINLINE void add_prime_field(const KoalaBearExtension<N> *a, const KoalaBear b, KoalaBearExtension<N> *result)
    {
        // Works even if result is the same as a
        result->coeffs[0] = KoalaBear::add(a->coeffs[0], b);
        for (int i = 1; i < N; i++)
        {
            result->coeffs[i] = a->coeffs[i];
        }
    }

    __device__ static MAYBE_NOINLINE void sub_prime_field(const KoalaBear a, const KoalaBearExtension<N> *b, KoalaBearExtension<N> *result)
    {
        // Works even if result is the same as b
        result->coeffs[0] = KoalaBear::sub(a, b->coeffs[0]);
        for (int i = 1; i < N; i++)
        {
            result->coeffs[i] = KoalaBear::sub(KoalaBear::from_canonical(0), b->coeffs[i]);
        }
    }

    __device__ static MAYBE_NOINLINE void sub_prime_field(const KoalaBearExtension<N> *a, const KoalaBear b, KoalaBearExtension<N> *result)
    {
        // Works even if result is the same as a
        result->coeffs[0] = KoalaBear::sub(a->coeffs[0], b);
        for (int i = 1; i < N; i++)
        {
            result->coeffs[i] = a->coeffs[i];
        }
    }
};

// Specialization for KoalaBear4 (N=4)
template <>
__device__ MAYBE_NOINLINE void KoalaBearExtension<4>::mul(const KoalaBearExtension<4> *a, const KoalaBearExtension<4> *b, KoalaBearExtension<4> *result)
{ // class "KoalaBearExtension<4>" has no member "mul"
    // Does not work if result is the same as a or b
    for (int i = 0; i < 4; i++)
    {
        result->coeffs[i] = {0};
    }

    // TODO Karatsuba
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            KoalaBear prod = KoalaBear::mul(a->coeffs[i], b->coeffs[j]);

            if (i + j < 4)
            {
                KoalaBear temp = KoalaBear::add(result->coeffs[i + j], prod);
                result->coeffs[i + j] = temp;
            }
            else
            {
                KoalaBear temp = KoalaBear::mul(prod, W_EXT4);
                result->coeffs[i + j - 4] = KoalaBear::add(result->coeffs[i + j - 4], temp);
            }
        }
    }
}




// Specialization for KoalaBear8 (N=8)
template <>
__device__ MAYBE_NOINLINE void KoalaBearExtension<8>::mul(const KoalaBearExtension<8> *a, const KoalaBearExtension<8> *b, KoalaBearExtension<8> *result)
{
    // Extension field multiplication (with 1 "karatsuba step")

    // Same implementation as your original KoalaBear8::mul
    KoalaBearExtension<8> A0_B0 = {0};
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            KoalaBear prod = KoalaBear::mul(a->coeffs[i], b->coeffs[j]);
            A0_B0.coeffs[i + j] = KoalaBear::add(A0_B0.coeffs[i + j], prod);
        }
    }

    KoalaBearExtension<8> A1_B1 = {0};
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            KoalaBear prod = KoalaBear::mul(a->coeffs[i + 4], b->coeffs[j + 4]);
            A1_B1.coeffs[i + j] = KoalaBear::add(A1_B1.coeffs[i + j], prod);
        }
    }

    KoalaBear A0_PLUS_A1[4];
    for (int i = 0; i < 4; i++)
    {
        A0_PLUS_A1[i] = KoalaBear::add(a->coeffs[i], a->coeffs[i + 4]);
    }

    KoalaBear B0_PLUS_B1[4];
    for (int i = 0; i < 4; i++)
    {
        B0_PLUS_B1[i] = KoalaBear::add(b->coeffs[i], b->coeffs[i + 4]);
    }

    KoalaBearExtension<8> A0_PLUS_A1_MULT_B0_PLUS_B1 = {0};
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            KoalaBear prod = KoalaBear::mul(A0_PLUS_A1[i], B0_PLUS_B1[j]);
            A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + j] = KoalaBear::add(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + j], prod);
        }
    }

    for (int i = 0; i < 4; i++)
    {
        result->coeffs[i] = KoalaBear::sub(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + 4], A0_B0.coeffs[i + 4]);
        result->coeffs[i] = KoalaBear::sub(result->coeffs[i], A1_B1.coeffs[i + 4]);
        result->coeffs[i] = KoalaBear::mul(result->coeffs[i], W_EXT8);
    }

    for (int i = 0; i < 4; i++)
    {
        result->coeffs[i + 4] = KoalaBear::sub(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i], A0_B0.coeffs[i]);
        result->coeffs[i + 4] = KoalaBear::sub(result->coeffs[i + 4], A1_B1.coeffs[i]);
    }

    KoalaBearExtension<8>::add(&A0_B0, result, result);
    KoalaBearExtension<8>::mul_prime_field(&A1_B1, W_EXT8, &A1_B1);
    KoalaBearExtension<8>::add(&A1_B1, result, result);
}

// Define your specific field types
typedef KoalaBearExtension<4> KoalaBear4;
typedef KoalaBearExtension<8> KoalaBear8;
