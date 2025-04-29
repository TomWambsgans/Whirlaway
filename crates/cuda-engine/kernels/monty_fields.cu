#include <stdio.h>
#include <stdint.h>

#ifdef USE_NOINLINE
#define MAYBE_NOINLINE __noinline__
#else
#define MAYBE_NOINLINE
#endif

// W is such that X^extension_degree - W is irreducible

#define W_KOALA_EXT4 KoalaBear::from_canonical(3)
#define W_KOALA_EXT8 KoalaBear::from_canonical(3)

#define W_BABY_EXT4 BabyBear::from_canonical(11)
#define W_BABY_EXT8 BabyBear::from_canonical(11)

template <uint MontyPrime, uint MontyBits, uint MontyMu>
struct MontyField
{
    uint32_t x; // montgomery representation

    const static int EXTENSION_DEGREE = 1;

    __device__ static constexpr MontyField<MontyPrime, MontyBits, MontyMu> one()
    {
        return MontyField<MontyPrime, MontyBits, MontyMu>::from_canonical(1);
    }

    __device__ static constexpr MontyField<MontyPrime, MontyBits, MontyMu> from(uint64_t x)
    {
        MontyField<MontyPrime, MontyBits, MontyMu> field = {0};
        field.x = x;
        return field;
    }

    __device__ static MontyField<MontyPrime, MontyBits, MontyMu> monty_reduce(uint64_t x)
    {
        uint64_t t = x * MontyMu & ((1ULL << MontyBits) - 1U);
        uint64_t u = t * MontyPrime;

        uint64_t x_sub_u = x - u;
        bool over = x < u;
        uint32_t x_sub_u_hi = (x_sub_u >> MontyBits);
        uint32_t corr = over ? MontyPrime : 0;
        return MontyField<MontyPrime, MontyBits, MontyMu>::from(x_sub_u_hi + corr);
    }

public:
    __device__ static constexpr MontyField<MontyPrime, MontyBits, MontyMu> from_canonical(uint32_t x)
    {
        return MontyField<MontyPrime, MontyBits, MontyMu>::from((uint32_t)(((uint64_t)x << MontyBits) % MontyPrime));
    }

    __device__ static MontyField<MontyPrime, MontyBits, MontyMu> add(const MontyField<MontyPrime, MontyBits, MontyMu> a, const MontyField<MontyPrime, MontyBits, MontyMu> b)
    {
        uint32_t sum = a.x + b.x;
        if (sum >= MontyPrime)
        {
            sum -= MontyPrime;
        }
        return MontyField<MontyPrime, MontyBits, MontyMu>::from(sum);
    }

    __device__ static MontyField<MontyPrime, MontyBits, MontyMu> mul(const MontyField<MontyPrime, MontyBits, MontyMu> a, const MontyField<MontyPrime, MontyBits, MontyMu> b)
    {
        uint64_t long_prod = (uint64_t)a.x * (uint64_t)b.x;
        return monty_reduce(long_prod);
    }

    __device__ static MontyField<MontyPrime, MontyBits, MontyMu> sub(const MontyField<MontyPrime, MontyBits, MontyMu> a, const MontyField<MontyPrime, MontyBits, MontyMu> b)
    {
        uint32_t diff = a.x - b.x;
        bool over = a.x < b.x; // Detect underflow
        uint32_t corr = over ? MontyPrime : 0;
        return MontyField<MontyPrime, MontyBits, MontyMu>::from(diff + corr);
    }
};

typedef MontyField<0x7f000001, 32, 0x81000001> KoalaBear;
typedef MontyField<0x78000001, 32, 0x88000001> BabyBear;

template <int N, typename BaseField>
struct ExtensionField
{
    const static int EXTENSION_DEGREE = N;
    BaseField coeffs[N]; // Polynomial coefficients

    __device__ static constexpr ExtensionField<N, BaseField> one()
    {
        ExtensionField<N, BaseField> res = {0};
        res.coeffs[0] = BaseField::one();
        return res;
    }

    __device__ static MAYBE_NOINLINE void add(ExtensionField<N, BaseField> *a, ExtensionField<N, BaseField> *b, ExtensionField<N, BaseField> *result)
    {
        // Works even if result is the same as a or b
        for (int i = 0; i < N; i++)
        {
            result->coeffs[i] = BaseField::add(a->coeffs[i], b->coeffs[i]);
        }
    }

    __device__ static MAYBE_NOINLINE void sub(const ExtensionField<N, BaseField> *a, const ExtensionField<N, BaseField> *b, ExtensionField<N, BaseField> *result)
    {
        // Works even if result is the same as a or b
        for (int i = 0; i < N; i++)
        {
            result->coeffs[i] = BaseField::sub(a->coeffs[i], b->coeffs[i]);
        }
    }

    __device__ static MAYBE_NOINLINE void mul(const ExtensionField<N, BaseField> *a, const ExtensionField<N, BaseField> *b, ExtensionField<N, BaseField> *result);

    __device__ static MAYBE_NOINLINE void mul_prime_field(const ExtensionField<N, BaseField> *a, const BaseField b, ExtensionField<N, BaseField> *result)
    {
        // Works even if result is the same as a
        for (int i = 0; i < N; i++)
        {
            result->coeffs[i] = BaseField::mul(a->coeffs[i], b);
        }
    }

    __device__ static MAYBE_NOINLINE void add_prime_field(const ExtensionField<N, BaseField> *a, const BaseField b, ExtensionField<N, BaseField> *result)
    {
        // Works even if result is the same as a
        result->coeffs[0] = BaseField::add(a->coeffs[0], b);
        for (int i = 1; i < N; i++)
        {
            result->coeffs[i] = a->coeffs[i];
        }
    }

    __device__ static MAYBE_NOINLINE void sub_prime_field(const BaseField a, const ExtensionField<N, BaseField> *b, ExtensionField<N, BaseField> *result)
    {
        // Works even if result is the same as b
        result->coeffs[0] = BaseField::sub(a, b->coeffs[0]);
        for (int i = 1; i < N; i++)
        {
            result->coeffs[i] = BaseField::sub(BaseField::from_canonical(0), b->coeffs[i]);
        }
    }

    __device__ static MAYBE_NOINLINE void sub_prime_field(const ExtensionField<N, BaseField> *a, const BaseField b, ExtensionField<N, BaseField> *result)
    {
        // Works even if result is the same as a
        result->coeffs[0] = BaseField::sub(a->coeffs[0], b);
        for (int i = 1; i < N; i++)
        {
            result->coeffs[i] = a->coeffs[i];
        }
    }
};
// Macro for N=4 extension field multiplication
#define DEFINE_EXTENSION_FIELD_MUL_N4(FIELD_TYPE, W_EXT4)                                                                                                                                    \
    template <>                                                                                                                                                                              \
    __device__ MAYBE_NOINLINE void ExtensionField<4, FIELD_TYPE>::mul(const ExtensionField<4, FIELD_TYPE> *a, const ExtensionField<4, FIELD_TYPE> *b, ExtensionField<4, FIELD_TYPE> *result) \
    {                                                                                                                                                                                        \
        /* Does not work if result is the same as a or b */                                                                                                                                  \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            result->coeffs[i] = {0};                                                                                                                                                         \
        }                                                                                                                                                                                    \
                                                                                                                                                                                             \
        /* TODO Karatsuba */                                                                                                                                                                 \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            for (int j = 0; j < 4; j++)                                                                                                                                                      \
            {                                                                                                                                                                                \
                FIELD_TYPE prod = FIELD_TYPE::mul(a->coeffs[i], b->coeffs[j]);                                                                                                               \
                                                                                                                                                                                             \
                if (i + j < 4)                                                                                                                                                               \
                {                                                                                                                                                                            \
                    FIELD_TYPE temp = FIELD_TYPE::add(result->coeffs[i + j], prod);                                                                                                          \
                    result->coeffs[i + j] = temp;                                                                                                                                            \
                }                                                                                                                                                                            \
                else                                                                                                                                                                         \
                {                                                                                                                                                                            \
                    FIELD_TYPE temp = FIELD_TYPE::mul(prod, W_EXT4);                                                                                                                         \
                    result->coeffs[i + j - 4] = FIELD_TYPE::add(result->coeffs[i + j - 4], temp);                                                                                            \
                }                                                                                                                                                                            \
            }                                                                                                                                                                                \
        }                                                                                                                                                                                    \
    }

// Macro for N=8 extension field multiplication
#define DEFINE_EXTENSION_FIELD_MUL_N8(FIELD_TYPE, W_EXT8)                                                                                                                                    \
    template <>                                                                                                                                                                              \
    __device__ MAYBE_NOINLINE void ExtensionField<8, FIELD_TYPE>::mul(const ExtensionField<8, FIELD_TYPE> *a, const ExtensionField<8, FIELD_TYPE> *b, ExtensionField<8, FIELD_TYPE> *result) \
    {                                                                                                                                                                                        \
        /* Does not work if result is the same as a or b */                                                                                                                                  \
        ExtensionField<8, FIELD_TYPE> A0_B0 = {0};                                                                                                                                           \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            for (int j = 0; j < 4; j++)                                                                                                                                                      \
            {                                                                                                                                                                                \
                FIELD_TYPE prod = FIELD_TYPE::mul(a->coeffs[i], b->coeffs[j]);                                                                                                               \
                A0_B0.coeffs[i + j] = FIELD_TYPE::add(A0_B0.coeffs[i + j], prod);                                                                                                            \
            }                                                                                                                                                                                \
        }                                                                                                                                                                                    \
                                                                                                                                                                                             \
        ExtensionField<8, FIELD_TYPE> A1_B1 = {0};                                                                                                                                           \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            for (int j = 0; j < 4; j++)                                                                                                                                                      \
            {                                                                                                                                                                                \
                FIELD_TYPE prod = FIELD_TYPE::mul(a->coeffs[i + 4], b->coeffs[j + 4]);                                                                                                       \
                A1_B1.coeffs[i + j] = FIELD_TYPE::add(A1_B1.coeffs[i + j], prod);                                                                                                            \
            }                                                                                                                                                                                \
        }                                                                                                                                                                                    \
                                                                                                                                                                                             \
        FIELD_TYPE A0_PLUS_A1[4];                                                                                                                                                            \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            A0_PLUS_A1[i] = FIELD_TYPE::add(a->coeffs[i], a->coeffs[i + 4]);                                                                                                                 \
        }                                                                                                                                                                                    \
                                                                                                                                                                                             \
        FIELD_TYPE B0_PLUS_B1[4];                                                                                                                                                            \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            B0_PLUS_B1[i] = FIELD_TYPE::add(b->coeffs[i], b->coeffs[i + 4]);                                                                                                                 \
        }                                                                                                                                                                                    \
                                                                                                                                                                                             \
        ExtensionField<8, FIELD_TYPE> A0_PLUS_A1_MULT_B0_PLUS_B1 = {0};                                                                                                                      \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            for (int j = 0; j < 4; j++)                                                                                                                                                      \
            {                                                                                                                                                                                \
                FIELD_TYPE prod = FIELD_TYPE::mul(A0_PLUS_A1[i], B0_PLUS_B1[j]);                                                                                                             \
                A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + j] = FIELD_TYPE::add(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + j], prod);                                                                  \
            }                                                                                                                                                                                \
        }                                                                                                                                                                                    \
                                                                                                                                                                                             \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            result->coeffs[i] = FIELD_TYPE::sub(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i + 4], A0_B0.coeffs[i + 4]);                                                                              \
            result->coeffs[i] = FIELD_TYPE::sub(result->coeffs[i], A1_B1.coeffs[i + 4]);                                                                                                     \
            result->coeffs[i] = FIELD_TYPE::mul(result->coeffs[i], W_EXT8);                                                                                                                  \
        }                                                                                                                                                                                    \
                                                                                                                                                                                             \
        for (int i = 0; i < 4; i++)                                                                                                                                                          \
        {                                                                                                                                                                                    \
            result->coeffs[i + 4] = FIELD_TYPE::sub(A0_PLUS_A1_MULT_B0_PLUS_B1.coeffs[i], A0_B0.coeffs[i]);                                                                                  \
            result->coeffs[i + 4] = FIELD_TYPE::sub(result->coeffs[i + 4], A1_B1.coeffs[i]);                                                                                                 \
        }                                                                                                                                                                                    \
                                                                                                                                                                                             \
        ExtensionField<8, FIELD_TYPE>::add(&A0_B0, result, result);                                                                                                                          \
        ExtensionField<8, FIELD_TYPE>::mul_prime_field(&A1_B1, W_EXT8, &A1_B1);                                                                                                              \
        ExtensionField<8, FIELD_TYPE>::add(&A1_B1, result, result);                                                                                                                          \
    }

// Apply the macros to define the specializations
DEFINE_EXTENSION_FIELD_MUL_N4(KoalaBear, W_KOALA_EXT4)
DEFINE_EXTENSION_FIELD_MUL_N4(BabyBear, W_BABY_EXT4)

DEFINE_EXTENSION_FIELD_MUL_N8(KoalaBear, W_KOALA_EXT8)
DEFINE_EXTENSION_FIELD_MUL_N8(BabyBear, W_BABY_EXT8)

typedef ExtensionField<4, KoalaBear> KoalaBear4;
typedef ExtensionField<8, KoalaBear> KoalaBear8;

typedef ExtensionField<4, BabyBear> BabyBear4;
typedef ExtensionField<8, BabyBear> BabyBear8;