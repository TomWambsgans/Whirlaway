#include "monty_fields.cu"

// TODO avoid duplications

#if !defined(FIELD)
#define FIELD 0
#endif

#if !defined(EXTENSION_DEGREE_A)
#define EXTENSION_DEGREE_A 1
#endif

#if !defined(EXTENSION_DEGREE_B)
#define EXTENSION_DEGREE_B 1
#endif

#if !defined(EXTENSION_DEGREE_C)
#define EXTENSION_DEGREE_C 1
#endif

// ──────────────────────────────────────────────────────────────────────────────
// Choose the family (KoalaBear   ↔ FIELD == 0
//                     BabyBear   ↔ FIELD == 1)
// ──────────────────────────────────────────────────────────────────────────────
#if FIELD == 0
// ─── KoalaBear family ────────────────────────────────────────────────────
template <unsigned Degree>
struct FieldSelector; // forward-declare

template <>
struct FieldSelector<1>
{
    using type = KoalaBear;
};
template <>
struct FieldSelector<4>
{
    using type = KoalaBear4;
};
template <>
struct FieldSelector<8>
{
    using type = KoalaBear8;
};

#elif FIELD == 1
// ─── BabyBear family ─────────────────────────────────────────────────────
template <unsigned Degree>
struct FieldSelector; // forward-declare

template <>
struct FieldSelector<1>
{
    using type = BabyBear;
};
template <>
struct FieldSelector<4>
{
    using type = BabyBear4;
};
template <>
struct FieldSelector<8>
{
    using type = BabyBear8;
};

#else
#error "Invalid value for FIELD."
#endif
// ──────────────────────────────────────────────────────────────────────────────

// Helper alias to avoid the verbose `typename …::type`
template <unsigned Degree>
using Field_t = typename FieldSelector<Degree>::type;

// ─── The three public aliases requested by the original code ─────────────────
using Field_A = Field_t<EXTENSION_DEGREE_A>;
using Field_B = Field_t<EXTENSION_DEGREE_B>;
using Field_C = Field_t<EXTENSION_DEGREE_C>;


#if EXTENSION_DEGREE_B == EXTENSION_DEGREE_C
#define FIELD_CONVERSION_B_C(src, dst) dst = src;
#elif EXTENSION_DEGREE_B == 1 && EXTENSION_DEGREE_C > 1
#define FIELD_CONVERSION_B_C(src, dst)                      \
    {                                                       \
        dst.coeffs[0] = src;                                \
        for (int i = 1; i < Field_C::EXTENSION_DEGREE; i++) \
        {                                                   \
            dst.coeffs[i] = {0};                            \
        }                                                   \
    }
#elif EXTENSION_DEGREE_B > 1 && EXTENSION_DEGREE_C >= EXTENSION_DEGREE_B
#define FIELD_CONVERSION_B_C(src, dst)                                              \
    {                                                                               \
        for (int i = 0; i < Field_B::EXTENSION_DEGREE; i++)                         \
        {                                                                           \
            dst.coeffs[i] = src->coeffs[i];                                         \
        }                                                                           \
        for (int i = Field_B::EXTENSION_DEGREE; i < Field_C::EXTENSION_DEGREE; i++) \
        {                                                                           \
            dst.coeffs[i] = {0};                                                    \
        }                                                                           \
    }
#endif

#if EXTENSION_DEGREE_A == 1 && EXTENSION_DEGREE_B == 1

#define ADD_AA(x, y, res) res = Field_A::add(x, y);
#define ADD_AB(x, y, res) res = Field_A::add(x, y);
#define ADD_BA(x, y, res) res = Field_B::add(x, y);
#define ADD_BB(x, y, res) res = Field_B::add(x, y);

#define SUB_AA(x, y, res) res = Field_A::sub(x, y);
#define SUB_AB(x, y, res) res = Field_A::sub(x, y);
#define SUB_BA(x, y, res) res = Field_B::sub(x, y);
#define SUB_BB(x, y, res) res = Field_B::sub(x, y);

#define MUL_AA(x, y, res) res = Field_A::mul(x, y);
#define MUL_BA(x, y, res) res = Field_B::mul(x, y);
#define MUL_AB(x, y, res) res = Field_A::mul(x, y);
#define MUL_BB(x, y, res) res = Field_B::mul(x, y);

#elif EXTENSION_DEGREE_A == 1 && EXTENSION_DEGREE_B > 1

#define ADD_AA(x, y, res) res = Field_A::add(x, y);
#define ADD_AB(x, y, res) Field_B::add_prime_field(&y, x, &res);
#define ADD_BA(x, y, res) Field_B::add_prime_field(&x, y, &res);
#define ADD_BB(x, y, res) Field_B::add(&x, &y, &res);

#define SUB_AA(x, y, res) res = Field_A::sub(x, y);
#define SUB_AB(x, y, res) Field_B::sub_prime_field(x, &y, &res);
#define SUB_BA(x, y, res) Field_B::sub_prime_field(&x, y, &res);
#define SUB_BB(x, y, res) Field_B::sub(&x, &y, &res);

#define MUL_AA(x, y, res) res = Field_A::mul(x, y);
#define MUL_BA(x, y, res) Field_B::mul_prime_field(&x, y, &res);
#define MUL_AB(x, y, res) Field_B::mul_prime_field(&y, x, &res);
#define MUL_BB(x, y, res) Field_B::mul(&x, &y, &res);

#elif EXTENSION_DEGREE_A > 1 && EXTENSION_DEGREE_B == 1

#define ADD_AA(x, y, res) Field_A::add(&x, &y, &res);
#define ADD_AB(x, y, res) Field_A::add_prime_field(&x, y, &res);
#define ADD_BA(x, y, res) Field_A::add_prime_field(x, &y, &res);
#define ADD_BB(x, y, res) res = Field_B::add(x, y);

#define SUB_AA(x, y, res) Field_A::sub(&x, &y, &res);
#define SUB_AB(x, y, res) Field_A::sub_prime_field(&x, y, &res);
#define SUB_BA(x, y, res) Field_A::sub_prime_field(x, &y, &res);
#define SUB_BB(x, y, res) res = Field_B::sub(x, y);

#define MUL_AA(x, y, res) Field_A::mul(&x, &y, &res);
#define MUL_BA(x, y, res) Field_A::mul_prime_field(&y, x, &res);
#define MUL_AB(x, y, res) Field_A::mul_prime_field(&x, y, &res);
#define MUL_BB(x, y, res) res = Field_B::mul(x, y);

#elif EXTENSION_DEGREE_A == EXTENSION_DEGREE_B

#define ADD_AA(x, y, res) Field_A::add(&x, &y, &res);
#define ADD_AB(x, y, res) Field_A::add(&x, &y, &res);
#define ADD_BA(x, y, res) Field_A::add(x, &y, &res);
#define ADD_BB(x, y, res) Field_A::add(&x, &y, &res);

#define SUB_AA(x, y, res) Field_A::sub(&x, &y, &res);
#define SUB_AB(x, y, res) Field_A::sub(&x, &y, &res);
#define SUB_BA(x, y, res) Field_A::sub(x, &y, &res);
#define SUB_BB(x, y, res) Field_A::sub(&x, &y, &res);

#define MUL_AA(x, y, res) Field_A::mul(&x, &y, &res);
#define MUL_BA(x, y, res) Field_A::mul(&x, &y, &res);
#define MUL_AB(x, y, res) Field_A::mul(&x, &y, &res);
#define MUL_BB(x, y, res) Field_A::mul(&x, &y, &res);

#else
#error "Invalid combination of EXTENSION_DEGREE_A and EXTENSION_DEGREE_B."

#endif

#if EXTENSION_DEGREE_A == 1 && EXTENSION_DEGREE_C == 1

#define ADD_AC(x, y, res) res = Field_A::add(x, y);
#define ADD_CA(x, y, res) res = Field_C::add(x, y);
#define ADD_CC(x, y, res) res = Field_C::add(x, y);

#define SUB_AC(x, y, res) res = Field_A::sub(x, y);
#define SUB_CA(x, y, res) res = Field_C::sub(x, y);
#define SUB_CC(x, y, res) res = Field_C::sub(x, y);

#define MUL_CA(x, y, res) res = Field_C::mul(x, y);
#define MUL_AC(x, y, res) res = Field_A::mul(x, y);
#define MUL_CC(x, y, res) res = Field_C::mul(x, y);

#elif EXTENSION_DEGREE_A == 1 && EXTENSION_DEGREE_C > 1
#define ADD_AC(x, y, res) Field_C::add_prime_field(&y, x, &res);
#define ADD_CA(x, y, res) Field_C::add_prime_field(&x, y, &res);
#define ADD_CC(x, y, res) Field_C::add(&x, &y, &res);

#define SUB_AA(x, y, res) res = Field_A::sub(x, y);
#define SUB_AC(x, y, res) Field_C::sub_prime_field(x, &y, &res);
#define SUB_CA(x, y, res) Field_C::sub_prime_field(&x, y, &res);
#define SUB_CC(x, y, res) Field_C::sub(&x, &y, &res);

#define MUL_CA(x, y, res) Field_C::mul_prime_field(&x, &y, &res);
#define MUL_AC(x, y, res) Field_C::mul_prime_field(&x, &y, &res);
#define MUL_CC(x, y, res) Field_C::mul(&x, &y, &res);

#elif EXTENSION_DEGREE_A > 1 && EXTENSION_DEGREE_C == 1

#define ADD_AC(x, y, res) Field_A::add_prime_field(&x, y, &res);
#define ADD_CA(x, y, res) Field_A::add_prime_field(x, &y, &res);
#define ADD_CC(x, y, res) res = Field_C::add(x, y);

#define SUB_AC(x, y, res) Field_A::sub_prime_field(&x, y, &res);
#define SUB_CA(x, y, res) Field_A::sub_prime_field(x, &y, &res);
#define SUB_CC(x, y, res) res = Field_C::sub(x, y);

#define MUL_CA(x, y, res) Field_A::mul_prime_field(x, &y, &res);
#define MUL_AC(x, y, res) Field_A::mul_prime_field(&x, y, &res);
#define MUL_CC(x, y, res) res = Field_C::mul(x, y);

#elif EXTENSION_DEGREE_A == EXTENSION_DEGREE_C

#define ADD_AC(x, y, res) Field_A::add(&x, &y, &res);
#define ADD_CA(x, y, res) Field_A::add(x, &y, &res);
#define ADD_CC(x, y, res) Field_A::add(&x, &y, &res);

#define SUB_AC(x, y, res) Field_A::sub(&x, &y, &res);
#define SUB_CA(x, y, res) Field_A::sub(x, &y, &res);
#define SUB_CC(x, y, res) Field_A::sub(&x, &y, &res);

#define MUL_CA(x, y, res) Field_A::mul(&x, &y, &res);
#define MUL_AC(x, y, res) Field_A::mul(&x, &y, &res);
#define MUL_CC(x, y, res) Field_A::mul(&x, &y, &res);

#else
#error "Invalid combination of EXTENSION_DEGREE_A and EXTENSION_DEGREE_C."

#endif

#if EXTENSION_DEGREE_B == 1 && EXTENSION_DEGREE_C == 1

#define ADD_BC(x, y, res) res = Field_B::add(x, y);
#define ADD_CB(x, y, res) res = Field_C::add(x, y);

#define SUB_BC(x, y, res) res = Field_B::sub(x, y);
#define SUB_CB(x, y, res) res = Field_C::sub(x, y);

#define MUL_CB(x, y, res) res = Field_C::mul(x, y);
#define MUL_BC(x, y, res) res = Field_B::mul(x, y);

#elif EXTENSION_DEGREE_B == 1 && EXTENSION_DEGREE_C > 1

#define ADD_BC(x, y, res) Field_C::add_prime_field(&y, x, &res);
#define ADD_CB(x, y, res) Field_C::add_prime_field(&x, y, &res);

#define SUB_BC(x, y, res) Field_C::sub_prime_field(x, &y, &res);
#define SUB_CB(x, y, res) Field_C::sub_prime_field(&x, y, &res);

#define MUL_CB(x, y, res) Field_C::mul_prime_field(&x, y, &res);
#define MUL_BC(x, y, res) Field_C::mul_prime_field(&y, x, &res);

#elif EXTENSION_DEGREE_B > 1 && EXTENSION_DEGREE_C == 1

#define ADD_BC(x, y, res) Field_B::add_prime_field(&x, y, &res);
#define ADD_CB(x, y, res) Field_B::add_prime_field(x, &y, &res);

#define SUB_BC(x, y, res) Field_B::sub_prime_field(&x, y, &res);
#define SUB_CB(x, y, res) Field_B::sub_prime_field(x, &y, &res);

#define MUL_CB(x, y, res) Field_B::mul_prime_field(x, &y, &res);
#define MUL_BC(x, y, res) Field_B::mul_prime_field(&x, y, &res);

#elif EXTENSION_DEGREE_B == EXTENSION_DEGREE_C

#define ADD_BC(x, y, res) Field_B::add(&x, &y, &res);
#define ADD_CB(x, y, res) Field_B::add(x, &y, &res);

#define SUB_BC(x, y, res) Field_B::sub(&x, &y, &res);
#define SUB_CB(x, y, res) Field_B::sub(x, &y, &res);

#define MUL_CB(x, y, res) Field_B::mul(&x, &y, &res);
#define MUL_BC(x, y, res) Field_B::mul(&x, &y, &res);

#else
#error "Invalid combination of EXTENSION_DEGREE_B and EXTENSION_DEGREE_C."

#endif

#define LARGER_TYPE(T1, T2) typename std::conditional<(sizeof(T1) > sizeof(T2)), T1, T2>::type
typedef LARGER_TYPE(Field_A, Field_B) LARGER_AB;

#if EXTENSION_DEGREE_A > EXTENSION_DEGREE_B
#define ADD_A_AND_MAX_AB(x, y, res) ADD_AA(x, y, res);
#define MUL_B_and_MAX_AB(x, y, res) MUL_BA(x, y, res);
#define ADD_MAX_AB(x, y, res) ADD_AA(x, y, res);
#define MUL_MAX_AB(x, y, res) MUL_AA(x, y, res);
#define SUB_MAX_AB(x, y, res) SUB_AA(x, y, res);

#else
#define ADD_A_AND_MAX_AB(x, y, res) ADD_AB(x, y, res);
#define MUL_B_and_MAX_AB(x, y, res) MUL_BB(x, y, res);
#define ADD_MAX_AB(x, y, res) ADD_BB(x, y, res);
#define MUL_MAX_AB(x, y, res) MUL_BB(x, y, res);
#define SUB_MAX_AB(x, y, res) SUB_BB(x, y, res);
#endif
