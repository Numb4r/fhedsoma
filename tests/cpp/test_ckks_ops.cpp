#include "ckks_ops.h"
#include <cstdio>
#include <cstdlib>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { ++g_fail; std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

static constexpr uint64_t Q = 998244353ULL;
static constexpr uint32_t N = 4;

// Reference polynomial multiplication (uses __uint128_t, not Montgomery).
static Poly ref_poly_mul(const Poly& a, const Poly& b) {
    Poly out(N, Q);
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            uint64_t prod = static_cast<uint64_t>((__uint128_t)a.coeffs[i] * b.coeffs[j] % Q);
            uint32_t k    = i + j;
            if (k < N)
                out.coeffs[k] = (out.coeffs[k] + prod) % Q;
            else
                out.coeffs[k - N] = (out.coeffs[k - N] + Q - prod) % Q;
        }
    return out;
}

// ---------------------------------------------------------------------------
// Multiply by the constant-1 plaintext — should be identity on c0 and c1.
// ---------------------------------------------------------------------------
static void test_mul_by_one() {
    Plaintext pt(Poly(N, Q, {1, 0, 0, 0}));
    Ciphertext ct(
        Poly(N, Q, {3, 7, 0, 1}),
        Poly(N, Q, {2, 0, 5, 0})
    );
    Ciphertext res = mul_plain(ct, pt);

    for (uint32_t i = 0; i < N; ++i) {
        CHECK_EQ(res.c0.coeffs[i], ct.c0.coeffs[i]);
        CHECK_EQ(res.c1.coeffs[i], ct.c1.coeffs[i]);
    }
}

// ---------------------------------------------------------------------------
// Multiply by the zero polynomial — should zero out both components.
// ---------------------------------------------------------------------------
static void test_mul_by_zero() {
    Plaintext pt(Poly(N, Q));        // all coefficients = 0
    Ciphertext ct(
        Poly(N, Q, {1, 2, 3, 4}),
        Poly(N, Q, {5, 6, 7, 8})
    );
    Ciphertext res = mul_plain(ct, pt);

    for (uint32_t i = 0; i < N; ++i) {
        CHECK_EQ(res.c0.coeffs[i], 0u);
        CHECK_EQ(res.c1.coeffs[i], 0u);
    }
}

// ---------------------------------------------------------------------------
// Multiply by a scalar constant k — should scale every coefficient by k.
// ---------------------------------------------------------------------------
static void test_scale_by_constant() {
    constexpr uint64_t k = 7;
    Plaintext pt(Poly(N, Q, {k, 0, 0, 0}));
    Ciphertext ct(
        Poly(N, Q, {1, 0, 0, 0}),
        Poly(N, Q, {0, 1, 0, 0})
    );
    Ciphertext res = mul_plain(ct, pt);

    // c0 = 1·k·X^0 → [7, 0, 0, 0]
    CHECK_EQ(res.c0.coeffs[0], k);
    for (uint32_t i = 1; i < N; ++i) CHECK_EQ(res.c0.coeffs[i], 0u);

    // c1 = X·k → [0, 7, 0, 0]
    CHECK_EQ(res.c1.coeffs[0], 0u);
    CHECK_EQ(res.c1.coeffs[1], k);
    for (uint32_t i = 2; i < N; ++i) CHECK_EQ(res.c1.coeffs[i], 0u);
}

// ---------------------------------------------------------------------------
// X^(N-1) * X  =  X^N  ≡  -1  (mod X^N+1)
// Verifies that the wrap-around negation is applied correctly.
// ---------------------------------------------------------------------------
static void test_wrap_negation() {
    // c0 = X^3: coefficients [0, 0, 0, 1]
    Ciphertext ct(Poly(N, Q, {0, 0, 0, 1}), Poly(N, Q));
    // pt = X:   coefficients [0, 1, 0, 0]
    Plaintext  pt(Poly(N, Q, {0, 1, 0, 0}));
    Ciphertext res = mul_plain(ct, pt);

    // X^3 * X = X^4 ≡ -1 mod (X^4+1)  → [q-1, 0, 0, 0]
    CHECK_EQ(res.c0.coeffs[0], Q - 1);
    for (uint32_t i = 1; i < N; ++i) CHECK_EQ(res.c0.coeffs[i], 0u);
}

// ---------------------------------------------------------------------------
// X^3 * X^3  =  X^6  =  X^4 · X^2  ≡  -X^2  (mod X^4+1)
// ---------------------------------------------------------------------------
static void test_wrap_negation_degree2() {
    Ciphertext ct(Poly(N, Q, {0, 0, 0, 1}), Poly(N, Q));
    Plaintext  pt(Poly(N, Q, {0, 0, 0, 1}));
    Ciphertext res = mul_plain(ct, pt);

    // Expected: [0, 0, q-1, 0]
    CHECK_EQ(res.c0.coeffs[0], 0u);
    CHECK_EQ(res.c0.coeffs[1], 0u);
    CHECK_EQ(res.c0.coeffs[2], Q - 1);
    CHECK_EQ(res.c0.coeffs[3], 0u);
}

// ---------------------------------------------------------------------------
// (1 + X)^2  =  1 + 2X + X^2  (no wrap, degree < N)
// ---------------------------------------------------------------------------
static void test_degree2_product() {
    Ciphertext ct(Poly(N, Q, {1, 1, 0, 0}), Poly(N, Q));
    Plaintext  pt(Poly(N, Q, {1, 1, 0, 0}));
    Ciphertext res = mul_plain(ct, pt);

    CHECK_EQ(res.c0.coeffs[0], 1u);
    CHECK_EQ(res.c0.coeffs[1], 2u);
    CHECK_EQ(res.c0.coeffs[2], 1u);
    CHECK_EQ(res.c0.coeffs[3], 0u);
}

// ---------------------------------------------------------------------------
// Agree with reference implementation on a random-ish pair.
// ---------------------------------------------------------------------------
static void test_agrees_with_reference() {
    Poly a(N, Q, {123456789 % Q, 87654321 % Q, 11111111 % Q, 99999999 % Q});
    Poly b(N, Q, {987654321 % Q, 12345678 % Q, 55555555 % Q, 33333333 % Q});

    Ciphertext ct(a, b);
    Plaintext  pt(Poly(N, Q, {246810121 % Q, 13579111 % Q, 77777777 % Q, 44444444 % Q}));

    Ciphertext got = mul_plain(ct, pt);
    Poly ref_c0 = ref_poly_mul(a, pt.poly);
    Poly ref_c1 = ref_poly_mul(b, pt.poly);

    for (uint32_t i = 0; i < N; ++i) {
        CHECK_EQ(got.c0.coeffs[i], ref_c0.coeffs[i]);
        CHECK_EQ(got.c1.coeffs[i], ref_c1.coeffs[i]);
    }
}

int main() {
    test_mul_by_one();
    test_mul_by_zero();
    test_scale_by_constant();
    test_wrap_negation();
    test_wrap_negation_degree2();
    test_degree2_product();
    test_agrees_with_reference();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
