#include "poly.h"
#include <cstdio>
#include <cstdlib>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { ++g_fail; std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

// ---------------------------------------------------------------------------
// Zero polynomial
// ---------------------------------------------------------------------------
static void test_zero() {
    Poly p(4, 17);
    CHECK_EQ(p.N, 4u);
    CHECK_EQ(p.q, 17u);
    for (uint32_t i = 0; i < 4; ++i)
        CHECK_EQ(p.coeffs[i], 0u);
}

// ---------------------------------------------------------------------------
// Addition wraps correctly (small prime)
// ---------------------------------------------------------------------------
static void test_add_small() {
    constexpr uint64_t q = 17;
    Poly a(4, q, {1, 10, 16, 0});
    Poly b(4, q, {5,  8,  1, 16});
    Poly c = a + b;

    CHECK_EQ(c.coeffs[0], 6u);   // 1  + 5  = 6
    CHECK_EQ(c.coeffs[1], 1u);   // 10 + 8  = 18 mod 17 = 1
    CHECK_EQ(c.coeffs[2], 0u);   // 16 + 1  = 17 mod 17 = 0
    CHECK_EQ(c.coeffs[3], 16u);  // 0  + 16 = 16
}

// ---------------------------------------------------------------------------
// Addition preserves modulus and degree metadata
// ---------------------------------------------------------------------------
static void test_add_metadata() {
    constexpr uint64_t q = 998244353ULL;
    Poly a(8, q);
    Poly b(8, q);
    Poly c = a + b;
    CHECK_EQ(c.N, 8u);
    CHECK_EQ(c.q, q);
    CHECK_EQ((uint32_t)c.coeffs.size(), 8u);
}

// ---------------------------------------------------------------------------
// Addition is commutative element-wise
// ---------------------------------------------------------------------------
static void test_add_commutative() {
    constexpr uint64_t q = 998244353ULL;
    Poly a(4, q, {123456789 % q, 0, q - 1, q / 2});
    Poly b(4, q, {987654321 % q, q - 1, 1, q / 2});
    Poly ab = a + b;
    Poly ba = b + a;
    for (uint32_t i = 0; i < 4; ++i)
        CHECK_EQ(ab.coeffs[i], ba.coeffs[i]);
}

// ---------------------------------------------------------------------------
// Adding zero polynomial is identity
// ---------------------------------------------------------------------------
static void test_add_zero_identity() {
    constexpr uint64_t q = 1000000007ULL;
    Poly a(4, q, {1, 2, 3, 4});
    Poly z(4, q);  // all zeros
    Poly c = a + z;
    for (uint32_t i = 0; i < 4; ++i)
        CHECK_EQ(c.coeffs[i], a.coeffs[i]);
}

// ---------------------------------------------------------------------------
// Invalid construction
// ---------------------------------------------------------------------------
static void test_invalid_size() {
    bool threw = false;
    try {
        Poly bad(4, 17, {1, 2, 3});  // size 3 != N=4
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    test_zero();
    test_add_small();
    test_add_metadata();
    test_add_commutative();
    test_add_zero_identity();
    test_invalid_size();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
