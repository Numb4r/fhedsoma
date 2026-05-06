#include "math_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>

// ---------------------------------------------------------------------------
// Minimal test harness — no external dependencies.
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { \
            ++g_pass; \
        } else { \
            ++g_fail; \
            std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

// ---------------------------------------------------------------------------
// mod_add / mod_sub — small prime
// ---------------------------------------------------------------------------

static void test_mod_add_sub_small() {
    constexpr uint64_t q = 17;

    CHECK_EQ(mod_add(0, 0, q), 0u);
    CHECK_EQ(mod_add(3, 4, q), 7u);
    CHECK_EQ(mod_add(10, 8, q), 1u);    // 18 mod 17 = 1
    CHECK_EQ(mod_add(16, 16, q), 15u);  // 32 mod 17 = 15
    CHECK_EQ(mod_add(0, 16, q), 16u);
    CHECK_EQ(mod_add(16, 0, q), 16u);

    CHECK_EQ(mod_sub(5, 3, q), 2u);
    CHECK_EQ(mod_sub(3, 5, q), 15u);    // 3 - 5 + 17
    CHECK_EQ(mod_sub(0, 0, q), 0u);
    CHECK_EQ(mod_sub(0, 1, q), 16u);    // -1 mod 17
    CHECK_EQ(mod_sub(16, 16, q), 0u);
}

// ---------------------------------------------------------------------------
// mod_add / mod_sub — large NTT prime (edge cases near q)
// ---------------------------------------------------------------------------

static void test_mod_add_sub_large() {
    constexpr uint64_t q = 998244353ULL;

    CHECK_EQ(mod_add(q - 1, 1, q), 0u);
    CHECK_EQ(mod_add(q - 1, q - 1, q), q - 2);
    CHECK_EQ(mod_add(0, 0, q), 0u);

    CHECK_EQ(mod_sub(0, q - 1, q), 1u);   // 0 - (q-1) + q = 1
    CHECK_EQ(mod_sub(q - 1, 0, q), q - 1);
    CHECK_EQ(mod_sub(q - 1, q - 1, q), 0u);
}

// ---------------------------------------------------------------------------
// MontgomeryCtx — round-trip and multiplication
// ---------------------------------------------------------------------------

// 998244353 = 119 * 2^23 + 1 (well-known NTT prime)
static constexpr uint64_t Q1 = 998244353ULL;
// 1000000007 (common 30-bit prime; validates that different moduli work)
static constexpr uint64_t Q2 = 1000000007ULL;
// 4611686018427387847 = 2^62 - 57 (large prime, stress-tests 62-bit range)
static constexpr uint64_t Q3 = 4611686018427387847ULL;

static uint64_t ref_mul(uint64_t a, uint64_t b, uint64_t q) {
    return static_cast<uint64_t>((__uint128_t)a * b % q);
}

static void test_round_trip(uint64_t q) {
    MontgomeryCtx ctx(q);
    uint64_t cases[] = {0, 1, 2, q / 2, q - 1};
    for (uint64_t a : cases) {
        CHECK_EQ(ctx.from_mont(ctx.to_mont(a)), a);
    }
}

static void test_mont_mul(uint64_t q) {
    MontgomeryCtx ctx(q);

    struct Pair { uint64_t a, b; };
    Pair cases[] = {
        {0, 0}, {1, 1}, {1, q - 1},
        {2, q / 2},
        {123456789 % q, 987654321 % q},
        {q - 1, q - 1},
    };

    for (auto [a, b] : cases) {
        uint64_t got = ctx.from_mont(ctx.mont_mul(ctx.to_mont(a), ctx.to_mont(b)));
        CHECK_EQ(got, ref_mul(a, b, q));
    }
}

static void test_mont_pow4(uint64_t q) {
    // a^4 computed as ((a^2)^2) in Montgomery domain
    MontgomeryCtx ctx(q);
    uint64_t a   = 42 % q;
    uint64_t a_m = ctx.to_mont(a);
    uint64_t a2  = ctx.mont_mul(a_m, a_m);
    uint64_t a4  = ctx.mont_mul(a2, a2);

    uint64_t expected = ref_mul(ref_mul(a, a, q), ref_mul(a, a, q), q);
    CHECK_EQ(ctx.from_mont(a4), expected);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    test_mod_add_sub_small();
    test_mod_add_sub_large();

    for (uint64_t q : {Q1, Q2, Q3}) {
        test_round_trip(q);
        test_mont_mul(q);
        test_mont_pow4(q);
    }

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
