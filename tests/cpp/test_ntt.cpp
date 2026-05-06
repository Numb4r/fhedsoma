#include "ntt.h"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { ++g_fail; \
               std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
    } while (0)
#define CHECK_EQ(a, b) CHECK((a) == (b))

// Standalone mod_pow — no dependency on NttCtx internals.
static uint64_t mpow(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1; b %= m;
    for (; e; e >>= 1, b = (__uint128_t)b * b % m)
        if (e & 1) r = (__uint128_t)r * b % m;
    return r;
}

// Well-known NTT prime: 998244353 = 119·2²³ + 1.
// Primitive root g = 3.  Supports N up to 2²².
static constexpr uint64_t Q1 = 998244353ULL;

// ── Bit-reversal ─────────────────────────────────────────────────────────────

// Hand-computed references:
//   N = 4 (2 bits): 0→0, 1→2, 2→1, 3→3
//   N = 8 (3 bits): 0→0, 1→4, 2→2, 3→6, 4→1, 5→5, 6→3, 7→7
//   N =16 (4 bits): 0→0, 1→8, 2→4, 3→12, 4→2, 5→10, 6→6, 7→14,
//                   8→1, 9→9,10→5,11→13,12→3,13→11,14→7,15→15

static void test_bit_rev_n4() {
    NttCtx ctx(4, Q1);
    const uint32_t expected[] = {0, 2, 1, 3};
    for (uint32_t i = 0; i < 4; ++i)
        CHECK_EQ(ctx.bit_rev[i], expected[i]);
}

static void test_bit_rev_n8() {
    NttCtx ctx(8, Q1);
    const uint32_t expected[] = {0, 4, 2, 6, 1, 5, 3, 7};
    for (uint32_t i = 0; i < 8; ++i)
        CHECK_EQ(ctx.bit_rev[i], expected[i]);
}

static void test_bit_rev_n16() {
    NttCtx ctx(16, Q1);
    const uint32_t expected[] = {
        0, 8, 4, 12, 2, 10, 6, 14,
        1, 9, 5, 13, 3, 11, 7, 15
    };
    for (uint32_t i = 0; i < 16; ++i)
        CHECK_EQ(ctx.bit_rev[i], expected[i]);
}

// bit_rev is an involution: applying it twice must be the identity.
static void test_bit_rev_is_involution() {
    for (uint32_t N : {4u, 8u, 16u, 32u, 64u}) {
        NttCtx ctx(N, Q1);
        for (uint32_t i = 0; i < N; ++i)
            CHECK_EQ(ctx.bit_rev[ctx.bit_rev[i]], i);
    }
}

// bit_rev must be a permutation of {0, …, N-1}.
static void test_bit_rev_is_permutation() {
    for (uint32_t N : {4u, 8u, 16u, 64u}) {
        NttCtx ctx(N, Q1);
        std::vector<uint32_t> sorted(ctx.bit_rev.begin(), ctx.bit_rev.end());
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t i = 0; i < N; ++i)
            CHECK_EQ(sorted[i], i);
    }
}

// ── Validation ───────────────────────────────────────────────────────────────

static void test_rejects_non_power_of_2() {
    for (uint32_t bad : {0u, 1u, 3u, 5u, 6u, 7u, 9u}) {
        bool threw = false;
        try { NttCtx ctx(bad, Q1); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
}

// q = 7 does not satisfy 7 ≡ 1 (mod 16) for N = 8.
static void test_rejects_bad_modulus() {
    bool threw = false;
    try { NttCtx ctx(8, 7); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// q = 17 satisfies 17 ≡ 1 (mod 8), so N = 4 is fine; but N = 16 requires
// q ≡ 1 (mod 32) which 17 does not satisfy.
static void test_rejects_modulus_too_small_for_n() {
    NttCtx ok(4, 17);          // 17 ≡ 1 (mod 8)  ← valid
    CHECK_EQ(ok.N, 4u);

    bool threw = false;
    try { NttCtx bad(16, 17); } // 17 ≢ 1 (mod 32) ← invalid
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// ── Primitive root properties ─────────────────────────────────────────────────

// ψ^N ≡ −1 (mod q)  — i.e. ψ has exact order 2N.
static void test_psi_power_n_is_minus_one() {
    for (uint32_t N : {4u, 8u, 16u, 64u}) {
        NttCtx ctx(N, Q1);
        CHECK_EQ(mpow(ctx.psi, N, Q1), Q1 - 1);
    }
}

// ψ^{2N} ≡ 1 (mod q)  — consistency with order 2N.
static void test_psi_power_2n_is_one() {
    for (uint32_t N : {4u, 8u, 16u, 64u}) {
        NttCtx ctx(N, Q1);
        CHECK_EQ(mpow(ctx.psi, 2ULL * N, Q1), 1u);
    }
}

// ψ · ψ_inv ≡ 1 (mod q).
static void test_psi_inverse_correct() {
    for (uint32_t N : {8u, 16u}) {
        NttCtx ctx(N, Q1);
        uint64_t prod = static_cast<uint64_t>(
            (__uint128_t)ctx.psi * ctx.psi_inv % Q1);
        CHECK_EQ(prod, 1u);
    }
}

// Verify with a second, smaller prime: q = 17, N = 4.
// 17 ≡ 1 (mod 8), so a primitive 8th root of unity exists mod 17.
static void test_small_prime_q17_n4() {
    NttCtx ctx(4, 17);
    // ψ^4 ≡ −1 ≡ 16 (mod 17)
    CHECK_EQ(mpow(ctx.psi, 4, 17), 16u);
    // ψ^8 ≡ 1 (mod 17)
    CHECK_EQ(mpow(ctx.psi, 8, 17), 1u);
    // ψ · ψ_inv ≡ 1 (mod 17)
    CHECK_EQ(ctx.psi * ctx.psi_inv % 17, 1u);
}

// ── Twiddle tables ────────────────────────────────────────────────────────────

// psi_pow[0] must always be to_mont(1) — ψ^0 = 1.
static void test_psi_pow_base_is_one() {
    NttCtx ctx(8, Q1);
    CHECK_EQ(ctx.mont.from_mont(ctx.psi_pow[0]), 1u);
    CHECK_EQ(ctx.mont.from_mont(ctx.psi_inv_pow[0]), 1u);
}

// For every k: from_mont(psi_pow[k]) == ψ^{bit_rev[k]} mod q.
static void test_psi_pow_values_match_bit_rev() {
    NttCtx ctx(8, Q1);
    for (uint32_t k = 0; k < 8; ++k) {
        uint64_t expected = mpow(ctx.psi, ctx.bit_rev[k], Q1);
        CHECK_EQ(ctx.mont.from_mont(ctx.psi_pow[k]), expected);
    }
}

// For every k: from_mont(psi_inv_pow[k]) == ψ^{-bit_rev[k]} mod q.
static void test_psi_inv_pow_values_match_bit_rev() {
    NttCtx ctx(8, Q1);
    for (uint32_t k = 0; k < 8; ++k) {
        uint64_t expected = mpow(ctx.psi_inv, ctx.bit_rev[k], Q1);
        CHECK_EQ(ctx.mont.from_mont(ctx.psi_inv_pow[k]), expected);
    }
}

// psi_pow[k] · psi_inv_pow[k] ≡ 1 (mod q) for all k.
static void test_twiddle_product_is_one() {
    NttCtx ctx(16, Q1);
    for (uint32_t k = 0; k < 16; ++k) {
        uint64_t p = ctx.mont.from_mont(ctx.psi_pow[k]);
        uint64_t q = ctx.mont.from_mont(ctx.psi_inv_pow[k]);
        uint64_t prod = static_cast<uint64_t>((__uint128_t)p * q % Q1);
        CHECK_EQ(prod, 1u);
    }
}

// ── NTT helpers used by transform tests ──────────────────────────────────────

// Deterministic random polynomial with coefficients in [0, q).
static std::vector<uint64_t> make_poly(uint32_t N, uint64_t q, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, q - 1);
    std::vector<uint64_t> a(N);
    for (auto& c : a) c = dist(rng);
    return a;
}

// Schoolbook negacyclic convolution: c = a * b mod (X^N + 1, q).
static std::vector<uint64_t> negacyclic_schoolbook(
    const std::vector<uint64_t>& a, const std::vector<uint64_t>& b,
    uint32_t N, uint64_t q)
{
    std::vector<uint64_t> c(N, 0);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            uint64_t prod = static_cast<uint64_t>((__uint128_t)a[i] * b[j] % q);
            uint32_t k = i + j;
            if (k < N)
                c[k] = mod_add(c[k], prod, q);   // normal term
            else
                c[k - N] = mod_sub(c[k - N], prod, q);  // X^N ≡ -1 fold
        }
    }
    return c;
}

// ── N_inv_mont ────────────────────────────────────────────────────────────────

// N · N^{-1} ≡ 1 (mod q).
static void test_n_inv_correct() {
    for (uint32_t N : {4u, 8u, 16u, 64u}) {
        NttCtx ctx(N, Q1);
        uint64_t n_inv = ctx.mont.from_mont(ctx.N_inv_mont);
        uint64_t prod  = static_cast<uint64_t>((__uint128_t)N * n_inv % Q1);
        CHECK_EQ(prod, 1u);
    }
}

// ── Forward NTT ───────────────────────────────────────────────────────────────

// NTT of the zero polynomial must be all zeros.
static void test_ntt_zero_poly() {
    NttCtx ctx(8, Q1);
    std::vector<uint64_t> a(8, 0);
    ntt_forward(a, ctx);
    for (uint32_t i = 0; i < 8; ++i)
        CHECK_EQ(a[i], 0u);
}

// A constant polynomial P(x) = c evaluates to c at every point.
// So NTT([c, 0, …, 0]) must be [c, c, …, c].
static void test_ntt_constant_poly() {
    NttCtx ctx(8, Q1);
    const uint64_t c = 12345;
    std::vector<uint64_t> a(8, 0);
    a[0] = c;
    ntt_forward(a, ctx);
    for (uint32_t i = 0; i < 8; ++i)
        CHECK_EQ(a[i], c);
}

// ── Round-trip: INTT(NTT(a)) == a ─────────────────────────────────────────────

// Fixed small polynomial, N = 8.
static void test_round_trip_n8() {
    NttCtx ctx(8, Q1);
    const std::vector<uint64_t> orig = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint64_t> a = orig;
    ntt_forward(a, ctx);
    ntt_inverse(a, ctx);
    for (uint32_t i = 0; i < 8; ++i)
        CHECK_EQ(a[i], orig[i]);
}

// Random polynomials for N = 4, 8, 16, 32, 64 — exercises every stage count.
static void test_round_trip_multiple_sizes() {
    for (uint32_t N : {4u, 8u, 16u, 32u, 64u}) {
        NttCtx ctx(N, Q1);
        std::vector<uint64_t> orig = make_poly(N, Q1, N);
        std::vector<uint64_t> a = orig;
        ntt_forward(a, ctx);
        ntt_inverse(a, ctx);
        for (uint32_t i = 0; i < N; ++i)
            CHECK_EQ(a[i], orig[i]);
    }
}

// Large random polynomial (N = 256) stresses all 8 butterfly stages.
static void test_round_trip_n256_random() {
    NttCtx ctx(256, Q1);
    std::vector<uint64_t> orig = make_poly(256, Q1, /*seed=*/42);
    std::vector<uint64_t> a = orig;
    ntt_forward(a, ctx);
    ntt_inverse(a, ctx);
    bool ok = true;
    for (uint32_t i = 0; i < 256; ++i) {
        if (a[i] != orig[i]) { ok = false; break; }
    }
    CHECK(ok);
}

// ── Negacyclic convolution via NTT ────────────────────────────────────────────
//
// Property: INTT(NTT(a) ⊙ NTT(b)) == a * b mod (X^N + 1, q)
// where ⊙ denotes coefficient-wise multiplication mod q.

// N = 4, hand-chosen inputs so the schoolbook result is easy to audit.
static void test_negacyclic_mul_n4() {
    NttCtx ctx(4, Q1);
    const std::vector<uint64_t> a = {1, 2, 3, 4};
    const std::vector<uint64_t> b = {5, 6, 7, 8};
    const std::vector<uint64_t> expected = negacyclic_schoolbook(a, b, 4, Q1);

    std::vector<uint64_t> a_hat = a, b_hat = b;
    ntt_forward(a_hat, ctx);
    ntt_forward(b_hat, ctx);
    std::vector<uint64_t> c_hat(4);
    for (uint32_t i = 0; i < 4; ++i)
        c_hat[i] = static_cast<uint64_t>((__uint128_t)a_hat[i] * b_hat[i] % Q1);
    ntt_inverse(c_hat, ctx);

    for (uint32_t i = 0; i < 4; ++i)
        CHECK_EQ(c_hat[i], expected[i]);
}

// N = 16, random inputs — catches any twiddle indexing error.
static void test_negacyclic_mul_n16() {
    NttCtx ctx(16, Q1);
    const std::vector<uint64_t> a = make_poly(16, Q1, /*seed=*/1);
    const std::vector<uint64_t> b = make_poly(16, Q1, /*seed=*/2);
    const std::vector<uint64_t> expected = negacyclic_schoolbook(a, b, 16, Q1);

    std::vector<uint64_t> a_hat = a, b_hat = b;
    ntt_forward(a_hat, ctx);
    ntt_forward(b_hat, ctx);
    std::vector<uint64_t> c_hat(16);
    for (uint32_t i = 0; i < 16; ++i)
        c_hat[i] = static_cast<uint64_t>((__uint128_t)a_hat[i] * b_hat[i] % Q1);
    ntt_inverse(c_hat, ctx);

    for (uint32_t i = 0; i < 16; ++i)
        CHECK_EQ(c_hat[i], expected[i]);
}

// ── Algebraic properties ──────────────────────────────────────────────────────

// NTT is Z_q-linear: NTT(a + b) == NTT(a) + NTT(b) (coefficient-wise mod q).
static void test_ntt_linearity() {
    NttCtx ctx(16, Q1);
    const std::vector<uint64_t> a = make_poly(16, Q1, 10);
    const std::vector<uint64_t> b = make_poly(16, Q1, 11);

    // Compute NTT(a + b).
    std::vector<uint64_t> apb(16);
    for (uint32_t i = 0; i < 16; ++i)
        apb[i] = mod_add(a[i], b[i], Q1);
    ntt_forward(apb, ctx);

    // Compute NTT(a) + NTT(b).
    std::vector<uint64_t> a_hat = a, b_hat = b;
    ntt_forward(a_hat, ctx);
    ntt_forward(b_hat, ctx);
    std::vector<uint64_t> sum(16);
    for (uint32_t i = 0; i < 16; ++i)
        sum[i] = mod_add(a_hat[i], b_hat[i], Q1);

    for (uint32_t i = 0; i < 16; ++i)
        CHECK_EQ(apb[i], sum[i]);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_bit_rev_n4();
    test_bit_rev_n8();
    test_bit_rev_n16();
    test_bit_rev_is_involution();
    test_bit_rev_is_permutation();

    test_rejects_non_power_of_2();
    test_rejects_bad_modulus();
    test_rejects_modulus_too_small_for_n();

    test_psi_power_n_is_minus_one();
    test_psi_power_2n_is_one();
    test_psi_inverse_correct();
    test_small_prime_q17_n4();

    test_psi_pow_base_is_one();
    test_psi_pow_values_match_bit_rev();
    test_psi_inv_pow_values_match_bit_rev();
    test_twiddle_product_is_one();

    test_n_inv_correct();

    test_ntt_zero_poly();
    test_ntt_constant_poly();

    test_round_trip_n8();
    test_round_trip_multiple_sizes();
    test_round_trip_n256_random();

    test_negacyclic_mul_n4();
    test_negacyclic_mul_n16();

    test_ntt_linearity();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
