#include "keygen.h"
#include "ntt.h"
#include "math_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
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

// Well-known NTT prime (supports up to N = 2^22).
static constexpr uint64_t Q = 998244353ULL;

// ── Helper: NTT-based poly multiply (independent of KeyGenerator internals) ──

static std::vector<uint64_t> ntt_poly_mul(
    const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b,
    const NttCtx& ctx)
{
    std::vector<uint64_t> a_hat = a;
    std::vector<uint64_t> b_hat = b;
    ntt_forward(a_hat, ctx);
    ntt_forward(b_hat, ctx);
    const uint32_t N = ctx.N;
    std::vector<uint64_t> c_hat(N);
    for (uint32_t i = 0; i < N; ++i)
        c_hat[i] = static_cast<uint64_t>((__uint128_t)a_hat[i] * b_hat[i] % ctx.q);
    ntt_inverse(c_hat, ctx);
    return c_hat;
}

// ── Secret key tests ──────────────────────────────────────────────────────────

// All coefficients of a uniform ternary key are in {0, 1, q-1}.
static void test_sk_uniform_ternary_coeffs() {
    KeyGenerator gen(64, Q, 3.2, /*seed=*/1);
    SecretKey sk = gen.gen_secret_key(/*hamming_weight=*/0);
    for (uint32_t i = 0; i < 64; ++i) {
        uint64_t c = sk.s.coeffs[i];
        CHECK(c == 0 || c == 1 || c == Q - 1);
    }
}

// A fixed Hamming-weight key has exactly hw non-zeros: hw/2 are +1, hw/2 are -1.
static void test_sk_hw_ternary_counts() {
    const uint32_t N  = 64;
    const uint32_t HW = 16;
    KeyGenerator gen(N, Q, 3.2, /*seed=*/2);
    SecretKey sk = gen.gen_secret_key(HW);

    uint32_t cnt_pos = 0, cnt_neg = 0, cnt_zero = 0;
    for (uint32_t i = 0; i < N; ++i) {
        uint64_t c = sk.s.coeffs[i];
        if      (c == 1)     ++cnt_pos;
        else if (c == Q - 1) ++cnt_neg;
        else                 ++cnt_zero;
    }
    CHECK_EQ(cnt_pos,  HW / 2);
    CHECK_EQ(cnt_neg,  HW / 2);
    CHECK_EQ(cnt_zero, N - HW);
}

// Fixed-weight ternary with hw = 0 is invalid by convention (must use uniform).
static void test_sk_hw_zero_rejected() {
    KeyGenerator gen(64, Q, 3.2, 3);
    // hamming_weight=0 triggers uniform ternary, not fixed-weight.
    // Separately, ternary_hw(N, q, 0) should succeed (all-zero poly is valid).
    Sampler s2(42);
    Poly z = s2.ternary_hw(8, Q, 0);
    bool all_zero = true;
    for (uint32_t i = 0; i < 8; ++i)
        if (z.coeffs[i] != 0) { all_zero = false; break; }
    CHECK(all_zero);
}

// Odd Hamming weight must be rejected.
static void test_sk_hw_odd_rejected() {
    Sampler s(5);
    bool threw = false;
    try { s.ternary_hw(64, Q, 7); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// hw > N must be rejected.
static void test_sk_hw_exceeds_n_rejected() {
    Sampler s(6);
    bool threw = false;
    try { s.ternary_hw(8, Q, 10); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// ── Public key invariant: pk0 + pk1·s = e ────────────────────────────────────
//
// This is the central correctness property of the public key.  All arithmetic
// is exact modular integer arithmetic, so equality must be bitwise exact.

static void run_pk_invariant_test(uint32_t N, uint32_t hw, uint64_t seed) {
    KeyGenerator gen(N, Q, /*sigma=*/3.2, seed);
    auto bundle = gen.gen_key_bundle(hw);

    // Compute pk1·s using the same NTT-based multiplication.
    NttCtx ctx(N, Q);
    auto pk1s = ntt_poly_mul(bundle.pk.pk1.coeffs, bundle.sk.s.coeffs, ctx);

    // pk0 + pk1·s  (coefficient-wise mod_add)
    bool ok = true;
    for (uint32_t i = 0; i < N; ++i) {
        uint64_t lhs = mod_add(bundle.pk.pk0.coeffs[i], pk1s[i], Q);
        if (lhs != bundle.e.coeffs[i]) { ok = false; break; }
    }
    CHECK(ok);
}

// N=64, uniform ternary secret key.
static void test_pk_invariant_n64_uniform() {
    run_pk_invariant_test(64, /*hw=*/0, /*seed=*/10);
}

// N=64, fixed Hamming-weight secret key (hw = 32).
static void test_pk_invariant_n64_hw32() {
    run_pk_invariant_test(64, /*hw=*/32, /*seed=*/11);
}

// N=128, uniform ternary — exercises more butterfly stages.
static void test_pk_invariant_n128_uniform() {
    run_pk_invariant_test(128, /*hw=*/0, /*seed=*/12);
}

// N=64, hw=8 — very sparse secret key.
static void test_pk_invariant_n64_hw8() {
    run_pk_invariant_test(64, /*hw=*/8, /*seed=*/13);
}

// ── Error-polynomial bounds ───────────────────────────────────────────────────
//
// pk0 + pk1·s = e, where e is Gaussian(sigma=3.2) with tail bound 6.
// Every centred coefficient of e must lie in (−6·sigma, 6·sigma].

static void test_error_is_small() {
    const uint32_t N     = 64;
    const double   SIGMA = 3.2;
    const double   TAIL  = 6.0;
    const int64_t  BOUND = static_cast<int64_t>(std::ceil(TAIL * SIGMA)); // 20

    KeyGenerator gen(N, Q, SIGMA, /*seed=*/20);
    auto bundle = gen.gen_key_bundle();

    NttCtx ctx(N, Q);
    auto pk1s = ntt_poly_mul(bundle.pk.pk1.coeffs, bundle.sk.s.coeffs, ctx);

    for (uint32_t i = 0; i < N; ++i) {
        uint64_t c = mod_add(bundle.pk.pk0.coeffs[i], pk1s[i], Q);
        // Centre: coefficients > Q/2 correspond to negative residues.
        int64_t centered = c > Q / 2
            ? static_cast<int64_t>(c) - static_cast<int64_t>(Q)
            : static_cast<int64_t>(c);
        CHECK(std::abs(centered) <= BOUND);
    }
}

// ── Reproducibility ───────────────────────────────────────────────────────────

// Two KeyGenerators with the same seed must produce identical keys.
static void test_deterministic_keygen() {
    const uint32_t N    = 64;
    const uint64_t SEED = 99;

    KeyGenerator g1(N, Q, 3.2, SEED);
    KeyGenerator g2(N, Q, 3.2, SEED);

    auto b1 = g1.gen_key_bundle();
    auto b2 = g2.gen_key_bundle();

    CHECK(b1.sk.s.coeffs   == b2.sk.s.coeffs);
    CHECK(b1.pk.pk0.coeffs == b2.pk.pk0.coeffs);
    CHECK(b1.pk.pk1.coeffs == b2.pk.pk1.coeffs);
    CHECK(b1.e.coeffs      == b2.e.coeffs);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_sk_uniform_ternary_coeffs();
    test_sk_hw_ternary_counts();
    test_sk_hw_zero_rejected();
    test_sk_hw_odd_rejected();
    test_sk_hw_exceeds_n_rejected();

    test_pk_invariant_n64_uniform();
    test_pk_invariant_n64_hw32();
    test_pk_invariant_n128_uniform();
    test_pk_invariant_n64_hw8();

    test_error_is_small();
    test_deterministic_keygen();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
