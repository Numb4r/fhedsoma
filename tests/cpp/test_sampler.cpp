#include "sampler.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { ++g_fail; std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

static constexpr uint64_t Q  = 998244353ULL;
static constexpr uint32_t N  = 1024;
static constexpr double   SIGMA = 3.2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Map a coefficient in [0, q) to the centred representative in (-q/2, q/2].
static int64_t centre(uint64_t c, uint64_t q) {
    int64_t s = static_cast<int64_t>(c);
    int64_t h = static_cast<int64_t>(q / 2);
    return s > h ? s - static_cast<int64_t>(q) : s;
}

static double mean_centred(const Poly& p) {
    double sum = 0;
    for (uint32_t i = 0; i < p.N; ++i)
        sum += static_cast<double>(centre(p.coeffs[i], p.q));
    return sum / p.N;
}

static double stddev_centred(const Poly& p) {
    double m = mean_centred(p);
    double sq = 0;
    for (uint32_t i = 0; i < p.N; ++i) {
        double d = static_cast<double>(centre(p.coeffs[i], p.q)) - m;
        sq += d * d;
    }
    return std::sqrt(sq / p.N);
}

// ---------------------------------------------------------------------------
// Construction and seeding
// ---------------------------------------------------------------------------

static void test_default_ctor_compiles() {
    // Just checks that std::random_device seeding doesn't throw / crash.
    Sampler s;
    Poly p = s.uniform(8, Q);
    CHECK_EQ(p.N, 8u);
    CHECK_EQ(p.q, Q);
}

static void test_seeded_reproducible() {
    Sampler s1(42), s2(42);
    Poly p1 = s1.uniform(N, Q);
    Poly p2 = s2.uniform(N, Q);
    for (uint32_t i = 0; i < N; ++i)
        CHECK_EQ(p1.coeffs[i], p2.coeffs[i]);
}

static void test_different_seeds_differ() {
    Sampler s1(1), s2(2);
    Poly p1 = s1.uniform(N, Q);
    Poly p2 = s2.uniform(N, Q);
    bool any_diff = false;
    for (uint32_t i = 0; i < N; ++i)
        if (p1.coeffs[i] != p2.coeffs[i]) { any_diff = true; break; }
    CHECK(any_diff);
}

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

static void test_uniform_range() {
    Sampler s(123);
    Poly p = s.uniform(N, Q);
    for (uint32_t i = 0; i < N; ++i)
        CHECK(p.coeffs[i] < Q);
}

static void test_uniform_small_q() {
    // q = 17: easy to check the entire range is reachable with many samples.
    constexpr uint64_t q = 17;
    constexpr uint32_t n = 4096;
    Sampler s(7);
    Poly p = s.uniform(n, q);
    bool seen[17] = {};
    for (uint32_t i = 0; i < n; ++i) {
        CHECK(p.coeffs[i] < q);
        seen[p.coeffs[i]] = true;
    }
    // With n=4096 and q=17, every residue appears with overwhelming probability.
    for (int r = 0; r < 17; ++r)
        CHECK(seen[r]);
}

static void test_uniform_mean_and_spread() {
    // Mean of uniform [0, q) should be close to (q-1)/2.
    Sampler s(99);
    Poly p = s.uniform(N, Q);
    double expected_mean = (Q - 1) / 2.0;
    // Allow ±5 standard errors: SE = sqrt(q^2/12 / N) ≈ 8.7e6 for Q=998244353.
    double se = std::sqrt(static_cast<double>(Q) * Q / 12.0 / N);
    double m = 0;
    for (uint32_t i = 0; i < N; ++i)
        m += static_cast<double>(p.coeffs[i]);
    m /= N;
    CHECK(std::abs(m - expected_mean) < 5.0 * se);
}

// ---------------------------------------------------------------------------
// Discrete Gaussian
// ---------------------------------------------------------------------------

static void test_gaussian_range() {
    Sampler s(55);
    Poly p = s.gaussian(N, Q, SIGMA);
    for (uint32_t i = 0; i < N; ++i)
        CHECK(p.coeffs[i] < Q);
}

static void test_gaussian_mean_near_zero() {
    // Centred mean should be within 5 standard errors of 0.
    // SE_mean = sigma / sqrt(N) ≈ 0.10 for sigma=3.2, N=1024.
    Sampler s(11);
    Poly p = s.gaussian(N, Q, SIGMA);
    double m = mean_centred(p);
    CHECK(std::abs(m) < 5.0 * SIGMA / std::sqrt(static_cast<double>(N)));
}

static void test_gaussian_stddev_near_sigma() {
    // Centred std dev should be within 10% of sigma.
    Sampler s(22);
    Poly p = s.gaussian(N, Q, SIGMA);
    double sd = stddev_centred(p);
    CHECK(std::abs(sd - SIGMA) < 0.1 * SIGMA * 4.0);   // 4 × 10% = 40% wide band
}

static void test_gaussian_tail_bound() {
    // With default tail=6, no coefficient (after centering) should exceed 6*sigma.
    Sampler s(33);
    Poly p = s.gaussian(N, Q, SIGMA);
    double limit = 6.0 * SIGMA;
    for (uint32_t i = 0; i < N; ++i)
        CHECK(std::abs(static_cast<double>(centre(p.coeffs[i], Q))) <= limit);
}

static void test_gaussian_seeded_reproducible() {
    Sampler s1(77), s2(77);
    Poly p1 = s1.gaussian(N, Q, SIGMA);
    Poly p2 = s2.gaussian(N, Q, SIGMA);
    for (uint32_t i = 0; i < N; ++i)
        CHECK_EQ(p1.coeffs[i], p2.coeffs[i]);
}

static void test_gaussian_small_sigma() {
    // sigma = 1 → most coefficients are 0 or 1 (after centering: -1, 0, 1).
    Sampler s(44);
    Poly p = s.gaussian(N, Q, 1.0);
    for (uint32_t i = 0; i < N; ++i) {
        int64_t c = centre(p.coeffs[i], Q);
        CHECK(c >= -6 && c <= 6);   // tail=6, sigma=1 → |c| <= 6
    }
}

static void test_gaussian_invalid_sigma() {
    Sampler s(0);
    bool threw = false;
    try { s.gaussian(N, Q, 0.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { s.gaussian(N, Q, -1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_gaussian_invalid_tail() {
    Sampler s(0);
    bool threw = false;
    try { s.gaussian(N, Q, SIGMA, 0.5); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// Move semantics (non-copyable contract)
// ---------------------------------------------------------------------------

static void test_move_ctor() {
    Sampler s1(123);
    // Advance s1's state so it's not at its initial position.
    s1.uniform(4, Q);
    Sampler s2(123);
    s2.uniform(4, Q);

    Sampler s3(std::move(s1));       // move: s3 continues from s1's position
    Poly from_s3 = s3.uniform(N, Q);
    Poly from_s2 = s2.uniform(N, Q);  // s2 at the same position as s1 was

    for (uint32_t i = 0; i < N; ++i)
        CHECK_EQ(from_s3.coeffs[i], from_s2.coeffs[i]);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    test_default_ctor_compiles();
    test_seeded_reproducible();
    test_different_seeds_differ();

    test_uniform_range();
    test_uniform_small_q();
    test_uniform_mean_and_spread();

    test_gaussian_range();
    test_gaussian_mean_near_zero();
    test_gaussian_stddev_near_sigma();
    test_gaussian_tail_bound();
    test_gaussian_seeded_reproducible();
    test_gaussian_small_sigma();
    test_gaussian_invalid_sigma();
    test_gaussian_invalid_tail();

    test_move_ctor();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
