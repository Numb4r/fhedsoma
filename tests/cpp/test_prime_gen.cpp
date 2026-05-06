#include "math_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Minimal test harness
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

// ---------------------------------------------------------------------------
// Independent Miller-Rabin for verification (same algorithm, separate copy)
// ---------------------------------------------------------------------------

static uint64_t test_powmod(uint64_t base, uint64_t exp, uint64_t mod) noexcept {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            result = static_cast<uint64_t>((__uint128_t)result * base % mod);
        base = static_cast<uint64_t>((__uint128_t)base * base % mod);
        exp >>= 1;
    }
    return result;
}

static bool test_mr_witness(uint64_t n, uint64_t a, uint64_t d, uint32_t r) noexcept {
    uint64_t x = test_powmod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (uint32_t i = 0; i < r - 1; ++i) {
        x = static_cast<uint64_t>((__uint128_t)x * x % n);
        if (x == n - 1) return true;
    }
    return false;
}

static bool is_prime(uint64_t n) noexcept {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    uint64_t d = n - 1;
    uint32_t r = 0;
    while ((d & 1) == 0) { d >>= 1; ++r; }
    constexpr uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (uint64_t a : witnesses) {
        if (a >= n) continue;
        if (!test_mr_witness(n, a, d, r)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Helper: verify all three properties for a given (bit_size, N) pair
// ---------------------------------------------------------------------------

static void verify(uint32_t bit_size, uint32_t N) {
    uint64_t q = 0;
    try {
        q = generate_proth_prime(bit_size, N);
    } catch (const std::exception& e) {
        ++g_fail;
        std::fprintf(stderr,
            "FAIL  generate_proth_prime(%u, %u) threw: %s\n",
            bit_size, N, e.what());
        return;
    }

    const uint64_t lo   = 1ULL << (bit_size - 1);
    const uint64_t hi   = (1ULL << bit_size) - 1ULL;
    const uint64_t step = 2ULL * N;

    // Bit-length check
    CHECK(q >= lo && q <= hi);
    // Congruence check
    CHECK(q % step == 1);
    // Primality check (independent)
    CHECK(is_prime(q));
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

static void test_small_primes() {
    // N=1 → step=2; any odd prime with the right bit-length qualifies.
    verify(4, 1);    // 4-bit: range [8,15]; odd primes: 11, 13
    verify(5, 1);    // 5-bit: range [16,31]; odd primes: 17,19,23,29,31
    verify(6, 1);
    verify(7, 1);
}

static void test_n4() {
    // N=4 → step=8; q ≡ 1 (mod 8)
    verify(10, 4);
    verify(16, 4);
    verify(20, 4);
}

static void test_ckks_params() {
    // N=4096 → step=8192; classic CKKS ring degree
    verify(30, 4096);
    verify(40, 4096);
    verify(50, 4096);
    verify(60, 4096);
}

static void test_large_n() {
    // N=32768 (2^15) → step=65536
    verify(30, 32768);
    verify(50, 32768);
    verify(60, 32768);
}

static void test_result_is_maximal() {
    // The returned prime must have no larger valid candidate below it that is
    // also prime.  We just check the returned value is ≡ 1 (mod 2N) and prime
    // (already done in verify), and that no skipped candidates above it in the
    // valid range are prime.
    constexpr uint32_t bit_size = 30;
    constexpr uint32_t N = 4096;
    constexpr uint64_t step = 2ULL * N;
    const uint64_t q = generate_proth_prime(bit_size, N);
    const uint64_t hi = (1ULL << bit_size) - 1ULL;

    // Walk from q+step up to hi; none of those should be prime with correct congruence.
    bool no_larger = true;
    for (uint64_t cand = q + step; cand <= hi; cand += step) {
        if (is_prime(cand)) { no_larger = false; break; }
    }
    CHECK(no_larger);
}

static void test_invalid_inputs() {
    auto throws = [](uint32_t bits, uint32_t n) -> bool {
        try { generate_proth_prime(bits, n); return false; }
        catch (const std::exception&) { return true; }
    };

    CHECK(throws(0, 4096));   // bit_size = 0
    CHECK(throws(1, 4096));   // bit_size = 1
    CHECK(throws(63, 4096));  // bit_size > 62
    CHECK(throws(10, 0));     // N = 0
    // N so large that 2N > entire bit_size range
    CHECK(throws(5, 1 << 10));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    test_small_primes();
    test_n4();
    test_ckks_params();
    test_large_n();
    test_result_is_maximal();
    test_invalid_inputs();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
