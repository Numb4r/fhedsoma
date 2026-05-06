#include "math_utils.h"
#include <stdexcept>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Compute -q^{-1} mod 2^64 via five Newton-Raphson iterations.
// Starting point q satisfies q^2 ≡ 1 (mod 8) for any odd q, giving 3-bit
// accuracy. Each step doubles precision; after 5 steps we exceed 64 bits.
static uint64_t compute_q_inv(uint64_t q) {
    uint64_t x = q;
    x *= 2 - q * x;   // ~6  bits
    x *= 2 - q * x;   // ~12 bits
    x *= 2 - q * x;   // ~24 bits
    x *= 2 - q * x;   // ~48 bits
    x *= 2 - q * x;   // ~96 bits → exact mod 2^64
    return -x;         // negate: gives -q^{-1} mod 2^64
}

// Compute R^2 mod q where R = 2^64.
static uint64_t compute_r2(uint64_t q) {
    __uint128_t r = (__uint128_t)1 << 64;  // R mod 2^128
    r %= q;                                 // R mod q
    r = (r * r) % q;                        // R^2 mod q
    return static_cast<uint64_t>(r);
}

// ---------------------------------------------------------------------------
// MontgomeryCtx
// ---------------------------------------------------------------------------

MontgomeryCtx::MontgomeryCtx(uint64_t q_) : q(q_) {
    q_inv = compute_q_inv(q);
    r2    = compute_r2(q);
}

// REDC: T * R^{-1} mod q.
// Proof sketch: m chosen so T + m*q ≡ 0 (mod R), then divide by R;
// result is in [0, 2q) so one conditional subtract suffices.
uint64_t MontgomeryCtx::redc(__uint128_t t) const noexcept {
    uint64_t m  = static_cast<uint64_t>(t) * q_inv;        // (T mod R) * q_inv mod R
    uint64_t hi = static_cast<uint64_t>((t + (__uint128_t)m * q) >> 64);
    return hi - (q & -(uint64_t)(hi >= q));
}

uint64_t MontgomeryCtx::to_mont(uint64_t a) const noexcept {
    // a * R^2 * R^{-1} mod q  =  a * R mod q
    return redc((__uint128_t)a * r2);
}

uint64_t MontgomeryCtx::from_mont(uint64_t a_m) const noexcept {
    // a_m * R^{-1} mod q  =  a mod q
    return redc((__uint128_t)a_m);
}

uint64_t MontgomeryCtx::mont_mul(uint64_t a_m, uint64_t b_m) const noexcept {
    // (a*R)(b*R) * R^{-1} mod q  =  a*b*R mod q
    return redc((__uint128_t)a_m * b_m);
}

// ---------------------------------------------------------------------------
// Miller-Rabin primality test (deterministic for all n < 2^64)
// ---------------------------------------------------------------------------

static uint64_t mr_powmod(uint64_t base, uint64_t exp, uint64_t mod) noexcept {
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

// Returns false if `a` is a Miller-Rabin witness for compositeness of n.
static bool mr_witness(uint64_t n, uint64_t a, uint64_t d, uint32_t r) noexcept {
    uint64_t x = mr_powmod(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (uint32_t i = 0; i < r - 1; ++i) {
        x = static_cast<uint64_t>((__uint128_t)x * x % n);
        if (x == n - 1) return true;
    }
    return false;
}

// Deterministic for n < 2^64 using the witness set from Pomerance et al.
static bool is_prime_mr(uint64_t n) noexcept {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 = d * 2^r with d odd.
    uint64_t d = n - 1;
    uint32_t r = 0;
    while ((d & 1) == 0) { d >>= 1; ++r; }

    // Witnesses sufficient for all n < 3.3 × 10^24 (> 2^62).
    constexpr uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (uint64_t a : witnesses) {
        if (a >= n) continue;
        if (!mr_witness(n, a, d, r)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// generate_proth_prime
// ---------------------------------------------------------------------------

uint64_t generate_proth_prime(uint32_t bit_size, uint32_t N) {
    if (bit_size < 2 || bit_size > 62)
        throw std::invalid_argument("bit_size must be in [2, 62]");
    if (N == 0)
        throw std::invalid_argument("N must be positive");

    const uint64_t step = 2ULL * N;          // candidates are spaced 2N apart
    const uint64_t lo   = 1ULL << (bit_size - 1);
    const uint64_t hi   = (1ULL << bit_size) - 1ULL;

    if (step > hi - lo)
        throw std::invalid_argument(
            "N is too large: no room for a bit_size-bit number ≡ 1 (mod 2N)");

    // Largest candidate ≡ 1 (mod 2N) that fits in bit_size bits.
    uint64_t k = (hi - 1) / step;
    uint64_t q = step * k + 1;

    while (q >= lo) {
        if (is_prime_mr(q))
            return q;
        if (q < lo + step)
            break;
        q -= step;
    }

    throw std::runtime_error(
        "generate_proth_prime: no prime found for the given bit_size and N");
}
