#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Branchless modular add / sub.
// Precondition: a, b in [0, q). q must be < 2^63 (all CKKS RNS primes satisfy this).
// ---------------------------------------------------------------------------

inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t q) noexcept {
    uint64_t c = a + b;
    // Subtract q exactly when c >= q; -(c >= q) is 0 or all-ones mask.
    return c - (q & -(uint64_t)(c >= q));
}

inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t q) noexcept {
    uint64_t c = a - b;
    // When a < b, c wraps in uint64; adding q corrects it back into [0, q).
    return c + (q & -(uint64_t)(a < b));
}

// ---------------------------------------------------------------------------
// Montgomery context for a fixed odd-prime modulus q < 2^62.
// Representatives live in Montgomery form: mont(a) = a * R mod q, R = 2^64.
// ---------------------------------------------------------------------------

// Generate a prime q with exactly bit_size bits satisfying q ≡ 1 (mod 2N).
// Searches downward from 2^bit_size - 1 in steps of 2N.
// Throws std::invalid_argument on bad inputs; std::runtime_error if no prime exists.
uint64_t generate_proth_prime(uint32_t bit_size, uint32_t N);

struct MontgomeryCtx {
    uint64_t q;      // modulus (odd prime)
    uint64_t q_inv;  // -q^{-1} mod 2^64   (for REDC inner step)
    uint64_t r2;     // R^2 mod q           (for conversion into Montgomery form)

    explicit MontgomeryCtx(uint64_t q);

    // a              -> a * R mod q
    uint64_t to_mont(uint64_t a) const noexcept;
    // a_m = a*R mod q -> a mod q
    uint64_t from_mont(uint64_t a_m) const noexcept;
    // (a*R mod q, b*R mod q) -> a*b*R mod q
    uint64_t mont_mul(uint64_t a_m, uint64_t b_m) const noexcept;

private:
    // Core REDC: returns T * R^{-1} mod q for T in [0, q*R).
    uint64_t redc(__uint128_t t) const noexcept;
};
