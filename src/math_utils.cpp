#include "math_utils.h"

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
