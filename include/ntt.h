#pragma once
#include <cstdint>
#include <vector>
#include "math_utils.h"
#include "poly.h"

// ---------------------------------------------------------------------------
// NttCtx — precomputed context for the negacyclic Number Theoretic Transform.
//
// Ring: Z_q[X]/(X^N+1), N a power of 2.
//
// The negacyclic NTT evaluates a polynomial at the N primitive (2N)-th roots
// of unity  ψ^1, ψ^3, ..., ψ^{2N-1}  where ψ satisfies ψ^N ≡ −1 (mod q).
// This is the ring-correct transform: it is compatible with multiplication
// in Z_q[X]/(X^N+1) without the sign-flip correction that a cyclic NTT needs.
//
// Required condition: q ≡ 1 (mod 2N) — guarantees a primitive (2N)-th root
// of unity exists in (Z/qZ)*.
//
// All twiddle entries are stored in Montgomery form so that the hot butterfly
// path never calls to_mont() at runtime.
// ---------------------------------------------------------------------------

struct NttCtx {
    uint32_t      N;          // polynomial degree (power of 2, ≥ 2)
    uint64_t      q;          // prime modulus; q ≡ 1 (mod 2N)
    uint64_t      psi;        // primitive 2N-th root of unity mod q (normal form)
    uint64_t      psi_inv;    // modular inverse of ψ (normal form)
    MontgomeryCtx mont;       // Montgomery multiplier for q

    // ── Bit-reversal permutation table ──────────────────────────────────────
    // bit_rev[i] = i with its log₂(N) bits reversed.
    // Used to reorder the twiddle table so the NTT butterfly can walk it
    // sequentially from index 1 to N-1.
    std::vector<uint32_t> bit_rev;

    // ── Twiddle tables (Montgomery form, bit-reversed power order) ───────────
    //
    //   psi_pow[k]     = to_mont( ψ^{bit_rev[k]} mod q )   — forward NTT
    //   psi_inv_pow[k] = to_mont( ψ^{-bit_rev[k]} mod q )  — inverse NTT
    //
    // Entry 0 is always to_mont(1).  The butterfly reads entries 1 … N-1
    // in order, one per radix-2 group, giving sequential cache access.
    std::vector<uint64_t> psi_pow;
    std::vector<uint64_t> psi_inv_pow;

    // ── Normalisation factor ─────────────────────────────────────────────────
    // N_inv_mont = to_mont( N^{-1} mod q ).
    // The inverse NTT multiplies every output coefficient by N^{-1} to undo
    // the implicit factor of N accumulated by the butterfly.
    uint64_t N_inv_mont;

    // Construct and precompute all tables for ring degree N and modulus q.
    // Throws std::invalid_argument if N is not a power of 2 (≥ 2) or if
    // q does not satisfy q ≡ 1 (mod 2N).
    NttCtx(uint32_t N, uint64_t q);
};

// ---------------------------------------------------------------------------
// In-place negacyclic NTT / INTT over Z_q[X]/(X^N+1).
//
// Both functions require a.size() == ctx.N and every a[i] in [0, ctx.q).
// The output satisfies the same range invariant.
// ---------------------------------------------------------------------------

// Forward NTT — Cooley–Tukey DIT.
// Maps the coefficient vector to its N negacyclic evaluation values
// (at the N primitive (2N)-th roots of unity ψ, ψ³, …, ψ^{2N-1}).
// Output ordering is bit-reversed so the inverse can be run without an
// explicit permutation step.
void ntt_forward(std::vector<uint64_t>& a, const NttCtx& ctx);

// Inverse NTT — Gentleman–Sande DIF with built-in 1/N normalisation.
// Left-inverse of ntt_forward: ntt_inverse(ntt_forward(a)) == a.
// Input must be in the bit-reversed evaluation order produced by ntt_forward
// (or a coefficient-wise product of two such vectors for ring multiplication).
void ntt_inverse(std::vector<uint64_t>& a, const NttCtx& ctx);

// ---------------------------------------------------------------------------
// Convenience: negacyclic polynomial multiplication via NTT — O(N log N).
// Returns a·b mod (X^N + 1, q), coefficients in [0, q).
// Requires a.N == b.N == ctx.N and a.q == b.q == ctx.q.
// ---------------------------------------------------------------------------
Poly poly_ntt_mul(const Poly& a, const Poly& b, const NttCtx& ctx);
