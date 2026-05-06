#include "ntt.h"
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Internal helpers — not exposed in the header.
// ---------------------------------------------------------------------------

// Binary exponentiation: base^exp mod m.
// Uses __uint128_t to prevent 64-bit overflow in the product.
static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1)
            result = static_cast<uint64_t>((__uint128_t)result * base % m);
        base = static_cast<uint64_t>((__uint128_t)base * base % m);
        exp >>= 1;
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^{-1} ≡ a^{q-2} (mod q).
// Valid only when q is prime and gcd(a, q) = 1.
static uint64_t mod_inv(uint64_t a, uint64_t q) {
    return mod_pow(a, q - 2, q);
}

// Collect the unique prime factors of n by trial division.
//
// For CKKS NTT primes of the form k·2^m + 1 (k small), n − 1 = k·2^m, so
// the trial loop terminates after removing 2^m and the small odd part k.
// For arbitrary 64-bit n the worst case is O(√n), which may be slow; a
// Miller–Rabin / Pollard-rho approach is needed for general-purpose use.
static std::vector<uint64_t> prime_factors(uint64_t n) {
    std::vector<uint64_t> fs;
    for (uint64_t p = 2; p * p <= n; ++p) {
        if (n % p == 0) {
            fs.push_back(p);
            while (n % p == 0) n /= p;
        }
    }
    if (n > 1) fs.push_back(n);   // remaining factor is prime
    return fs;
}

// Find the smallest primitive root of Z_q (assumes q is an odd prime).
//
// g is a primitive root iff  g^{(q-1)/p} ≢ 1 (mod q)  for every prime p
// dividing q − 1.  For well-known NTT primes (e.g. 998244353) the smallest
// primitive root is found on the first or second candidate.
static uint64_t primitive_root(uint64_t q) {
    const uint64_t phi     = q - 1;
    const auto     factors = prime_factors(phi);

    for (uint64_t g = 2; g < q; ++g) {
        bool ok = true;
        for (uint64_t p : factors) {
            if (mod_pow(g, phi / p, q) == 1) { ok = false; break; }
        }
        if (ok) return g;
    }
    throw std::runtime_error(
        "primitive_root: exhausted candidates for q = " + std::to_string(q));
}

// Derive the primitive (2N)-th root of unity ψ from the primitive root g.
//
// Because g has order q − 1 in (Z/qZ)*, the element
//     ψ = g^{(q−1)/(2N)}
// has order exactly 2N.  Since g is a primitive root it is always a
// quadratic non-residue, so g^{(q−1)/2} ≡ −1 (mod q), which propagates to
// ψ^N ≡ −1.  A runtime assertion verifies this.
static uint64_t primitive_2n_root(uint64_t q, uint32_t N) {
    const uint64_t g   = primitive_root(q);
    const uint64_t psi = mod_pow(g, (q - 1) / (2ULL * N), q);

    if (mod_pow(psi, N, q) != q - 1)
        throw std::runtime_error(
            "primitive_2n_root: ψ^N ≠ −1 (mod q); verify q ≡ 1 (mod 2N)");
    return psi;
}

// Build the log₂(N)-bit-reversal permutation.
//
// bit_rev[i] is obtained by writing i in binary with exactly log₂(N) bits
// and reading those bits in reverse order.
//
// Example (N = 8, 3 bits): 0→0, 1→4, 2→2, 3→6, 4→1, 5→5, 6→3, 7→7.
static std::vector<uint32_t> make_bit_rev(uint32_t N) {
    // __builtin_ctz gives the number of trailing zero bits, which equals
    // log₂(N) for a positive power of 2.
    const uint32_t log2N = static_cast<uint32_t>(__builtin_ctz(N));

    std::vector<uint32_t> br(N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t x = i, r = 0;
        for (uint32_t b = 0; b < log2N; ++b) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        br[i] = r;
    }
    return br;
}

// ---------------------------------------------------------------------------
// NttCtx constructor
// ---------------------------------------------------------------------------

NttCtx::NttCtx(uint32_t N_, uint64_t q_)
    : N(N_), q(q_), mont(q_)
{
    // ── Validate ────────────────────────────────────────────────────────────
    if (N < 2 || (N & (N - 1)) != 0)
        throw std::invalid_argument(
            "NttCtx: N must be a power of 2 and ≥ 2 (got " +
            std::to_string(N) + ")");

    if ((q - 1) % (2ULL * N) != 0)
        throw std::invalid_argument(
            "NttCtx: q must satisfy q ≡ 1 (mod 2N)  "
            "(q = " + std::to_string(q) +
            ", 2N = " + std::to_string(2ULL * N) + ")");

    // ── 1. Bit-reversal table ────────────────────────────────────────────────
    bit_rev = make_bit_rev(N);

    // ── 2. Primitive 2N-th roots ─────────────────────────────────────────────
    psi     = primitive_2n_root(q, N);
    psi_inv = mod_inv(psi, q);

    // ── 3. Twiddle tables (bit-reversed power order, Montgomery form) ────────
    //
    // Compute the raw sequential powers first:
    //   raw_fwd[k] = ψ^k mod q
    //   raw_inv[k] = ψ^{-k} mod q
    // Then permute by the bit-reversal index to obtain the access-order table:
    //   psi_pow[k]     = raw_fwd[ bit_rev[k] ]
    //   psi_inv_pow[k] = raw_inv[ bit_rev[k] ]
    //
    // The iterative NTT butterfly reads psi_pow[1], psi_pow[2], …, psi_pow[N-1]
    // sequentially (one twiddle per radix-2 group), so this layout is optimal
    // for cache prefetching.
    const uint64_t psi_m     = mont.to_mont(psi);
    const uint64_t psi_inv_m = mont.to_mont(psi_inv);

    std::vector<uint64_t> raw_fwd(N), raw_inv(N);
    raw_fwd[0] = raw_inv[0] = mont.to_mont(1);
    for (uint32_t k = 1; k < N; ++k) {
        raw_fwd[k] = mont.mont_mul(raw_fwd[k - 1], psi_m);
        raw_inv[k] = mont.mont_mul(raw_inv[k - 1], psi_inv_m);
    }

    psi_pow.resize(N);
    psi_inv_pow.resize(N);
    for (uint32_t k = 0; k < N; ++k) {
        psi_pow[k]     = raw_fwd[bit_rev[k]];
        psi_inv_pow[k] = raw_inv[bit_rev[k]];
    }

    // ── 4. N⁻¹ mod q in Montgomery form ─────────────────────────────────────
    N_inv_mont = mont.to_mont(mod_inv(static_cast<uint64_t>(N), q));
}

// ---------------------------------------------------------------------------
// Forward NTT — Cooley–Tukey DIT butterfly.
//
// The outer variable m counts the number of independent radix-2 groups.
// Within each group i, every butterfly pair (x0[j], x1[j]) is scaled by the
// same twiddle psi_pow[m + i] = to_mont(ψ^{bit_rev[m+i]}).  Because the
// twiddle is already in Montgomery form and x0/x1 are normal residues, one
// mont_mul gives x1 * ψ^k mod q — no full to_mont conversion needed.
//
// Input:  natural-order coefficient vector, entries in [0, q).
// Output: bit-reversed NTT evaluation vector, entries in [0, q).
// ---------------------------------------------------------------------------
void ntt_forward(std::vector<uint64_t>& a, const NttCtx& ctx) {
    const uint32_t N = ctx.N;
    for (uint32_t m = 1; m < N; m <<= 1) {
        const uint32_t gap = N / (2 * m);   // half the group size
        for (uint32_t i = 0; i < m; ++i) {
            const uint64_t wi = ctx.psi_pow[m + i];  // Montgomery-form twiddle
            uint64_t* __restrict__ x0 = a.data() + 2 * i * gap;
            uint64_t* __restrict__ x1 = x0 + gap;
            for (uint32_t j = 0; j < gap; ++j) {
                const uint64_t u = x0[j];
                const uint64_t v = ctx.mont.mont_mul(x1[j], wi);
                x0[j] = mod_add(u, v, ctx.q);
                x1[j] = mod_sub(u, v, ctx.q);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inverse NTT — Gentleman–Sande DIF butterfly + 1/N normalisation.
//
// The butterfly is the algebraic transpose of the forward one: first add/sub
// (sum → x0, difference → x1), then multiply x1 by the inverse twiddle.
// Running m from N/2 down to 1 is the Gentleman–Sande schedule.
//
// After log₂N stages, every coefficient has been multiplied by N implicitly;
// a single pass of N multiplications by N_inv_mont corrects this.
//
// Input:  bit-reversed NTT evaluation vector, entries in [0, q).
// Output: natural-order coefficient vector, entries in [0, q).
// ---------------------------------------------------------------------------
void ntt_inverse(std::vector<uint64_t>& a, const NttCtx& ctx) {
    const uint32_t N = ctx.N;
    for (uint32_t m = N >> 1; m >= 1; m >>= 1) {
        const uint32_t gap = N / (2 * m);
        for (uint32_t i = 0; i < m; ++i) {
            const uint64_t wi = ctx.psi_inv_pow[m + i];
            uint64_t* __restrict__ x0 = a.data() + 2 * i * gap;
            uint64_t* __restrict__ x1 = x0 + gap;
            for (uint32_t j = 0; j < gap; ++j) {
                const uint64_t u = x0[j];
                const uint64_t v = x1[j];
                x0[j] = mod_add(u, v, ctx.q);
                x1[j] = ctx.mont.mont_mul(mod_sub(u, v, ctx.q), wi);
            }
        }
    }
    // Divide every coefficient by N to undo the implicit N-fold accumulation.
    for (uint32_t i = 0; i < N; ++i)
        a[i] = ctx.mont.mont_mul(a[i], ctx.N_inv_mont);
}

// ---------------------------------------------------------------------------
// poly_ntt_mul — negacyclic ring multiplication via forward NTT + pointwise
// product + inverse NTT.  All three steps are O(N log N); the pointwise step
// uses __uint128_t to avoid overflow for any 64-bit prime q.
// ---------------------------------------------------------------------------
Poly poly_ntt_mul(const Poly& a, const Poly& b, const NttCtx& ctx) {
    std::vector<uint64_t> a_hat = a.coeffs;
    std::vector<uint64_t> b_hat = b.coeffs;
    ntt_forward(a_hat, ctx);
    ntt_forward(b_hat, ctx);
    const uint32_t N = ctx.N;
    std::vector<uint64_t> c_hat(N);
    for (uint32_t i = 0; i < N; ++i)
        c_hat[i] = static_cast<uint64_t>((__uint128_t)a_hat[i] * b_hat[i] % ctx.q);
    ntt_inverse(c_hat, ctx);
    return Poly(N, ctx.q, std::move(c_hat));
}
