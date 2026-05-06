#include "ckks_ops.h"
#include "math_utils.h"
#include <vector>

// ---------------------------------------------------------------------------
// Schoolbook polynomial multiplication in Z_q[X]/(X^N+1).
//
// Reduction rule: X^N ≡ -1, so any product term a[i]*b[j] with i+j >= N
// contributes negatively to coefficient i+j-N.
//
// All arithmetic is carried out in Montgomery domain to avoid the inner-loop
// division that a plain mod-reduction would require.  The accumulator stays in
// Montgomery form throughout; a single from_mont() call per output coefficient
// converts it back at the end.
//
// Complexity: O(N^2).  This will be replaced by an NTT-based O(N log N)
// implementation once the NTT layer is available.
// ---------------------------------------------------------------------------
static Poly poly_mul(const Poly& a, const Poly& b) {
    const uint32_t   N = a.N;
    const uint64_t   q = a.q;
    MontgomeryCtx mont(q);

    // Convert both operands to Montgomery form once.
    std::vector<uint64_t> am(N), bm(N);
    for (uint32_t i = 0; i < N; ++i) {
        am[i] = mont.to_mont(a.coeffs[i]);
        bm[i] = mont.to_mont(b.coeffs[i]);
    }

    // Accumulate products in Montgomery form.
    // mont_mul(am[i], bm[j]) = a[i]*b[j]*R mod q — still a Montgomery residue —
    // so mod_add / mod_sub on rm[] are well-formed residue operations.
    std::vector<uint64_t> rm(N, 0);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            uint64_t prod = mont.mont_mul(am[i], bm[j]);
            uint32_t k    = i + j;
            if (k < N)
                rm[k]     = mod_add(rm[k],     prod, q);
            else
                rm[k - N] = mod_sub(rm[k - N], prod, q);
        }
    }

    Poly out(N, q);
    for (uint32_t i = 0; i < N; ++i)
        out.coeffs[i] = mont.from_mont(rm[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Ciphertext × Plaintext
// ---------------------------------------------------------------------------
Ciphertext mul_plain(const Ciphertext& ct, const Plaintext& pt) {
    return Ciphertext(
        poly_mul(ct.c0, pt.poly),
        poly_mul(ct.c1, pt.poly)
    );
}
