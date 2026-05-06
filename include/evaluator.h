#pragma once
#include <cstdint>
#include "ckks_types.h"

// ---------------------------------------------------------------------------
// Evaluator — homomorphic arithmetic on CKKS ciphertexts.
//
// Homomorphic addition is exact in Z_q:
//
//   c0_sum = c0_a + c0_b  (mod q)
//   c1_sum = c1_a + c1_b  (mod q)
//
// Correctness identity:
//   c0_sum + c1_sum · s  =  (c0_a + c1_a·s) + (c0_b + c1_b·s)
//                        ≈   m_a + m_b
//
// The noise budget decreases linearly with each addition: after k additions
// of fresh ciphertexts the per-slot error is bounded by k · N/(2Δ) plus
// k encryption-noise terms.  No bootstrapping is required for plain addition.
//
// All coefficient arithmetic uses the branchless mod_add primitive from
// math_utils.h and is annotated with #pragma omp simd for AVX2 vectorisation.
// ---------------------------------------------------------------------------
class Evaluator {
public:
    // N, q — ring parameters; must match the Encryptor / Decryptor used to
    //         produce the ciphertexts.
    Evaluator(uint32_t N, uint64_t q);

    // In-place ciphertext addition: ct1 += ct2 (coefficient-wise, mod q).
    // Both components c0 and c1 are updated.
    // Throws std::invalid_argument if (N, q) of either operand does not match
    // the Evaluator.
    void add_inplace(Ciphertext& ct1, const Ciphertext& ct2) const;

    // Returns a new Ciphertext equal to ct1 + ct2.
    // Equivalent to copying ct1 and calling add_inplace; uses Poly::operator+
    // directly to avoid two separate allocation passes.
    Ciphertext add(const Ciphertext& ct1, const Ciphertext& ct2) const;

private:
    uint32_t N_;
    uint64_t q_;

    void check_params(const Ciphertext& ct) const;
};
