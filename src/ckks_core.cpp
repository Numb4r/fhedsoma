#include "encryptor.h"
#include "decryptor.h"
#include "math_utils.h"

// ==========================================================================
// Encryptor
// ==========================================================================

Encryptor::Encryptor(uint32_t N, uint64_t q, const PublicKey& pk,
                     double sigma, uint64_t seed)
    : N_(N), q_(q), sigma_(sigma)
    , pk_(pk)
    , sampler_(seed != 0 ? Sampler(seed) : Sampler())
    , ntt_ctx_(N, q)
{}

// --------------------------------------------------------------------------
// encrypt
//
// Draw fresh ternary u and Gaussian errors e0, e1 for every call so that
// encrypting the same plaintext twice yields statistically independent
// ciphertexts (semantic security under LWE).
//
// All polynomial multiplications delegate to poly_ntt_mul (O(N log N));
// additions are coefficient-wise mod_add (O(N)).
// --------------------------------------------------------------------------
Ciphertext Encryptor::encrypt(const Plaintext& pt) {
    Poly u  = sampler_.ternary_uniform(N_, q_);
    Poly e0 = sampler_.gaussian(N_, q_, sigma_);
    Poly e1 = sampler_.gaussian(N_, q_, sigma_);

    // c1 = pk1·u + e1
    Poly pk1u = poly_ntt_mul(pk_.pk1, u, ntt_ctx_);
    Poly c1(N_, q_);
    for (uint32_t i = 0; i < N_; ++i)
        c1.coeffs[i] = mod_add(pk1u.coeffs[i], e1.coeffs[i], q_);

    // c0 = pk0·u + e0 + m
    Poly pk0u = poly_ntt_mul(pk_.pk0, u, ntt_ctx_);
    Poly c0(N_, q_);
    for (uint32_t i = 0; i < N_; ++i)
        c0.coeffs[i] = mod_add(
            mod_add(pk0u.coeffs[i], e0.coeffs[i], q_),
            pt.poly.coeffs[i], q_);

    return Ciphertext(std::move(c0), std::move(c1));
}

// ==========================================================================
// Decryptor
// ==========================================================================

Decryptor::Decryptor(uint32_t N, uint64_t q, const SecretKey& sk)
    : N_(N), q_(q)
    , sk_(sk)
    , ntt_ctx_(N, q)
{}

// --------------------------------------------------------------------------
// decrypt
//
// m' = c0 + c1·s
//
// Correctness:
//   c0 + c1·s = (pk0·u + e0 + m) + (pk1·u + e1)·s
//             = u·(pk0 + pk1·s) + e0 + e1·s + m
//             = u·e_key + e0 + e1·s + m
//
// The noise term u·e_key + e0 + e1·s has coefficients bounded by
// ~2·N·tail·sigma in the worst case (well below q/2 for CKKS parameters),
// so the centring step inside Encoder::decode recovers the correct slot
// values up to that small noise.
// --------------------------------------------------------------------------
Plaintext Decryptor::decrypt(const Ciphertext& ct) const {
    Poly c1s = poly_ntt_mul(ct.c1, sk_.s, ntt_ctx_);
    Poly m(N_, q_);
    for (uint32_t i = 0; i < N_; ++i)
        m.coeffs[i] = mod_add(ct.c0.coeffs[i], c1s.coeffs[i], q_);
    return Plaintext(std::move(m));
}
