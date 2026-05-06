#pragma once
#include <cstdint>
#include "ckks_types.h"
#include "keygen.h"
#include "ntt.h"
#include "sampler.h"

// ---------------------------------------------------------------------------
// Encryptor — CKKS public-key encryption.
//
// Given Plaintext m (an encoded polynomial) and PublicKey (pk0, pk1):
//
//   u   ← ternary_uniform(N, q)      ephemeral mask
//   e0  ← DiscreteGaussian(sigma)    independent noise
//   e1  ← DiscreteGaussian(sigma)
//
//   c1  = pk1·u + e1
//   c0  = pk0·u + e0 + m
//
// Decryption invariant (holds exactly in Z_q arithmetic):
//   c0 + c1·s  =  u·(pk0 + pk1·s) + e0 + e1·s + m
//              =  u·e_key + e0 + e1·s + m
//              ≈  m
//
// Both polynomial multiplications use NTT (O(N log N)).
// ---------------------------------------------------------------------------
class Encryptor {
public:
    // N, q   — ring parameters; must match the key pair and Encoder.
    // pk     — public key; Encryptor stores its own copy.
    // sigma  — encryption noise std dev (match the KeyGenerator sigma).
    // seed   — 0: std::random_device (production); other: deterministic (tests).
    Encryptor(uint32_t N, uint64_t q, const PublicKey& pk,
              double sigma = 3.2, uint64_t seed = 0);

    Encryptor(const Encryptor&)            = delete;
    Encryptor& operator=(const Encryptor&) = delete;

    // Encrypt pt under pk.  Draws a fresh u, e0, e1 every call.
    Ciphertext encrypt(const Plaintext& pt);

private:
    uint32_t  N_;
    uint64_t  q_;
    double    sigma_;
    PublicKey pk_;        // owned copy of the public key
    Sampler   sampler_;
    NttCtx    ntt_ctx_;
};
