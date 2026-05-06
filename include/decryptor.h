#pragma once
#include <cstdint>
#include "ckks_types.h"
#include "keygen.h"
#include "ntt.h"

// ---------------------------------------------------------------------------
// Decryptor — CKKS secret-key decryption.
//
// Given Ciphertext (c0, c1) and SecretKey s:
//   m' = c0 + c1·s   (one NTT-based polynomial multiplication)
//
// m' equals the original plaintext polynomial m plus the small encryption
// noise (u·e_key + e0 + e1·s).  Pass m' to Encoder::decode to recover
// the approximate slot values.
// ---------------------------------------------------------------------------
class Decryptor {
public:
    // N, q — ring parameters.
    // sk   — secret key; Decryptor stores its own copy.
    Decryptor(uint32_t N, uint64_t q, const SecretKey& sk);

    Decryptor(const Decryptor&)            = delete;
    Decryptor& operator=(const Decryptor&) = delete;

    // Decrypt ct: returns a Plaintext whose poly satisfies p ≈ original_m.
    // The result is in Z_q form; pass to Encoder::decode for slot values.
    Plaintext decrypt(const Ciphertext& ct) const;

private:
    uint32_t  N_;
    uint64_t  q_;
    SecretKey sk_;       // owned copy of the secret key
    NttCtx    ntt_ctx_;
};
