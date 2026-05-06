#pragma once
#include "ckks_types.h"

// Ciphertext × Plaintext  →  (c0·pt, c1·pt)
// Both components are multiplied by the plaintext polynomial in Z_q[X]/(X^N+1).
// Requires ct and pt to share the same N and q.
Ciphertext mul_plain(const Ciphertext& ct, const Plaintext& pt);
