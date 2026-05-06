#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "poly.h"

// A CKKS plaintext is a single polynomial m(X) in Z_q[X]/(X^N+1).
// The scale Δ is tracked externally by the encoder (not stored here).
struct Plaintext {
    Poly poly;
    explicit Plaintext(Poly p) : poly(std::move(p)) {}
};

// A CKKS ciphertext is a pair (c0, c1) such that
//   c0 + c1 * s  ≈  m  (mod q)
// where s is the secret key polynomial.
//
// Binary wire format (to_bytes / from_bytes):
//   [c0: poly_header || coeffs][c1: poly_header || coeffs]
// where poly_header = [N: uint32 LE (4 B)][q: uint64 LE (8 B)]
// and each coeff is uint64 LE (8 B).  Total: 2 × (12 + N×8) bytes.
struct Ciphertext {
    Poly c0, c1;
    Ciphertext(Poly c0_, Poly c1_) : c0(std::move(c0_)), c1(std::move(c1_)) {}

    // Serialize to a flat byte buffer (little-endian, self-describing).
    std::vector<uint8_t> to_bytes() const;

    // Deserialize from raw bytes.  Throws std::runtime_error on truncation,
    // trailing bytes, or (N, q) mismatch between c0 and c1.
    static Ciphertext from_bytes(const uint8_t* data, std::size_t size);
};
