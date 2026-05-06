#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "math_utils.h"

// Polynomial ring Z_q[X]/(X^N + 1).
// N must be a power of 2. Coefficients live in [0, q).
struct Poly {
    uint64_t q;
    uint32_t N;
    std::vector<uint64_t> coeffs;  // size N

    // Zero polynomial.
    Poly(uint32_t N, uint64_t q);
    // From an explicit coefficient vector (size must equal N).
    Poly(uint32_t N, uint64_t q, std::vector<uint64_t> c);

    // Coefficient-wise modular addition.
    Poly operator+(const Poly& rhs) const;
};
