#include "poly.h"

Poly::Poly(uint32_t N, uint64_t q) : q(q), N(N), coeffs(N, 0) {}

Poly::Poly(uint32_t N, uint64_t q, std::vector<uint64_t> c)
    : q(q), N(N), coeffs(std::move(c)) {
    if (coeffs.size() != N)
        throw std::invalid_argument("coeffs size must equal N");
}

Poly Poly::operator+(const Poly& rhs) const {
    Poly out(N, q);
    // Each iteration is independent and mod_add is branchless, so the loop
    // maps directly onto AVX2 vpsubq/vpcmpgtq/vpand with -march=native.
    // omp simd makes the vectorisation contract explicit for all compilers.
    #pragma omp simd
    for (uint32_t i = 0; i < N; ++i)
        out.coeffs[i] = mod_add(coeffs[i], rhs.coeffs[i], q);
    return out;
}
