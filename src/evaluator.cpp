#include "evaluator.h"
#include "math_utils.h"
#include <stdexcept>

Evaluator::Evaluator(uint32_t N, uint64_t q) : N_(N), q_(q) {}

void Evaluator::check_params(const Ciphertext& ct) const {
    if (ct.c0.N != N_ || ct.c0.q != q_ || ct.c1.N != N_ || ct.c1.q != q_)
        throw std::invalid_argument(
            "Ciphertext ring parameters do not match Evaluator");
}

// ---------------------------------------------------------------------------
// add_inplace — ct1 += ct2 (in-place, coefficient-wise mod q).
//
// Uses the same branchless mod_add as Poly::operator+ and carries the
// #pragma omp simd contract so the compiler maps both loops onto AVX2
// vpsubq/vpcmpgtq/vpand with -march=native.
// ---------------------------------------------------------------------------
void Evaluator::add_inplace(Ciphertext& ct1, const Ciphertext& ct2) const {
    check_params(ct1);
    check_params(ct2);
    #pragma omp simd
    for (uint32_t i = 0; i < N_; ++i)
        ct1.c0.coeffs[i] = mod_add(ct1.c0.coeffs[i], ct2.c0.coeffs[i], q_);
    #pragma omp simd
    for (uint32_t i = 0; i < N_; ++i)
        ct1.c1.coeffs[i] = mod_add(ct1.c1.coeffs[i], ct2.c1.coeffs[i], q_);
}

// ---------------------------------------------------------------------------
// add — returns ct1 + ct2 as a new Ciphertext.
//
// Delegates to Poly::operator+ (which carries its own #pragma omp simd loop)
// to avoid two separate copy + mutation passes.
// ---------------------------------------------------------------------------
Ciphertext Evaluator::add(const Ciphertext& ct1, const Ciphertext& ct2) const {
    check_params(ct1);
    check_params(ct2);
    return Ciphertext(ct1.c0 + ct2.c0, ct1.c1 + ct2.c1);
}
