#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

Sampler::Sampler()
    : rng_(std::random_device{}()) {}

Sampler::Sampler(uint64_t seed)
    : rng_(seed) {}

// ---------------------------------------------------------------------------
// Uniform
// ---------------------------------------------------------------------------

Poly Sampler::uniform(uint32_t N, uint64_t q) {
    std::uniform_int_distribution<uint64_t> dist(0, q - 1);
    Poly out(N, q);
    for (uint32_t i = 0; i < N; ++i)
        out.coeffs[i] = dist(rng_);
    return out;
}

// ---------------------------------------------------------------------------
// Discrete Gaussian
// ---------------------------------------------------------------------------

Poly Sampler::gaussian(uint32_t N, uint64_t q, double sigma, double tail) {
    if (sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive");
    if (tail < 1.0)
        throw std::invalid_argument("tail bound must be >= 1");

    std::normal_distribution<double> dist(0.0, sigma);
    const double          bound = tail * sigma;
    const int64_t         sq    = static_cast<int64_t>(q);

    Poly out(N, q);
    for (uint32_t i = 0; i < N; ++i) {
        // Rejection loop: discards outliers beyond `tail` standard deviations.
        // Terminates in one iteration with probability > 1 - 2e-9 (tail = 6).
        int64_t x;
        do { x = std::llround(dist(rng_)); } while (std::abs(x) > static_cast<int64_t>(bound));

        // Lift into [0, q): C++ % truncates toward zero, so negative x needs +q.
        int64_t r = x % sq;
        if (r < 0) r += sq;
        out.coeffs[i] = static_cast<uint64_t>(r);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Ternary uniform
// ---------------------------------------------------------------------------

Poly Sampler::ternary_uniform(uint32_t N, uint64_t q) {
    // Draw from {0, 1, 2} with equal probability; map 1→1, 2→(q-1 = -1 mod q).
    std::uniform_int_distribution<int> dist(0, 2);
    Poly out(N, q);
    for (uint32_t i = 0; i < N; ++i) {
        int x = dist(rng_);
        if (x == 1)       out.coeffs[i] = 1;
        else if (x == 2)  out.coeffs[i] = q - 1;
        // x == 0 → coefficient stays 0
    }
    return out;
}

// ---------------------------------------------------------------------------
// Fixed Hamming-weight ternary
// ---------------------------------------------------------------------------

Poly Sampler::ternary_hw(uint32_t N, uint64_t q, uint32_t hw) {
    if (hw > N)
        throw std::invalid_argument("hamming weight exceeds N");
    if (hw % 2 != 0)
        throw std::invalid_argument("hamming weight must be even");

    // Partial Fisher–Yates: select hw positions uniformly at random.
    std::vector<uint32_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    for (uint32_t i = 0; i < hw; ++i) {
        std::uniform_int_distribution<uint32_t> pick(i, N - 1);
        std::swap(idx[i], idx[pick(rng_)]);
    }

    Poly out(N, q);
    for (uint32_t i = 0; i < hw / 2; ++i)
        out.coeffs[idx[i]] = 1;
    for (uint32_t i = hw / 2; i < hw; ++i)
        out.coeffs[idx[i]] = q - 1;   // -1 mod q
    return out;
}
