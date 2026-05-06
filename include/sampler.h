#pragma once
#include <cstdint>
#include <random>
#include "poly.h"

// ---------------------------------------------------------------------------
// Sampler — pseudo-random polynomial generator for CKKS key/noise generation.
//
// Thread-safety contract
// ----------------------
// A Sampler is NOT safe to call concurrently from multiple threads on the
// same instance.  For FL workers, create one Sampler per thread:
//
//   // main thread: build a seeded sampler per worker
//   std::vector<Sampler> samplers;
//   for (int t = 0; t < n_threads; ++t)
//       samplers.emplace_back(base_seed ^ t);
//
// Copying is disabled to prevent two objects silently sharing PRNG state and
// producing correlated outputs.  Move is allowed for the pattern above.
// ---------------------------------------------------------------------------

class Sampler {
public:
    // Seed from std::random_device — suitable for production.
    Sampler();
    // Deterministic seed — suitable for tests and reproducible experiments.
    explicit Sampler(uint64_t seed);

    Sampler(const Sampler&)            = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&)                 = default;
    Sampler& operator=(Sampler&&)      = default;

    // Uniform polynomial: each coefficient drawn uniformly from [0, q).
    // Uses internal rejection sampling so the distribution is exactly uniform
    // even when q does not divide 2^64.
    Poly uniform(uint32_t N, uint64_t q);

    // Discrete Gaussian polynomial.
    //
    // Each coefficient is sampled as round(X) where X ~ N(0, sigma²), then
    // reduced into [0, q).  A rejection loop discards the rare |X| > tail*sigma
    // outliers (default tail = 6; P(reject) < 2e-9 per coefficient).
    //
    // Typical CKKS value: sigma = 3.2.
    // sigma must be > 0; tail must be >= 1.
    Poly gaussian(uint32_t N, uint64_t q, double sigma, double tail = 6.0);

    // Uniform ternary polynomial.
    // Each coefficient is drawn independently and uniformly from {-1, 0, 1},
    // stored in [0, q) as {0, 1, q-1} respectively.
    Poly ternary_uniform(uint32_t N, uint64_t q);

    // Fixed Hamming-weight ternary polynomial.
    // Exactly hw coefficients are non-zero: hw/2 are set to +1 and hw/2 to -1
    // (q-1 in [0,q) representation), placed at uniformly random positions.
    // hw must be even and ≤ N.
    Poly ternary_hw(uint32_t N, uint64_t q, uint32_t hw);

private:
    std::mt19937_64 rng_;
};
