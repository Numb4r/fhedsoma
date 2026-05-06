#pragma once
#include <complex>
#include <cstdint>
#include <vector>
#include "poly.h"

// ---------------------------------------------------------------------------
// Encoder — maps between C^{N/2} slot values and polynomials in Z_q[X]/(X^N+1).
//
// The canonical embedding σ evaluates a polynomial at the N primitive
// (2N)-th roots of unity ω_j = exp(iπ(2j+1)/N).  Because ω_{N-1-j} =
// conj(ω_j), only N/2 independent complex slots exist.
//
// encode(z)  = σ^{-1}:  slot vector  →  scaled, rounded polynomial
// decode(p)  = σ    :   polynomial   →  approximate slot vector
//
// The only rounding error is the coefficient quantisation at encode time.
// With scale Δ the per-slot error after a round-trip is bounded by N/(2Δ).
// ---------------------------------------------------------------------------

class Encoder {
public:
    // N     : ring degree, must be a power of 2 (≥ 2).
    // q     : coefficient modulus; must exceed 2 * scale * max|slot value|.
    // scale : encoding scale Δ.  Typical value: 2^40.
    Encoder(uint32_t N, uint64_t q, double scale);

    // Encode a vector of at most N/2 complex values.
    // Shorter inputs are zero-padded to N/2 slots.
    Poly encode(const std::vector<std::complex<double>>& z) const;

    // Convenience overload: real values are treated as complex with Im = 0.
    Poly encode(const std::vector<double>& z) const;

    // Decode a polynomial to N/2 complex slot values.
    std::vector<std::complex<double>> decode(const Poly& p) const;

    uint32_t slots()  const noexcept { return N_ / 2; }
    uint32_t degree() const noexcept { return N_; }

private:
    uint32_t N_;
    uint64_t q_;
    double   scale_;

    // zeta_[k]     = exp(+iπk/N)  — twist applied before IFFT in decode
    // zeta_inv_[k] = exp(−iπk/N)  — twist applied after  FFT  in encode
    std::vector<std::complex<double>> zeta_;
    std::vector<std::complex<double>> zeta_inv_;

    // In-place Cooley–Tukey radix-2 FFT.
    //   inverse=false:  a[k] = Σ_j x[j] · e^{−2πijk/N}  (no normalisation)
    //   inverse=true :  a[j] = (1/N) Σ_k x[k] · e^{+2πijk/N}
    static void fft(std::vector<std::complex<double>>& a, bool inverse);
};
