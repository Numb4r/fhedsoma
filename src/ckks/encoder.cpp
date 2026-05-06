#include "encoder.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// In-place Cooley–Tukey radix-2 DIT FFT.
//
// inverse=false : X[k] = Σ_j x[j] · e^{−2πijk/N}   (standard DFT, no /N)
// inverse=true  : x[j] = (1/N) Σ_k X[k] · e^{+2πijk/N}  (IDFT, normalised)
//
// Both directions use the same butterfly structure; the only differences are
// the sign of the twiddle angle and the final 1/N rescaling.
// ---------------------------------------------------------------------------
void Encoder::fft(std::vector<std::complex<double>>& a, bool inverse) {
    const uint32_t N = static_cast<uint32_t>(a.size());

    // Bit-reversal permutation.
    for (uint32_t i = 1, j = 0; i < N; ++i) {
        uint32_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Butterfly stages.
    for (uint32_t len = 2; len <= N; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inverse ? 1.0 : -1.0);
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (uint32_t i = 0; i < N; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (uint32_t j = 0; j < len / 2; ++j) {
                auto u = a[i + j];
                auto v = a[i + j + len / 2] * w;
                a[i + j]             = u + v;
                a[i + j + len / 2]   = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        double inv_n = 1.0 / N;
        for (auto& x : a) x *= inv_n;
    }
}

// ---------------------------------------------------------------------------
// Constructor — validate parameters and precompute twiddle tables.
// ---------------------------------------------------------------------------
Encoder::Encoder(uint32_t N, uint64_t q, double scale)
    : N_(N), q_(q), scale_(scale)
{
    if (N < 2 || (N & (N - 1)) != 0)
        throw std::invalid_argument("N must be a power of 2 and >= 2");
    if (scale <= 0.0)
        throw std::invalid_argument("scale must be positive");

    zeta_.resize(N);
    zeta_inv_.resize(N);
    for (uint32_t k = 0; k < N; ++k) {
        double theta  = M_PI * k / N;
        zeta_[k]      = {std::cos(theta),  std::sin(theta)};
        zeta_inv_[k]  = {std::cos(theta), -std::sin(theta)};
    }
}

// ---------------------------------------------------------------------------
// encode — slot vector z ∈ C^{N/2}  →  polynomial in Z_q[X]/(X^N+1)
//
// The inverse canonical embedding σ^{-1} is:
//   1. Lift z to a conjugate-symmetric N-vector  v[j] = z[j] for j < N/2,
//      v[N−1−j] = conj(z[j]) for j < N/2.
//   2. Apply the forward FFT:  V[k] = Σ_j v[j] · e^{−2πijk/N}
//   3. Recover polynomial coefficient:  a_k = V[k] · ζ^{−k} / N
//   4. Scale, round, and reduce mod q.
//
// Because v is conjugate-symmetric, Im(a_k) ≈ 0 up to floating-point noise;
// we take only the real part before rounding.
// ---------------------------------------------------------------------------
Poly Encoder::encode(const std::vector<std::complex<double>>& z) const {
    const uint32_t m = N_ / 2;
    if (z.size() > m)
        throw std::invalid_argument(
            "input length exceeds N/2 = " + std::to_string(m) + " slots");

    // Step 1 — build conjugate-symmetric N-vector (zero-pad missing slots).
    std::vector<std::complex<double>> v(N_, {0.0, 0.0});
    for (uint32_t i = 0; i < static_cast<uint32_t>(z.size()); ++i)
        v[i] = z[i];
    for (uint32_t i = 0; i < m; ++i)
        v[N_ - 1 - i] = std::conj(v[i]);

    // Step 2 — forward FFT.
    fft(v, /*inverse=*/false);

    // Step 3 & 4 — twist, normalise, scale, round, reduce mod q.
    const int64_t sq  = static_cast<int64_t>(q_);
    const double  inv = 1.0 / static_cast<double>(N_);

    Poly out(N_, q_);
    for (uint32_t k = 0; k < N_; ++k) {
        double a_k = (v[k] * zeta_inv_[k]).real() * inv;
        int64_t c  = std::llround(scale_ * a_k);
        // Lift negatives into [0, q).
        int64_t r  = c % sq;
        if (r < 0) r += sq;
        out.coeffs[k] = static_cast<uint64_t>(r);
    }
    return out;
}

Poly Encoder::encode(const std::vector<double>& z) const {
    std::vector<std::complex<double>> zc(z.size());
    for (std::size_t i = 0; i < z.size(); ++i)
        zc[i] = {z[i], 0.0};
    return encode(zc);
}

// ---------------------------------------------------------------------------
// decode — polynomial p ∈ Z_q[X]/(X^N+1)  →  slot vector z ∈ C^{N/2}
//
// The canonical embedding σ is:
//   1. Centre coefficients into (−q/2, q/2] and divide by scale.
//   2. Twist:  b_k = a_k · ζ^k  where ζ^k = exp(+iπk/N).
//   3. Unnormalised IFFT:  v = N · IFFT(b)  =  Σ_k b_k · e^{+2πijk/N}.
//   4. Return the first N/2 entries (the N/2 independent slot values).
// ---------------------------------------------------------------------------
std::vector<std::complex<double>> Encoder::decode(const Poly& p) const {
    if (p.N != N_ || p.q != q_)
        throw std::invalid_argument("Poly parameters do not match Encoder");

    const int64_t sq     = static_cast<int64_t>(q_);
    const int64_t half_q = sq / 2;

    // Step 1 & 2 — centre, unscale, twist.
    std::vector<std::complex<double>> b(N_);
    for (uint32_t k = 0; k < N_; ++k) {
        int64_t c = static_cast<int64_t>(p.coeffs[k]);
        if (c > half_q) c -= sq;
        double  a_k = static_cast<double>(c) / scale_;
        b[k] = a_k * zeta_[k];
    }

    // Step 3 — IFFT then multiply by N to undo the 1/N normalisation.
    fft(b, /*inverse=*/true);
    const double nd = static_cast<double>(N_);
    for (auto& x : b) x *= nd;

    // Step 4 — return the N/2 independent slots.
    return {b.begin(), b.begin() + N_ / 2};
}
