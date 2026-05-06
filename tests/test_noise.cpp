// test_noise.cpp — CKKS decryption noise characterisation.
//
// This program measures the infinity norm of the noise introduced by the
// public-key encryption scheme:
//
//   noise_polynomial  =  m' − m   (coefficient-wise, centred mod q)
//   infinity_norm     =  max_i |m'_i − m_i|
//
// where m is a plaintext with known small integer coefficients, and m' is
// the polynomial recovered after a full encrypt → decrypt round-trip.
//
// The output is printed to stdout so the result can be read by scripts,
// CI systems, or the developer directly.

#include "decryptor.h"
#include "encryptor.h"
#include "keygen.h"
#include "poly.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Interpret a [0, q) residue as a signed integer in (−q/2, q/2].
// ---------------------------------------------------------------------------
static int64_t centred(uint64_t c, uint64_t q) {
    int64_t sc = static_cast<int64_t>(c);
    int64_t sq = static_cast<int64_t>(q);
    return sc > sq / 2 ? sc - sq : sc;
}

int main() {
    // ── Parameters ──────────────────────────────────────────────────────────
    const uint32_t N     = 64;
    const uint64_t Q     = 998244353ULL;   // 119·2²³+1; supports NTT for N ≤ 2²²
    const double   SIGMA = 3.2;            // standard CKKS error width
    const uint64_t KG_SEED  = 42;          // key-generation seed (deterministic)
    const uint64_t ENC_SEED = 1337;        // encryption seed    (deterministic)

    std::cout << "CKKS Decryption Noise Analysis\n"
              << "================================\n"
              << "  N     = " << N     << "\n"
              << "  q     = " << Q     << "  (998244353 = 119·2^23 + 1)\n"
              << "  sigma = " << SIGMA << "\n\n";

    // ── Step 1: generate key pair ────────────────────────────────────────────
    KeyGenerator keygen(N, Q, SIGMA, KG_SEED);
    SecretKey sk = keygen.gen_secret_key();        // uniform ternary
    PublicKey pk = keygen.gen_public_key(sk);

    // ── Step 2: build plaintext with known small integer coefficients ────────
    //
    // Coefficients cycle through {1, 2, …, 13} — small positive integers well
    // below q/2 ≈ 5·10⁸.  Because every m[i] < q/2, the centred value equals
    // m[i] exactly, making the noise calculation unambiguous.
    Poly m(N, Q);
    for (uint32_t i = 0; i < N; ++i)
        m.coeffs[i] = static_cast<uint64_t>((i % 13) + 1);   // values in [1, 13]

    std::cout << "Plaintext m: coefficients cycle through {1, …, 13} (" << N
              << " coefficients total)\n\n";

    // ── Step 3: encrypt ──────────────────────────────────────────────────────
    Encryptor encryptor(N, Q, pk, SIGMA, ENC_SEED);
    Ciphertext ct = encryptor.encrypt(Plaintext(m));

    // ── Step 4: decrypt ──────────────────────────────────────────────────────
    Decryptor decryptor(N, Q, sk);
    Poly m_prime = decryptor.decrypt(ct).poly;

    // ── Step 5: compute noise statistics ─────────────────────────────────────
    //
    // noise[i] = centred(m'[i], q) − centred(m[i], q)
    //          = centred(m'[i], q) − m[i]          (since m[i] < q/2)
    //
    // Infinity norm = max |noise[i]|.
    std::vector<int64_t> noise(N);
    for (uint32_t i = 0; i < N; ++i)
        noise[i] = centred(m_prime.coeffs[i], Q) - centred(m.coeffs[i], Q);

    const int64_t inf_norm = *std::max_element(
        noise.begin(), noise.end(),
        [](int64_t a, int64_t b) { return std::abs(a) < std::abs(b); });
    const int64_t inf_norm_abs = std::abs(inf_norm);

    const double avg_abs = [&] {
        double s = 0;
        for (auto n : noise) s += std::abs(static_cast<double>(n));
        return s / N;
    }();

    // Theoretical worst-case: |u·e_key|_∞ + |e1·s|_∞ + |e0|_∞
    //   ≤ N·(6σ) + N·(6σ) + 6σ  =  2N·6σ + 6σ
    const int64_t theoretical_bound =
        static_cast<int64_t>(std::ceil(2.0 * N * 6.0 * SIGMA + 6.0 * SIGMA));

    // ── Step 6: print results ─────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Decryption noise per coefficient (centred):\n"
              << "  Infinity norm  max|m'[i] - m[i]|  = " << inf_norm_abs       << "\n"
              << "  Average        avg|m'[i] - m[i]|  = " << avg_abs            << "\n"
              << "  Theoretical worst-case bound       = " << theoretical_bound  << "\n"
              << "  Noise / bound                      = "
              << std::setprecision(4)
              << static_cast<double>(inf_norm_abs) / theoretical_bound * 100.0
              << " %\n\n";

    std::cout << "Per-coefficient noise  [index : noise_value]:\n";
    for (uint32_t i = 0; i < N; ++i) {
        std::cout << "  [" << std::setw(2) << i << "] "
                  << std::setw(6) << noise[i] << "\n";
    }

    std::cout << "\nSummary: infinity norm of decryption noise = "
              << inf_norm_abs << "\n";

    return 0;
}
