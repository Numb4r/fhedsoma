// test_fl_aggregation.cpp — Canonical FedAvg simulation (no server-side scaling).
//
// FedAvg protocol
// ───────────────
//   1. Each of 90 clients holds a private local-weights vector w_c ∈ (−1,1)^32.
//   2. Each client independently computes the averaging step in plaintext:
//        v_c[j] = w_c[j] / NUM_CLIENTS          ← client-side 1/90 scaling
//      encodes v_c with Δ = 2^40, and encrypts under the shared public key.
//   3. The server is a pure accumulator — it ONLY calls Evaluator::add_inplace:
//        ct_agg = ct_0 + ct_1 + … + ct_89
//   4. The result is decrypted and decoded by the server (or a trusted party).
//   5. Every decoded slot must match the canonical FedAvg value
//        fedavg[j] = (w_0[j] + w_1[j] + … + w_89[j]) / 90
//      within SLOT_TOL = 1e-4.
//
// Why there is no server-side scaling
// ─────────────────────────────────────
//   Distributing the 1/N division to the clients means the server performs no
//   scalar multiplication on ciphertexts.  It cannot distinguish one client's
//   contribution from another, and the homomorphic sum automatically equals the
//   FedAvg result without any post-hoc rescaling step.
//
// Prime selection — Q = 15 × 2^48 + 1 = 4 222 124 650 659 841
// ─────────────────────────────────────────────────────────────
//  • 52-bit verified prime; Q ≡ 1 (mod 128) → NTT works for N = 64.
//  • Q − 1 = 3 × 5 × 2^48: prime_factors() terminates in ~50 trial-divisions.
//  • MontgomeryCtx requires q < 2^62 — satisfied (Q < 2^52).
//
// Overflow budget
// ───────────────
//   Client-side pre-scaling by 1/90 means each client contributes at most
//   Δ/90 per polynomial coefficient.  Summing 90 such contributions yields at
//   most Δ per coefficient — far tighter than unscaled summation:
//
//       max encoded coeff per client  = Δ × (1/90) × max|w| ≈ 1.2 × 10^10
//       max aggregate coeff (sum 90)  = Δ                   ≈ 1.1 × 10^12
//       Q/2                                                  ≈ 2.1 × 10^15
//       safety factor  Q/2 / Δ ≈ 1 920×  (vs. 21× for unscaled summation)
//
// Noise budget
// ─────────────
//   Encryption noise is independent of the plaintext magnitude.
//   Per fresh ciphertext (6σ tail, N = 64, σ = 3.2):
//     |u·e_key|_∞ + |e1·s|_∞ + |e0|_∞ ≤ 2·N·6σ + 6σ ≈ 2 477
//   After 90 additions:                                ≤ 222 930
//   Per-slot decoding error: 222 930 / Δ ≈ 2.0 × 10^-7
//   SLOT_TOL = 1e-4 provides a ~493× safety margin.

#include "decryptor.h"
#include "encoder.h"
#include "encryptor.h"
#include "evaluator.h"
#include "keygen.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { ++g_fail; \
               std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
    } while (0)

// ── Parameters ────────────────────────────────────────────────────────────────
static constexpr uint32_t N           = 64;
static constexpr uint64_t Q           = 4222124650659841ULL;  // 15 * 2^48 + 1
static constexpr double   SCALE       = double(1ULL << 40);   // Δ = 2^40
static constexpr double   SIGMA       = 3.2;
static constexpr uint32_t NUM_CLIENTS = 90;
static constexpr uint32_t SLOTS       = N / 2;    // 32 real-valued slots
static constexpr double   SLOT_TOL    = 1e-4;

// ── Overflow budget check ─────────────────────────────────────────────────────
// After client-side pre-scaling by 1/90, the homomorphic sum encodes the
// FedAvg, whose coefficients are bounded by Δ (not 90×Δ).
// This check fails loudly if the parameters are changed to an unsafe combination.
static void test_overflow_budget() {
    // Max aggregate polynomial coefficient = Δ × max|w| × 1 = Δ × 1.0.
    // (The 1/90 scaling on each client exactly cancels the ×90 from summation.)
    const double max_aggregate_coeff = SCALE;
    const double half_q              = static_cast<double>(Q) / 2.0;
    const double safety_factor       = half_q / max_aggregate_coeff;

    std::fprintf(stderr,
        "      Q/2 = %.3e   max_agg_coeff = Δ = %.3e   safety_factor = %.0f×\n",
        half_q, max_aggregate_coeff, safety_factor);

    CHECK(max_aggregate_coeff < half_q);  // 1 check
}

// ── Canonical FedAvg precision test ──────────────────────────────────────────
static void test_fedavg_precision() {
    // Key infrastructure — fully deterministic for reproducibility.
    KeyGenerator keygen(N, Q, SIGMA, /*seed=*/1);
    SecretKey    sk = keygen.gen_secret_key();
    PublicKey    pk = keygen.gen_public_key(sk);
    Encoder      enc(N, Q, SCALE);
    Encryptor    encryptor(N, Q, pk, SIGMA, /*seed=*/2);
    Decryptor    decryptor(N, Q, sk);
    Evaluator    eval(N, Q);

    // Generate 90 local-weights vectors — the unscaled private client data.
    std::mt19937_64 data_rng(42);
    std::uniform_real_distribution<double> udist(-1.0, 1.0);

    std::vector<std::vector<double>> weights(NUM_CLIENTS,
                                             std::vector<double>(SLOTS));
    for (auto& w : weights)
        for (double& x : w)
            x = udist(data_rng);

    // Canonical FedAvg reference (plaintext, computed exactly).
    //   fedavg[j] = (w_0[j] + w_1[j] + … + w_89[j]) / 90
    std::vector<double> fedavg(SLOTS, 0.0);
    for (const auto& w : weights)
        for (uint32_t i = 0; i < SLOTS; ++i)
            fedavg[i] += w[i];
    for (double& v : fedavg)
        v /= static_cast<double>(NUM_CLIENTS);

    // Each client scales by 1/NUM_CLIENTS in plaintext, then encodes + encrypts.
    // The server exclusively uses add_inplace — no scalar multiplication.
    const double inv_n = 1.0 / static_cast<double>(NUM_CLIENTS);

    // Reuse a single scratch buffer to avoid NUM_CLIENTS extra heap allocations.
    std::vector<double> scaled(SLOTS);

    for (uint32_t i = 0; i < SLOTS; ++i) scaled[i] = weights[0][i] * inv_n;
    Ciphertext agg = encryptor.encrypt(Plaintext(enc.encode(scaled)));

    for (uint32_t c = 1; c < NUM_CLIENTS; ++c) {
        for (uint32_t i = 0; i < SLOTS; ++i) scaled[i] = weights[c][i] * inv_n;
        eval.add_inplace(agg,
            encryptor.encrypt(Plaintext(enc.encode(scaled))));
    }

    // Decrypt and decode: the result must approximate fedavg[].
    const auto result = enc.decode(decryptor.decrypt(agg).poly);

    // Per-slot precision against the canonical FedAvg reference.
    double max_real_err = 0.0;
    double max_imag_err = 0.0;
    for (uint32_t i = 0; i < SLOTS; ++i) {
        const double err_re = std::abs(result[i].real() - fedavg[i]);
        const double err_im = std::abs(result[i].imag());
        max_real_err = std::max(max_real_err, err_re);
        max_imag_err = std::max(max_imag_err, err_im);
        CHECK(err_re < SLOT_TOL);   // 32 checks — real part matches FedAvg
        CHECK(err_im < SLOT_TOL);   // 32 checks — imaginary part ≈ 0 (real inputs)
    }

    std::fprintf(stderr,
        "      FedAvg (90 clients, Δ=2^40, client-side 1/90):  "
        "max_real_err = %.2e   max_imag_err = %.2e   (tol %.2e)\n",
        max_real_err, max_imag_err, SLOT_TOL);

    // Sanity: fedavg[j] ~ N(0, Var[U]/90).  std ≈ 0.061 per slot; the maximum
    // over 32 independent slots is ~0.16 — essentially never below 0.01.
    double max_signal = 0.0;
    for (double v : fedavg)
        max_signal = std::max(max_signal, std::abs(v));
    CHECK(max_signal > 0.01);   // 1 check — FedAvg is non-trivially non-zero
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_overflow_budget();   //  1 check
    test_fedavg_precision();  // 65 checks
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
