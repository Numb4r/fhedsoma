// test_stress_aggregation.cpp — Large-scale FL stress tests + overflow-boundary sweep.
//
// Section 1 — multi-config realistic FedAvg
//   Exercises N ∈ {1024, 4096} with hundreds of clients and all N/2 slots filled.
//   Each client pre-scales its weights by 1/NUM_CLIENTS so the homomorphic sum
//   equals the canonical FedAvg without any server-side scalar multiply.
//
// Section 2 — overflow budget checks
//   For each config, verify that max_aggregate_coeff (= Δ after pre-scaling) is
//   safely below Q/2.  This generalises test_overflow_budget() from
//   test_fl_aggregation.cpp to the larger parameter sets.
//
// Section 3 — failure-boundary sweep (N=64, no pre-scaling)
//   To map the overflow envelope we intentionally omit client-side 1/K scaling:
//   each client contributes up to Δ per coefficient, so the aggregate grows as K·Δ.
//   The theoretical boundary is K_OVERFLOW = Q / (2·Δ) ≈ 1920 for the N=64 prime.
//   We sweep K ∈ {100, 500, 1000, 1500, 1800, 1900} and record the largest K at which
//   every decoded slot matches the plaintext sum within ERROR_TOL = 0.1.
//   The positive assertion is CHECK(k_max_safe >= K_SAFE) with K_SAFE = 1500, which
//   documents that the system handles 1500 unscaled additions without overflow.

#include "decryptor.h"
#include "encoder.h"
#include "encryptor.h"
#include "evaluator.h"
#include "keygen.h"
#include "math_utils.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

// ── Assertion harness (identical to every other test in this suite) ──────────
static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { ++g_fail; \
               std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
    } while (0)

// ── Shared constants ──────────────────────────────────────────────────────────
static constexpr double SCALE      = double(1ULL << 40);  // Δ = 2^40
static constexpr double SIGMA      = 3.2;
static constexpr double SLOT_TOL   = 1e-3;   // per-slot FedAvg error tolerance

// ── Section 1 + 2: realistic FedAvg for one parameter set ────────────────────
struct Config {
    uint32_t N;
    uint32_t bits;
    uint32_t num_clients;
};

static void test_fedavg_config(const Config& cfg) {
    const uint64_t q     = generate_proth_prime(cfg.bits, cfg.N);
    const uint32_t slots = cfg.N / 2;

    std::fprintf(stderr,
        "  [N=%u, bits=%u, clients=%u, slots=%u]\n"
        "      q = %llu  (Q/2 = %.3e  safety = %.0f×)\n",
        cfg.N, cfg.bits, cfg.num_clients, slots,
        (unsigned long long)q,
        static_cast<double>(q) / 2.0,
        (static_cast<double>(q) / 2.0) / SCALE);

    // ── Overflow budget check (Section 2) ───────────────────────────────────
    // After client-side 1/K pre-scaling the aggregate coefficient ≤ Δ regardless
    // of how many clients participate.
    CHECK(SCALE < static_cast<double>(q) / 2.0);  // 1 check per config

    // ── Key infrastructure — deterministic seeds for reproducibility ─────────
    KeyGenerator keygen(cfg.N, q, SIGMA, /*seed=*/1);
    SecretKey    sk  = keygen.gen_secret_key();
    PublicKey    pk  = keygen.gen_public_key(sk);
    Encoder      enc(cfg.N, q, SCALE);
    Encryptor    encryptor(cfg.N, q, pk, SIGMA, /*seed=*/2);
    Decryptor    decryptor(cfg.N, q, sk);
    Evaluator    eval(cfg.N, q);

    // ── Generate client weight vectors ───────────────────────────────────────
    std::mt19937_64 data_rng(42);
    std::uniform_real_distribution<double> udist(-1.0, 1.0);

    std::vector<std::vector<double>> weights(cfg.num_clients,
                                             std::vector<double>(slots));
    for (auto& w : weights)
        for (double& x : w)
            x = udist(data_rng);

    // Canonical FedAvg reference (plaintext)
    std::vector<double> fedavg(slots, 0.0);
    for (const auto& w : weights)
        for (uint32_t i = 0; i < slots; ++i)
            fedavg[i] += w[i];
    for (double& v : fedavg)
        v /= static_cast<double>(cfg.num_clients);

    // ── Encrypt and aggregate ────────────────────────────────────────────────
    const double inv_n = 1.0 / static_cast<double>(cfg.num_clients);
    std::vector<double> scaled(slots);

    for (uint32_t i = 0; i < slots; ++i) scaled[i] = weights[0][i] * inv_n;
    Ciphertext agg = encryptor.encrypt(Plaintext(enc.encode(scaled)));

    for (uint32_t c = 1; c < cfg.num_clients; ++c) {
        for (uint32_t i = 0; i < slots; ++i) scaled[i] = weights[c][i] * inv_n;
        eval.add_inplace(agg, encryptor.encrypt(Plaintext(enc.encode(scaled))));
    }

    // ── Decrypt and check precision ──────────────────────────────────────────
    const auto result = enc.decode(decryptor.decrypt(agg).poly);

    double max_real_err = 0.0;
    double max_imag_err = 0.0;
    for (uint32_t i = 0; i < slots; ++i) {
        const double err_re = std::abs(result[i].real() - fedavg[i]);
        const double err_im = std::abs(result[i].imag());
        max_real_err = std::max(max_real_err, err_re);
        max_imag_err = std::max(max_imag_err, err_im);
        CHECK(err_re < SLOT_TOL);   // slots checks
        CHECK(err_im < SLOT_TOL);   // slots checks
    }

    // Sanity: FedAvg of U(-1,1) over many clients is non-trivially non-zero.
    double max_signal = 0.0;
    for (double v : fedavg)
        max_signal = std::max(max_signal, std::abs(v));
    CHECK(max_signal > 1e-3);  // 1 check

    std::fprintf(stderr,
        "      max_real_err = %.3e   max_imag_err = %.3e   (tol %.3e)\n",
        max_real_err, max_imag_err, SLOT_TOL);
}

// ── Section 3: overflow-boundary sweep (N=64, no pre-scaling) ────────────────
//
// Without 1/K client-side scaling, each encrypted weight contributes up to Δ per
// coefficient.  The aggregate sum grows as K·Δ and overflows Q/2 near K ≈ 1920.
// We sweep K values, compare the decoded homomorphic sum against the plaintext sum,
// and record the largest K at which every slot's error is below ERROR_TOL.
//
// SIGNED overflow error is catastrophic (≫ 1.0 decoded units) so ERROR_TOL = 0.1
// cleanly separates the safe from the broken regime.

static void test_overflow_boundary_sweep() {
    static constexpr uint32_t N_SW    = 64;
    static constexpr uint64_t Q_SW    = 4222124650659841ULL;  // 15·2^48+1, ≡1 (mod 128)
    static constexpr uint32_t SLOTS_SW = N_SW / 2;           // 32 slots
    static constexpr double   ERROR_TOL = 0.1;
    static constexpr uint32_t K_SAFE   = 1500;  // minimum safe client count

    // Theoretical overflow boundary: K_OVF = Q / (2·Δ) ≈ 1920.
    const double k_ovf_theory = static_cast<double>(Q_SW) / (2.0 * SCALE);
    std::fprintf(stderr,
        "\n  [Overflow-boundary sweep  N=%u, Q=15·2^48+1]\n"
        "      Theoretical K_overflow = Q/(2·Δ) ≈ %.0f\n",
        N_SW, k_ovf_theory);

    KeyGenerator keygen(N_SW, Q_SW, SIGMA, /*seed=*/7);
    SecretKey    sk  = keygen.gen_secret_key();
    PublicKey    pk  = keygen.gen_public_key(sk);
    Encoder      enc(N_SW, Q_SW, SCALE);
    Encryptor    encryptor(N_SW, Q_SW, pk, SIGMA, /*seed=*/8);
    Decryptor    decryptor(N_SW, Q_SW, sk);
    Evaluator    eval(N_SW, Q_SW);

    // Sweep K values — stay deliberately close to the theoretical boundary.
    const std::vector<uint32_t> sweep = {100, 500, 1000, 1500, 1800, 1900};
    uint32_t k_max_safe = 0;

    for (uint32_t K : sweep) {
        std::mt19937_64 rng(K);  // different seed per K for variety
        std::uniform_real_distribution<double> udist(-1.0, 1.0);

        // Generate K weight vectors and the plaintext reference sum.
        std::vector<std::vector<double>> weights(K, std::vector<double>(SLOTS_SW));
        for (auto& w : weights) for (double& x : w) x = udist(rng);

        std::vector<double> pt_sum(SLOTS_SW, 0.0);
        for (const auto& w : weights)
            for (uint32_t i = 0; i < SLOTS_SW; ++i)
                pt_sum[i] += w[i];

        // Encrypt and sum — NO pre-scaling.
        Ciphertext agg = encryptor.encrypt(Plaintext(enc.encode(weights[0])));
        for (uint32_t c = 1; c < K; ++c)
            eval.add_inplace(agg, encryptor.encrypt(Plaintext(enc.encode(weights[c]))));

        const auto decoded = enc.decode(decryptor.decrypt(agg).poly);

        double max_err = 0.0;
        for (uint32_t i = 0; i < SLOTS_SW; ++i)
            max_err = std::max(max_err,
                               std::abs(decoded[i].real() - pt_sum[i]));

        const bool safe = (max_err < ERROR_TOL);
        if (safe) k_max_safe = K;

        std::fprintf(stderr,
            "      K = %4u  max_err = %.3e  overflow_ratio = %.3f  %s\n",
            K,
            max_err,
            static_cast<double>(K) * SCALE / (static_cast<double>(Q_SW) / 2.0),
            safe ? "OK" : "OVERFLOW");
    }

    // The system must remain correct at K = K_SAFE (= 1500) unscaled additions.
    CHECK(k_max_safe >= K_SAFE);  // 1 check

    std::fprintf(stderr,
        "      k_max_safe = %u  (K_SAFE = %u)\n", k_max_safe, K_SAFE);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    // Configs: (N, prime_bits, num_clients)
    const std::vector<Config> configs = {
        {1024, 52, 200},
        {4096, 52, 100},
    };

    std::fprintf(stderr,
        "\n=== Section 1+2: Multi-config realistic FedAvg ===\n");
    for (const auto& cfg : configs)
        test_fedavg_config(cfg);

    std::fprintf(stderr,
        "\n=== Section 3: Overflow-boundary sweep ===\n");
    test_overflow_boundary_sweep();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
