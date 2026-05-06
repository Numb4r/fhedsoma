#include "keygen.h"
#include "math_utils.h"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

KeyGenerator::KeyGenerator(uint32_t N, uint64_t q, double sigma, uint64_t seed)
    : N_(N), q_(q), sigma_(sigma)
    , sampler_(seed != 0 ? Sampler(seed) : Sampler())
    , ntt_ctx_(N, q)
{}

// ---------------------------------------------------------------------------
// NTT-based negacyclic polynomial multiplication
//
// Computes a·b mod (X^N + 1, q) in O(N log N):
//   1. Forward NTT both operands (coefficient → evaluation form).
//   2. Pointwise product mod q in evaluation form.
//   3. Inverse NTT (evaluation → coefficient form) with built-in 1/N scaling.
//
// All arithmetic is exact modular integer arithmetic — no floating-point error.
// ---------------------------------------------------------------------------

Poly KeyGenerator::ntt_mul(const Poly& a, const Poly& b) const {
    std::vector<uint64_t> a_hat = a.coeffs;
    std::vector<uint64_t> b_hat = b.coeffs;

    ntt_forward(a_hat, ntt_ctx_);
    ntt_forward(b_hat, ntt_ctx_);

    std::vector<uint64_t> c_hat(N_);
    for (uint32_t i = 0; i < N_; ++i)
        c_hat[i] = static_cast<uint64_t>((__uint128_t)a_hat[i] * b_hat[i] % q_);

    ntt_inverse(c_hat, ntt_ctx_);
    return Poly(N_, q_, std::move(c_hat));
}

// ---------------------------------------------------------------------------
// Secret key
// ---------------------------------------------------------------------------

SecretKey KeyGenerator::gen_secret_key(uint32_t hamming_weight) {
    Poly s = hamming_weight == 0
        ? sampler_.ternary_uniform(N_, q_)
        : sampler_.ternary_hw(N_, q_, hamming_weight);
    return SecretKey(std::move(s));
}

// ---------------------------------------------------------------------------
// Public key (shared core)
//
// Generates:
//   a   ← uniform(Z_q[X]/(X^N+1))
//   e   ← DiscreteGaussian(sigma)
//   pk1 = a
//   pk0 = e − a·s  mod q
//
// The ring identity  pk0 + pk1·s = (e − a·s) + a·s = e  holds exactly.
// ---------------------------------------------------------------------------

std::pair<PublicKey, Poly> KeyGenerator::make_public_key_(const SecretKey& sk) {
    Poly a = sampler_.uniform(N_, q_);
    Poly e = sampler_.gaussian(N_, q_, sigma_);

    // a·s  mod (X^N+1, q)  — O(N log N) via NTT
    Poly as = ntt_mul(a, sk.s);

    // pk0 = e − a·s  (coefficient-wise mod_sub so result stays in [0, q))
    Poly pk0(N_, q_);
    for (uint32_t i = 0; i < N_; ++i)
        pk0.coeffs[i] = mod_sub(e.coeffs[i], as.coeffs[i], q_);

    return { PublicKey(std::move(pk0), std::move(a)), std::move(e) };
}

PublicKey KeyGenerator::gen_public_key(const SecretKey& sk) {
    return make_public_key_(sk).first;
}

KeyGenerator::KeyBundle KeyGenerator::gen_key_bundle(uint32_t hamming_weight) {
    SecretKey sk = gen_secret_key(hamming_weight);
    auto [pk, e] = make_public_key_(sk);
    return { std::move(sk), std::move(pk), std::move(e) };
}
