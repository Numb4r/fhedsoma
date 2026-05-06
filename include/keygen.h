#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "ntt.h"
#include "poly.h"
#include "sampler.h"

// ---------------------------------------------------------------------------
// CKKS key types
// ---------------------------------------------------------------------------

// Secret key: ternary polynomial s ∈ {-1, 0, 1}^N.
// Coefficients are stored as [0, q) residues: -1 is represented as q-1.
//
// Binary wire format (to_bytes / from_bytes):
//   [s: poly_header || coeffs]
// where poly_header = [N: uint32 LE (4 B)][q: uint64 LE (8 B)]
// and each coeff is uint64 LE (8 B).  Total: 12 + N×8 bytes.
struct SecretKey {
    Poly s;
    explicit SecretKey(Poly s_) : s(std::move(s_)) {}

    // Serialize to a flat byte buffer (little-endian, self-describing).
    std::vector<uint8_t> to_bytes() const;

    // Deserialize from raw bytes.  Throws std::runtime_error on truncation
    // or trailing bytes.
    static SecretKey from_bytes(const uint8_t* data, std::size_t size);
};

// Public key: pair (pk0, pk1) satisfying  pk0 + pk1·s = e  (mod q)
// where e is a small Gaussian error and s is the matching secret key.
//
//   pk1 = a               (uniform random in Z_q[X]/(X^N+1))
//   pk0 = -a·s + e        (hides s behind the error)
//
// Any party holding pk can encrypt; only the holder of sk can decrypt.
//
// Binary wire format (to_bytes / from_bytes):
//   [pk0: poly_header || coeffs][pk1: poly_header || coeffs]
// where poly_header = [N: uint32 LE (4 B)][q: uint64 LE (8 B)]
// and each coeff is uint64 LE (8 B).  Total: 2 × (12 + N×8) bytes.
struct PublicKey {
    Poly pk0;   // = -a·s + e
    Poly pk1;   // = a  (uniform)
    PublicKey(Poly pk0_, Poly pk1_)
        : pk0(std::move(pk0_)), pk1(std::move(pk1_)) {}

    // Serialize to a flat byte buffer (little-endian, self-describing).
    std::vector<uint8_t> to_bytes() const;

    // Deserialize from raw bytes.  Throws std::runtime_error on truncation,
    // trailing bytes, or (N, q) mismatch between pk0 and pk1.
    static PublicKey from_bytes(const uint8_t* data, std::size_t size);
};

// ---------------------------------------------------------------------------
// KeyGenerator
// ---------------------------------------------------------------------------
class KeyGenerator {
public:
    // N     — ring degree (power of 2; must satisfy q ≡ 1 mod 2N)
    // q     — prime modulus
    // sigma — error standard deviation (CKKS convention: 3.2)
    // seed  — 0 uses std::random_device; any other value is deterministic
    KeyGenerator(uint32_t N, uint64_t q, double sigma = 3.2, uint64_t seed = 0);

    KeyGenerator(const KeyGenerator&)            = delete;
    KeyGenerator& operator=(const KeyGenerator&) = delete;

    // ── Key generation ────────────────────────────────────────────────────────

    // Generate a ternary secret key.
    //   hamming_weight == 0  →  uniform ternary (each coeff in {-1,0,1} w.p. 1/3)
    //   hamming_weight  > 0  →  fixed-weight (exactly hw non-zeros: hw/2 are +1,
    //                           hw/2 are -1; hw must be even and ≤ N)
    SecretKey gen_secret_key(uint32_t hamming_weight = 0);

    // Generate a public key from an existing secret key.
    // Uses one NTT-based polynomial multiplication (O(N log N)).
    PublicKey gen_public_key(const SecretKey& sk);

    // ── Test / research API ───────────────────────────────────────────────────

    // Returns the secret key, public key, AND the internal error polynomial e
    // that satisfies  pk0 + pk1·sk.s = e  exactly.
    // Never expose e in production — it reveals information about sk.
    struct KeyBundle {
        SecretKey sk;
        PublicKey pk;
        Poly      e;   // pk0 + pk1·sk.s == e  (exact modular identity)
    };
    KeyBundle gen_key_bundle(uint32_t hamming_weight = 0);

private:
    uint32_t  N_;
    uint64_t  q_;
    double    sigma_;
    Sampler   sampler_;
    NttCtx    ntt_ctx_;

    // NTT-based negacyclic multiplication: returns a·b mod (X^N+1, q).
    // O(N log N) using the precomputed twiddle tables in ntt_ctx_.
    Poly ntt_mul(const Poly& a, const Poly& b) const;

    // Shared implementation: returns {PublicKey, e}.
    std::pair<PublicKey, Poly> make_public_key_(const SecretKey& sk);
};
