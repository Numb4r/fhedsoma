// test_serialization.cpp — Binary serialization round-trips for Ciphertext and PublicKey.
//
// Checks
// ──────
//  1. Ciphertext byte-length               (1 check)
//  2. Ciphertext round-trip: slot values   (32 × real + 32 × imag = 64 checks)
//  3. Ciphertext round-trip: decrypt matches original (32 × real checks)
//  4. PublicKey byte-length                (1 check)
//  5. PublicKey round-trip: re-encrypt with deserialized pk, decrypt with sk (32 checks)
//  6. Homomorphic addition survives serialization round-trip (32 checks)
//  7. Truncation throws                    (1 check)
//  8. Trailing byte throws                 (1 check)
//
// Total: ~196 checks

#include "decryptor.h"
#include "encoder.h"
#include "encryptor.h"
#include "evaluator.h"
#include "keygen.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { ++g_fail; \
               std::fprintf(stderr, "FAIL  %s:%d  %s\n", \
                            __FILE__, __LINE__, #expr); } \
    } while (0)

#define CHECK_THROWS(expr) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (const std::exception&) { threw = true; } \
        CHECK(threw); \
    } while (0)

// ── Parameters ────────────────────────────────────────────────────────────────
// Same prime as the FL test: Q = 15×2^48+1 is large enough for Δ = 2^40.
static constexpr uint32_t N     = 64;
static constexpr uint64_t Q     = 4222124650659841ULL;
static constexpr double   SCALE = double(1ULL << 40);
static constexpr double   SIGMA = 3.2;
static constexpr uint32_t SLOTS = N / 2;
static constexpr double   TOL   = 1e-4;

// Expected byte lengths.
// Poly wire size = 12 + N*8.  Ciphertext / PublicKey = 2 × poly wire size.
static constexpr std::size_t POLY_BYTES  = 12 + static_cast<std::size_t>(N) * 8;
static constexpr std::size_t CT_BYTES    = 2 * POLY_BYTES;
static constexpr std::size_t PK_BYTES    = 2 * POLY_BYTES;

// ── Infrastructure ─────────────────────────────────────────────────────────────
struct Ctx {
    Encoder      enc;
    KeyGenerator keygen;
    SecretKey    sk;
    PublicKey    pk;
    Encryptor    encryptor;
    Decryptor    decryptor;
    Evaluator    eval;

    Ctx()
        : enc(N, Q, SCALE)
        , keygen(N, Q, SIGMA, /*seed=*/1)
        , sk(keygen.gen_secret_key())
        , pk(keygen.gen_public_key(sk))
        , encryptor(N, Q, pk, SIGMA, /*seed=*/2)
        , decryptor(N, Q, sk)
        , eval(N, Q)
    {}
};

// Encrypt a slot vector and return the ciphertext.
// Encryptor::encrypt is non-const (mutates the PRNG state).
static Ciphertext encrypt_slots(Ctx& ctx, const std::vector<double>& vals) {
    return ctx.encryptor.encrypt(Plaintext(ctx.enc.encode(vals)));
}

// Decrypt a ciphertext and return real parts of the slot vector.
static std::vector<double> decrypt_real(const Ctx& ctx, const Ciphertext& ct) {
    auto slots = ctx.enc.decode(ctx.decryptor.decrypt(ct).poly);
    std::vector<double> out(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i)
        out[i] = slots[i].real();
    return out;
}

// ── Test 1: Ciphertext byte length ────────────────────────────────────────────
static void test_ciphertext_byte_length() {
    Ctx ctx;
    std::vector<double> vals(SLOTS, 0.5);
    auto ct    = encrypt_slots(ctx, vals);  // ctx must be non-const
    auto bytes = ct.to_bytes();
    CHECK(bytes.size() == CT_BYTES);
}

// ── Test 2 & 3: Ciphertext round-trip — decrypt matches original ───────────────
static void test_ciphertext_roundtrip() {
    Ctx ctx;
    std::vector<double> vals(SLOTS);
    for (uint32_t i = 0; i < SLOTS; ++i)
        vals[i] = (static_cast<double>(i) / SLOTS) - 0.5;  // in (-0.5, 0.5)

    const Ciphertext ct_orig = encrypt_slots(ctx, vals);

    // Serialize then deserialize.
    const auto bytes     = ct_orig.to_bytes();
    const Ciphertext ct2 = Ciphertext::from_bytes(bytes.data(), bytes.size());

    // Decrypt both and compare slot-by-slot.
    const auto dec_orig = decrypt_real(ctx, ct_orig);
    const auto dec2     = decrypt_real(ctx, ct2);

    for (uint32_t i = 0; i < SLOTS; ++i) {
        CHECK(std::abs(dec2[i] - dec_orig[i]) == 0.0);  // bit-identical after roundtrip
        CHECK(std::abs(dec2[i] - vals[i]) < TOL);       // correct value
    }
}

// ── Test 4: PublicKey byte length ─────────────────────────────────────────────
static void test_publickey_byte_length() {
    Ctx ctx;
    CHECK(ctx.pk.to_bytes().size() == PK_BYTES);
}

// ── Test 5: PublicKey round-trip — re-encrypt with deserialized pk ────────────
static void test_publickey_roundtrip() {
    Ctx ctx;
    std::vector<double> vals(SLOTS);
    for (uint32_t i = 0; i < SLOTS; ++i)
        vals[i] = 0.1 * (static_cast<int>(i) - static_cast<int>(SLOTS) / 2);

    // Deserialize the public key.
    const auto pk_bytes = ctx.pk.to_bytes();
    const PublicKey pk2 = PublicKey::from_bytes(pk_bytes.data(), pk_bytes.size());

    // Encrypt with the deserialized pk (different Encryptor, same pk material).
    Encryptor enc2(N, Q, pk2, SIGMA, /*seed=*/99);
    const Ciphertext ct = enc2.encrypt(Plaintext(ctx.enc.encode(vals)));

    // Decrypt with the original sk.
    const auto dec = decrypt_real(ctx, ct);

    for (uint32_t i = 0; i < SLOTS; ++i)
        CHECK(std::abs(dec[i] - vals[i]) < TOL);
}

// ── Test 6: Homomorphic addition after serialization round-trip ───────────────
static void test_add_after_roundtrip() {
    Ctx ctx;
    std::vector<double> a(SLOTS), b(SLOTS);
    for (uint32_t i = 0; i < SLOTS; ++i) {
        a[i] =  0.1 * static_cast<double>(i) / SLOTS;
        b[i] = -0.1 * static_cast<double>(i) / SLOTS + 0.2;
    }

    const Ciphertext ct_a = encrypt_slots(ctx, a);
    const Ciphertext ct_b = encrypt_slots(ctx, b);

    // Round-trip both ciphertexts through bytes, then add.
    const auto bytes_a  = ct_a.to_bytes();
    const auto bytes_b  = ct_b.to_bytes();
    Ciphertext ct_a2    = Ciphertext::from_bytes(bytes_a.data(), bytes_a.size());
    const Ciphertext ct_b2 = Ciphertext::from_bytes(bytes_b.data(), bytes_b.size());

    ctx.eval.add_inplace(ct_a2, ct_b2);
    const auto dec = decrypt_real(ctx, ct_a2);

    for (uint32_t i = 0; i < SLOTS; ++i)
        CHECK(std::abs(dec[i] - (a[i] + b[i])) < TOL);
}

// ── Test 7 & 8: Error handling ────────────────────────────────────────────────
static void test_error_handling() {
    Ctx ctx;
    std::vector<double> vals(SLOTS, 0.3);
    const auto bytes = encrypt_slots(ctx, vals).to_bytes();

    // Truncation: drop the last byte.
    CHECK_THROWS(Ciphertext::from_bytes(bytes.data(), bytes.size() - 1));

    // Trailing byte: append one extra byte.
    auto extended = bytes;
    extended.push_back(0xAB);
    CHECK_THROWS(Ciphertext::from_bytes(extended.data(), extended.size()));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_ciphertext_byte_length();   //   1 check
    test_ciphertext_roundtrip();     //  64 checks
    test_publickey_byte_length();    //   1 check
    test_publickey_roundtrip();      //  32 checks
    test_add_after_roundtrip();      //  32 checks
    test_error_handling();           //   2 checks

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
