#include "decryptor.h"
#include "encoder.h"
#include "encryptor.h"
#include "keygen.h"
#include "math_utils.h"
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { ++g_fail; \
               std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
    } while (0)
#define CHECK_EQ(a, b) CHECK((a) == (b))

// ── Shared parameters ────────────────────────────────────────────────────────

// 998244353 = 119·2²³+1  ≡ 1 (mod 2N) for all N ≤ 2²².
static constexpr uint64_t Q     = 998244353ULL;
// scale = 2^20 → max encoded coefficient ≈ 2^20.  Q/2 ≈ 5·10^8 ≫ 2^20, so
// the noise budget is > 500× the encoded signal.
static constexpr double   SCALE = 1 << 20;
// Standard CKKS error width.
static constexpr double   SIGMA = 3.2;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Centre an [0,q) residue into (−q/2, q/2].
static int64_t centre(uint64_t c, uint64_t q) {
    int64_t sc = static_cast<int64_t>(c);
    int64_t sq = static_cast<int64_t>(q);
    return sc > sq / 2 ? sc - sq : sc;
}

// Full encrypt→decrypt pipeline.  Returns the decrypted Poly.
static Poly enc_dec(uint32_t N, const Poly& pt_poly,
                    uint64_t kg_seed, uint64_t enc_seed)
{
    KeyGenerator keygen(N, Q, SIGMA, kg_seed);
    SecretKey    sk = keygen.gen_secret_key();
    PublicKey    pk = keygen.gen_public_key(sk);

    Encryptor  encryptor(N, Q, pk, SIGMA, enc_seed);
    Decryptor  decryptor(N, Q, sk);

    Ciphertext ct      = encryptor.encrypt(Plaintext(pt_poly));
    Plaintext  dec_pt  = decryptor.decrypt(ct);
    return dec_pt.poly;
}

// ── Polynomial-level noise tests ─────────────────────────────────────────────
//
// The decrypted polynomial m' must satisfy m'[i] ≈ m[i] in centred form.
// Worst-case noise bound (6σ tail, N=64):
//   |u·e_key|_∞  ≤  N · 6σ  =  64 · 20  =  1280
//   |e1·s|_∞     ≤  N · 6σ  =  1280
//   |e0|_∞       ≤  6σ       =  20
//   Total                     ≤  2580
// We allow 3000 (generous margin > 2580).

static constexpr int64_t POLY_NOISE_BOUND = 3000;

// N=64, uniform ternary secret, non-trivial plaintext.
static void test_poly_noise_n64_real() {
    const uint32_t N = 64;
    Encoder enc(N, Q, SCALE);
    Poly pt = enc.encode(std::vector<double>{1.0, -0.5, 0.25, -0.75, 0.9});

    Poly dec = enc_dec(N, pt, /*kg_seed=*/1, /*enc_seed=*/2);

    bool ok = true;
    for (uint32_t i = 0; i < N; ++i) {
        int64_t noise = std::abs(centre(dec.coeffs[i], Q) - centre(pt.coeffs[i], Q));
        if (noise > POLY_NOISE_BOUND) { ok = false; break; }
    }
    CHECK(ok);
}

// N=128, verify the bound still holds with more butterfly stages.
static void test_poly_noise_n128_real() {
    const uint32_t N = 128;
    // N=128: worst-case noise ≤ 128·20·2 + 20 = 5140 → allow 7000
    const int64_t BOUND = 7000;
    Encoder enc(N, Q, SCALE);
    Poly pt = enc.encode(std::vector<double>{0.3, -0.7, 0.6});

    Poly dec = enc_dec(N, pt, 3, 4);

    bool ok = true;
    for (uint32_t i = 0; i < N; ++i) {
        int64_t noise = std::abs(centre(dec.coeffs[i], Q) - centre(pt.coeffs[i], Q));
        if (noise > BOUND) { ok = false; break; }
    }
    CHECK(ok);
}

// ── Slot-level round-trip tests ───────────────────────────────────────────────
//
// Full pipeline: slot values → encode → encrypt → decrypt → decode → slot values.
// Tolerance accounts for both quantisation and encryption noise at decode level.
//
// Expected per-slot error:
//   |δz_j| ≤ POLY_NOISE_BOUND / SCALE + N/(2·SCALE)
//           ≤ 3000/2^20 + 32/2^20  ≈  2.9×10⁻³
// We use TOL = 1e-2 (≈ 3× the bound — reliably passes even in adversarial draws).

static constexpr double SLOT_TOL = 1e-2;

static void check_slots(const std::vector<std::complex<double>>& z,
                        const std::vector<std::complex<double>>& z_hat,
                        double tol)
{
    for (std::size_t i = 0; i < z.size(); ++i) {
        double err_re = std::abs(z_hat[i].real() - z[i].real());
        double err_im = std::abs(z_hat[i].imag() - z[i].imag());
        CHECK(err_re <= tol);
        CHECK(err_im <= tol);
    }
}

// Real input vector — the most common CKKS use case.
static void test_roundtrip_real_slots() {
    const uint32_t N = 64;
    Encoder    enc(N, Q, SCALE);
    const std::vector<double> z = {0.5, -0.3, 0.7, -0.1, 0.9, -0.8, 0.2};

    KeyGenerator keygen(N, Q, SIGMA, 10);
    SecretKey    sk = keygen.gen_secret_key();
    PublicKey    pk = keygen.gen_public_key(sk);
    Encryptor    encryptor(N, Q, pk, SIGMA, 20);
    Decryptor    decryptor(N, Q, sk);

    Plaintext pt(enc.encode(z));
    auto z_hat = enc.decode(decryptor.decrypt(encryptor.encrypt(pt)).poly);

    std::vector<std::complex<double>> zc(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) zc[i] = {z[i], 0.0};
    check_slots(zc, z_hat, SLOT_TOL);
}

// Complex input vector.
static void test_roundtrip_complex_slots() {
    const uint32_t N = 64;
    Encoder    enc(N, Q, SCALE);
    const std::vector<std::complex<double>> z = {
        {0.6, 0.4}, {-0.3, 0.8}, {0.1, -0.9}, {-0.5, 0.2}
    };

    KeyGenerator keygen(N, Q, SIGMA, 30);
    SecretKey    sk = keygen.gen_secret_key();
    PublicKey    pk = keygen.gen_public_key(sk);
    Encryptor    encryptor(N, Q, pk, SIGMA, 40);
    Decryptor    decryptor(N, Q, sk);

    Plaintext pt(enc.encode(z));
    auto z_hat = enc.decode(decryptor.decrypt(encryptor.encrypt(pt)).poly);

    check_slots(z, z_hat, SLOT_TOL);
}

// The zero polynomial encrypts and decrypts to approximately zero.
static void test_roundtrip_zero() {
    const uint32_t N = 64;
    Poly zero_pt(N, Q);  // all coefficients = 0

    Poly dec = enc_dec(N, zero_pt, 50, 60);

    for (uint32_t i = 0; i < N; ++i) {
        int64_t c = centre(dec.coeffs[i], Q);
        CHECK(std::abs(c) <= POLY_NOISE_BOUND);
    }
}

// Encrypting the same plaintext twice must yield different ciphertexts
// (fresh u, e0, e1 each call — semantic security requirement).
static void test_fresh_randomness_per_encryption() {
    const uint32_t N = 64;
    Encoder enc(N, Q, SCALE);
    Poly pt_poly = enc.encode(std::vector<double>{1.0, -1.0, 0.5});

    KeyGenerator keygen(N, Q, SIGMA, 70);
    SecretKey sk = keygen.gen_secret_key();
    PublicKey pk = keygen.gen_public_key(sk);
    Encryptor encryptor(N, Q, pk, SIGMA, /*seed=*/0);  // random_device seed

    Ciphertext ct1 = encryptor.encrypt(Plaintext(pt_poly));
    Ciphertext ct2 = encryptor.encrypt(Plaintext(pt_poly));

    // The ciphertexts are independent random variables; any coefficient differs
    // with overwhelming probability.  Check c0 and c1 both differ.
    bool c0_differs = false, c1_differs = false;
    for (uint32_t i = 0; i < N; ++i) {
        if (ct1.c0.coeffs[i] != ct2.c0.coeffs[i]) c0_differs = true;
        if (ct1.c1.coeffs[i] != ct2.c1.coeffs[i]) c1_differs = true;
    }
    CHECK(c0_differs);
    CHECK(c1_differs);
}

// Deterministic (seeded) encryptors give reproducible ciphertexts.
static void test_deterministic_encryption() {
    const uint32_t N = 64;
    Encoder enc(N, Q, SCALE);
    Poly pt_poly = enc.encode(std::vector<double>{0.5, -0.5});

    KeyGenerator keygen(N, Q, SIGMA, 80);
    SecretKey sk = keygen.gen_secret_key();
    PublicKey pk = keygen.gen_public_key(sk);

    const uint64_t SEED = 99;
    Encryptor enc1(N, Q, pk, SIGMA, SEED);
    Encryptor enc2(N, Q, pk, SIGMA, SEED);

    Ciphertext ct1 = enc1.encrypt(Plaintext(pt_poly));
    Ciphertext ct2 = enc2.encrypt(Plaintext(pt_poly));

    CHECK(ct1.c0.coeffs == ct2.c0.coeffs);
    CHECK(ct1.c1.coeffs == ct2.c1.coeffs);
}

// Multiple successive encryptions are all correctly decryptable.
static void test_multiple_messages() {
    const uint32_t N = 64;
    Encoder enc(N, Q, SCALE);

    KeyGenerator keygen(N, Q, SIGMA, 100);
    SecretKey sk = keygen.gen_secret_key();
    PublicKey pk = keygen.gen_public_key(sk);
    Encryptor encryptor(N, Q, pk, SIGMA, 101);
    Decryptor decryptor(N, Q, sk);

    const std::vector<std::vector<double>> messages = {
        {1.0, -1.0},
        {0.0,  0.5, -0.5},
        {0.25, 0.75, -0.25, -0.75},
    };

    for (auto& z : messages) {
        Poly pt_poly = enc.encode(z);
        auto z_hat = enc.decode(
            decryptor.decrypt(encryptor.encrypt(Plaintext(pt_poly))).poly);

        for (std::size_t i = 0; i < z.size(); ++i)
            CHECK(std::abs(z_hat[i].real() - z[i]) <= SLOT_TOL);
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_poly_noise_n64_real();
    test_poly_noise_n128_real();

    test_roundtrip_real_slots();
    test_roundtrip_complex_slots();
    test_roundtrip_zero();

    test_fresh_randomness_per_encryption();
    test_deterministic_encryption();
    test_multiple_messages();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
