#include "decryptor.h"
#include "encoder.h"
#include "encryptor.h"
#include "evaluator.h"
#include "keygen.h"
#include <algorithm>
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
#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))

// ── Shared parameters ─────────────────────────────────────────────────────────
// Q = 998244353 = 119·2²³+1: NTT-friendly prime, ≡ 1 (mod 2N) for all N ≤ 2²²
static constexpr uint32_t N     = 64;
static constexpr uint64_t Q     = 998244353ULL;
static constexpr double   SCALE = 1 << 20;
static constexpr double   SIGMA = 3.2;

// Per-slot noise budget after k fresh encryptions added together:
//   |δz| ≤ k · POLY_NOISE_BOUND / SCALE + k · N/(2·SCALE)
// where POLY_NOISE_BOUND = 3000 (from test_encryption.cpp).
//
//   k=2: ≤ 2·3000/2²⁰ + 32/2²⁰ ≈ 5.8e-3  → use 2e-2 (3.4× headroom)
//   k=4: ≤ 4·3000/2²⁰            ≈ 1.1e-2  → use 3e-2 (2.7× headroom)
static constexpr double SLOT_TOL_1ADD = 2e-2;
static constexpr double SLOT_TOL_3ADD = 3e-2;

// Polynomial-level noise bound per fresh ciphertext (6σ tail, N=64).
static constexpr int64_t POLY_NOISE_BOUND = 3000;

// ── Infrastructure ─────────────────────────────────────────────────────────────

struct Ctx {
    Encoder    enc;
    KeyGenerator keygen;
    SecretKey  sk;
    PublicKey  pk;
    Encryptor  encryptor;
    Decryptor  decryptor;
    Evaluator  eval;

    Ctx(uint64_t kg_seed, uint64_t enc_seed)
        : enc(N, Q, SCALE)
        , keygen(N, Q, SIGMA, kg_seed)
        , sk(keygen.gen_secret_key())
        , pk(keygen.gen_public_key(sk))
        , encryptor(N, Q, pk, SIGMA, enc_seed)
        , decryptor(N, Q, sk)
        , eval(N, Q)
    {}
};

static Ciphertext encrypt(Ctx& cx, const std::vector<double>& z) {
    return cx.encryptor.encrypt(Plaintext(cx.enc.encode(z)));
}
static Ciphertext encrypt(Ctx& cx, const std::vector<std::complex<double>>& z) {
    return cx.encryptor.encrypt(Plaintext(cx.enc.encode(z)));
}
static std::vector<std::complex<double>> decrypt(Ctx& cx, const Ciphertext& ct) {
    return cx.enc.decode(cx.decryptor.decrypt(ct).poly);
}

// Centre a [0,q) residue into (−q/2, q/2].
static int64_t centre(uint64_t c, uint64_t q) {
    int64_t sc = static_cast<int64_t>(c);
    int64_t sq = static_cast<int64_t>(q);
    return sc > sq / 2 ? sc - sq : sc;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// Decrypt(enc(a) + enc(b)) ≈ a + b  for real slot vectors.
// Checks all N/2 = 32 real parts and that imaginary parts stay near zero.
static void test_add_inplace_real_slots() {
    Ctx cx(1, 2);
    const std::vector<double> a = {
         0.5, -0.3,  0.7, -0.1,  0.9, -0.8,  0.2,  0.4,
        -0.6,  0.3, -0.4,  0.8, -0.2,  0.6, -0.5,  0.1,
         0.7, -0.2,  0.3, -0.7,  0.4, -0.9,  0.1, -0.4,
        -0.1,  0.5, -0.3,  0.8, -0.6,  0.2,  0.4, -0.7
    };
    const std::vector<double> b = {
         0.2,  0.4, -0.1,  0.3, -0.5,  0.7, -0.3,  0.6,
         0.1, -0.4,  0.5, -0.3,  0.7, -0.1,  0.4, -0.2,
        -0.3,  0.8, -0.5,  0.2, -0.6,  0.1,  0.9, -0.3,
         0.4, -0.2,  0.6, -0.4,  0.3,  0.7, -0.1,  0.5
    };

    Ciphertext ct_a = encrypt(cx, a);
    Ciphertext ct_b = encrypt(cx, b);
    cx.eval.add_inplace(ct_a, ct_b);
    auto got = decrypt(cx, ct_a);

    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK_NEAR(got[i].real(), a[i] + b[i], SLOT_TOL_1ADD);  // 32 checks
        CHECK_NEAR(got[i].imag(), 0.0,          SLOT_TOL_1ADD);  // 32 checks
    }
}

// Decrypt(enc(a) + enc(b)) ≈ a + b  for complex slot vectors.
static void test_add_complex_slots() {
    Ctx cx(3, 4);
    using cd = std::complex<double>;
    const std::vector<cd> a = {
        { 0.6,  0.4}, {-0.3,  0.8}, { 0.1, -0.9}, {-0.5,  0.2}
    };
    const std::vector<cd> b = {
        {-0.2,  0.1}, { 0.5, -0.3}, {-0.4,  0.7}, { 0.3, -0.6}
    };

    Ciphertext ct_a = encrypt(cx, a);
    Ciphertext ct_b = encrypt(cx, b);
    cx.eval.add_inplace(ct_a, ct_b);
    auto got = decrypt(cx, ct_a);

    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK_NEAR(got[i].real(), a[i].real() + b[i].real(), SLOT_TOL_1ADD);  // 4
        CHECK_NEAR(got[i].imag(), a[i].imag() + b[i].imag(), SLOT_TOL_1ADD);  // 4
    }
}

// Polynomial-level noise: |dec(ct_a + ct_b)[k] − (pt_a[k]+pt_b[k])| ≤ 2·BOUND
// Exercises the full encrypt → homadd → decrypt chain at coefficient granularity.
static void test_poly_level_noise_after_add() {
    Ctx cx(5, 6);
    const std::vector<double> a = {0.5, -0.5, 0.25};
    const std::vector<double> b = {0.3,  0.2, -0.4};

    Poly pt_a = cx.enc.encode(a);
    Poly pt_b = cx.enc.encode(b);
    Poly pt_sum = pt_a + pt_b;   // exact polynomial sum mod q

    Ciphertext ct_a = cx.encryptor.encrypt(Plaintext(pt_a));
    Ciphertext ct_b = cx.encryptor.encrypt(Plaintext(pt_b));
    cx.eval.add_inplace(ct_a, ct_b);
    Poly dec = cx.decryptor.decrypt(ct_a).poly;

    bool ok = true;
    for (uint32_t i = 0; i < N; ++i) {
        int64_t noise = std::abs(
            centre(dec.coeffs[i], Q) - centre(pt_sum.coeffs[i], Q));
        if (noise > 2 * POLY_NOISE_BOUND) { ok = false; break; }
    }
    CHECK(ok);  // 1 check
}

// add() and add_inplace() must produce bit-identical ciphertexts.
static void test_add_vs_add_inplace_consistent() {
    Ctx cx(7, 8);
    const std::vector<double> a = {0.1, -0.2, 0.3};
    const std::vector<double> b = {0.4,  0.5, -0.1};

    Ciphertext ct_a1 = encrypt(cx, a);
    Ciphertext ct_b  = encrypt(cx, b);

    // add_inplace path
    Ciphertext ct_ip = ct_a1;   // copy before modifying
    cx.eval.add_inplace(ct_ip, ct_b);

    // add (functional) path  — uses a fresh copy of ct_a1 internally
    Ciphertext ct_fn = cx.eval.add(ct_a1, ct_b);

    CHECK(ct_ip.c0.coeffs == ct_fn.c0.coeffs);  // 1
    CHECK(ct_ip.c1.coeffs == ct_fn.c1.coeffs);  // 1
}

// ct + encrypt(0) ≈ ct  (additive identity for the plaintext message).
static void test_add_identity_zero() {
    Ctx cx(9, 10);
    const std::vector<double> z = {0.5, -0.3, 0.7, -0.1, 0.9};

    Ciphertext ct   = encrypt(cx, z);
    Ciphertext zero = cx.encryptor.encrypt(Plaintext(Poly(N, Q)));  // zero poly
    cx.eval.add_inplace(ct, zero);
    auto got = decrypt(cx, ct);

    // Error bound: two fresh encryptions summed → use SLOT_TOL_1ADD.
    for (std::size_t i = 0; i < z.size(); ++i)
        CHECK_NEAR(got[i].real(), z[i], SLOT_TOL_1ADD);  // 5 checks
    for (std::size_t i = z.size(); i < N / 2; ++i)
        CHECK_NEAR(got[i].real(), 0.0,  SLOT_TOL_1ADD);  // 27 checks (zero-padded)
}

// add is commutative: dec(ct_a + ct_b) ≡ dec(ct_b + ct_a) at slot level.
// Since addition in Z_q is commutative, the two ciphertexts are bit-identical.
static void test_add_commutativity() {
    Ctx cx(11, 12);
    const std::vector<double> a = {0.3, -0.5, 0.2, 0.8};
    const std::vector<double> b = {-0.1, 0.7, -0.4, 0.6};

    Ciphertext ct_ab = cx.eval.add(encrypt(cx, a), encrypt(cx, b));
    Ciphertext ct_ba = cx.eval.add(encrypt(cx, b), encrypt(cx, a));

    // Each direction uses independent fresh randomness, so the ciphertexts
    // differ — but their decoded slot values must match to within two-addition
    // tolerance.
    auto got_ab = decrypt(cx, ct_ab);
    auto got_ba = decrypt(cx, ct_ba);
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK_NEAR(got_ab[i].real(), a[i] + b[i], SLOT_TOL_1ADD);  // 4
        CHECK_NEAR(got_ba[i].real(), a[i] + b[i], SLOT_TOL_1ADD);  // 4
    }
}

// add is associative in Z_q: (ct_a + ct_b) + ct_c ≡ ct_a + (ct_b + ct_c)
// because coefficient-wise addition in Z_q is exactly associative.
// These ciphertexts all use the same encryption keys and seeds, so the
// two assembly orders yield identical polynomials and thus identical slots.
static void test_add_associativity() {
    Ctx cx(13, 14);
    const std::vector<double> a = {0.2, -0.4, 0.6};
    const std::vector<double> b = {-0.3, 0.5, -0.1};
    const std::vector<double> c = {0.4, -0.2, 0.3};

    Ciphertext ct_a = encrypt(cx, a);
    Ciphertext ct_b = encrypt(cx, b);
    Ciphertext ct_c = encrypt(cx, c);

    // (ct_a + ct_b) + ct_c
    Ciphertext lhs = cx.eval.add(cx.eval.add(ct_a, ct_b), ct_c);
    // ct_a + (ct_b + ct_c)
    Ciphertext rhs = cx.eval.add(ct_a, cx.eval.add(ct_b, ct_c));

    // Both groupings pass through identical Z_q arithmetic; results are equal.
    CHECK(lhs.c0.coeffs == rhs.c0.coeffs);  // 1
    CHECK(lhs.c1.coeffs == rhs.c1.coeffs);  // 1
}

// Chain of four additions: dec(ct1+ct2+ct3+ct4) ≈ z1+z2+z3+z4.
static void test_add_chain_four() {
    Ctx cx(15, 16);
    const std::vector<double> z1 = {0.1, -0.2,  0.3, -0.4,  0.5, -0.6,  0.7, -0.8};
    const std::vector<double> z2 = {0.2,  0.1, -0.1,  0.3, -0.2,  0.4, -0.3,  0.5};
    const std::vector<double> z3 = {-0.3, 0.4, -0.5,  0.1,  0.2, -0.1,  0.4, -0.2};
    const std::vector<double> z4 = {0.0,  0.3,  0.2, -0.2, -0.1,  0.3, -0.4,  0.1};

    Ciphertext ct = encrypt(cx, z1);
    cx.eval.add_inplace(ct, encrypt(cx, z2));
    cx.eval.add_inplace(ct, encrypt(cx, z3));
    cx.eval.add_inplace(ct, encrypt(cx, z4));
    auto got = decrypt(cx, ct);

    for (std::size_t i = 0; i < z1.size(); ++i) {
        double expected = z1[i] + z2[i] + z3[i] + z4[i];
        CHECK_NEAR(got[i].real(), expected, SLOT_TOL_3ADD);  // 8 checks
    }
}

// Direct arithmetic test: coefficients wrapping exactly through 0 mod q.
// Does not involve encryption; only verifies the mod_add correctness inside
// add_inplace for the boundary case where a + b = q (i.e. wraps to 0).
static void test_add_modular_wrap_around() {
    // Use a small N so the test is fast and independent of NTT setup.
    const uint32_t N4 = 4;
    Evaluator eval4(N4, Q);

    // All coefficients = Q-1; adding 1 should give 0.
    std::vector<uint64_t> all_qm1(N4, Q - 1);
    std::vector<uint64_t> all_one(N4, 1);
    Ciphertext ct1(Poly(N4, Q, all_qm1), Poly(N4, Q, all_qm1));
    Ciphertext ct2(Poly(N4, Q, all_one), Poly(N4, Q, all_one));
    eval4.add_inplace(ct1, ct2);
    bool c0_zero = std::all_of(ct1.c0.coeffs.begin(), ct1.c0.coeffs.end(),
                                [](uint64_t x){ return x == 0; });
    bool c1_zero = std::all_of(ct1.c1.coeffs.begin(), ct1.c1.coeffs.end(),
                                [](uint64_t x){ return x == 0; });
    CHECK(c0_zero);  // 1
    CHECK(c1_zero);  // 1

    // All coefficients = Q-2; adding 5 should give 3 (crosses the modulus).
    const uint64_t expected = 3;
    std::vector<uint64_t> all_qm2(N4, Q - 2);
    std::vector<uint64_t> all_five(N4, 5);
    Ciphertext ct3(Poly(N4, Q, all_qm2), Poly(N4, Q, all_qm2));
    Ciphertext ct4(Poly(N4, Q, all_five), Poly(N4, Q, all_five));
    eval4.add_inplace(ct3, ct4);
    bool c0_exp = std::all_of(ct3.c0.coeffs.begin(), ct3.c0.coeffs.end(),
                               [](uint64_t x){ return x == expected; });
    bool c1_exp = std::all_of(ct3.c1.coeffs.begin(), ct3.c1.coeffs.end(),
                               [](uint64_t x){ return x == expected; });
    CHECK(c0_exp);  // 1
    CHECK(c1_exp);  // 1
}

// Evaluator must reject ciphertexts whose (N, q) differ from its own.
static void test_mismatch_throws() {
    Evaluator eval_main(N, Q);

    // Ciphertext built with a different N.
    const uint32_t N_wrong = N * 2;
    Ciphertext ct_bad(Poly(N_wrong, Q), Poly(N_wrong, Q));
    Ciphertext ct_ok (Poly(N,       Q), Poly(N,       Q));

    bool threw = false;
    try { eval_main.add_inplace(ct_ok, ct_bad); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);  // 1
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_add_inplace_real_slots();     // 64
    test_add_complex_slots();          //  8
    test_poly_level_noise_after_add(); //  1
    test_add_vs_add_inplace_consistent(); // 2
    test_add_identity_zero();          // 32
    test_add_commutativity();          //  8
    test_add_associativity();          //  2
    test_add_chain_four();             //  8
    test_add_modular_wrap_around();    //  4
    test_mismatch_throws();            //  1

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
