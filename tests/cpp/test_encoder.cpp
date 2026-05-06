#include "encoder.h"
#include <cmath>
#include <cstdio>
#include <complex>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) \
    do { \
        if (expr) { ++g_pass; } \
        else { \
            ++g_fail; \
            std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); \
        } \
    } while (0)

#define CHECK_EQ(a, b)    CHECK((a) == (b))
#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------
static constexpr uint32_t N     = 16;
static constexpr uint64_t Q     = 4611686018427387847ULL;  // 2^62 - 57
static constexpr double   SCALE = 1ULL << 40;
// Per-slot round-trip error bound: N / (2 * scale) = 16 / 2^41 ≈ 7.3e-12.
static constexpr double   TOL   = 1e-9;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static double max_err_real(
    const std::vector<std::complex<double>>& got,
    const std::vector<double>&               expected)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        err = std::max(err, std::abs(got[i].real() - expected[i]));
    return err;
}

static double max_err_complex(
    const std::vector<std::complex<double>>& got,
    const std::vector<std::complex<double>>& expected)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        err = std::max(err, std::abs(got[i] - expected[i]));
    return err;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
static void test_constructor_valid() {
    Encoder enc(N, Q, SCALE);
    CHECK_EQ(enc.slots(), N / 2);
    CHECK_EQ(enc.degree(), N);
}

static void test_constructor_invalid_N() {
    bool threw = false;
    try { Encoder enc(3, Q, SCALE); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { Encoder enc(0, Q, SCALE); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_constructor_invalid_scale() {
    bool threw = false;
    try { Encoder enc(N, Q, -1.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// Encode produces a valid Poly
// ---------------------------------------------------------------------------
static void test_encode_returns_valid_poly() {
    Encoder enc(N, Q, SCALE);
    std::vector<double> z(N / 2, 0.5);
    Poly p = enc.encode(z);
    CHECK_EQ(p.N, N);
    CHECK_EQ(p.q, Q);
    for (uint32_t i = 0; i < N; ++i)
        CHECK(p.coeffs[i] < Q);
}

static void test_encode_zero_gives_zero_poly() {
    Encoder enc(N, Q, SCALE);
    Poly p = enc.encode(std::vector<double>(N / 2, 0.0));
    for (uint32_t i = 0; i < N; ++i)
        CHECK_EQ(p.coeffs[i], 0u);
}

static void test_encode_too_long_throws() {
    Encoder enc(N, Q, SCALE);
    bool threw = false;
    try { enc.encode(std::vector<double>(N / 2 + 1, 0.0)); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// Round-trip: encode then decode recovers the original values.
// ---------------------------------------------------------------------------
static void test_roundtrip_real_zeros() {
    Encoder enc(N, Q, SCALE);
    std::vector<double> z(N / 2, 0.0);
    auto got = enc.decode(enc.encode(z));
    CHECK_EQ(got.size(), N / 2);
    for (auto& v : got)
        CHECK_NEAR(v.real(), 0.0, TOL);
}

static void test_roundtrip_real_constant() {
    Encoder enc(N, Q, SCALE);
    std::vector<double> z(N / 2, 0.25);
    auto got = enc.decode(enc.encode(z));
    for (std::size_t i = 0; i < z.size(); ++i)
        CHECK_NEAR(got[i].real(), z[i], TOL);
}

static void test_roundtrip_real_vector() {
    Encoder enc(N, Q, SCALE);
    std::vector<double> z = {0.1, -0.3, 0.7, -0.5, 0.2, -0.8, 0.4, -0.6};
    Poly p   = enc.encode(z);
    auto got = enc.decode(p);
    double err = max_err_real(got, z);
    std::fprintf(stderr, "      roundtrip real: max|err| = %.2e  (tol %.2e)\n", err, TOL);
    CHECK(err < TOL);
}

static void test_roundtrip_complex_vector() {
    Encoder enc(N, Q, SCALE);
    using cd = std::complex<double>;
    std::vector<cd> z = {
        {0.1,  0.2}, {-0.3, 0.4}, {0.5, -0.6}, {-0.7, 0.8},
        {0.15, -0.1}, {0.3,  0.3}, {-0.2, 0.5}, {0.4, -0.2}
    };
    auto got = enc.decode(enc.encode(z));
    double err = max_err_complex(got, z);
    std::fprintf(stderr, "      roundtrip cplx: max|err| = %.2e  (tol %.2e)\n", err, TOL);
    CHECK(err < TOL);
}

static void test_roundtrip_short_input_zero_padded() {
    Encoder enc(N, Q, SCALE);
    // Encode 2 real values; remaining N/2-2 slots should decode to ≈ 0.
    std::vector<double> z = {0.5, -0.5};
    auto got = enc.decode(enc.encode(z));
    CHECK_NEAR(got[0].real(), 0.5, TOL);
    CHECK_NEAR(got[1].real(), -0.5, TOL);
    for (std::size_t i = 2; i < got.size(); ++i)
        CHECK_NEAR(got[i].real(), 0.0, TOL);
}

// ---------------------------------------------------------------------------
// Linearity: encode(a) + encode(b) ≈ encode(a + b)   (addition is exact mod q)
// ---------------------------------------------------------------------------
static void test_encode_is_linear() {
    Encoder enc(N, Q, SCALE);
    std::vector<double> a = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8};
    std::vector<double> b = {0.2,  0.1,-0.1,  0.3,-0.2,  0.4,-0.3,  0.5};
    std::vector<double> apb(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) apb[i] = a[i] + b[i];

    Poly pa   = enc.encode(a);
    Poly pb   = enc.encode(b);
    Poly psum = pa + pb;                    // polynomial addition (mod q)
    Poly papb = enc.encode(apb);

    auto dec_sum  = enc.decode(psum);
    auto dec_papb = enc.decode(papb);

    // Both should match: |dec_sum[i] - dec_papb[i]| ≈ quantisation noise
    for (std::size_t i = 0; i < a.size(); ++i)
        CHECK_NEAR(dec_sum[i].real(), dec_papb[i].real(), TOL * 4);
}

// ---------------------------------------------------------------------------
// Decode rejects mismatched Poly
// ---------------------------------------------------------------------------
static void test_decode_wrong_params_throws() {
    Encoder enc(N, Q, SCALE);
    Poly wrong(N + 4, Q);   // different N
    bool threw = false;
    try { enc.decode(wrong); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_constructor_valid();
    test_constructor_invalid_N();
    test_constructor_invalid_scale();

    test_encode_returns_valid_poly();
    test_encode_zero_gives_zero_poly();
    test_encode_too_long_throws();

    test_roundtrip_real_zeros();
    test_roundtrip_real_constant();
    test_roundtrip_real_vector();
    test_roundtrip_complex_vector();
    test_roundtrip_short_input_zero_padded();

    test_encode_is_linear();

    test_decode_wrong_params_throws();

    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
