#include "ckks_types.h"
#include "keygen.h"
#include <stdexcept>

// ─── Little-endian codec helpers ──────────────────────────────────────────────

static void write_le32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back( v        & 0xFFu);
    buf.push_back((v >>  8) & 0xFFu);
    buf.push_back((v >> 16) & 0xFFu);
    buf.push_back((v >> 24) & 0xFFu);
}

static void write_le64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

static uint32_t read_le32(const uint8_t* p) {
    return  static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) <<  8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

static uint64_t read_le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

// ─── Poly serialization helpers ───────────────────────────────────────────────
//
// Wire layout per Poly:
//   [N: uint32 LE (4 B)][q: uint64 LE (8 B)][coeffs[0..N-1]: uint64 LE (8 B each)]
// Total: 12 + N×8 bytes.

static void poly_append(std::vector<uint8_t>& buf, const Poly& p) {
    buf.reserve(buf.size() + 12 + static_cast<std::size_t>(p.N) * 8);
    write_le32(buf, p.N);
    write_le64(buf, p.q);
    for (uint32_t i = 0; i < p.N; ++i)
        write_le64(buf, p.coeffs[i]);
}

// Reads one Poly from data[off..size), advances off.
// If expected_N / expected_q are non-zero, validates them against the header.
static Poly poly_read(const uint8_t* data, std::size_t size, std::size_t& off,
                      uint32_t expected_N = 0, uint64_t expected_q = 0) {
    if (off + 12 > size)
        throw std::runtime_error("serialization: truncated poly header");

    const uint32_t N = read_le32(data + off);  off += 4;
    const uint64_t q = read_le64(data + off);  off += 8;

    if (expected_N != 0 && N != expected_N)
        throw std::runtime_error("serialization: N mismatch between polynomials");
    if (expected_q != 0 && q != expected_q)
        throw std::runtime_error("serialization: q mismatch between polynomials");
    if (off + static_cast<std::size_t>(N) * 8 > size)
        throw std::runtime_error("serialization: truncated poly coefficients");

    std::vector<uint64_t> coeffs(N);
    for (uint32_t i = 0; i < N; ++i) {
        coeffs[i] = read_le64(data + off);
        off += 8;
    }
    return Poly(N, q, std::move(coeffs));
}

// ─── Ciphertext ───────────────────────────────────────────────────────────────

std::vector<uint8_t> Ciphertext::to_bytes() const {
    std::vector<uint8_t> buf;
    poly_append(buf, c0);
    poly_append(buf, c1);
    return buf;
}

Ciphertext Ciphertext::from_bytes(const uint8_t* data, std::size_t size) {
    std::size_t off = 0;
    Poly p0 = poly_read(data, size, off);
    Poly p1 = poly_read(data, size, off, p0.N, p0.q);
    if (off != size)
        throw std::runtime_error("serialization: trailing bytes in ciphertext");
    return Ciphertext(std::move(p0), std::move(p1));
}

// ─── PublicKey ────────────────────────────────────────────────────────────────

std::vector<uint8_t> PublicKey::to_bytes() const {
    std::vector<uint8_t> buf;
    poly_append(buf, pk0);
    poly_append(buf, pk1);
    return buf;
}

PublicKey PublicKey::from_bytes(const uint8_t* data, std::size_t size) {
    std::size_t off = 0;
    Poly p0 = poly_read(data, size, off);
    Poly p1 = poly_read(data, size, off, p0.N, p0.q);
    if (off != size)
        throw std::runtime_error("serialization: trailing bytes in public key");
    return PublicKey(std::move(p0), std::move(p1));
}

// ─── SecretKey ────────────────────────────────────────────────────────────────

std::vector<uint8_t> SecretKey::to_bytes() const {
    std::vector<uint8_t> buf;
    poly_append(buf, s);
    return buf;
}

SecretKey SecretKey::from_bytes(const uint8_t* data, std::size_t size) {
    std::size_t off = 0;
    Poly p = poly_read(data, size, off);
    if (off != size)
        throw std::runtime_error("serialization: trailing bytes in secret key");
    return SecretKey(std::move(p));
}
