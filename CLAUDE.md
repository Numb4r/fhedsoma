# CLAUDE.md

## Project: High-Performance CKKS for Federated Learning
A performance-oriented C++ implementation of the CKKS homomorphic encryption scheme
with Python bindings, optimized for additive HE in FL environments.

## Architecture Goals
- **Backend:** C++17 for all core arithmetic.
- **Arithmetic:** NTT-based O(N log N) polynomial multiplication; Montgomery reduction throughout.
- **FL Focus:** Optimized for `Ciphertext + Ciphertext` and `Ciphertext × Plaintext` (gradient aggregation, weight scaling).
- **Flexibility:** Raw polynomial access for research and custom FL protocols.

## Tech Stack
- **C++:** custom coefficient buffers with `#pragma omp simd` hints; optional Eigen.
- **Bindings:** `pybind11` for the Python interface.
- **Build:** CMake 3.18+; single static library `ckks_core` linked by all executables.

## Development Strategy
1. **Module 1 — Math / NTT:** Ring Z_q[X]/(X^N+1), Montgomery reduction, NTT.
2. **Module 2 — Sampling:** Discrete Gaussian, Uniform, Ternary samplers.
3. **Module 3 — CKKS Core:** Encoder, KeyGenerator, Encryptor, Decryptor.
4. **Module 4 — FL Ops:** Ciphertext addition, scalar multiplication.
5. **Module 5 — Python API:** pybind11 wrappers for all public types.
6. **Module 6 — FL Library:** `ckks_federated` — high-level Python layer for Flower/FL workflows.

---

## Commands

```bash
# ── Build (C++ only, no Python bindings) ────────────────────────────────────
mkdir build && cd build
cmake .. && make -j$(nproc)

# ── Build with Python bindings ───────────────────────────────────────────────
mkdir build && cd build
cmake .. -DCKKS_BUILD_PYTHON=ON \
         -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") \
  && make -j$(nproc)

# ── Run all C++ unit tests ───────────────────────────────────────────────────
cd build && ctest --output-on-failure

# ── Run the decryption-noise diagnostic (prints infinity norm to stdout) ─────
cd build && ./tests/cpp/test_noise

# ── Python tests ─────────────────────────────────────────────────────────────
python tests/test_precision.py
python tests/python/test_poly.py
python test_py_ckks_simplifly.py
python test_py_serialization.py
```

---

## Current API

### Python (`ckks_python` extension)

Built at `build/ckks_python.cpython-<ver>-<arch>.so`.

```python
import sys; sys.path.insert(0, "build")
import ckks_python
```

#### `ckks_python.Poly`

Polynomial in Z_q[X]/(X^N+1). Coefficients are unsigned 64-bit integers in **[0, q)**.

| Signature | Description |
|-----------|-------------|
| `Poly(N, q)` | Zero polynomial of degree `N` mod `q`. |
| `Poly(N, q, coeffs: np.ndarray)` | Construct from a 1-D uint64 NumPy array of length `N`. |
| `p + other → Poly` | Coefficient-wise modular addition (new Poly). |
| `p.coeffs → np.ndarray[uint64]` | **Zero-copy** writable view of the internal buffer. |
| `p.coeffs = arr` | Replace all coefficients from a 1-D array of length `N`. |
| `p.N → int` | Ring degree (read-only). |
| `p.q → int` | Modulus (read-only). |

#### `ckks_python.Plaintext`

| Signature | Description |
|-----------|-------------|
| `Plaintext(poly: Poly)` | Wrap a `Poly` as a CKKS plaintext. |
| `pt.poly → np.ndarray[uint64]` | **Zero-copy** read-only uint64 view of the encoded polynomial. |

#### `ckks_python.Ciphertext`

Wire format: `[c0: N u32 LE, q u64 LE, N×u64 LE coeffs][c1: same]` — 2×(12+N×8) bytes total.

| Signature | Description |
|-----------|-------------|
| `Ciphertext(c0: Poly, c1: Poly)` | Construct directly from two polynomials (rarely needed). |
| `ct.c0 → np.ndarray[uint64]` | **Zero-copy** read-only view of the c0 polynomial. |
| `ct.c1 → np.ndarray[uint64]` | **Zero-copy** read-only view of the c1 polynomial. |
| `ct.to_bytes() → bytes` | Serialize to a `bytes` object (little-endian, self-describing). |
| `Ciphertext.from_bytes(data: bytes) → Ciphertext` | Deserialize; raises `RuntimeError` on truncation, trailing bytes, or (N,q) mismatch. |

#### `ckks_python.SecretKey`

| Signature | Description |
|-----------|-------------|
| `sk.s → np.ndarray[uint64]` | **Zero-copy** read-only view of the ternary secret key polynomial. |
| `sk.to_bytes() → bytes` | Serialize to a `bytes` object for secure local storage. |
| `SecretKey.from_bytes(data: bytes) → SecretKey` | Deserialize from disk; raises `RuntimeError` on malformed data. |

#### `ckks_python.PublicKey`

Wire format: same layout as `Ciphertext` — `[pk0 poly][pk1 poly]`.

| Signature | Description |
|-----------|-------------|
| `pk.pk0 → np.ndarray[uint64]` | **Zero-copy** read-only view of `pk0 = -a·s + e`. |
| `pk.pk1 → np.ndarray[uint64]` | **Zero-copy** read-only view of `pk1 = a` (uniform). |
| `pk.to_bytes() → bytes` | Serialize to a `bytes` object (little-endian, self-describing). |
| `PublicKey.from_bytes(data: bytes) → PublicKey` | Deserialize; raises `RuntimeError` on truncation, trailing bytes, or (N,q) mismatch. |

#### `ckks_python.Encoder`

| Signature | Description |
|-----------|-------------|
| `Encoder(N, q, scale)` | Construct encoder for N slots, modulus q, scaling factor Δ. |
| `encode(z: np.ndarray[float64]) → Poly` | Real float64 array → scaled polynomial. |
| `encode(z: np.ndarray[complex128]) → Poly` | Complex128 array → scaled polynomial. |
| `decode(p: Poly) → np.ndarray[complex128]` | Polynomial → slot vector (length N/2). |
| `slots → int` | Number of complex slots = N/2. |
| `degree → int` | Ring degree N. |

#### `ckks_python.KeyGenerator`

| Signature | Description |
|-----------|-------------|
| `KeyGenerator(N, q, sigma=3.2, seed=0)` | Non-copyable. `seed=0` uses `std::random_device`. |
| `gen_secret_key(hamming_weight=0) → SecretKey` | `hamming_weight=0`: uniform ternary; `>0`: fixed-weight (must be even). |
| `gen_public_key(sk) → PublicKey` | One NTT-based multiplication. |

#### `ckks_python.Encryptor`

| Signature | Description |
|-----------|-------------|
| `Encryptor(N, q, pk, sigma=3.2, seed=0)` | Non-copyable. Stores a copy of `pk`. |
| `encrypt(pt: Plaintext) → Ciphertext` | Fresh encryption using `pk`. Draws new `u`, `e0`, `e1` each call. |

#### `ckks_python.Decryptor`

| Signature | Description |
|-----------|-------------|
| `Decryptor(N, q, sk)` | Non-copyable. Stores a copy of `sk`. |
| `decrypt(ct: Ciphertext) → Plaintext` | Returns `c0 + c1·s` as a `Plaintext`. Pass `.poly` coeffs to `Encoder.decode`. |

#### `ckks_python.Evaluator`

| Signature | Description |
|-----------|-------------|
| `Evaluator(N, q)` | Stateless; all methods are thread-safe for distinct ciphertexts. |
| `add_inplace(ct1, ct2)` | `ct1 += ct2` in-place, coefficient-wise mod q. Raises on (N, q) mismatch. |
| `add(ct1, ct2) → Ciphertext` | Returns a new `Ciphertext` equal to `ct1 + ct2`. |

---

### `ckks_federated` — FL high-level library (`ckks_federated/__init__.py`)

High-level Python layer built on top of `ckks_python`. Provides key persistence and
auto-chunking tensor encryption for Flower / containerised FL workflows.

```python
from ckks_federated import CKKSContext, CKKSTensor
```

#### `CKKSContext`

| Signature | Description |
|-----------|-------------|
| `CKKSContext(N, bits, scale_bits=40)` | Generate fresh keys. `q` is derived via `generate_prime(bits, N)`. |
| `CKKSContext.from_keys(N, q, scale, pk_bytes, sk_bytes=None, public_only=False)` | Reconstruct from serialised keys — no new keys generated. |
| `ctx.save(path)` | Write `params.pkl`, `pk.bin`, and (if sk present) `sk.bin` to `path/`. |
| `CKKSContext.load(path, public_only=False)` | Load context from disk. `public_only=True` skips SK/Decryptor. |
| `ctx.__getstate__() / __setstate__()` | Full `pickle` support (including `multiprocessing` and Flower internals). |
| `ctx.make_public()` | Drop SK and Decryptor in-place (server-side blind mode). |
| `ctx.export_public_key() → bytes` | Return `pk.to_bytes()`. |
| `ctx.import_public_key(pk_bytes)` | Replace pk and rebuild Encryptor from bytes. |

#### `CKKSTensor`

Auto-chunking encrypted tensor. Splits arrays larger than N/2 into multiple ciphertexts
and reassembles on decryption. Carries `original_size` as metadata.

| Signature | Description |
|-----------|-------------|
| `CKKSTensor(ctx, data)` | Encrypt a NumPy/TF array, a list of `Ciphertext` objects, or deserialize from a list of `bytes`. |
| `t.decrypt() → np.ndarray` | Decrypt, concatenate chunks, strip imaginary part, trim to `original_size`. |
| `t.serialize() → list[bytes]` | Payload for gRPC: `[original_size (8 B)] + [ct.to_bytes() …]`. |
| `t1 + t2 → CKKSTensor` | Homomorphic addition (new object). |
| `t1 += t2` | Homomorphic addition in-place (`add_inplace`). |

---

### C++ only (not exposed to Python)

#### Module 1 — Math / NTT (`math_utils.h`, `poly.h`, `ntt.h`)

| Symbol | Description |
|--------|-------------|
| `mod_add(a, b, q) → uint64_t` | Branchless modular addition; no branch, maps to AVX2. |
| `mod_sub(a, b, q) → uint64_t` | Branchless modular subtraction. |
| `struct MontgomeryCtx` | Precomputed Montgomery context for a fixed prime q. Methods: `to_mont`, `from_mont`, `mont_mul`. |
| `struct Poly` | Polynomial in Z_q[X]/(X^N+1). Fields: `N`, `q`, `coeffs`. Supports `operator+`. |
| `struct NttCtx` | Precomputed NTT context: bit-reversal table, twiddle tables (`psi_pow`, `psi_inv_pow`), `N_inv_mont`. Constructed once per (N, q) pair; reused across transforms. |
| `ntt_forward(a, ctx)` | In-place Cooley–Tukey DIT. Natural-order input → bit-reversed NTT evaluations. O(N log N). |
| `ntt_inverse(a, ctx)` | In-place Gentleman–Sande DIF + 1/N scaling. Bit-reversed input → natural-order coefficients. O(N log N). |
| `poly_ntt_mul(a, b, ctx) → Poly` | Negacyclic ring multiplication via NTT. Returns a·b mod (X^N+1, q). O(N log N). |

#### Module 2 — Sampling (`sampler.h`)

| Symbol | Description |
|--------|-------------|
| `class Sampler` | Non-copyable PRNG wrapper (`mt19937_64`). One instance per thread. |
| `Sampler()` | Seed from `std::random_device` (production). |
| `Sampler(uint64_t seed)` | Deterministic seed (testing / reproducibility). |
| `uniform(N, q) → Poly` | Each coefficient drawn uniformly from [0, q). |
| `gaussian(N, q, sigma, tail=6) → Poly` | Discrete Gaussian: round(N(0, σ²)), rejection-sampled within ±tail·σ. |
| `ternary_uniform(N, q) → Poly` | Each coefficient i.i.d. in {0, 1, q−1} with probability 1/3 each. |
| `ternary_hw(N, q, hw) → Poly` | Fixed Hamming weight: exactly hw non-zeros (hw/2 are +1, hw/2 are q−1). hw must be even and ≤ N. |

#### Module 3 — CKKS Core

**Types** (`ckks_types.h`):

| Symbol | Description |
|--------|-------------|
| `struct Plaintext` | Wraps a single `Poly` representing a scaled, encoded message. |
| `struct Ciphertext` | Pair `(c0, c1)` of `Poly`; invariant: `c0 + c1·s ≈ m (mod q)`. |

**Key generation** (`keygen.h`):

| Symbol | Description |
|--------|-------------|
| `struct SecretKey` | Ternary polynomial s ∈ {-1, 0, 1}^N stored as [0, q) residues. |
| `struct PublicKey` | Pair `(pk0, pk1)` where `pk1 = a` (uniform) and `pk0 = -a·s + e`. Invariant: `pk0 + pk1·s = e`. |
| `class KeyGenerator(N, q, sigma=3.2, seed=0)` | Owns one `Sampler` and one `NttCtx`. |
| `gen_secret_key(hw=0) → SecretKey` | hw=0: uniform ternary. hw>0: fixed-weight ternary (hw must be even). |
| `gen_public_key(sk) → PublicKey` | One NTT-based multiplication. |
| `gen_key_bundle(hw=0) → KeyBundle` | Returns `{sk, pk, e}` — exposes the error polynomial for unit tests only. |

**Encryption** (`encryptor.h`):

| Symbol | Description |
|--------|-------------|
| `class Encryptor(N, q, pk, sigma=3.2, seed=0)` | Stores a copy of `pk`; owns one `Sampler` and one `NttCtx`. |
| `encrypt(pt) → Ciphertext` | Draws fresh `u` (ternary), `e0`, `e1` (Gaussian). Computes `c1 = pk1·u + e1`, `c0 = pk0·u + e0 + m`. Two NTT multiplications. |

**Decryption** (`decryptor.h`):

| Symbol | Description |
|--------|-------------|
| `class Decryptor(N, q, sk)` | Stores a copy of `sk`; owns one `NttCtx`. |
| `decrypt(ct) → Plaintext` | Computes `m' = c0 + c1·s`. One NTT multiplication. |

**Homomorphic evaluation** (`evaluator.h`):

| Symbol | Description |
|--------|-------------|
| `class Evaluator(N, q)` | Owns no state beyond (N, q). All methods are const and thread-safe for distinct ciphertexts. |
| `add_inplace(ct1, ct2)` | `ct1 += ct2`: coefficient-wise modular addition of c0 and c1 using branchless `mod_add`; `#pragma omp simd` vectorised. Throws if (N, q) mismatch. |
| `add(ct1, ct2) → Ciphertext` | Returns `ct1 + ct2` as a new Ciphertext; delegates to `Poly::operator+`. |

**Schoolbook ops** (`ckks_ops.h`):

| Symbol | Description |
|--------|-------------|
| `mul_plain(ct, pt) → Ciphertext` | Ciphertext × Plaintext via schoolbook poly-mul in the Montgomery domain. O(N²) — kept as a reference implementation; use `poly_ntt_mul` for performance. |

---

## Implementation Status

| Module | Component | Status |
|--------|-----------|--------|
| 1 — Math / NTT | `mod_add`, `mod_sub`, Montgomery reduction (`MontgomeryCtx`) | ✅ done |
| 1 — Math / NTT | `Poly` ring type with modular addition | ✅ done |
| 1 — Math / NTT | `NttCtx`: bit-reversal table, twiddle precomputation, `N_inv_mont` | ✅ done |
| 1 — Math / NTT | `ntt_forward` / `ntt_inverse` (Cooley–Tukey DIT / Gentleman–Sande DIF) | ✅ done |
| 1 — Math / NTT | `poly_ntt_mul` — free-function negacyclic ring multiply | ✅ done |
| 2 — Sampling | `uniform`, `gaussian` samplers | ✅ done |
| 2 — Sampling | `ternary_uniform`, `ternary_hw` samplers | ✅ done |
| 3 — CKKS Core | `Encoder` — `encode` / `decode` (C++ + Python) | ✅ done |
| 3 — CKKS Core | `SecretKey`, `PublicKey`, `KeyGenerator` (NTT-based) | ✅ done |
| 3 — CKKS Core | `Encryptor` / `Decryptor` (public-key, NTT-based) | ✅ done |
| 3 — CKKS Core | `mul_plain` schoolbook C++ reference | ✅ done |
| 4 — FL Ops | Ciphertext addition (`Evaluator::add_inplace`, `add`) | ✅ done |
| 4 — FL Ops | Scalar multiplication | ⬜ not started |
| 5 — Python API | `Poly` binding (`py::array_t`, zero-copy view) | ✅ done |
| 5 — Python API | `Encoder` binding (typed float64 + complex128 overloads) | ✅ done |
| 5 — Python API | `Plaintext`, `Ciphertext` (zero-copy poly views) | ✅ done |
| 5 — Python API | `SecretKey`, `PublicKey` (zero-copy views) | ✅ done |
| 5 — Python API | `KeyGenerator`, `Encryptor`, `Decryptor` (non-copyable) | ✅ done |
| 5 — Python API | `Evaluator` (`add_inplace`, `add`) | ✅ done |
| 6 — Serialization | `Ciphertext::to_bytes` / `from_bytes` (C++ + Python) | ✅ done |
| 6 — Serialization | `PublicKey::to_bytes` / `from_bytes` (C++ + Python) | ✅ done |
| 6 — Serialization | `SecretKey::to_bytes` / `from_bytes` (C++ + Python) | ✅ done |
| 7 — FL Library | `CKKSContext` — key generation, `make_public`, `import_public_key` | ✅ done |
| 7 — FL Library | `CKKSContext.from_keys` — reconstruct from serialised keys | ✅ done |
| 7 — FL Library | `CKKSContext.save` / `load` — disk persistence (3-file layout) | ✅ done |
| 7 — FL Library | `CKKSContext.__getstate__` / `__setstate__` — pickle support | ✅ done |
| 7 — FL Library | `CKKSTensor` — auto-chunking encrypt/decrypt/serialize | ✅ done |
| 7 — FL Library | `CKKSTensor.__add__` / `__iadd__` — homomorphic addition operators | ✅ done |

---

## Test Suite

### C++ (CTest)

All C++ tests are in `tests/cpp/` and registered with CTest. Run with:

```bash
cd build && ctest --output-on-failure        # all suites (C++ + Python; Python needs -DCKKS_BUILD_PYTHON=ON)
cd build && ctest -R test_noise -V           # single suite, verbose
```

| Executable | Source | What it tests |
|------------|--------|---------------|
| `test_math_utils` | `tests/cpp/test_math_utils.cpp` | `mod_add`, `mod_sub`, `MontgomeryCtx` (53 checks) |
| `test_poly` | `tests/cpp/test_poly.cpp` | `Poly` construction, addition, zero-copy (22 checks) |
| `test_ntt` | `tests/cpp/test_ntt.cpp` | Bit-reversal, twiddle tables, round-trips, convolution, linearity (490 checks) |
| `test_sampler` | `tests/cpp/test_sampler.cpp` | Statistical properties, tail bounds, reproducibility (11 290 checks) |
| `test_encoder` | `tests/cpp/test_encoder.cpp` | Encode/decode round-trips, slot accuracy (76 checks) |
| `test_ckks_ops` | `tests/cpp/test_ckks_ops.cpp` | Schoolbook `mul_plain` correctness (44 checks) |
| `test_keygen` | `tests/cpp/test_keygen.cpp` | Ternary distributions, `pk0 + pk1·s = e` identity (142 checks) |
| `test_encryption` | `tests/cpp/test_encryption.cpp` | Encrypt/decrypt round-trips, noise bounds, reproducibility (101 checks) |
| `test_homomorphic_add` | `tests/cpp/test_homomorphic_add.cpp` | Homomorphic addition: slot-level correctness, poly-level noise, commutativity, associativity, modular wrap-around (130 checks) |
| `test_serialization` | `tests/cpp/test_serialization.cpp` | Ciphertext, PublicKey, and SecretKey round-trips, expected byte lengths, homomorphic add after deserialization, truncation and trailing-byte error handling (231 checks) |
| `test_prime_gen` | `tests/cpp/test_prime_gen.cpp` | NTT-friendly prime generation: bit size, congruence q ≡ 1 (mod 2N), primality (48 checks) |
| `test_fl_aggregation` | `tests/test_fl_aggregation.cpp` | FL stress test: 90 clients, Δ=2^40, 90 homomorphic additions, per-slot error < 1e-4; overflow budget check (66 checks) |
| `test_stress_aggregation` | `tests/test_stress_aggregation.cpp` | N ∈ {1024, 4096} FedAvg with 100–200 clients; overflow-boundary sweep at N=64 (5125 checks) |
| `test_noise` | `tests/test_noise.cpp` | **Diagnostic:** prints per-coefficient decryption noise and infinity norm to stdout. Not a pass/fail test. |

### Python

Three of these run under CTest when configured with `-DCKKS_BUILD_PYTHON=ON`
(`py_test_stress_federated`, `py_test_overflow_federated`, `py_test_precision`).

| File | What it tests |
|------|---------------|
| `tests/test_stress_federated.py` | `CKKSContext`/`CKKSTensor` at scale: auto-chunking, 100-client FedAvg, serialize, save/load, blind server, K sweep (25 checks) |
| `tests/test_overflow_federated.py` | Reproduces modular overflow: high-mean vector fails by exactly q/Δ once K·mean > q/(2Δ); zero-mean and single-slot data and client-side 1/K do not (10 checks) |
| `tests/python/test_poly.py` | `Poly` Python binding: construction, addition, zero-copy view, setter (10 checks) |
| `test_py_ckks_simplifly.py` | End-to-end encode → encrypt → decrypt → decode; numerical noise verification |
| `test_py_serialization.py` | `CKKSContext` save/load, ciphertext serialization, homomorphic add after deserialization (6 checks) |
| `tests/test_precision.py` | Precision under 10 homomorphic additions; error < 2^-20 margin (1 check) |
