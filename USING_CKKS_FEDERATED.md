# USING_CKKS_FEDERATED — Integration Guide

> **Purpose of this file:** A self-contained reference for an agent (Claude Code or a human
> engineer) integrating `ckks_federated` into a *different* project.  Copy this file into your
> project root or paste it into your own `CLAUDE.md`.
>
> For internal development of the library itself, see `CLAUDE.md`.

---

## 1. What this library is

`ckks_federated` is a high-performance, C++-backed implementation of the **CKKS homomorphic
encryption scheme** with Python bindings, built specifically for **additive aggregation in
Federated Learning**.

It exposes two layers:

| Layer | Module | Purpose |
|---|---|---|
| Low-level | `ckks_python` (C++ extension `.so`) | Ring arithmetic, NTT, samplers, keys, encode/encrypt/decrypt, evaluator |
| High-level | `ckks_federated` (pure Python) | `CKKSContext` + `CKKSTensor` — auto-chunking, key persistence, Flower-ready serialization |

**The 90% use case:** encrypt NumPy weight tensors on each FL client, ship them to the
server, and add them together homomorphically without the server ever seeing the plaintext.

---

## 2. ⚠️ Limitations — read before writing any code

These are **hard constraints** of the current implementation, not configuration knobs.
Ignoring them produces silently incorrect results or runtime errors.

### 2.1 Additive only

- Supported operation: **`Ciphertext + Ciphertext`** (coefficient-wise modular addition).
- **Not supported** (not implemented): ciphertext × ciphertext multiplication,
  ciphertext × scalar, SIMD rotations, relinearization, key-switching, bootstrapping.
- A schoolbook `mul_plain` (ciphertext × plaintext) exists in C++ only, is O(N²), and is
  **not exposed to Python**.
- **Consequence for FedAvg:** the `1/K` averaging divisor must be applied **client-side in
  plaintext** before encryption.  The server can only call `+=`; it cannot rescale.

### 2.2 Single modulus, zero multiplicative depth

- There is no RNS/CRT modulus chain and no rescaling operation.
- Encoding precision: `~Δ⁻¹` per coefficient.  Noise from one fresh encryption decodes to
  `O(N·σ/Δ)` per slot (`σ=3.2`, `Δ=2^40` → ~3×10⁻⁹ per slot for N=1024).
- After K homomorphic additions, noise is `O(K·N·σ/Δ)` — negligible for FL workloads
  (e.g. N=4096, K=10,000 clients: max decoded error < 10⁻⁴).

### 2.3 Overflow budget

Every polynomial coefficient must remain `< q/2` after all additions.  With Δ=2^40 and a
52-bit prime, the safe additive budget is:

```
unscaled additions before overflow  =  q / (2·Δ)  ≈  2^52 / 2^41  =  2048
```

**With client-side `1/K` pre-scaling the budget is effectively unlimited** — the aggregate
coefficient is bounded by Δ regardless of how many clients participate.  Always pre-scale.

### 2.4 Parameter constraints

| Constraint | Reason |
|---|---|
| `N` must be a power of two | NTT requires it |
| `q` must be prime with `q ≡ 1 (mod 2N)` | NTT primitive root requirement |
| `q < 2^62` | Montgomery reduction uses 64-bit words |
| `q` is derived automatically by `generate_prime(bits, N)` | Do not hand-craft `q` |

### 2.5 Real data only (via `CKKSTensor`)

`CKKSTensor.decrypt()` returns `np.real(decoded)` — the imaginary parts are discarded.
Complex slot vectors are accessible only through the low-level `Encoder.decode()` API.

### 2.6 Security responsibilities of the caller

- **Key generation:** `seed=0` (default) uses OS entropy — use this in production.
  Non-zero seeds are deterministic; use only for unit tests.
- **Security level:** the library does **not** enforce NIST/HE standard parameter sets.
  The caller chooses `(N, bits)`.  Minimum recommended: `N=4096, bits=52` for 128-bit
  computational security.  `N=1024` is fine for experiments; do not use in production.
- **Secret key:** never send `sk.to_bytes()` to the server.  Use `ctx.make_public()` or
  `CKKSContext.load(path, public_only=True)` for the server-side context.

### 2.7 Platform notes

- Compiled with `-march=native`: the `.so` is **not portable across CPU architectures**.
  Rebuild on each target machine.
- No GPU support.  Single-threaded per operation (OpenMP-SIMD vector hints only).
- Requires a C++17 compiler (GCC 9+ or Clang 10+) and Python 3.8+.

---

## 3. Installation

### Option A — pip (recommended for consumer projects)

Requires a C++17 compiler and pybind11 accessible.

```bash
# From a zip distribution
unzip ckks_project.zip
cd ckks_project
pip3 install .
```

```bash
# Or directly from the repo directory
cd /path/to/CKKS_python
pip3 install .
```

This compiles `ckks_python` (the C++ extension) and installs both `ckks_python` and
`ckks_federated` into the active Python environment.

**Declared dependencies** (`setup.py`):
`numpy`, `flwr` (Flower), `tensorflow`.
If your project does not use Flower or TensorFlow, you can install them separately:
```bash
pip3 install numpy
pip3 install .   # flwr/tensorflow are optional at runtime for the core HE logic
```

### Option B — CMake build (dev / CI / embedding)

```bash
cd /path/to/CKKS_python
mkdir -p build && cd build
cmake .. -DCKKS_BUILD_PYTHON=ON \
         -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
```

The compiled extension is placed at `build/ckks_python.cpython-<ver>-<arch>.so`.
To import it without installing:

```python
import sys
sys.path.insert(0, "/path/to/CKKS_python/build")
import ckks_python
from ckks_federated import CKKSContext, CKKSTensor
```

### Verifying the install

```python
import ckks_python
print(dir(ckks_python))
# Should include: Ciphertext, Decryptor, Encoder, Encryptor, Evaluator,
#                 KeyGenerator, Plaintext, Poly, PublicKey, SecretKey, generate_prime
```

---

## 4. Quick start

### 4.1 Minimal round trip

```python
import numpy as np
from ckks_federated import CKKSContext, CKKSTensor

# 1. Create a context (generates fresh keys internally)
ctx = CKKSContext(N=4096, bits=52, scale_bits=40)
# slots = N/2 = 2048 per ciphertext; q is auto-generated

# 2. Encrypt a NumPy array (any size — auto-chunked)
weights = np.random.uniform(-1.0, 1.0, size=10_000).astype(np.float64)
tensor = CKKSTensor(ctx, weights)
print(f"chunks: {len(tensor.ciphertexts)}")   # ceil(10000 / 2048) = 5

# 3. Decrypt and verify
recovered = tensor.decrypt()
print(f"shape: {recovered.shape}")             # (10000,)
print(f"max error: {np.abs(recovered - weights).max():.2e}")  # ~1e-8
```

### 4.2 FedAvg pattern (canonical multi-client aggregation)

> This is the only supported aggregation pattern.
> The server performs **only addition** — no rescaling.

```python
import numpy as np
from ckks_federated import CKKSContext, CKKSTensor

N_CLIENTS = 100
WEIGHT_SIZE = 10_000   # e.g. a flattened ResNet layer

ctx = CKKSContext(N=4096, bits=52, scale_bits=40)

# --- On each client (independently, in parallel) ---
rng = np.random.default_rng(42)
weights = [rng.uniform(-1.0, 1.0, size=WEIGHT_SIZE) for _ in range(N_CLIENTS)]

inv_k = 1.0 / N_CLIENTS   # ← pre-scaling happens HERE, in plaintext, before encryption

client_tensors = [CKKSTensor(ctx, w * inv_k) for w in weights]

# --- On the server (only uses +=, never sees plaintext) ---
agg = client_tensors[0]
for t in client_tensors[1:]:
    agg += t

# --- Trusted decryptor (client / aggregator) ---
fedavg = agg.decrypt()    # shape (WEIGHT_SIZE,)

# Verify against plaintext reference
reference = np.mean(weights, axis=0)
print(f"max error: {np.abs(fedavg - reference).max():.2e}")   # < 1e-4
```

---

## 5. Federated workflow — serialization, persistence, blind server

### 5.1 Serialize ciphertexts for network transport (gRPC / Flower)

```python
# Client: encrypt and serialize
tensor = CKKSTensor(ctx, weights * inv_k)
payload: list[bytes] = tensor.serialize()
# payload[0] = original_size as 8-byte little-endian int
# payload[1..] = ciphertext bytes

# Server: receive bytes, reconstruct, aggregate
received = CKKSTensor(ctx, payload)    # bytes-list branch
agg += received
```

### 5.2 Key persistence

```python
# Save (call once on the trusted key-generation node)
ctx.save("/secure/path/to/keys/")
# Writes:  params.pkl  pk.bin  sk.bin

# Client reload (full context, can encrypt AND decrypt)
ctx_client = CKKSContext.load("/secure/path/to/keys/")

# Server reload (public-only, can ONLY aggregate)
ctx_server = CKKSContext.load("/secure/path/to/keys/", public_only=True)
```

### 5.3 Blind-server flow

```python
import pickle

# One way: make_public() on an in-memory copy
ctx_server = pickle.loads(pickle.dumps(ctx))   # deep copy via pickle
ctx_server.make_public()
assert ctx_server.sk is None
assert ctx_server.decryptor is None

# Server can still aggregate
agg = CKKSTensor(ctx_server,
                 [ctx_server.evaluator.add(ct1, ct2)
                  for ct1, ct2 in zip(t1.ciphertexts, t2.ciphertexts)],
                 original_size=t1.original_size)

# Server CANNOT decrypt
try:
    agg.decrypt()
except PermissionError:
    print("blocked, as expected")

# Client decrypts using its own full context
result = CKKSTensor(ctx_client, agg.ciphertexts,
                    original_size=agg.original_size).decrypt()
```

### 5.4 Reconstruct context from raw key bytes

```python
ctx2 = CKKSContext.from_keys(
    N=ctx.N, q=ctx.q, scale=ctx.scale,
    pk_bytes=ctx.export_public_key(),
    sk_bytes=ctx.sk.to_bytes(),          # omit for public-only
)
```

---

## 6. API reference

### 6.1 `ckks_federated.CKKSContext`

```python
from ckks_federated import CKKSContext
```

| Signature | Description |
|---|---|
| `CKKSContext(N, bits, scale_bits=40)` | Generate fresh keys. `q = generate_prime(bits, N)`, `scale = 2**scale_bits`. |
| `CKKSContext.from_keys(N, q, scale, pk_bytes, sk_bytes=None, public_only=False)` | Reconstruct from serialised bytes; no key generation. |
| `ctx.save(path)` | Write `params.pkl`, `pk.bin`, and (if sk present) `sk.bin` to `path/`. |
| `CKKSContext.load(path, public_only=False)` | Load context from disk. `public_only=True` skips SK/Decryptor. |
| `ctx.make_public()` | Drop SK and Decryptor in-place (server blind mode). |
| `ctx.export_public_key() → bytes` | `pk.to_bytes()`. |
| `ctx.import_public_key(pk_bytes)` | Replace pk and rebuild Encryptor. |
| `ctx.N`, `ctx.q`, `ctx.scale`, `ctx.bits` | Parameters (read). |
| `ctx.encoder`, `ctx.encryptor`, `ctx.decryptor`, `ctx.evaluator`, `ctx.sk`, `ctx.pk` | Underlying components (access directly if needed). |
| Pickle support | `__getstate__` / `__setstate__` — works with `multiprocessing`, Flower internals. |

### 6.2 `ckks_federated.CKKSTensor`

```python
from ckks_federated import CKKSTensor
```

| Signature | Description |
|---|---|
| `CKKSTensor(ctx, data)` | `data` can be: NumPy/list array → encrypt with auto-chunking; `list[bytes]` → deserialize from `serialize()` payload; `list[Ciphertext]` → wrap existing ciphertexts (pass `original_size=` explicitly). |
| `t.decrypt() → np.ndarray` | Decrypt, concatenate chunks, strip imaginary part, trim to `original_size`. Raises `PermissionError` if context has no SK. |
| `t.serialize() → list[bytes]` | `[original_size (8 B LE)] + [ct.to_bytes() …]`. For gRPC / Flower. |
| `t1 + t2 → CKKSTensor` | Homomorphic addition (new object). Checks chunk-count equality. |
| `t1 += t2` | In-place homomorphic addition. Does **not** check chunk-count equality — caller must ensure matching sizes. |
| `t.original_size` | Integer: number of elements in the original array (metadata). |
| `t.ciphertexts` | `list[ckks_python.Ciphertext]` — the underlying encrypted chunks. |

### 6.3 Low-level `ckks_python` (C++ extension)

```python
import sys; sys.path.insert(0, "build")   # if not installed via pip
import ckks_python
```

#### Prime generation

| Signature | Description |
|---|---|
| `ckks_python.generate_prime(bits, N) → int` | Returns the largest prime `q` with exactly `bits` bits satisfying `q ≡ 1 (mod 2N)`. Throws on failure. |

#### `Poly`

| Signature | Description |
|---|---|
| `Poly(N, q)` | Zero polynomial of degree N mod q. |
| `Poly(N, q, coeffs: np.ndarray[uint64])` | Construct from a length-N uint64 array. |
| `p + other → Poly` | Coefficient-wise modular addition. |
| `p.coeffs → np.ndarray[uint64]` | **Zero-copy writable view** of the internal buffer. |
| `p.N`, `p.q` | Ring degree and modulus (read-only). |

#### `Encoder`

| Signature | Description |
|---|---|
| `Encoder(N, q, scale)` | Δ is the scaling factor (typically `2**40`). |
| `encode(z: np.ndarray[float64]) → Poly` | Real array (≤ N/2 elements, zero-padded) → polynomial. |
| `encode(z: np.ndarray[complex128]) → Poly` | Complex array → polynomial. |
| `decode(p: Poly) → np.ndarray[complex128]` | Polynomial → slot vector (length N/2). |
| `encoder.slots → int` | N/2. |

#### `KeyGenerator`

| Signature | Description |
|---|---|
| `KeyGenerator(N, q, sigma=3.2, seed=0)` | Non-copyable. `seed=0` → OS entropy. |
| `gen_secret_key(hamming_weight=0) → SecretKey` | `hw=0`: uniform ternary. `hw>0`: fixed-weight (must be even). |
| `gen_public_key(sk) → PublicKey` | One NTT-based multiplication. |

#### `Encryptor`

| Signature | Description |
|---|---|
| `Encryptor(N, q, pk, sigma=3.2, seed=0)` | Non-copyable. Stores a copy of `pk`. |
| `encrypt(pt: Plaintext) → Ciphertext` | Fresh encryption; draws new `u`, `e0`, `e1` each call. |

#### `Decryptor`

| Signature | Description |
|---|---|
| `Decryptor(N, q, sk)` | Non-copyable. Stores a copy of `sk`. |
| `decrypt(ct: Ciphertext) → Plaintext` | Returns `c0 + c1·s` as a `Plaintext`. |

#### `Evaluator`

| Signature | Description |
|---|---|
| `Evaluator(N, q)` | Stateless; all methods are thread-safe for distinct ciphertexts. |
| `add_inplace(ct1, ct2)` | `ct1 += ct2` in-place. Raises `ValueError` on `(N, q)` mismatch. |
| `add(ct1, ct2) → Ciphertext` | Returns a new ciphertext equal to `ct1 + ct2`. |

#### `Plaintext` / `Ciphertext`

| Signature | Description |
|---|---|
| `Plaintext(poly: Poly)` | Wrap a `Poly` as a CKKS plaintext. |
| `pt.poly → np.ndarray[uint64]` | Zero-copy read-only view of the encoded polynomial. |
| `Ciphertext(c0: Poly, c1: Poly)` | Construct directly from two polynomials. |
| `ct.c0`, `ct.c1 → np.ndarray[uint64]` | Zero-copy read-only views. |
| `ct.to_bytes() → bytes` | Serialize (little-endian, self-describing). |
| `Ciphertext.from_bytes(data) → Ciphertext` | Deserialize; raises `RuntimeError` on malformed data. |

#### `SecretKey` / `PublicKey`

| Signature | Description |
|---|---|
| `sk.s → np.ndarray[uint64]` | Zero-copy view of the ternary secret polynomial. |
| `sk.to_bytes()` / `SecretKey.from_bytes(data)` | Serialize / deserialize. |
| `pk.pk0`, `pk.pk1 → np.ndarray[uint64]` | Zero-copy views. |
| `pk.to_bytes()` / `PublicKey.from_bytes(data)` | Serialize / deserialize. |

---

## 7. Parameter selection cheat-sheet

The **only** user-facing parameters are `N` (ring degree) and `bits` (prime bit-size).
Everything else is derived.

| N | bits | scale_bits | Slots (N/2) | Max array per CT | Unscaled add. budget¹ | Typical use |
|---|------|-----------|-------------|-----------------|----------------------|-------------|
| 1024 | 52 | 40 | 512 | 512 float64 | 2 048 | Experiments, small vectors |
| 2048 | 52 | 40 | 1024 | 1024 float64 | 2 048 | Small models |
| 4096 | 52 | 40 | 2048 | 2048 float64 | 2 048 | Production FL (128-bit sec) |
| 4096 | 55 | 40 | 2048 | 2048 float64 | 16 384 | Higher add. budget |
| 8192 | 55 | 40 | 4096 | 4096 float64 | 16 384 | Large models |
| 8192 | 60 | 40 | 4096 | 4096 float64 | 524 288 | Very large add. budget |

> ¹ **Unscaled addition budget** = Q / (2·Δ) ≈ 2^bits / 2^(scale_bits+1).
> With client-side **1/K pre-scaling** this limit never applies — use pre-scaling.

**Arrays larger than N/2** are automatically split into multiple ciphertext chunks by
`CKKSTensor`; `decrypt()` reassembles them transparently.

---

## 8. Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: python` | System uses `python3` | Use `python3` everywhere |
| `import ckks_python` fails after moving to another machine | Built with `-march=native` | Rebuild on the target host |
| `add_inplace` silently produces wrong result | Tensors have different chunk counts | Ensure both tensors were created with the same `original_size` / slot layout |
| `decrypt()` returns values near 0 for all slots | Server-side context has no SK | Use the client-side `ctx` for decryption |
| Large imaginary parts in low-level decode | Using `Encoder.decode` directly — imaginary is real noise | Use `CKKSTensor` which calls `np.real()`, or discard `.imag` manually |
| Aggregated result is wildly wrong (not just noisy) | Overflow: K × Δ > q/2 without pre-scaling | Always pre-scale client data by `1/K` before encryption |
| `ValueError: (N, q) mismatch` from `add_inplace` | Mixing ciphertexts from different contexts | Use one shared `CKKSContext` per FL round |
| Non-deterministic failures with fixed seed | `seed=0` used — this is correct (OS entropy) | Set an explicit `seed=1` or similar for reproducible tests |

---

## 9. Running the tests

```bash
# All C++ unit + stress tests
cd build && ctest --output-on-failure

# Python high-level stress test (CKKSContext / CKKSTensor)
python3 tests/test_stress_federated.py

# Python serialization smoke test
python3 test_py_serialization.py

# End-to-end encode → encrypt → decrypt → decode
python3 test_py_ckks_simplifly.py
```

---

## 10. Further reading

| Resource | What it covers |
|---|---|
| `CLAUDE.md` (this repo) | Internal architecture, module roadmap, implementation status, C++ API surface, build system |
| `tests/test_stress_federated.py` | Runnable examples of every high-level use case (chunking, multi-client FedAvg, serialize, save/load, blind server, overflow sweep) |
| `tests/test_fl_aggregation.cpp` | Canonical C++ FedAvg — the reference for the noise/overflow budget math |
| `ckks_federated/__init__.py` | Full source of the high-level layer |
| `bindings/module.cpp` | pybind11 wrappers — ground truth for Python-callable signatures |
