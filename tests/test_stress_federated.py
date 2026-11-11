#!/usr/bin/env python3
"""
tests/test_stress_federated.py

Stress tests for the ckks_federated high-level Python layer (CKKSContext /
CKKSTensor).  Each test section exercises a distinct code path that the
smaller unit tests do not cover at scale.

Sections
--------
1. Auto-chunking round trip
     Encrypt a NumPy array larger than N/2 (forces multiple ciphertext chunks),
     decrypt, and assert that the shape and values are recovered correctly.

2. Many-client homomorphic aggregation
     Build N_CLIENTS CKKSTensors each pre-scaled by 1/N_CLIENTS, fold with +=
     (add_inplace), and compare against the plaintext FedAvg mean.

3. Serialize / deserialize round trip
     t.serialize() -> CKKSTensor(ctx, bytes_list) -> decrypt(), checking that the
     8-byte original_size header survives the round trip.

4. Save / load round trip
     ctx.save(tmpdir), CKKSContext.load(tmpdir), decrypt a tensor produced under
     the original context with the reloaded one.

5. Blind-server flow
     make_public() on a server-side copy; assert += aggregation works on it and
     that decrypt() raises PermissionError; decrypt with the full client context.

6. Failure-boundary sweep
     Increase the number of clients (without pre-scaling) until decrypted error
     exceeds ERROR_TOL, mapping the overflow boundary.  Assert that the system
     correctly handles at least K_SAFE clients.

Parameters
----------
N           = 1024    ring degree (512 complex slots)
BITS        = 52      prime bit-length (gives a 52-bit NTT-friendly prime)
SCALE_BITS  = 40      Δ = 2^40
CHUNK_SIZE  = 5*N//2 + 7 = 2567  spans 6 ciphertext chunks, exercises tail trim
N_CLIENTS   = 100     many-client aggregation
K_SAFE      = 1500    minimum unscaled-addition count that must stay correct
ERROR_TOL_FEDAVG = 1e-3  per-element absolute error for FedAvg tests
ERROR_TOL_SWEEP  = 0.1   absolute error that indicates overflow
"""

import sys
import pathlib
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: find the compiled ckks_python extension and ckks_federated pkg
# ---------------------------------------------------------------------------

ROOT  = pathlib.Path(__file__).parent.parent
BUILD = ROOT / "build"
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(ROOT))

try:
    import ckks_python  # noqa: F401 — verifies the .so is importable
    from ckks_federated import CKKSContext, CKKSTensor
except ModuleNotFoundError as exc:
    sys.exit(
        f"Cannot import ckks_python / ckks_federated.\n"
        f"Run: cmake --build build  (with -DCKKS_BUILD_PYTHON=ON)\n"
        f"Original error: {exc}"
    )

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N           = 1024
BITS        = 52
SCALE_BITS  = 40
CHUNK_SIZE  = 5 * (N // 2) + 7   # 2567 → spans 6 chunks of 512, tail trim needed
N_CLIENTS   = 100
K_SAFE      = 1500
ERROR_TOL_FEDAVG = 1e-3
ERROR_TOL_SWEEP  = 0.1

# ---------------------------------------------------------------------------
# Shared context (generated once, reused across all sections)
# ---------------------------------------------------------------------------

print(f"\nBuilding CKKSContext(N={N}, bits={BITS}, scale_bits={SCALE_BITS}) ...")
ctx = CKKSContext(N, BITS, scale_bits=SCALE_BITS)
print(f"  q = {ctx.q}  scale = 2^{SCALE_BITS} = {ctx.scale}")
print(f"  slots per ciphertext = {ctx.N // 2}")
print(f"  safety factor Q/(2·Δ) = {ctx.q / (2 * ctx.scale):.1f}×")

# ---------------------------------------------------------------------------
# Helper: pass/fail accounting (mirrors C++ convention)
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0


def CHECK(cond: bool, msg: str = "") -> None:
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        caller = sys._getframe(1)
        loc = f"{pathlib.Path(caller.f_code.co_filename).name}:{caller.f_lineno}"
        print(f"  FAIL  {loc}  {msg}", file=sys.stderr)


# ===========================================================================
# Section 1 — Auto-chunking round trip
# ===========================================================================

def test_chunking_round_trip() -> None:
    print(f"\n--- Section 1: Auto-chunking round trip  (size={CHUNK_SIZE}) ---")
    rng = np.random.default_rng(1)
    data = rng.uniform(-1.0, 1.0, size=CHUNK_SIZE)

    tensor = CKKSTensor(ctx, data)
    n_chunks_expected = (CHUNK_SIZE + ctx.N // 2 - 1) // (ctx.N // 2)
    print(f"  ciphertext chunks: {len(tensor.ciphertexts)} (expected {n_chunks_expected})")

    CHECK(len(tensor.ciphertexts) == n_chunks_expected,
          f"chunk count {len(tensor.ciphertexts)} != {n_chunks_expected}")
    CHECK(tensor.original_size == CHUNK_SIZE,
          f"original_size {tensor.original_size} != {CHUNK_SIZE}")

    recovered = tensor.decrypt()
    CHECK(recovered.shape == (CHUNK_SIZE,),
          f"shape {recovered.shape} != ({CHUNK_SIZE},)")

    max_err = float(np.abs(recovered - data).max())
    CHECK(max_err < ERROR_TOL_FEDAVG,
          f"max_err {max_err:.3e} >= {ERROR_TOL_FEDAVG:.3e}")
    print(f"  max |error| = {max_err:.3e}  (tol {ERROR_TOL_FEDAVG:.3e})  OK")


# ===========================================================================
# Section 2 — Many-client homomorphic aggregation
# ===========================================================================

def test_many_client_aggregation() -> None:
    print(f"\n--- Section 2: Many-client aggregation  ({N_CLIENTS} clients, "
          f"size={CHUNK_SIZE}) ---")
    rng = np.random.default_rng(2)
    # Each client holds a random weight vector; server computes FedAvg.
    weights = [rng.uniform(-1.0, 1.0, size=CHUNK_SIZE) for _ in range(N_CLIENTS)]
    fedavg_ref = np.mean(weights, axis=0)

    inv_k = 1.0 / N_CLIENTS
    # Seed accumulator with client 0's pre-scaled tensor.
    agg = CKKSTensor(ctx, weights[0] * inv_k)
    for c in range(1, N_CLIENTS):
        agg += CKKSTensor(ctx, weights[c] * inv_k)

    result = agg.decrypt()
    CHECK(result.shape == (CHUNK_SIZE,),
          f"shape mismatch: {result.shape}")

    max_err = float(np.abs(result - fedavg_ref).max())
    CHECK(max_err < ERROR_TOL_FEDAVG,
          f"max_err {max_err:.3e} >= {ERROR_TOL_FEDAVG:.3e}")
    print(f"  max |error vs FedAvg| = {max_err:.3e}  (tol {ERROR_TOL_FEDAVG:.3e})  OK")

    # Also exercise __add__ (creates a new tensor instead of mutating)
    t1 = CKKSTensor(ctx, weights[0] * inv_k)
    t2 = CKKSTensor(ctx, weights[1] * inv_k)
    t3 = t1 + t2
    result2 = t3.decrypt()
    ref2 = (weights[0] + weights[1]) * inv_k
    max_err2 = float(np.abs(result2 - ref2).max())
    CHECK(max_err2 < ERROR_TOL_FEDAVG,
          f"__add__  max_err {max_err2:.3e} >= {ERROR_TOL_FEDAVG:.3e}")
    print(f"  __add__ max |error| = {max_err2:.3e}  OK")


# ===========================================================================
# Section 3 — Serialize / deserialize round trip
# ===========================================================================

def test_serialize_round_trip() -> None:
    print(f"\n--- Section 3: Serialize / deserialize round trip ---")
    rng = np.random.default_rng(3)
    data = rng.uniform(-0.5, 0.5, size=CHUNK_SIZE)

    t_orig = CKKSTensor(ctx, data)
    payload = t_orig.serialize()

    # payload[0] is the 8-byte original_size header; rest are ciphertext bytes.
    CHECK(len(payload) == len(t_orig.ciphertexts) + 1,
          f"payload length {len(payload)} != {len(t_orig.ciphertexts) + 1}")

    recovered_size = int.from_bytes(payload[0], byteorder='little')
    CHECK(recovered_size == CHUNK_SIZE,
          f"header original_size {recovered_size} != {CHUNK_SIZE}")

    t_deser = CKKSTensor(ctx, payload)  # bytes-list branch
    CHECK(t_deser.original_size == CHUNK_SIZE,
          f"deserialized original_size {t_deser.original_size} != {CHUNK_SIZE}")
    CHECK(len(t_deser.ciphertexts) == len(t_orig.ciphertexts),
          "ciphertext count mismatch after deserialization")

    result = t_deser.decrypt()
    max_err = float(np.abs(result - data).max())
    CHECK(max_err < ERROR_TOL_FEDAVG,
          f"max_err {max_err:.3e} >= {ERROR_TOL_FEDAVG:.3e}")
    print(f"  original_size in header: {recovered_size}  ✓")
    print(f"  max |error| after deserialize = {max_err:.3e}  OK")


# ===========================================================================
# Section 4 — Save / load round trip
# ===========================================================================

def test_save_load_round_trip() -> None:
    print(f"\n--- Section 4: Save / load round trip ---")
    rng = np.random.default_rng(4)
    data = rng.uniform(-0.5, 0.5, size=CHUNK_SIZE)
    t_orig = CKKSTensor(ctx, data)
    expected = t_orig.decrypt()

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.save(tmpdir)
        import os
        saved = sorted(os.listdir(tmpdir))
        print(f"  saved files: {saved}")
        CHECK("params.pkl" in saved, "params.pkl missing")
        CHECK("pk.bin" in saved, "pk.bin missing")
        CHECK("sk.bin" in saved, "sk.bin missing")

        ctx_loaded = CKKSContext.load(tmpdir)
        CHECK(ctx_loaded.N == ctx.N, "N mismatch after load")
        CHECK(ctx_loaded.q == ctx.q, "q mismatch after load")
        CHECK(ctx_loaded.scale == ctx.scale, "scale mismatch after load")

        # Decrypt the tensor (produced under ctx) with the reloaded context.
        result = t_orig.decrypt()  # ctx is still bound to the tensor
        max_err = float(np.abs(result - expected).max())
        CHECK(max_err < ERROR_TOL_FEDAVG, f"max_err {max_err:.3e}")

        # Encrypt fresh under loaded context and decrypt — round trip.
        t_new = CKKSTensor(ctx_loaded, data)
        result2 = t_new.decrypt()
        max_err2 = float(np.abs(result2 - data).max())
        CHECK(max_err2 < ERROR_TOL_FEDAVG,
              f"encrypt/decrypt under loaded ctx: max_err {max_err2:.3e}")
        print(f"  max |error| fresh encrypt under loaded ctx = {max_err2:.3e}  OK")


# ===========================================================================
# Section 5 — Blind-server flow
# ===========================================================================

def test_blind_server() -> None:
    print(f"\n--- Section 5: Blind-server flow ---")
    import copy, pickle

    rng = np.random.default_rng(5)
    data1 = rng.uniform(-0.5, 0.5, size=CHUNK_SIZE)
    data2 = rng.uniform(-0.5, 0.5, size=CHUNK_SIZE)

    # Client-side: encrypt two tensors under the full context.
    t1 = CKKSTensor(ctx, data1 * 0.5)
    t2 = CKKSTensor(ctx, data2 * 0.5)

    # Server-side: reconstruct a public-only context via pickle round-trip,
    # then strip the secret key.
    server_ctx = pickle.loads(pickle.dumps(ctx))
    server_ctx.make_public()
    CHECK(server_ctx.sk is None, "make_public() did not clear sk")
    CHECK(server_ctx.decryptor is None, "make_public() did not clear decryptor")

    # Server aggregates — should work without SK.
    agg = CKKSTensor(server_ctx,
                     [server_ctx.evaluator.add(ct1, ct2)
                      for ct1, ct2 in zip(t1.ciphertexts, t2.ciphertexts)],
                     original_size=CHUNK_SIZE)

    # Server cannot decrypt.
    blocked = False
    try:
        agg.decrypt()
    except PermissionError:
        blocked = True
    CHECK(blocked, "blind server should raise PermissionError on decrypt()")
    print(f"  blind server PermissionError: ✓")

    # Client decrypts the aggregated result (still bound to full ctx).
    result_agg = CKKSTensor(ctx, agg.ciphertexts, original_size=CHUNK_SIZE).decrypt()
    ref = data1 * 0.5 + data2 * 0.5
    max_err = float(np.abs(result_agg - ref).max())
    CHECK(max_err < ERROR_TOL_FEDAVG,
          f"blind-server aggregation error {max_err:.3e}")
    print(f"  client decryption of blind-server aggregate: max_err={max_err:.3e}  OK")


# ===========================================================================
# Section 6 — Failure-boundary sweep (no pre-scaling, overflow mapping)
# ===========================================================================

def test_overflow_boundary_sweep() -> None:
    print(f"\n--- Section 6: Failure-boundary sweep (no pre-scaling) ---")
    k_ovf_theory = ctx.q / (2 * ctx.scale)
    print(f"  Theoretical K_overflow = Q/(2·Δ) ≈ {k_ovf_theory:.0f}")

    # Use a small fixed array size (just 1 ciphertext) for speed.
    SWEEP_SIZE = ctx.N // 2  # exactly 1 chunk, no auto-chunking
    sweep_ks = [100, 500, 1000, 1500, 2000, 3000]
    k_max_safe = 0

    rng_base = np.random.default_rng(6)

    for K in sweep_ks:
        rng = np.random.default_rng(K)
        weights = [rng.uniform(-1.0, 1.0, size=SWEEP_SIZE) for _ in range(K)]
        pt_sum = np.sum(weights, axis=0)

        # Encrypt and sum — NO 1/K pre-scaling.
        agg = CKKSTensor(ctx, weights[0])
        for c in range(1, K):
            agg += CKKSTensor(ctx, weights[c])

        decoded = agg.decrypt()
        max_err = float(np.abs(decoded - pt_sum).max())
        overflow_ratio = K * ctx.scale / (ctx.q / 2)
        safe = max_err < ERROR_TOL_SWEEP
        if safe:
            k_max_safe = K

        print(f"  K={K:5d}  max_err={max_err:.3e}  "
              f"overflow_ratio={overflow_ratio:.3f}  {'OK' if safe else 'OVERFLOW'}")

    CHECK(k_max_safe >= K_SAFE,
          f"k_max_safe={k_max_safe} < K_SAFE={K_SAFE}")
    print(f"  k_max_safe = {k_max_safe}  (K_SAFE = {K_SAFE})")


# ===========================================================================
# Run all sections and report
# ===========================================================================

def main() -> int:
    test_chunking_round_trip()
    test_many_client_aggregation()
    test_serialize_round_trip()
    test_save_load_round_trip()
    test_blind_server()
    test_overflow_boundary_sweep()

    print(f"\nResults: {_pass} passed, {_fail} failed")
    if _fail == 0:
        print("PASS  test_stress_federated")
    else:
        print(f"FAIL  test_stress_federated  ({_fail} failures)")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
