#!/usr/bin/env python3
"""
tests/test_overflow_federated.py

Reproduces and bounds the modular-overflow failure of single-prime CKKS
aggregation.

Why test_stress_federated.py (K=3000, zero-mean data) does not fail
-------------------------------------------------------------------
The limit q/(2*Delta) applies to every polynomial COEFFICIENT of the summed
plaintext, not to slots. The encoder uses the canonical embedding, so
|a_k| <= (2/N) * sum_j |z_j|. Zero-mean random slots, or one large slot, are
diluted by N/2 and stay far below the limit. A packet with a high MEAN is the
worst case: a constant vector v encodes to the constant polynomial v*Delta, so
coefficient 0 grows as K*v*Delta.

Sections
--------
1. Constant vector, K below the limit        -> correct
2. Constant vector, K above the limit        -> wrong by exactly q/Delta in every slot
3. Same K, zero-mean random vector           -> correct (shows coefficient, not slot, is the limit)
4. One huge slot (diluted by N/2)            -> correct beyond the per-slot limit
5. Same overloaded K with client-side 1/K    -> correct (the fix)
6. Mean sweep at fixed K                     -> failure starts where K*mean crosses q/(2*Delta)
"""

import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT))

try:
    from ckks_federated import CKKSContext, CKKSTensor
except ModuleNotFoundError as exc:
    sys.exit(f"Cannot import ckks_federated: {exc}\nBuild with -DCKKS_BUILD_PYTHON=ON")

N, BITS, SCALE_BITS = 1024, 52, 40
TOL = 1e-3

_pass = _fail = 0


def CHECK(cond: bool, msg: str = "") -> None:
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL: {msg}")


def aggregate_sum(ctx: CKKSContext, vectors: list) -> np.ndarray:
    """Encrypt each vector, add homomorphically (server side), decrypt."""
    agg = CKKSTensor(ctx, vectors[0])
    for v in vectors[1:]:
        agg += CKKSTensor(ctx, v)
    return agg.decrypt()


def max_err(ctx: CKKSContext, vectors: list) -> float:
    return float(np.max(np.abs(aggregate_sum(ctx, vectors) - np.sum(vectors, axis=0))))


def main() -> int:
    ctx = CKKSContext(N=N, bits=BITS, scale_bits=SCALE_BITS)
    delta = float(2 ** SCALE_BITS)
    limit = ctx.q / (2 * delta)          # max |coefficient| in value units
    wrap = ctx.q / delta                 # error added when a coefficient wraps
    slots = N // 2
    print(f"q={ctx.q}  limit q/(2Δ)={limit:.1f}  wrap error q/Δ={wrap:.1f}\n")

    k_ok, k_bad = int(limit * 0.9), int(limit * 1.1)

    print(f"--- 1. constant 1.0, K={k_ok} (< limit)")
    e = max_err(ctx, [np.ones(slots)] * k_ok)
    print(f"  max err = {e:.3e}")
    CHECK(e < TOL, "below limit must be correct")

    print(f"--- 2. constant 1.0, K={k_bad} (> limit)")
    out = aggregate_sum(ctx, [np.ones(slots)] * k_bad)
    err = out - k_bad
    print(f"  max err = {np.max(np.abs(err)):.3f}  (q/Δ = {wrap:.3f})")
    CHECK(np.max(np.abs(err)) > 1.0, "above limit must fail")
    CHECK(np.allclose(np.abs(err), wrap, atol=TOL), "error must be exactly ±q/Δ in every slot")

    print(f"--- 3. zero-mean random, K={k_bad} (same K as section 2)")
    rng = np.random.default_rng(0)
    e = max_err(ctx, [rng.standard_normal(slots) for _ in range(k_bad)])
    print(f"  max err = {e:.3e}")
    CHECK(e < TOL, "zero-mean data must not overflow")

    print(f"--- 4. one slot = 1.0 (rest 0), K={k_bad}")
    v = np.zeros(slots); v[0] = 1.0
    e = max_err(ctx, [v] * k_bad)
    print(f"  max err = {e:.3e}")
    CHECK(e < TOL, "single slot is diluted by N/2")

    print(f"--- 5. constant 1.0, K={k_bad}, client-side 1/K")
    out = aggregate_sum(ctx, [np.ones(slots) / k_bad] * k_bad)
    e = float(np.max(np.abs(out - 1.0)))
    print(f"  max err = {e:.3e}")
    CHECK(e < TOL, "pre-scaling must remove overflow")

    k = 1000
    print(f"--- 6. mean sweep, K={k}; predicted failure above mean = {limit / k:.3f}")
    for mean in (0.5 * limit / k, 0.95 * limit / k, 1.05 * limit / k, 2.0 * limit / k):
        e = max_err(ctx, [np.full(slots, mean)] * k)
        ok = e < TOL
        print(f"  mean={mean:.3f}  K*mean/limit={k * mean / limit:.2f}  max err={e:.3e}  {'OK' if ok else 'OVERFLOW'}")
        CHECK(ok == (k * mean < limit), f"outcome at K*mean/limit={k * mean / limit:.2f} disagrees with prediction")

    print(f"\nResults: {_pass} passed, {_fail} failed")
    print("PASS  test_overflow_federated" if _fail == 0 else "FAIL  test_overflow_federated")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
