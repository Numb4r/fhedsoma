#!/usr/bin/env python3
"""
tests/test_precision.py

Precision test for CKKS polynomial arithmetic.

The encode → add (C++) → decode pipeline is the floating-point accuracy layer
that sits beneath a full CKKS scheme.  This test verifies that 10 polynomial
additions accumulate less than 2^{-20} absolute error per coefficient.

Parameters
----------
N      = 16          polynomial degree (power of 2, NTT-compatible)
Q      = 2^62 - 57   62-bit prime; large enough that 10 additions never
                     overflow (10 * SCALE << Q/2)
SCALE  = 2^40        encoding scale Δ; sets the precision floor at 1/Δ ≈ 9e-13
N_ADD  = 10          number of additions performed
TOL    = 2^-20       ≈ 9.5e-7; error bound to assert against

Error budget per coefficient
-----------------------------
  encode rounding : ≤ 0.5 / Δ  per operand
  10 additions    : ≤ 10 × 0.5 / Δ = 5 / 2^40 ≈ 4.5e-12  << TOL
"""
import sys
import pathlib

import numpy as np

BUILD = pathlib.Path(__file__).parent.parent / "build"
sys.path.insert(0, str(BUILD))

try:
    import ckks_python
except ModuleNotFoundError as exc:
    sys.exit(
        f"Cannot import ckks_python from {BUILD}.\n"
        "Run: cd build && make  (with -DCKKS_BUILD_PYTHON=ON)\n"
        f"Original error: {exc}"
    )

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N     = 16
Q     = 4_611_686_018_427_387_847   # 2^62 − 57
SCALE = 1 << 40                     # 2^40
N_ADD = 10
TOL   = 2 ** -20

# ---------------------------------------------------------------------------
# Encoder / decoder
# ---------------------------------------------------------------------------

def encode(v: np.ndarray) -> ckks_python.Poly:
    """
    Map a real vector v[0..N-1] → polynomial in Z_q[X]/(X^N+1).

    Each coefficient is  round(v[i] * Δ) mod q, stored as an unsigned residue
    in [0, q) so negative floats map to q − |round(v[i]*Δ)|.
    """
    if len(v) != N:
        raise ValueError(f"vector length {len(v)} != N={N}")
    scaled = np.rint(v * SCALE).astype(np.int64)
    # Lift negative residues into [0, q).
    coeffs = np.where(scaled < 0, scaled + Q, scaled).astype(np.uint64)
    return ckks_python.Poly(N, Q, coeffs)


def decode(poly: ckks_python.Poly) -> np.ndarray:
    """
    Recover a real vector from a polynomial.

    Coefficients are centred back into (−q/2, q/2] before dividing by Δ,
    so that residues representing negative numbers are correctly signed.
    """
    raw = poly.coeffs.astype(np.int64)   # safe: Q < 2^63
    half_q = Q >> 1
    centred = np.where(raw > half_q, raw - Q, raw)
    return centred.astype(np.float64) / SCALE

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_precision_10_additions() -> None:
    rng = np.random.default_rng(42)
    v = rng.uniform(-0.5, 0.5, size=N)   # values well within safe range

    # Encode the same vector N_ADD times, then sum the polynomials via C++.
    polys = [encode(v) for _ in range(N_ADD)]
    total: ckks_python.Poly = polys[0]
    for p in polys[1:]:
        total = total + p

    decoded  = decode(total)
    expected = v * N_ADD

    err     = np.abs(decoded - expected)
    max_err = float(err.max())

    print(f"  scale     = 2^40 = {SCALE}")
    print(f"  additions = {N_ADD}")
    print(f"  max |error| per coefficient = {max_err:.3e}")
    print(f"  tolerance  2^-20            = {TOL:.3e}")
    print(f"  margin     (tol / err)      = {TOL / max_err:.1f}×")

    assert max_err < TOL, (
        f"Precision failure: max error {max_err:.3e} exceeds "
        f"2^-20 = {TOL:.3e}"
    )


if __name__ == "__main__":
    try:
        test_precision_10_additions()
        print("PASS  test_precision_10_additions")
    except AssertionError as exc:
        print(f"FAIL  {exc}")
        sys.exit(1)
