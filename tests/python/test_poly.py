"""Tests for the pybind11 Poly bindings — NumPy array interface."""
import sys
import pathlib
import numpy as np

# Find the built extension next to this file or in the build directory.
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "build"))

import ckks_python

# ---------------------------------------------------------------------------
# Construction from NumPy array
# ---------------------------------------------------------------------------

def test_construct_from_numpy():
    arr = np.array([1, 10, 16, 0], dtype=np.uint64)
    p = ckks_python.Poly(4, 17, arr)
    assert list(p.coeffs) == [1, 10, 16, 0]

def test_construct_forcecast():
    # int64 source — forcecast converts to uint64 transparently.
    arr = np.array([1, 2, 3, 4], dtype=np.int64)
    p = ckks_python.Poly(4, 17, arr)
    assert list(p.coeffs) == [1, 2, 3, 4]

def test_construct_size_mismatch():
    arr = np.array([1, 2, 3], dtype=np.uint64)
    try:
        ckks_python.Poly(4, 17, arr)
        assert False, "expected invalid_argument"
    except (ValueError, Exception):
        pass

# ---------------------------------------------------------------------------
# coeffs property — zero-copy view
# ---------------------------------------------------------------------------

def test_coeffs_is_numpy_array():
    p = ckks_python.Poly(4, 17, np.array([1, 2, 3, 4], dtype=np.uint64))
    c = p.coeffs
    assert isinstance(c, np.ndarray)
    assert c.dtype == np.uint64

def test_coeffs_zero_copy_read():
    """View shares memory with the Poly — no copy on read."""
    arr = np.array([1, 2, 3, 4], dtype=np.uint64)
    p = ckks_python.Poly(4, 17, arr)
    view = p.coeffs
    # Mutate through the view and confirm the Poly reflects the change.
    view[0] = 99
    assert p.coeffs[0] == 99

def test_coeffs_setter():
    p = ckks_python.Poly(4, 17)
    p.coeffs = np.array([5, 6, 7, 8], dtype=np.uint64)
    assert list(p.coeffs) == [5, 6, 7, 8]

def test_coeffs_setter_size_mismatch():
    p = ckks_python.Poly(4, 17)
    try:
        p.coeffs = np.array([1, 2, 3], dtype=np.uint64)
        assert False, "expected invalid_argument"
    except (ValueError, Exception):
        pass

# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------

def test_add_wraps():
    q = 17
    p1 = ckks_python.Poly(4, q, np.array([1, 10, 16, 0], dtype=np.uint64))
    p2 = ckks_python.Poly(4, q, np.array([5,  8,  1, 16], dtype=np.uint64))
    p3 = p1 + p2
    assert list(p3.coeffs) == [6, 1, 0, 16]

def test_add_large_prime():
    q = 998244353
    a = ckks_python.Poly(4, q, np.array([q-1, q-1, 0, 1], dtype=np.uint64))
    b = ckks_python.Poly(4, q, np.array([1,   0,   0, q-1], dtype=np.uint64))
    c = a + b
    assert list(c.coeffs) == [0, q-1, 0, 0]

def test_add_does_not_mutate_inputs():
    q = 17
    arr1 = np.array([3, 4, 5, 6], dtype=np.uint64)
    p1 = ckks_python.Poly(4, q, arr1)
    p2 = ckks_python.Poly(4, q, np.array([1, 1, 1, 1], dtype=np.uint64))
    _ = p1 + p2
    assert list(p1.coeffs) == [3, 4, 5, 6]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
