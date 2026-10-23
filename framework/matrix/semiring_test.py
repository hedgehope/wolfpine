"""Tropical products against a brute-force triple loop."""

import numpy as np
import pytest

from framework.matrix import semiring as sr
from framework.matrix.formats import get_format
from framework.matrix.mac import tile_mac

pytest.importorskip("ml_dtypes")


def _brute(C, A, B, reduce):
    M, K = A.shape
    N = B.shape[1]
    out = C.astype(np.float64).copy()
    for i in range(M):
        for j in range(N):
            for k in range(K):
                out[i, j] = reduce(out[i, j], A[i, k] + B[k, j])
    return out


@pytest.mark.parametrize("semiring, reduce", [("min_plus", min), ("max_plus", max)])
@pytest.mark.parametrize(
    "in_fmt, acc_fmt",
    [("fp64", "fp64"), ("fp32", "fp32"), ("bf16", "fp32"), ("fp16", "fp16")],
)
def test_tropical_matches_brute_force(semiring, reduce, in_fmt, acc_fmt):
    rng = np.random.default_rng(8)
    C, A, B = (
        rng.uniform(-4, 4, (5, 6)),
        rng.uniform(-4, 4, (5, 9)),
        rng.uniform(-4, 4, (9, 6)),
    )
    in_f, acc_f = get_format(in_fmt), get_format(acc_fmt)
    a, b = in_f.round(A).astype(np.float64), in_f.round(B).astype(np.float64)
    c = acc_f.round(C).astype(np.float64)
    # Sums of two inputs rounded to the compute dtype, then into the accumulator
    cdt = np.float64 if acc_fmt == "fp64" else np.float32
    want = acc_f.round(
        _brute(c, a.astype(cdt), b.astype(cdt), lambda x, y: reduce(x, float(cdt(y))))
    )
    got = tile_mac(C, A, B, in_fmt, acc_fmt, semiring=semiring)
    np.testing.assert_array_equal(got, want)
    # Min/max are exact and rounding is monotonic: chunking changes nothing
    np.testing.assert_array_equal(
        tile_mac(C, A, B, in_fmt, acc_fmt, semiring=semiring, k_chunk=2), got
    )


@pytest.mark.parametrize("semiring, reduce", [("min_plus", min), ("max_plus", max)])
def test_tropical_int8(semiring, reduce):
    rng = np.random.default_rng(9)
    C, A, B = (
        rng.integers(-300, 300, (4, 4)),
        rng.integers(-128, 128, (4, 7)),
        rng.integers(-128, 128, (7, 4)),
    )
    got = tile_mac(C, A, B, "int8", "int32", semiring=semiring)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, _brute(C, A, B, reduce))


def test_shortest_paths_by_repeated_squaring():
    inf = np.inf
    D = np.array([[0, 3, inf, 7], [8, 0, 2, inf], [5, inf, 0, 1], [2, inf, inf, 0]])
    P = D
    for _ in range(2):  # D^(2^2) covers every path of up to 4 edges
        P = tile_mac(None, P, P, "fp32", "fp32", semiring="min_plus")
    np.testing.assert_array_equal(
        P, [[0, 3, 5, 6], [5, 0, 2, 3], [3, 6, 0, 1], [2, 5, 7, 0]]
    )


def test_empty_k_is_the_identity():
    assert (
        sr.tropical_product(np.empty((2, 0)), np.empty((0, 3)), "min_plus").min()
        == np.inf
    )
    out = tile_mac(
        None,
        np.empty((2, 0), int),
        np.empty((0, 3), int),
        "int8",
        "int32",
        semiring="max_plus",
    )
    assert (out == np.iinfo(np.int32).min).all()
    C = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(
        tile_mac(
            C, np.empty((2, 0)), np.empty((0, 3)), "fp32", "fp32", semiring="min_plus"
        ),
        C,
    )
