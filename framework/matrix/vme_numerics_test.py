"""The ``"zvt"`` strategy (Zvt v0.4's accumulation rule) through ``tile_mac``: the cases that tell it apart."""

import importlib.util

import numpy as np
import pytest

from framework.matrix.formats import get_format
from framework.matrix.mac import NUMERICS, tile_mac
from framework.matrix.vme_numerics import add_rto_fp32, rto_fp32, rvbna_dot

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


def test_registered():
    assert "zvt" in NUMERICS


def test_fp32_rounds_the_product_before_the_add():
    # a * a = 1 + 2^-22 + 2^-46 exactly; rounded to fp32 first it loses the 2^-46, so C + p = 0.
    a = np.float32(1 + 2**-23)
    C = np.array([[-(1 + 2**-22)]], np.float32)
    got = tile_mac(
        C, np.array([[a]]), np.array([[a]]), "fp32", "fp32", numerics="zvt", kmax=1
    )
    assert got[0, 0] == 0.0
    fused = np.float64(a) * np.float64(a) + np.float64(C[0, 0])
    assert np.float32(fused) == np.float32(
        2.0**-46
    )  # a fused multiply-add would keep it


def test_fp64_k_order_is_increasing_k():
    A = np.array([[1.0, 1e-17, -1.0]])
    B = np.array([[1.0], [1.0], [1.0]])
    got = tile_mac(np.zeros((1, 1)), A, B, "fp64", "fp64", numerics="zvt", kmax=1)
    assert got[0, 0] == (0.0 + 1.0 + 1e-17) - 1.0 == 0.0


def test_rvbna_loses_a_cancelled_tail_where_an_exact_sum_would_not():
    """RVBNA aligns to the largest product: after 1 - 1 cancels, a tiny third product is one jammed LSB."""
    e5m2 = get_format("fp8_e5m2")
    A = e5m2.round(np.array([[1.0, 1.0, 2.0**-14]]))
    B = e5m2.round(np.array([[1.0], [-1.0], [2.0**-14]]))
    r = rvbna_dot(A, B, "fp8_e5m2", "fp8_e5m2", 4)
    # q - 1 + g = 25 fractional bits below the max reference exponent (n = 4: g = 2, products have 2 integer bits).
    assert r[0, 0] == np.float32(2.0**-25)
    assert np.float32(2.0**-28) != r[0, 0]  # the exact sum


def test_round_to_odd_add_and_overflow():
    c = np.array([1.0], np.float32)
    s = np.array([2.0**-30], np.float32)
    assert add_rto_fp32(c, s)[0] == np.float32(
        1 + 2**-23
    )  # RTO, not RNE (which gives 1.0)
    big = np.array([3.4e38], np.float32)
    assert np.isposinf(
        add_rto_fp32(big, big)[0]
    )  # RVBNA's RTO: overflow is an infinity
    assert rto_fp32(np.array([2.0**-160]))[0] == np.float32(
        2.0**-149
    )  # gradual underflow, odd
    assert rto_fp32(np.array([-(2.0**-160)]))[0] == -np.float32(2.0**-149)


def test_narrow_inputs_go_through_rvbna_with_static_n():
    bf16 = get_format("bf16")
    rng = np.random.default_rng(3)
    A = bf16.round(rng.standard_normal((4, 2)))
    B = bf16.round(rng.standard_normal((2, 5)))
    C = rng.standard_normal((4, 5)).astype(np.float32)
    got = tile_mac(C, A, B, "bf16", "fp32", numerics="zvt", fmt_b="bf16", kmax=2)
    want = add_rto_fp32(C, rvbna_dot(A, B, "bf16", "bf16", 2))
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    # bf16 products of unit-scale inputs are exact in 26 aligned bits: the RTO of the exact sum.
    exact = A.astype(np.float64) @ B.astype(np.float64)
    assert np.array_equal(rvbna_dot(A, B, "bf16", "bf16", 2), rto_fp32(exact))


def test_rejects_non_zvt_types():
    with pytest.raises(ValueError, match="not a Zvt type"):
        tile_mac(
            np.zeros((1, 1), np.float32),
            np.ones((1, 1)),
            np.ones((1, 1)),
            "fp64",
            "fp32",
            numerics="zvt",
        )
    with pytest.raises(ValueError, match="k_chunk"):
        tile_mac(
            None,
            np.ones((1, 1)),
            np.ones((1, 1)),
            "fp32",
            "fp32",
            numerics="zvt",
            k_chunk=1,
        )
