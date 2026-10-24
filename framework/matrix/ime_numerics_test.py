"""The ``"ime"`` numerics strategy: exactness on the fast and rational paths, against the rational reference."""

import importlib.util

import numpy as np
import pytest

from framework.matrix.ime_numerics import add_round, int_mac, round_odd
from framework.matrix.mac import NUMERICS, tile_mac

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)

RNG = np.random.default_rng(11)


def _ref():
    from framework.matrix.ime_isa_test import _ref_round, ref_fp

    return _ref_round, ref_fp


def _spread(shape, dtype, lo=-30, hi=30):
    v = RNG.standard_normal(shape) * 2.0 ** RNG.integers(lo, hi, shape)
    return v.astype(dtype)


def test_registered():
    assert "ime" in NUMERICS


@pytest.mark.parametrize("rnd", ["frm", "rto", "xct"])
@pytest.mark.parametrize("g", [1, 2, 4])
def test_fp32_groups_with_wide_spreads_take_the_exact_path(g, rnd):
    """fp32 products in a group of 2+ rarely sum exactly in fp64: the rational path must agree."""
    _, ref_fp = _ref()
    A = _spread((3, 8), np.float32)
    B = _spread((8, 4), np.float32)
    C = _spread((3, 4), np.float32)
    got = tile_mac(C, A, B, "fp32", "fp32", numerics="ime", group=g, rnd=rnd)
    assert np.array_equal(
        got.astype(np.float64), ref_fp(C, A, B.T, "fp32", g=g, rnd=rnd)
    )


@pytest.mark.parametrize("rnd", ["frm", "rto", "xct"])
def test_bf16_into_bf16_and_fp64(rnd):
    _, ref_fp = _ref()
    import ml_dtypes

    bf = ml_dtypes.bfloat16
    A = _spread((4, 4), bf, -8, 8)
    B = _spread((4, 4), bf, -8, 8)
    for acc, dt in (("bf16", bf), ("fp64", np.float64)):
        C = _spread((4, 4), dt, -4, 4)
        got = tile_mac(C, A, B, "bf16", acc, numerics="ime", group=2, rnd=rnd)
        assert np.array_equal(
            got.astype(np.float64), ref_fp(C, A, B.T, acc, g=2, rnd=rnd)
        )


def test_fp64_fused_and_grouped_use_rationals():
    _, ref_fp = _ref()
    A, B, C = (_spread(s, np.float64) for s in ((3, 4), (4, 3), (3, 3)))
    for g, rnd in ((1, "xct"), (2, "frm"), (2, "rto")):
        got = tile_mac(C, A, B, "fp64", "fp64", numerics="ime", group=g, rnd=rnd)
        assert np.array_equal(got, ref_fp(C, A, B.T, "fp64", g=g, rnd=rnd))


def test_frm_and_xct_differ_where_fusing_matters():
    # c + a*b with a*b inexact in fp32: unfused (frm) rounds twice, fused (xct) once.
    a = np.array([[1 + 2.0**-12]], np.float32)
    b = np.array([[1 + 2.0**-12]], np.float32)
    c = np.array([[-1.0]], np.float32)
    frm = tile_mac(c, a, b, "fp32", "fp32", numerics="ime", rnd="frm")
    xct = tile_mac(c, a, b, "fp32", "fp32", numerics="ime", rnd="xct")
    assert xct[0, 0] == np.float32(2.0**-11 + 2.0**-24)
    assert frm[0, 0] == np.float32(2.0**-11)


def test_round_odd_and_add_round_helpers():
    from framework.matrix.formats import get_format

    f32 = get_format("fp32")
    x = np.array([1 + 2.0**-30, -(1 + 2.0**-30), 3.0, 1e39, 2.0**-160])
    got = round_odd(x, f32)
    assert got[0] == np.float32(1 + 2.0**-23)
    assert got[1] == -got[0]
    assert got[2] == 3.0
    assert got[3] == np.finfo(np.float32).max  # round to odd never overflows to inf
    assert got[4].view(np.uint32) == 1  # the smallest subnormal
    # 1 + 2^-24 + 2^-60 is just above a tie: RNE goes up, which a plain fp64 sum then fp32 cast misses.
    s = add_round(np.array([1 + 2.0**-24]), np.array([2.0**-60]), f32)
    assert s[0] == np.float32(1 + 2.0**-23)


def test_bad_options():
    a = np.ones((2, 4), np.float32)
    with pytest.raises(ValueError, match="k_chunk"):
        tile_mac(None, a, a.T, "fp32", "fp32", numerics="ime", k_chunk=2)
    with pytest.raises(ValueError, match="psm"):
        tile_mac(None, a, a.T, "fp32", "fp32", numerics="ime", psm=1)
    with pytest.raises(ValueError, match="groups"):
        tile_mac(None, a, a.T, "fp32", "fp32", numerics="ime", group=3)


def test_int_mac_wraps():
    A = np.array([[127, -128]], np.int8)
    B = np.array([[255], [255]], np.uint8)
    C = np.array([[2**31 - 1]], np.int32)
    got = int_mac(C, A, B, 32)
    want = (2**31 - 1 + 127 * 255 - 128 * 255) % 2**32
    assert int(got[0, 0]) == want - 2**32 if want >= 2**31 else want
