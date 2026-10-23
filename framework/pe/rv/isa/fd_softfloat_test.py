"""Randomised cross-checks of ``fd_softfloat`` against independent references.

* RNE add / sub / mul / div / sqrt: numpy's float32 / float64 arithmetic, which
  is IEEE correctly rounded (and does not flush subnormals).
* Fused multiply-add: ``fractions.Fraction`` for fp64 (``float()`` of a
  Fraction is correctly rounded), ``framework.matrix.mac.fma_round_once`` for
  fp32.
* Directed rounding and the inexact flag: the exact Fraction result, bracketed
  by its two neighbouring representable values found with ``np.nextafter``, so
  every mode's answer is picked without sharing any code with the module.

The rounding and flag edge cases (overflow, tininess, ties, signalling NaNs)
are pinned one by one in ``fd_isa_test.py``, through the core.
"""

import math
import random
import struct
from fractions import Fraction

import numpy as np
import pytest

from framework.pe.rv.isa import fd_softfloat as sf

N = 1500

_DTYPE = {sf.F32: np.float32, sf.F64: np.float64}
_UINT = {sf.F32: np.uint32, sf.F64: np.uint64}


def to_np(bits, fmt):
    return np.array([bits], dtype=_UINT[fmt]).view(_DTYPE[fmt])[0]


def to_bits(value, fmt):
    return int(np.array([value], dtype=_DTYPE[fmt]).view(_UINT[fmt])[0])


def exact(bits, fmt):
    return Fraction(float(to_np(bits, fmt)))


def random_finite(rng, fmt, exp_lo=0, exp_hi=None):
    """A random finite bit pattern with biased exponent in [exp_lo, exp_hi]."""
    if exp_hi is None:
        exp_hi = (1 << fmt.exp_bits) - 2
    sign = rng.getrandbits(1)
    exp = rng.randint(exp_lo, exp_hi)
    frac = rng.getrandbits(fmt.frac_bits)
    if rng.random() < 0.1:
        frac &= ~((1 << rng.randint(0, fmt.frac_bits)) - 1)  # short mantissas: ties
    return (sign << (fmt.bits - 1)) | (exp << fmt.frac_bits) | frac


def check_nan_or_equal(got, ref, fmt):
    if np.isnan(ref):
        assert got == fmt.canonical_nan
    else:
        assert got == to_bits(ref, fmt), (hex(got), ref)


@pytest.mark.parametrize("fmt", [sf.F32, sf.F64], ids=["s", "d"])
def test_rne_matches_numpy(fmt):
    rng = random.Random(1 + fmt.bits)
    with np.errstate(all="ignore"):
        for _ in range(N):
            a = random_finite(rng, fmt)
            b = random_finite(rng, fmt)
            na, nb = to_np(a, fmt), to_np(b, fmt)
            for op, ref in (
                (sf.add(a, b, fmt, sf.RNE), na + nb),
                (sf.add(a, b, fmt, sf.RNE, negate_b=True), na - nb),
                (sf.mul(a, b, fmt, sf.RNE), na * nb),
                (sf.div(a, b, fmt, sf.RNE), na / nb),
                (sf.sqrt(a, fmt, sf.RNE), np.sqrt(na)),
            ):
                got, flags = op
                check_nan_or_equal(got, ref, fmt)
                # Overflow from finite operands is exactly an infinite result.
                assert bool(flags & sf.OF) == bool(np.isinf(ref))


def test_fma_fp64_matches_fraction():
    fmt = sf.F64
    rng = random.Random(7)
    for _ in range(N):
        a = random_finite(rng, fmt, 600, 1400)
        b = random_finite(rng, fmt, 600, 1400)
        # An addend near the product's magnitude, so cancellation happens.
        prod = exact(a, fmt) * exact(b, fmt)
        if rng.random() < 0.5 and prod:
            c = to_bits(-float(prod) * (1 + rng.uniform(-1e-12, 1e-12)), fmt)
        else:
            c = random_finite(rng, fmt, 600, 1400)
        want = prod + exact(c, fmt)
        got, flags = sf.fma(a, b, c, fmt, sf.RNE)
        if want == 0:
            assert got & ~(1 << 63) == 0
            continue
        ref = float(want)  # correctly rounded
        assert got == to_bits(ref, fmt), (hex(a), hex(b), hex(c))
        assert bool(flags & sf.NX) == (Fraction(ref) != want)


def test_fma_fp32_matches_fma_round_once():
    from framework.matrix.mac import fma_round_once

    fmt = sf.F32
    rng = random.Random(8)
    with np.errstate(all="ignore"):
        for _ in range(N):
            a = random_finite(rng, fmt, 60, 190)
            b = random_finite(rng, fmt, 60, 190)
            c = random_finite(rng, fmt, 0, 254)
            if rng.random() < 0.5:
                c = to_bits(-np.float32(to_np(a, fmt) * to_np(b, fmt)), fmt)
            ref = fma_round_once(to_np(a, fmt), to_np(b, fmt), to_np(c, fmt), "fp32")
            got, _ = sf.fma(a, b, c, fmt, sf.RNE)
            if ref == 0:
                assert got & 0x7FFFFFFF == 0
            else:
                assert got == to_bits(np.float32(ref), fmt), (hex(a), hex(b), hex(c))


def ref_round(x, fmt, rm):
    """The exact non-zero Fraction ``x`` rounded to ``fmt`` under ``rm``."""
    dtype = _DTYPE[fmt]
    lo = dtype(float(x))
    while Fraction(float(lo)) > x:
        lo = np.nextafter(lo, dtype(-np.inf))
    while Fraction(float(np.nextafter(lo, dtype(np.inf)))) <= x:
        lo = np.nextafter(lo, dtype(np.inf))
    if Fraction(float(lo)) == x:
        return to_bits(lo, fmt), False
    hi = np.nextafter(lo, dtype(np.inf))
    lo_b, hi_b = to_bits(lo, fmt), to_bits(hi, fmt)
    # A zero neighbour takes the sign of x.
    if x > 0 and float(lo) == 0:
        lo_b = 0
    if x < 0 and float(hi) == 0:
        hi_b = 1 << (fmt.bits - 1)
    toward_zero, away = (lo_b, hi_b) if x > 0 else (hi_b, lo_b)
    if rm == sf.RTZ:
        return toward_zero, True
    if rm == sf.RDN:
        return lo_b, True
    if rm == sf.RUP:
        return hi_b, True
    d_lo, d_hi = x - Fraction(float(lo)), Fraction(float(hi)) - x
    if d_lo != d_hi:
        return (lo_b if d_lo < d_hi else hi_b), True
    if rm == sf.RMM:
        return away, True
    return (lo_b if lo_b & 1 == 0 else hi_b), True


@pytest.mark.parametrize("rm", [sf.RNE, sf.RTZ, sf.RDN, sf.RUP, sf.RMM])
@pytest.mark.parametrize("fmt", [sf.F32, sf.F64], ids=["s", "d"])
def test_every_rounding_mode_matches_exact_reference(fmt, rm):
    rng = random.Random(100 * rm + fmt.bits)
    top = (1 << fmt.exp_bits) - 2
    mid = top // 2
    span = top // 4
    for _ in range(N // 3):
        # Exponents kept clear of overflow, but low enough to reach subnormals.
        a = random_finite(rng, fmt, mid - span - 30, mid + span)
        b = random_finite(rng, fmt, mid - span, mid + span)
        if rng.random() < 0.2:
            a = rng.getrandbits(fmt.frac_bits)  # a subnormal operand
        xa, xb = exact(a, fmt), exact(b, fmt)
        cases = [
            (sf.add(a, b, fmt, rm), xa + xb),
            (sf.mul(a, b, fmt, rm), xa * xb),
        ]
        if xb:
            cases.append((sf.div(a, b, fmt, rm), xa / xb))
        for (got, flags), want in cases:
            if want == 0:
                continue
            ref, inexact = ref_round(want, fmt, rm)
            assert got == ref, (hex(a), hex(b), hex(got), hex(ref))
            assert bool(flags & sf.NX) == inexact


@pytest.mark.parametrize("rm", [sf.RNE, sf.RTZ, sf.RDN, sf.RUP, sf.RMM])
@pytest.mark.parametrize("fmt", [sf.F32, sf.F64], ids=["s", "d"])
def test_sqrt_brackets_the_root(fmt, rm):
    """The result r and its neighbour bracket sqrt(x) as each mode requires."""
    rng = random.Random(200 + rm + fmt.bits)
    dtype = _DTYPE[fmt]
    for _ in range(N // 3):
        a = random_finite(rng, fmt) & ~(1 << (fmt.bits - 1))
        x = exact(a, fmt)
        if x == 0:
            continue
        got, flags = sf.sqrt(a, fmt, rm)
        r = to_np(got, fmt)
        r2 = Fraction(float(r)) ** 2
        if r2 == x:
            assert not flags & sf.NX
            continue
        assert flags == sf.NX
        up = Fraction(float(np.nextafter(r, dtype(np.inf)))) ** 2
        down = Fraction(float(np.nextafter(r, dtype(0)))) ** 2
        if rm in (sf.RTZ, sf.RDN):
            assert r2 < x < up
        elif rm == sf.RUP:
            assert down < x < r2
        else:  # nearest: |r - sqrt(x)| <= half an ulp; no ties are possible
            ref = to_bits(np.sqrt(to_np(a, fmt)), fmt)
            assert got == ref


def test_integer_conversions_round_every_mode():
    rng = random.Random(5)
    for _ in range(N):
        a = random_finite(rng, sf.F64, 1000, 1100)
        x = exact(a, sf.F64)
        for rm, fn in (
            (sf.RTZ, math.trunc),
            (sf.RDN, math.floor),
            (sf.RUP, math.ceil),
            (sf.RNE, round),
        ):
            value, flags = sf.to_int(a, sf.F64, rm, 64, True)
            want = fn(x)
            if -(1 << 63) <= want < 1 << 63:
                assert value == want
                assert flags == (sf.NX if want != x else 0)
            else:
                assert flags == sf.NV


def test_from_int_matches_python_float():
    rng = random.Random(6)
    for _ in range(N):
        v = rng.randint(-(1 << 63), (1 << 64) - 1)
        bits, flags = sf.from_int(v, sf.F64, sf.RNE)
        assert bits == struct.unpack("<Q", struct.pack("<d", float(v)))[0]
        assert bool(flags & sf.NX) == (int(float(v)) != v)
