"""The host-float fast path in ``fd_softfloat`` is the exact path, bit for bit and flag for flag.

``add`` / ``mul`` / ``div`` / ``sqrt`` / ``fma`` try the fast path under RNE and
fall back to ``*_soft`` (the exact integer path, itself checked against
independent references in ``fd_softfloat_test.py``). Every case here runs both
and compares ``(bits, fflags)``, over operand generators aimed at the places a
fast path would go wrong: ties and short significands, cancellation, exponent
spreads wide enough that a double cannot hold the exact fp32 sum, the
overflow and underflow thresholds, exact quotients and perfect squares, and
every class (zeros, subnormals, infinities, NaNs). Each generator also asserts
the fast path actually answered a good share of its cases, so a regression
that silently always defers is caught too.
"""

import random

import pytest

from framework.pe.rv.isa import fd_softfloat as sf

N = 4000
FORMATS = {"f32": sf.F32, "f64": sf.F64}


def _bits(fmt, sign, exp, frac):
    return (sign << (fmt.bits - 1)) | (exp << fmt.frac_bits) | frac


def _any(rng, fmt):
    """Any bit pattern, weighted so the special classes turn up often."""
    r = rng.random()
    top = (1 << fmt.exp_bits) - 1
    if r < 0.05:
        return _bits(fmt, rng.getrandbits(1), 0, 0)  # zero
    if r < 0.10:
        return _bits(
            fmt, rng.getrandbits(1), 0, rng.getrandbits(fmt.frac_bits)
        )  # subnormal
    if r < 0.13:
        return _bits(fmt, rng.getrandbits(1), top, 0)  # infinity
    if r < 0.16:
        return _bits(
            fmt, rng.getrandbits(1), top, rng.getrandbits(fmt.frac_bits) | 1
        )  # NaN
    return rng.getrandbits(fmt.bits)


def _normal(rng, fmt, lo=1, hi=None, short=0.3):
    hi = (1 << fmt.exp_bits) - 2 if hi is None else hi
    frac = rng.getrandbits(fmt.frac_bits)
    if rng.random() < short:  # short significands: exact results and ties
        frac &= ~((1 << rng.randint(0, fmt.frac_bits)) - 1)
    return _bits(fmt, rng.getrandbits(1), rng.randint(lo, hi), frac)


def _near(rng, fmt, x, spread):
    """A normal value within ``spread`` binades of ``x`` (for cancellation and ties)."""
    e = (x >> fmt.frac_bits) & ((1 << fmt.exp_bits) - 1)
    e = min(max(e + rng.randint(-spread, spread), 1), (1 << fmt.exp_bits) - 2)
    frac = (x & ((1 << fmt.frac_bits) - 1)) ^ rng.getrandbits(
        rng.randint(0, fmt.frac_bits)
    )
    return _bits(fmt, rng.getrandbits(1), e, frac)


def _gen_pairs(rng, fmt, kind):
    top = (1 << fmt.exp_bits) - 2
    bias = fmt.bias
    if kind == "any":
        return _any(rng, fmt), _any(rng, fmt)
    if kind == "normal":
        return _normal(rng, fmt), _normal(rng, fmt)
    if kind == "mid":  # moderate exponents: the bulk of real programs
        return _normal(rng, fmt, bias - 20, bias + 20), _normal(
            rng, fmt, bias - 20, bias + 20
        )
    if kind == "close":  # cancellation and ties
        a = _normal(rng, fmt, bias - 30, bias + 30)
        return a, _near(rng, fmt, a, 2)
    if kind == "spread":  # exact fp32 sums past 53 bits; fp64 sticky cases
        a = _normal(rng, fmt, bias - 10, bias + 10)
        return a, _near(rng, fmt, a, 2 * fmt.precision + 10)
    if kind == "high":  # overflow threshold
        return _normal(rng, fmt, top - 3, top), _normal(rng, fmt, bias - 2, bias + 4)
    if kind == "low":  # underflow threshold
        return _normal(rng, fmt, 1, 4), _normal(rng, fmt, bias - 4, bias + 2)
    raise ValueError(kind)


KINDS = ("any", "normal", "mid", "close", "spread", "high", "low")
#: Kinds whose cases should mostly be answered by the fast path (for the ops that have one).
FAST_KINDS = ("mid", "close")


@pytest.mark.parametrize("fmt_name", FORMATS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_binary_ops_match_the_exact_path(op, kind, fmt_name):
    fmt = FORMATS[fmt_name]
    rng = random.Random(f"{op}-{kind}-{fmt_name}")
    fast64, fast32 = {
        "add": (
            lambda a, b: sf._fast_add64(a, b, False),
            lambda a, b: sf._fast_add32(a, b, False),
        ),
        "sub": (
            lambda a, b: sf._fast_add64(a, b, True),
            lambda a, b: sf._fast_add32(a, b, True),
        ),
        "mul": (sf._fast_mul64, sf._fast_mul32),
        "div": (sf._fast_div64, sf._fast_div32),
    }[op]
    fast = fast64 if fmt is sf.F64 else fast32
    hits = 0
    for _ in range(N):
        a, b = _gen_pairs(rng, fmt, kind)
        if op == "div" and kind in ("mid", "close") and rng.random() < 0.3:
            # Exact quotients: a = q * b with a short q.
            q = _normal(rng, fmt, fmt.bias - 4, fmt.bias + 4, short=1.0)
            prod, _ = sf.mul_soft(q, b, fmt, sf.RNE)
            if (prod >> fmt.frac_bits) & ((1 << fmt.exp_bits) - 1) not in (
                0,
                (1 << fmt.exp_bits) - 1,
            ):
                a = prod
        for rm in (sf.RNE, sf.RNE, sf.RTZ, sf.RDN, sf.RUP, sf.RMM):
            if op in ("add", "sub"):
                neg = op == "sub"
                got = sf.add(a, b, fmt, rm, neg)
                want = sf.add_soft(a, b, fmt, rm, neg)
            else:
                got = getattr(sf, op)(a, b, fmt, rm)
                want = getattr(sf, op + "_soft")(a, b, fmt, rm)
            assert got == want, (op, fmt_name, hex(a), hex(b), rm, got, want)
        hits += fast(a, b) is not None
    if kind in FAST_KINDS:
        assert hits > N // 2, f"fast path answered only {hits}/{N}"


@pytest.mark.parametrize("fmt_name", FORMATS)
@pytest.mark.parametrize("kind", ["any", "normal", "mid", "low", "high", "square"])
def test_sqrt_matches_the_exact_path(kind, fmt_name):
    fmt = FORMATS[fmt_name]
    rng = random.Random(f"sqrt-{kind}-{fmt_name}")
    fast = sf._fast_sqrt64 if fmt is sf.F64 else sf._fast_sqrt32
    hits = 0
    for _ in range(N):
        if kind == "square":  # perfect squares: exact roots
            r = _normal(rng, fmt, fmt.bias - 30, fmt.bias + 30, short=1.0) & ~(
                1 << (fmt.bits - 1)
            )
            a, _ = sf.mul_soft(r, r, fmt, sf.RNE)
        else:
            a = _gen_pairs(rng, fmt, kind)[0]
        for rm in (sf.RNE, sf.RTZ, sf.RDN, sf.RUP, sf.RMM):
            assert sf.sqrt(a, fmt, rm) == sf.sqrt_soft(a, fmt, rm), (
                fmt_name,
                hex(a),
                rm,
            )
        hits += fast(a) is not None
    if kind in ("mid", "square"):
        assert hits > N // 3, f"fast path answered only {hits}/{N}"


@pytest.mark.parametrize("fmt_name", FORMATS)
@pytest.mark.parametrize("kind", KINDS)
def test_fma_matches_the_exact_path(kind, fmt_name):
    fmt = FORMATS[fmt_name]
    rng = random.Random(f"fma-{kind}-{fmt_name}")
    hits = 0
    for _ in range(N):
        a, b = _gen_pairs(rng, fmt, kind)
        r = rng.random()
        if r < 0.4:
            # c close to -a*b: cancellation, and sums wider than a double.
            p, _ = sf.mul_soft(a, b, fmt, sf.RNE)
            c = (
                _near(rng, fmt, p, 3)
                if (p >> fmt.frac_bits) & ((1 << fmt.exp_bits) - 1)
                else _normal(rng, fmt)
            )
        elif r < 0.7:
            c = _gen_pairs(rng, fmt, kind)[0]
        else:
            c = _any(rng, fmt)
        np_, na = rng.getrandbits(1), rng.getrandbits(1)
        for rm in (sf.RNE, sf.RNE, sf.RTZ, sf.RDN, sf.RUP, sf.RMM):
            got = sf.fma(a, b, c, fmt, rm, np_, na)
            want = sf.fma_soft(a, b, c, fmt, rm, np_, na)
            assert got == want, (
                fmt_name,
                hex(a),
                hex(b),
                hex(c),
                rm,
                np_,
                na,
                got,
                want,
            )
        fast = sf._fast_fma32 if fmt is sf.F32 else sf._fast_fma64
        hits += fast(a, b, c, np_, na) is not None
    if kind in FAST_KINDS:
        assert hits > N // 3, f"fast path answered only {hits}/{N}"


def test_fp32_sum_on_a_midpoint_with_a_residual_defers():
    """``s`` exactly halfway between two singles with a nonzero TwoSum error: the error picks the side."""
    one_and_half_ulp = 1.0 + 2.0**-24  # midpoint of 1.0 and its single successor
    assert sf._fast_sum32(one_and_half_ulp, 2.0**-80) is None
    assert sf._fast_sum32(one_and_half_ulp, -(2.0**-80)) is None
    # Off the midpoint, a residual cannot change the rounding: inexact, correct side.
    bits, flags = sf._fast_sum32(1.0 + 2.0**-24 + 2.0**-40, 2.0**-80)
    assert (bits, flags) == (0x3F800001, sf.NX)
    bits, flags = sf._fast_sum32(1.0 + 2.0**-25, 2.0**-80)
    assert (bits, flags) == (0x3F800000, sf.NX)
    # An exact midpoint (err == 0) is a plain RNE tie, which the single rounding
    # decides: (1 + 2^-12)^2 - 2^-11 = 1 + 2^-24 exactly, which ties to 1.0.
    a = 0x3F800800  # 1 + 2^-12
    c = 0xBA000000  # -2^-11
    assert sf._fast_fma32(a, a, c, False, False) == (0x3F800000, sf.NX)
    assert sf.fma_soft(a, a, c, sf.F32, sf.RNE) == (0x3F800000, sf.NX)


def test_fast_path_ignores_other_rounding_modes_and_formats():
    one = 0x3FF0000000000000
    third = 0x3FD5555555555555
    for rm in (sf.RTZ, sf.RDN, sf.RUP, sf.RMM):
        assert sf.add(one, third, sf.F64, rm) == sf.add_soft(one, third, sf.F64, rm)
    # bf16 never takes it.
    assert sf.add(0x3F80, 0x3F80, sf.BF16, sf.RNE) == sf.add_soft(
        0x3F80, 0x3F80, sf.BF16, sf.RNE
    )


def test_flags_on_the_fast_path():
    one = 0x3FF0000000000000
    three = 0x4008000000000000
    assert sf._fast_div64(one, three) == (0x3FD5555555555555, sf.NX)
    assert sf._fast_div64(three, one) == (three, 0)
    assert sf._fast_mul64(three, three) == (0x4022000000000000, 0)
    assert sf._fast_sqrt64(0x4022000000000000) == (three, 0)  # sqrt(9) = 3 exactly
    assert sf._fast_sqrt64(0x4000000000000000)[1] == sf.NX  # sqrt(2)
    assert sf._fast_add64(one, 0x3CA0000000000000, False) == (
        one,
        sf.NX,
    )  # 1 + 2^-53: a tie, to even
    assert sf._fast_sqrt64(0xBFF0000000000000) is None  # negative: NV, exact path
