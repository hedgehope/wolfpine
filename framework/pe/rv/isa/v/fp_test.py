"""RVV floating point: the numpy fast paths against softfloat, and the FP instructions end to end.

The fast paths in ``fp.py`` claim to give softfloat's bits and flags exactly;
the property tests here run both on random bit patterns, adversarial values
(near-cancelling FMAs, ties, subnormals, specials) and every rounding mode.
"""

import numpy as np
import pytest

from framework.pe.rv.isa import fd_softfloat as sf
from framework.pe.rv.isa.v import fp
from framework.pe.rv.isa.v.vtesting import OPFVF, OPFVV, UINT, VCore, opv

RNG = np.random.default_rng(11)
_FL = {16: np.float16, 32: np.float32, 64: np.float64}


def operands(sew, n, kind):
    """Bit patterns: raw random, moderate values, or moderate values with specials sprinkled in."""
    fmt = fp.FMT[sew]
    if kind == "raw":
        return RNG.integers(0, 1 << sew, n, dtype=np.uint64).astype(UINT[sew])
    x = RNG.standard_normal(n) * 2.0 ** RNG.integers(-12, 12, n)
    with np.errstate(all="ignore"):
        b = x.astype(_FL[sew]).view(UINT[sew]).copy()
    if kind == "special":
        specials = np.array(
            [
                0,
                1 << (sew - 1),
                fmt.canonical_nan,
                ((1 << fmt.exp_bits) - 1) << fmt.frac_bits,
                1,
                1 << fmt.frac_bits,
                (((1 << fmt.exp_bits) - 1) << fmt.frac_bits) | 1,
            ],  # +0 -0 qNaN inf min-sub min-normal sNaN
            dtype=np.uint64,
        ).astype(UINT[sew])
        idx = RNG.integers(0, n, n // 3)
        b[idx] = specials[RNG.integers(0, len(specials), len(idx))]
    return b


def _slow(f, *cols):
    flags, out = 0, []
    for args in zip(*(c.tolist() for c in cols)):
        r, fl = f(*args)
        out.append(r)
        flags |= fl
    return out, flags


@pytest.mark.parametrize("sew", [16, 32, 64])
@pytest.mark.parametrize("rm", range(5))
@pytest.mark.parametrize("kind", ["raw", "value", "special"])
def test_fast_paths_equal_softfloat(sew, rm, kind):
    fmt = fp.FMT[sew]
    for _ in range(6):
        n = 64
        a, b, c = operands(sew, n, kind), operands(sew, n, kind), operands(sew, n, kind)
        for op in ("add", "sub", "mul", "div"):
            got, flags = fp.binary(op, a, b, sew, rm)
            want, wflags = _slow(
                lambda x, y, o=op: fp._sf_binary(o, x, y, fmt, rm), a, b
            )
            assert got.tolist() == want, op
            assert flags == wflags, op
        got, flags = fp.fma(a, b, c, sew, rm, True, False)
        want, wflags = _slow(
            lambda x, y, z: sf.fma(x, y, z, fmt, rm, True, False), a, b, c
        )
        assert got.tolist() == want
        assert flags == wflags
        got, flags = fp.sqrt(a, sew, rm)
        want, wflags = _slow(lambda x: sf.sqrt(x, fmt, rm), a)
        assert got.tolist() == want
        assert flags == wflags


@pytest.mark.parametrize("sew", [32, 64])
def test_fma_fast_path_on_near_cancellation_and_ties(sew):
    """c = -(a*b) rounded, nudged by a few ulps either way: the hardest case for the RO emulation."""
    fmt = fp.FMT[sew]
    dt = _FL[sew]
    for _ in range(20):
        a = operands(sew, 64, "value")
        b = operands(sew, 64, "value")
        with np.errstate(all="ignore"):
            p = (a.view(dt).astype(np.float64) * b.view(dt).astype(np.float64)).astype(
                dt
            )
        c = (-p).view(UINT[sew]) + RNG.integers(-3, 4, 64).astype(UINT[sew])
        for rm in (sf.RNE, sf.RDN):
            got, flags = fp.fma(a, b, c, sew, rm)
            want, wflags = _slow(lambda x, y, z: sf.fma(x, y, z, fmt, rm), a, b, c)
            assert got.tolist() == want
            assert flags == wflags


@pytest.mark.parametrize("src,dst", [(16, 32), (32, 64), (64, 32), (32, 16), (64, 16)])
@pytest.mark.parametrize("rm", [sf.RNE, sf.RTZ, sf.RUP, fp.ROD])
def test_float_conversions(src, dst, rm):
    a = operands(src, 256, "special")
    got, flags = fp.convert(a, fp.FMT[src], fp.FMT[dst], rm)
    srm = sf.RTZ if rm == fp.ROD else rm
    want, wflags = _slow(lambda x: sf.convert(x, fp.FMT[src], fp.FMT[dst], srm), a)
    if rm == fp.ROD:
        # Round-to-odd: RTZ with the last bit set when inexact (finite results).
        for i, (g, w) in enumerate(zip(got.tolist(), want)):
            fx = sf.convert(int(a[i]), fp.FMT[src], fp.FMT[dst], sf.RTZ)[1] & sf.NX
            assert g == (
                w | 1
                if fx and not fp.nonfinite(np.array([w], np.uint64), fp.FMT[dst])[0]
                else w
            )
    else:
        assert got.tolist() == want
    assert flags == wflags


@pytest.mark.parametrize("sew", [16, 32, 64])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("rm", range(5))
def test_float_to_int_and_back(sew, signed, rm):
    fmt = fp.FMT[sew]
    a = operands(sew, 128, "special")
    got, flags = fp.to_int(a, fmt, rm, sew, signed)
    want, wflags = _slow(lambda x: sf.to_int(x, fmt, rm, sew, signed), a)
    assert [int(g) & ((1 << sew) - 1) for g in got] == [
        w & ((1 << sew) - 1) for w in want
    ]
    assert flags == wflags
    ints = RNG.integers(0, 1 << 63, 128, dtype=np.uint64) >> RNG.integers(
        0, 63, 128
    ).astype(np.uint64)
    ints = ints & np.uint64((1 << sew) - 1)
    got, flags = fp.from_int(ints, signed, sew, fmt, rm)

    def ref(v):
        v &= (1 << sew) - 1
        if signed and v >> (sew - 1):
            v -= 1 << sew
        return sf.from_int(v, fmt, rm)

    want, wflags = _slow(ref, ints)
    assert got.tolist() == want
    assert flags == wflags


@pytest.mark.parametrize("sew", [16, 32, 64])
def test_compare_minmax_classify(sew):
    fmt = fp.FMT[sew]
    a, b = operands(sew, 200, "special"), operands(sew, 200, "special")
    for op, sfop in (("eq", "eq"), ("lt", "lt"), ("le", "le")):
        got, flags = fp.compare(a, b, sew, op)
        want, wflags = _slow(lambda x, y, o=sfop: sf.compare(x, y, fmt, o), a, b)
        assert got.astype(int).tolist() == want
        assert flags == wflags
    for is_max in (False, True):
        got, flags = fp.minmax(a, b, sew, is_max)
        want, wflags = _slow(lambda x, y: sf.minmax(x, y, fmt, is_max), a, b)
        assert got.tolist() == want
        assert flags == wflags
    assert fp.classify(a, sew).tolist() == [sf.fclass(int(x), fmt) for x in a]


@pytest.mark.parametrize("sew", [16, 32, 64])
@pytest.mark.parametrize("rm", [sf.RNE, sf.RUP])
def test_ordered_sum_is_sequential_softfloat(sew, rm):
    fmt = fp.FMT[sew]
    for kind in ("value", "special"):
        vals = operands(sew, 40, kind)
        start = int(operands(sew, 1, "value")[0])
        got, flags = fp.ordered_sum(start, vals, sew, rm)
        acc, wflags = start, 0
        for x in vals.tolist():
            acc, f = sf.add(acc, x, fmt, rm)
            wflags |= f
        assert (got, flags) == (acc, wflags)


# -- instructions ---------------------------------------------------------------------------


def _fset(c, base, values, sew, nregs=1):
    c.vset_data(
        base, np.asarray(values, dtype=_FL[sew]).view(UINT[sew]), UINT[sew], nregs
    )


def test_vfadd_masked_raises_flags_only_for_active_elements():
    c = VCore()
    c.vset(32, 1)
    _fset(c, 8, [1, 2, 3, 4, 5, 6, 7, np.inf], 32)
    _fset(
        c, 9, [1, 2, 3, 4, 5, 6, 7, -np.inf], 32
    )  # inf - inf -> NV, but element 7 is masked off
    _fset(c, 10, [9] * 8, 32)
    c.set_mask([1] * 7 + [0])
    c.clear_fflags()
    c.exec(opv(OPFVV, 0x00, 10, 8, 9, vm=0))
    assert list(c.v(10, np.float32)) == [2, 4, 6, 8, 10, 12, 14, 9]
    assert c.fflags() == 0
    c.exec(opv(OPFVV, 0x00, 10, 8, 9, vm=1))
    assert c.fflags() == sf.NV
    assert np.isnan(c.v(10, np.float32)[7])


@pytest.mark.parametrize(
    "f6,expect",
    [
        (0x2C, lambda s, a, d: s * a + d),  # vfmacc: vs1*vs2 + vd
        (0x2D, lambda s, a, d: -(s * a) - d),  # vfnmacc
        (0x2E, lambda s, a, d: s * a - d),  # vfmsac
        (0x2F, lambda s, a, d: -(s * a) + d),  # vfnmsac
        (0x28, lambda s, a, d: s * d + a),  # vfmadd: vs1*vd + vs2
        (0x29, lambda s, a, d: -(s * d) - a),  # vfnmadd
        (0x2A, lambda s, a, d: s * d - a),  # vfmsub
        (0x2B, lambda s, a, d: -(s * d) + a),  # vfnmsub
    ],
)
def test_fma_family_operand_roles(f6, expect):
    c = VCore()
    c.vset(64, 1)
    a, d = [1.5, -2.0, 3.25, 0.5], [10.0, 20.0, -30.0, 40.0]
    _fset(c, 8, a, 64)
    _fset(c, 16, d, 64)
    c.f(3, np.float64(2.0).view(np.uint64).item())
    c.exec(opv(OPFVF, f6, 16, 8, 3))
    assert list(c.v(16, np.float64)[:4]) == [expect(2.0, x, y) for x, y in zip(a, d)]


def test_single_rounding_in_vfmacc():
    """a*b + c with an exact product that a separate multiply would round: fused keeps it."""
    c = VCore()
    c.vset(32, 1)
    x = np.float32(1 + 2**-12)
    _fset(c, 8, [x] * 8, 32)
    _fset(c, 9, [x] * 8, 32)
    _fset(c, 10, [-(1 + 2**-11)] * 8, 32)
    c.exec(opv(OPFVV, 0x2C, 10, 8, 9))
    assert c.v(10, np.float32)[0] == np.float32(2**-24)


def test_dynamic_rounding_mode_and_illegal_frm():
    c = VCore()
    c.vset(32, 1)
    _fset(c, 8, [1.0] * 8, 32)
    _fset(c, 9, [2**-30] * 8, 32)
    c.set_frm(sf.RUP)
    c.exec(opv(OPFVV, 0x00, 10, 8, 9))
    assert c.v(10, np.uint32)[0] == np.float32(1.0).view(np.uint32) + 1
    c.set_frm(5)
    assert not c.legal(opv(OPFVV, 0x00, 10, 8, 9))


def test_sew16_fp_needs_zvfh_and_sew8_is_never_fp():
    c = VCore(extensions=("v",))
    c.vset(16, 1)
    assert not c.legal(opv(OPFVV, 0x00, 10, 8, 9))
    c = VCore()
    c.vset(16, 1)
    assert c.legal(opv(OPFVV, 0x00, 10, 8, 9))
    c.vset(8, 1)
    assert not c.legal(opv(OPFVV, 0x00, 10, 8, 9))


def test_scalar_operand_nan_boxing():
    c = VCore()
    c.vset(32, 1)
    _fset(c, 8, [1.0] * 8, 32)
    c.f(3, 0x000000003F800000)  # not boxed: reads as the canonical NaN
    c.exec(opv(OPFVF, 0x00, 10, 8, 3))
    assert c.v(10, np.uint32)[0] == 0x7FC00000
    c.f(3, 0xFFFFFFFF3F800000)
    c.exec(opv(OPFVF, 0x00, 10, 8, 3))
    assert c.v(10, np.float32)[0] == 2.0


def test_widening_fp_and_bf16():
    c = VCore()
    c.vset(16, 1)
    h = np.array([1.5, -2.25, 65504.0, 2**-24] * 4, dtype=np.float16)
    c.vset_data(8, h.view(np.uint16), UINT[16])
    c.vset_data(9, h.view(np.uint16), UINT[16])
    c.exec(opv(OPFVV, 0x38, 16, 8, 9))  # vfwmul.vv -> fp32, exact
    assert list(c.v(16, np.float32, 2)[:16]) == [
        np.float32(x) * np.float32(x) for x in h
    ]
    c.exec(opv(OPFVV, 0x12, 18, 8, 0x0C))  # vfwcvt.f.f.v
    assert list(c.v(18, np.float32, 2)[:16]) == [np.float32(x) for x in h]
    # vfncvtbf16.f.f.w then vfwmaccbf16.vv
    c.vset_data(
        18,
        np.array([1.0, 3.0, -0.1, 7.75] * 4, np.float32).view(np.uint32),
        UINT[32],
        2,
    )
    c.exec(opv(OPFVV, 0x12, 20, 18, 0x1D))
    bf = c.v(20, np.uint16)[:4].tolist()
    assert bf == [0x3F80, 0x4040, 0xBDCD, 0x40F8]
    c.vset_data(22, np.zeros(16, np.float32).view(np.uint32), UINT[32], 2)
    c.exec(opv(OPFVV, 0x3B, 22, 20, 20))  # vfwmaccbf16: 0 + b*b
    sq = [
        np.float32(np.frombuffer(np.uint32(x << 16).tobytes(), np.float32)[0]) ** 2
        for x in bf
    ]
    assert list(c.v(22, np.float32, 2)[:4]) == sq


def test_fp_reductions():
    c = VCore()
    c.vset(32, 2)
    vals = np.float32([1e8, 1.0, -1e8, 1.0] * 4)
    _fset(c, 8, vals, 32, 2)
    _fset(c, 4, [0.5], 32)
    c.exec(opv(OPFVV, 0x03, 2, 8, 4))  # vfredosum: sequential
    acc = np.float32(0.5)
    for v in vals:
        acc = np.float32(acc + v)
    assert c.v(2, np.float32)[0] == acc
    c.exec(opv(OPFVV, 0x01, 3, 8, 4))  # vfredusum: Spike's order is the same sequence
    assert c.v(3, np.float32)[0] == acc
    c.exec(opv(OPFVV, 0x07, 5, 8, 4))  # vfredmax
    assert c.v(5, np.float32)[0] == np.float32(1e8)
    c.set_mask([0] * 16)
    _fset(c, 4, [np.nan], 32)
    c.vs.vrf[4, :4] = np.frombuffer(
        np.uint32(0x7F800001).tobytes(), np.uint8
    )  # an sNaN start
    c.clear_fflags()
    c.exec(
        opv(OPFVV, 0x01, 3, 8, 4, vm=0)
    )  # no active element: usum canonicalises the NaN start
    assert c.v(3, np.uint32)[0] == 0x7FC00000
    assert c.fflags() == sf.NV
    c.exec(opv(OPFVV, 0x05, 3, 8, 4, vm=0))  # min/max leave it as it is
    assert c.v(3, np.uint32)[0] == 0x7F800001


def test_estimates_match_spike_examples():
    """vfrec7 / vfrsqrt7 on a few values whose estimates the spec's tables give directly."""
    c = VCore()
    c.vset(32, 1)
    _fset(c, 8, [1.0, 2.0, 4.0, 0.25, -1.0, 0.0, np.inf, 3.0], 32)
    c.exec(opv(OPFVV, 0x13, 9, 8, 0x05))  # vfrec7.v
    rec = c.v(9, np.float32)
    assert list(rec[:4]) == [
        np.float32(0.99609375),
        np.float32(0.498046875),
        np.float32(0.2490234375),
        np.float32(3.984375),
    ]
    c.exec(opv(OPFVV, 0x13, 10, 8, 0x04))  # vfrsqrt7.v
    rs = c.v(10, np.float32)
    assert rs[0] == np.float32(0.99609375)
    assert np.isnan(rs[4])
    assert rs[5] == np.inf
    assert rs[6] == 0
