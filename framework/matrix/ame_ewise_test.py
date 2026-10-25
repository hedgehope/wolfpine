"""The element-wise accumulator ops (custom-3: features ewise / ewise_tx) against exact goldens.

Real instruction words run through an ``RV64I`` with the unit attached (the
``ame_isa_test`` harness). Goldens are independent of the implementation:
fp32 add / sub / mul from numpy's own fp32 arithmetic (IEEE, RNE), and the
fused and transcendental ops from exact rationals, rounded to nearest even
by comparing against the neighbouring representable values.
"""

import importlib.util
from fractions import Fraction

import numpy as np
import pytest

from framework.matrix import formats
from framework.matrix.ame_isa import attach
from framework.matrix.ame_isa_test import ALL_FEATURES, FP32, FP64, INT32, X_VAL
from framework.matrix.ame_isa_test import _exec as _exec_word
from framework.matrix.encodings import DECODER, TABLE
from framework.matrix.unit import IllegalMatrixInstruction, MatrixUnit
from framework.pe.rv.isa.isa64_testing import make_core

F32, F64 = formats.get_format("fp32"), formats.get_format("fp64")
R = 4
OPS = {"add": 0, "sub": 1, "mul": 2, "max": 3, "min": 4, "rsub": 5}


def _core(m=R, n=R, k=R, **kw):
    args = dict(
        m=m,
        n=n,
        k=k,
        tiles=4,
        accs=4,
        in_formats=("fp64", "fp32"),
        acc_formats=("fp64", "fp32", "int32"),
        features=ALL_FEATURES,
    )
    args.update(kw)
    core, _ = make_core([])
    return core, attach(core, MatrixUnit(**args))


def _exec(core, name, regs=None, **ops):
    if "as_" in ops:
        ops["as"] = ops.pop("as_")
    _exec_word(core, name, regs, **ops)


def _rand(seed, dtype=np.float32, shape=(R, R)):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) * np.exp2(rng.integers(-8, 8, shape))
    return x.astype(dtype)


def _rne_exact(v, dtype):
    """The exact rational ``v`` rounded to nearest even in ``dtype``."""
    c = dtype(float(v))  # within one ulp of the answer
    best = None
    for cand in (np.nextafter(c, dtype(-np.inf)), c, np.nextafter(c, dtype(np.inf))):
        if not np.isfinite(cand):
            continue
        d = abs(Fraction(float(cand)) - v)
        bits = cand.view(np.uint32 if dtype == np.float32 else np.uint64)
        key = (d, int(bits) & 1)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1]


def _rne_by_midpoints(dtype, x, f):
    """The RNE ``dtype`` value of ``f(x)`` for monotone ``f`` with an exact midpoint test.

    ``f`` maps a Fraction midpoint ``m`` to the sign of ``f(x) - m`` exactly.
    """
    c = dtype(f(None))
    lo, hi = np.nextafter(c, dtype(-np.inf)), np.nextafter(c, dtype(np.inf))
    cands = sorted({lo, c, hi})
    best = cands[0]
    for a, b in zip(cands, cands[1:]):
        m = (Fraction(float(a)) + Fraction(float(b))) / 2
        s = f(m)
        if s > 0 or (s == 0 and int(b.view(np.uint32)) % 2 == 0):
            best = b
    return best


# -- the table -------------------------------------------------------------------


def test_the_ewise_ops_are_in_custom3_behind_their_features():
    names = {"aew.aa", "aew.at", "aew.ax", "amacc.aa", "amacc.at", "aewu"}
    got = {
        i.name
        for i in TABLE.instructions.values()
        if i.feature in ("ewise", "ewise_tx")
    }
    assert got == names
    for name in names:
        insn = TABLE.instructions[name]
        assert insn.opcode == TABLE.opcodes["custom-3"]
        assert insn.feature == ("ewise_tx" if name == "aewu" else "ewise")


# -- binary ops ----------------------------------------------------------------------


@pytest.mark.parametrize("op", sorted(OPS))
def test_aew_aa_fp32_matches_ieee_fp32_arithmetic(op):
    core, unit = _core()
    a, b = _rand(1), _rand(2)
    unit.acc_view(1, F32)[...] = a
    unit.acc_view(2, F32)[...] = b
    _exec(core, "aew.aa", ad=0, as1=1, as2=2, fmt=FP32, op=OPS[op])
    want = {
        "add": a + b,
        "sub": a - b,
        "mul": a * b,
        "max": np.maximum(a, b),
        "min": np.minimum(a, b),
        "rsub": b - a,
    }[op]
    np.testing.assert_array_equal(unit.acc_view(0, F32), want)
    assert unit.stats["ewise_ops"] == 1
    assert unit.stats["ewise_elems"] == R * R


@pytest.mark.parametrize("op", ["add", "mul", "rsub"])
def test_aew_aa_fp64_and_in_place(op):
    core, unit = _core()
    a, b = _rand(3, np.float64), _rand(4, np.float64)
    unit.acc_view(1, F64)[...] = a
    unit.acc_view(2, F64)[...] = b
    _exec(core, "aew.aa", ad=1, as1=1, as2=2, fmt=FP64, op=OPS[op])  # ad = as1
    want = {"add": a + b, "mul": a * b, "rsub": b - a}[op]
    np.testing.assert_array_equal(unit.acc_view(1, F64), want)


def test_aew_add_breaks_a_near_tie_by_the_sticky_bits():
    """1 + (2^-24 + 2^-47): just above the midpoint of 1 and 1 + 2^-23, so RNE goes up."""
    core, unit = _core()
    unit.acc_view(1, F32)[...] = np.float32(1.0)
    unit.acc_view(2, F32)[...] = np.float32(2.0**-24 + 2.0**-47)
    _exec(core, "aew.aa", ad=0, as1=1, as2=2, fmt=FP32, op=OPS["add"])
    assert (unit.acc_view(0, F32) == np.float32(1 + 2.0**-23)).all()


def test_max_min_are_ieee_maximum_minimum():
    core, unit = _core()
    a = np.array([[np.nan, 0.0, -0.0, 1.0]] * R, np.float32)
    b = np.array([[1.0, -0.0, 0.0, np.nan]] * R, np.float32)
    unit.acc_view(1, F32)[...] = a
    unit.acc_view(2, F32)[...] = b
    _exec(core, "aew.aa", ad=0, as1=1, as2=2, fmt=FP32, op=OPS["max"])
    mx = unit.acc_view(0, F32)[0].copy()
    _exec(core, "aew.aa", ad=0, as1=1, as2=2, fmt=FP32, op=OPS["min"])
    mn = unit.acc_view(0, F32)[0].copy()
    assert np.isnan(mx[0])
    assert np.isnan(mx[3])
    assert np.isnan(mn[0])
    assert np.isnan(mn[3])
    assert mx[1] == 0
    assert not np.signbit(mx[1])
    assert not np.signbit(mx[2])
    assert mn[1] == 0
    assert np.signbit(mn[1])
    assert np.signbit(mn[2])


def test_aew_at_reads_the_tiles_top_left_m_by_n_block():
    """A non-square 4 x 3 x 5 unit: the tile operand is its top-left M x N (4 x 3) block."""
    core, unit = _core(m=4, n=3, k=5)
    a = _rand(5, shape=(4, 3))
    t = _rand(6, shape=(5, 5))
    unit.acc_view(1, F32)[...] = a
    unit.tiles[2].fill(0)
    unit.tiles[2, :5, : 5 * 4] = t.view(np.uint8).reshape(5, 20)
    _exec(core, "aew.at", ad=0, as1=1, ts=2, fmt=FP32, op=OPS["mul"])
    np.testing.assert_array_equal(unit.acc_view(0, F32), a * t[:4, :3])


def test_aew_at_fp64_block_must_fit_the_tile():
    """Tiles sized for fp32 cannot hold an M x N fp64 block: illegal."""
    core, unit = _core(in_formats=("fp32",))
    with pytest.raises(IllegalMatrixInstruction, match="does not fit"):
        _exec(core, "aew.at", ad=0, as1=1, ts=2, fmt=FP64, op=0)


def test_aew_ax_scalar_operand_and_x0_clamp():
    core, unit = _core()
    a = _rand(7)
    unit.acc_view(1, F32)[...] = a
    s = np.float32(0.375)
    bits = int(s.view(np.uint32)) | (0xDEAD << 32)  # high bits are ignored
    _exec(
        core,
        "aew.ax",
        {X_VAL: bits},
        ad=0,
        as1=1,
        value=X_VAL,
        fmt=FP32,
        op=OPS["rsub"],
    )
    np.testing.assert_array_equal(unit.acc_view(0, F32), s - a)
    # rs2 = x0 reads +0.0: max(a, 0)
    _exec(core, "aew.ax", ad=0, as1=1, value=0, fmt=FP32, op=OPS["max"])
    np.testing.assert_array_equal(unit.acc_view(0, F32), np.maximum(a, np.float32(0)))


# -- fused multiply-add ---------------------------------------------------------------


@pytest.mark.parametrize("neg", [0, 1])
@pytest.mark.parametrize("src", ["aa", "at"])
def test_amacc_fp32_is_fused(neg, src):
    core, unit = _core()
    rng = np.random.default_rng(8)
    # products whose low half matters: a * b nearly cancels c
    a = (1 + rng.integers(0, 1 << 23, (R, R)) * 2.0**-23).astype(np.float32)
    b = (1 + rng.integers(0, 1 << 23, (R, R)) * 2.0**-23).astype(np.float32)
    c = (-(a.astype(np.float64) * b) * (1 if not neg else -1)).astype(np.float32)
    unit.acc_view(0, F32)[...] = c
    unit.acc_view(1, F32)[...] = a
    if src == "aa":
        unit.acc_view(2, F32)[...] = b
        _exec(core, "amacc.aa", ad=0, as1=1, as2=2, fmt=FP32, neg=neg)
    else:
        unit.tiles[3].fill(0)
        unit.tile_view(3, "a", F32)[...] = b
        _exec(core, "amacc.at", ad=0, as1=1, ts=3, fmt=FP32, neg=neg)
    sign = -1 if neg else 1
    want = np.array(
        [
            _rne_exact(
                Fraction(float(ci)) + sign * Fraction(float(ai)) * Fraction(float(bi)),
                np.float32,
            )
            for ai, bi, ci in zip(a.ravel(), b.ravel(), c.ravel())
        ],
        np.float32,
    ).reshape(R, R)
    np.testing.assert_array_equal(unit.acc_view(0, F32), want)
    assert (want != 0).any()  # the residual survives: an unfused op would give 0


def test_amacc_fp64_is_fused_exactly():
    core, unit = _core()
    rng = np.random.default_rng(9)
    a = 1 + rng.integers(0, 1 << 52, (R, R)) * 2.0**-52
    b = 1 + rng.integers(0, 1 << 52, (R, R)) * 2.0**-52
    c = -(a * b)
    unit.acc_view(0, F64)[...] = c
    unit.acc_view(1, F64)[...] = a
    unit.acc_view(2, F64)[...] = b
    _exec(core, "amacc.aa", ad=0, as1=1, as2=2, fmt=FP64, neg=0)
    want = np.array(
        [
            _rne_exact(Fraction(ci) + Fraction(ai) * Fraction(bi), np.float64)
            for ai, bi, ci in zip(a.ravel(), b.ravel(), c.ravel())
        ]
    ).reshape(R, R)
    np.testing.assert_array_equal(unit.acc_view(0, F64), want)
    assert (want != 0).all()  # the fp64 rounding error of a * b, recovered


# -- transcendentals ---------------------------------------------------------------


def _fp32_samples(seed, n=R * R):
    rng = np.random.default_rng(seed)
    return np.exp2(rng.uniform(-60, 60, n)).astype(np.float32) * rng.uniform(
        1, 2, n
    ).astype(np.float32)


def _run_unary(op, x, fmt=FP32, view=F32):
    core, unit = _core()
    unit.acc_view(1, view)[...] = x.reshape(R, R)
    _exec(core, "aewu", ad=0, as1=1, fmt=fmt, op=op)
    return unit.acc_view(0, view).ravel().copy()


@pytest.mark.parametrize("seed", range(4))
def test_recip_and_sqrt_fp32_are_correctly_rounded(seed):
    x = _fp32_samples(seed)
    got = _run_unary(0, x)
    want = [_rne_exact(1 / Fraction(float(v)), np.float32) for v in x]
    np.testing.assert_array_equal(got, np.array(want, np.float32))
    got = _run_unary(2, x)
    # sqrt(x) vs midpoint m: sign(x - m^2)
    want = [
        _rne_by_midpoints(
            np.float32,
            v,
            lambda m, v=v: (
                np.sqrt(np.float64(v)) if m is None else Fraction(float(v)) - m * m
            ),
        )
        for v in x
    ]
    np.testing.assert_array_equal(got, np.array(want, np.float32))


@pytest.mark.parametrize("seed", range(8))
def test_rsqrt_fp32_is_correctly_rounded_on_samples(seed):
    """The documented bound allows a miss within 2^-50 of a boundary; random samples never hit one."""
    x = _fp32_samples(100 + seed)
    got = _run_unary(1, x)
    # 1/sqrt(x) vs midpoint m: sign(1 - m^2 x)
    want = [
        _rne_by_midpoints(
            np.float32,
            v,
            lambda m, v=v: (
                1 / np.sqrt(np.float64(v))
                if m is None
                else 1 - m * m * Fraction(float(v))
            ),
        )
        for v in x
    ]
    np.testing.assert_array_equal(got, np.array(want, np.float32))


def test_rsqrt_fp64_is_within_its_bound():
    rng = np.random.default_rng(11)
    x = np.exp2(rng.uniform(-200, 200, R * R)) * rng.uniform(1, 2, R * R)
    got = _run_unary(1, x, FP64, F64)
    exact = 1 / np.sqrt(np.longdouble(x))
    ulp = np.spacing(got)
    assert (np.abs(got - exact) <= 1.5 * ulp).all()


def test_transcendental_specials():
    x = np.array([0.0, -0.0, -1.0, np.inf] * R, np.float32)
    r = _run_unary(1, x)  # rsqrt
    assert r[0] == np.inf
    assert r[1] == -np.inf
    assert np.isnan(r[2])
    assert r[3] == 0
    assert not np.signbit(r[3])
    q = _run_unary(0, x)  # recip
    assert q[0] == np.inf
    assert q[1] == -np.inf
    assert q[2] == -1
    assert q[3] == 0
    s = _run_unary(2, x)  # sqrt
    assert s[0] == 0
    assert np.signbit(s[1])
    assert np.isnan(s[2])
    assert s[3] == np.inf


@pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)
def test_bf16_accumulator_rounds_once_to_bf16():
    bf16 = formats.get_format("bf16")
    core, unit = _core(acc_formats=("fp32", "bf16"))
    code = TABLE.formats["bf16"]
    a = bf16.round(_rand(12).astype(np.float64))
    b = bf16.round(_rand(13).astype(np.float64))
    unit.acc_view(1, bf16)[...] = a
    unit.acc_view(2, bf16)[...] = b
    _exec(core, "aew.aa", ad=0, as1=1, as2=2, fmt=code, op=OPS["add"])
    want = bf16.round(a.astype(np.float64) + b.astype(np.float64))
    np.testing.assert_array_equal(
        unit.acc_view(0, bf16).view(np.uint16), want.view(np.uint16)
    )
    _exec(
        core, "aewu", ad=0, as1=1, fmt=code, op=1
    )  # rsqrt of a: NaN for negatives, else RNE
    want = bf16.round(1 / np.sqrt(np.abs(a.astype(np.float64))))
    got = unit.acc_view(0, bf16).astype(np.float64)
    pos = a.astype(np.float64) > 0
    np.testing.assert_array_equal(got[pos], want.astype(np.float64)[pos])
    assert np.isnan(got[~pos]).all()


# -- gating ---------------------------------------------------------------------------


def test_ewise_needs_its_feature():
    others = tuple(f for f in ALL_FEATURES if f not in ("ewise", "ewise_tx"))
    core, _ = _core(features=others)
    with pytest.raises(IllegalMatrixInstruction, match="'ewise' feature"):
        _exec(core, "aew.aa", ad=0, as1=1, as2=2, fmt=FP32, op=0)
    core, _ = _core(features=others + ("ewise",))
    _exec(core, "aew.aa", ad=0, as1=1, as2=2, fmt=FP32, op=0)
    with pytest.raises(IllegalMatrixInstruction, match="'ewise_tx' feature"):
        _exec(core, "aewu", ad=0, as1=1, fmt=FP32, op=1)


@pytest.mark.parametrize(
    "name, ops, match",
    [
        ("aew.aa", dict(ad=0, as1=1, as2=2, fmt=INT32, op=0), "not a float"),
        ("aew.aa", dict(ad=0, as1=1, as2=2, fmt=FP32, op=6), "op 6"),
        ("aewu", dict(ad=0, as1=1, fmt=FP32, op=3), "op 3"),
        ("amacc.aa", dict(ad=0, as1=1, as2=4, fmt=FP32, neg=0), "accumulator 4"),
        ("aew.at", dict(ad=0, as1=1, ts=4, fmt=FP32, op=0), "tile 4"),
    ],
)
def test_illegal_forms(name, ops, match):
    core, _ = _core()
    with pytest.raises(IllegalMatrixInstruction, match=match):
        _exec(core, name, **ops)


def test_reserved_rs2_of_aewu_is_illegal():
    core, _ = _core()
    word = DECODER.encode("aewu", ad=0, as1=1, fmt=FP32, op=1) | (1 << 20)
    from framework.pe.rv.isa.isa64_testing import run

    with pytest.raises(IllegalMatrixInstruction):
        run(word, None, core=core)
