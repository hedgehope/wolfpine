"""The Ztt-tracking AME additions (custom-2) and the acc_mem gate, against numpy.

Real instruction words run through an ``RV64I`` with the unit attached (the
``ame_isa_test`` harness). Most cases use a square 4 x 4 x 4 unit, as Ztt
squares are; the transposed mma also runs on ``ame_isa_test``'s non-square
4 x 3 x 5 unit, so a shape mixup cannot pass.
"""

import numpy as np
import pytest

from framework.matrix import formats
from framework.matrix.ame_isa import attach
from framework.matrix.ame_isa_test import (
    ALL_FEATURES,
    FP32,
    FP64,
    INT32,
    X_BASE,
    X_STRIDE,
    _exec,
    _mem_ops,
)
from framework.matrix.encodings import DECODER, TABLE
from framework.matrix.mac import tile_mac
from framework.matrix.unit import IllegalMatrixInstruction, MatrixUnit
from framework.pe.rv.isa.isa64_testing import DATA, make_core, reg, run

F32, F64, I32 = (formats.get_format(n) for n in ("fp32", "fp64", "int32"))
R = 4
NEW = ("acc_move", "convert", "mma_transpose", "reduce", "shift2")


def _core(**kw):
    args = dict(
        m=R,
        n=R,
        k=R,
        tiles=4,
        accs=2,
        in_formats=("fp64", "fp32", "int8", "int32"),
        acc_formats=("fp64", "fp32", "int32"),
        features=ALL_FEATURES,
    )
    args.update(kw)
    core, memory = make_core([])
    unit = attach(core, MatrixUnit(**args))
    return core, memory, unit


def _rand(seed, shape=(R, R)):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# -- table ----------------------------------------------------------------------


def test_the_new_instructions_avoid_ztts_slots():
    """Ztt v0.6 is all custom-1 funct3 = 000; every addition is in custom-2."""
    for insn in TABLE.instructions.values():
        if insn.feature in NEW:
            assert insn.opcode == TABLE.opcodes["custom-2"], insn.name
    ztt_slot = TABLE.opcodes["custom-1"]
    for insn in TABLE.instructions.values():
        if insn.opcode == ztt_slot and insn.funct3 == 0:
            assert (
                insn.name == "mma"
            )  # the one pre-existing overlap, documented in the table


# -- acc <-> tile moves (Ztt mmov.m.a / mmov.a.m) ----------------------------------


@pytest.mark.parametrize("b", [0, 1])
def test_amov_round_trips_an_accumulator_bit_for_bit(b):
    core, _, unit = _core()
    role = "b" if b else "a"
    acc = _rand(1)
    acc[0, 0] = np.float32(np.nan)  # a NaN payload survives: it is a bit copy
    unit.acc_view(1, F32)[...] = acc
    unit.tiles[2].fill(0xFF)
    _exec(core, "amov.t.a", td=2, fmt=FP32, b=b, **{"as": 1})
    np.testing.assert_array_equal(
        unit.tile_view(2, role, F32).view(np.uint32), acc.view(np.uint32)
    )
    # bytes outside the 4 x 4 fp32 view are zeroed (the tile is sized for fp64)
    assert not unit.tiles[2, :, R * 4 :].any()
    assert unit.stats["acc_to_tile_bytes"] == R * R * 4
    _exec(core, "amov.a.t", ad=0, ts=2, fmt=FP32, b=b)
    np.testing.assert_array_equal(
        unit.acc_view(0, F32).view(np.uint32), acc.view(np.uint32)
    )
    assert unit.stats["tile_to_acc_bytes"] == R * R * 4


def test_amov_needs_a_format_that_is_also_a_tile_format():
    core, _, _ = _core(in_formats=("fp32",), acc_formats=("fp32", "fp64"))
    with pytest.raises(IllegalMatrixInstruction, match="input \\(tile\\) format"):
        _exec(core, "amov.t.a", td=0, fmt=FP64, b=0, **{"as": 0})


def test_amov_needs_an_accumulator_shaped_view():
    """On a 4 x 3 x 5 unit neither tile view is the accumulator's 4 x 3."""
    core, _, _ = _core(m=4, n=3, k=5)
    with pytest.raises(IllegalMatrixInstruction, match="not the accumulator"):
        _exec(core, "amov.t.a", td=0, fmt=FP32, b=0, **{"as": 0})


# -- conversion (Ztt mconv.ew) --------------------------------------------------------


def test_tconv_narrows_with_one_rne_rounding_and_widens_exactly():
    core, _, unit = _core()
    x = np.random.default_rng(3).standard_normal((R, R))
    unit.tile_view(1, "a", F64)[...] = x
    _exec(core, "tconv", td=1, ts=1, b=0, to=FP32, **{"from": FP64})
    np.testing.assert_array_equal(unit.tile_view(1, "a", F32), x.astype(np.float32))
    assert not unit.tiles[1, :, R * 4 :].any()
    _exec(core, "tconv", td=2, ts=1, b=0, to=FP64, **{"from": FP32})
    np.testing.assert_array_equal(
        unit.tile_view(2, "a", F64), x.astype(np.float32).astype(np.float64)
    )


def test_tconv_to_bf16_matches_the_format_rounding():
    pytest.importorskip("ml_dtypes")
    core, _, unit = _core(in_formats=("fp32", "bf16"), acc_formats=("fp32",))
    bf16 = formats.get_format("bf16")
    x = _rand(4) * np.float32(1e3)
    unit.tile_view(0, "b", F32)[...] = x
    _exec(core, "tconv", td=3, ts=0, b=1, to=TABLE.formats["bf16"], **{"from": FP32})
    np.testing.assert_array_equal(
        unit.tile_view(3, "b", bf16).view(np.uint16), bf16.round(x).view(np.uint16)
    )


def test_tconv_between_float_and_int_is_illegal():
    core, _, _ = _core()
    with pytest.raises(IllegalMatrixInstruction, match="float <-> int"):
        _exec(core, "tconv", td=0, ts=1, b=0, to=INT32, **{"from": FP32})


# -- transposed multiply (Ztt mmulatacc.2d / mmulbtacc.2d) -----------------------------


@pytest.mark.parametrize("shape", [(4, 4, 4), (4, 3, 5)], ids=["square", "4x3x5"])
def test_mma_at_and_bt(shape):
    m, n, k = shape
    core, _, unit = _core(m=m, n=n, k=k)
    at = _rand(5, (k, m))  # mma.at reads tile 0 as K x M
    b = _rand(6, (k, n))
    unit.tiles[0, :k, : m * 4] = at.view(np.uint8).reshape(k, m * 4)
    unit.tile_view(1, "b", F32)[...] = b
    _exec(core, "mma.at", ad=0, ta=0, tb=1, **{"in": FP32, "acc": FP32})
    np.testing.assert_array_equal(
        unit.acc_view(0, F32), tile_mac(None, at.T, b, "fp32", "fp32")
    )
    a = _rand(7, (m, k))
    bt = _rand(8, (n, k))  # mma.bt reads tile 3 as N x K
    unit.tile_view(2, "a", F32)[...] = a
    unit.tiles[3, :n, : k * 4] = bt.view(np.uint8).reshape(n, k * 4)
    _exec(core, "mma.bt", ad=1, ta=2, tb=3, **{"in": FP32, "acc": FP32})
    np.testing.assert_array_equal(
        unit.acc_view(1, F32), tile_mac(None, a, bt.T, "fp32", "fp32")
    )
    assert unit.stats["mma"] == 2


# -- reductions and prefixes (Ztt mreduce* / mprefix*) -------------------------------


def _seq(x, fn, fmt):
    """Reference: a sequential fold along axis 1, every step rounded to ``fmt``."""
    out = np.empty_like(x)
    run = x[:, 0]
    out[:, 0] = run
    for j in range(1, x.shape[1]):
        run = fmt.round(fn(run, x[:, j])).astype(x.dtype)
        out[:, j] = run
    return out


@pytest.mark.parametrize("cols", [0, 1], ids=["row", "col"])
@pytest.mark.parametrize("op, fn", [(0, np.add), (1, np.maximum), (2, np.minimum)])
def test_treduce_broadcasts_a_sequential_fold(cols, op, fn):
    core, _, unit = _core()
    x = _rand(9).astype(np.float64)
    unit.tile_view(1, "a", F32)[...] = x
    _exec(core, "treduce", td=2, ts=1, op=op, fmt=FP32, cols=cols, b=0)
    xs = x.T if cols else x
    want = np.repeat(_seq(xs, fn, F32)[:, -1:], R, axis=1)
    np.testing.assert_array_equal(unit.tile_view(2, "a", F32), want.T if cols else want)


@pytest.mark.parametrize("cols", [0, 1], ids=["row", "col"])
@pytest.mark.parametrize("op, fn", [(0, np.add), (1, np.maximum)])
def test_tprefix_is_an_inclusive_sequential_scan(cols, op, fn):
    core, _, unit = _core()
    x = _rand(10).astype(np.float64)
    unit.tile_view(1, "b", F32)[...] = x
    _exec(core, "tprefix", td=1, ts=1, op=op, fmt=FP32, cols=cols, b=1)
    xs = x.T if cols else x
    want = _seq(xs, fn, F32)
    np.testing.assert_array_equal(unit.tile_view(1, "b", F32), want.T if cols else want)


def test_integer_prefix_wraps():
    core, _, unit = _core()
    unit.tile_view(0, "a", I32)[...] = np.int32(2**31 - 1)
    _exec(core, "tprefix", td=0, ts=0, op=0, fmt=INT32, cols=0, b=0)
    assert unit.tile_view(0, "a", I32)[0, 1] == np.int32(-2)


def test_undefined_reduce_ops_are_illegal():
    core, _, _ = _core()
    with pytest.raises(IllegalMatrixInstruction, match="op 3"):
        _exec(core, "treduce", td=0, ts=1, op=3, fmt=FP32, cols=0, b=0)
    with pytest.raises(IllegalMatrixInstruction, match="op 2"):
        _exec(core, "tprefix", td=0, ts=1, op=2, fmt=FP32, cols=0, b=0)


# -- the funnel shift (tshift2: not in Ztt) -----------------------------------------


@pytest.mark.parametrize("cols", [0, 1], ids=["rows", "cols"])
@pytest.mark.parametrize("amount", [0, 1, 3, 4])
def test_tshift2_is_a_window_into_the_concatenation(cols, amount):
    core, _, unit = _core()
    a, b = _rand(11), _rand(12)
    unit.tile_view(1, "a", F32)[...] = a
    unit.tile_view(2, "a", F32)[...] = b
    _exec(core, "tshift2", td=1, ta=1, tb=2, esz=2, cols=cols, amount=amount)
    cat = np.concatenate([a, b], axis=cols)
    want = cat[:, amount : amount + R] if cols else cat[amount : amount + R]
    np.testing.assert_array_equal(unit.tile_view(1, "a", F32), want)


def test_tshift2_limits():
    core, _, _ = _core()
    with pytest.raises(IllegalMatrixInstruction, match="amount 5"):
        _exec(core, "tshift2", td=0, ta=1, tb=2, esz=2, cols=0, amount=5)
    core, _, _ = _core(in_formats=("fp32",), acc_formats=("fp32",))
    with pytest.raises(IllegalMatrixInstruction, match="wider than"):
        _exec(core, "tshift2", td=0, ta=1, tb=2, esz=3, cols=0, amount=1)
    core, _, _ = _core(m=4, n=3, k=5)
    with pytest.raises(IllegalMatrixInstruction, match="square"):
        _exec(core, "tshift2", td=0, ta=1, tb=2, esz=2, cols=0, amount=1)


# -- gating ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature, name, ops",
    [
        ("acc_move", "amov.t.a", {"td": 0, "as": 0, "fmt": FP32, "b": 0}),
        ("acc_move", "amov.a.t", {"ad": 0, "ts": 0, "fmt": FP32, "b": 0}),
        ("convert", "tconv", {"td": 0, "ts": 0, "to": FP32, "from": FP32, "b": 0}),
        (
            "mma_transpose",
            "mma.at",
            {"ad": 0, "ta": 0, "tb": 1, "in": FP32, "acc": FP32},
        ),
        (
            "mma_transpose",
            "mma.bt",
            {"ad": 0, "ta": 0, "tb": 1, "in": FP32, "acc": FP32},
        ),
        (
            "reduce",
            "treduce",
            {"td": 0, "ts": 0, "op": 0, "fmt": FP32, "cols": 0, "b": 0},
        ),
        (
            "reduce",
            "tprefix",
            {"td": 0, "ts": 0, "op": 0, "fmt": FP32, "cols": 0, "b": 0},
        ),
        (
            "shift2",
            "tshift2",
            {"td": 0, "ta": 1, "tb": 2, "esz": 2, "cols": 0, "amount": 1},
        ),
    ],
)
def test_each_addition_is_illegal_unless_listed(feature, name, ops):
    others = tuple(f for f in ALL_FEATURES if f != feature)
    core, _, _ = _core(features=others)
    with pytest.raises(IllegalMatrixInstruction, match=f"'{feature}' feature"):
        _exec(core, name, **ops)
    core, _, _ = _core(features=(feature,))
    _exec(core, name, **ops)


def test_acc_mem_false_makes_aload_and_astore_illegal():
    core, _, unit = _core(features=(), acc_mem=False)
    assert "acc_mem" not in unit.features
    for name, ops in (
        ("aload", {"ad": 0}),
        ("astore", {"as": 0}),
    ):
        with pytest.raises(IllegalMatrixInstruction, match="'acc_mem' feature"):
            _exec(
                core,
                name,
                _mem_ops(DATA, 16),
                base=X_BASE,
                stride=X_STRIDE,
                fmt=FP32,
                **ops,
            )
    core, _, unit = _core(features=())
    assert "acc_mem" in unit.features  # on by default
    _exec(
        core,
        "astore",
        _mem_ops(DATA, 16),
        base=X_BASE,
        stride=X_STRIDE,
        fmt=FP32,
        **{"as": 0},
    )


def test_acc_mem_is_reported_by_mcfg():
    word = DECODER.encode("mcfg", rd=13, sel=TABLE.config["FEATURES"])
    for acc_mem in (True, False):
        core, _, unit = _core(features=("shift",), acc_mem=acc_mem)
        run(word, core=core)
        want = 1 << TABLE.features["shift"]
        if acc_mem:
            want |= 1 << TABLE.features["acc_mem"]
        assert reg(core, 13) == want
