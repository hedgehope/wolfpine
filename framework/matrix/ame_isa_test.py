"""AME matrix unit: decode (from the table), per-instruction semantics vs numpy, and gating.

Each test runs real instruction words through an ``RV64I`` with the unit
attached, so decode, the GPR/memory side and the unit's state are exercised
together. The unit is deliberately non-square (M=4, N=3, K=5) so a shape mixup
cannot pass.
"""

import copy
import random
import sys

import numpy as np
import pytest
import yaml

from framework.matrix import formats
from framework.matrix.ame_isa import RV_AME_ISA, attach
from framework.matrix.encodings import (
    DECODER,
    ENCODINGS_PATH,
    TABLE,
    Decoder,
    EncodingTable,
    EncodingTableError,
)
from framework.matrix.mac import tile_mac
from framework.matrix.semiring import tropical_accumulate
from framework.matrix.unit import IllegalMatrixInstruction, MatrixUnit
from framework.pe.rv.isa.isa64_testing import DATA, make_core, reg, run

M, N, K = 4, 3, 5
X_BASE, X_STRIDE, X_VAL, X_RD = 10, 11, 12, 13
FP32, FP64, INT8, INT32 = (TABLE.formats[n] for n in ("fp32", "fp64", "int8", "int32"))
ALL_FEATURES = tuple(TABLE.features)


def _unit(**kw):
    args = dict(
        m=M,
        n=N,
        k=K,
        tiles=4,
        accs=2,
        in_formats=("fp32", "int8"),
        acc_formats=("fp32", "int32"),
        features=ALL_FEATURES,
    )
    args.update(kw)
    return MatrixUnit(**args)


def _core(**kw):
    core, memory = make_core([])
    unit = attach(core, _unit(**kw))
    return core, memory, unit


def _exec(core, name, regs=None, **ops):
    run(DECODER.encode(name, **ops), regs, core=core)


def _mem_ops(base=DATA, stride=0):
    return {X_BASE: base, X_STRIDE: stride}


# -- decode, driven from the table ------------------------------------------


@pytest.mark.parametrize("name", sorted(TABLE.instructions))
def test_every_table_instruction_round_trips_through_the_decoder(name):
    insn = TABLE.instructions[name]
    rng = random.Random(name)
    for _ in range(20):
        ops = {op.name: rng.randrange(1 << op.width) for op in insn.operands}
        word = DECODER.encode(name, **ops)
        assert word & 0x7F == insn.opcode
        assert (word >> 12) & 7 == insn.funct3
        decoded, got = DECODER.decode(word)
        assert decoded is insn
        assert got == ops


@pytest.mark.parametrize("name", sorted(TABLE.instructions))
def test_a_set_reserved_bit_is_illegal(name):
    insn = TABLE.instructions[name]
    reserved = insn.mask & ~0x707F
    if not reserved:
        pytest.skip(f"{name} has no reserved bits")
    bit = reserved & -reserved
    assert DECODER.decode(insn.match | bit) is None
    core, _, _ = _core()
    with pytest.raises(IllegalMatrixInstruction, match="no matrix instruction"):
        run(insn.match | bit, core=core)


def test_an_unassigned_funct3_in_a_matrix_opcode_is_illegal():
    core, _, _ = _core()
    with pytest.raises(IllegalMatrixInstruction):
        run(TABLE.opcodes["custom-1"] | (7 << 12), core=core)


def test_other_opcodes_are_declined():
    core, _, _ = _core()
    addi = (5 << 20) | (1 << 7) | 0x13
    assert RV_AME_ISA.run(core.register_file, core.visible_memory, False, addi) is False
    bare, _ = make_core([])
    word = DECODER.encode("azero", ad=0)
    assert RV_AME_ISA.run(bare.register_file, bare.visible_memory, False, word) is False


def test_the_decoder_follows_an_edited_table():
    raw = yaml.safe_load(ENCODINGS_PATH.read_text())
    edited = copy.deepcopy(raw)
    edited["instructions"]["mma"]["funct3"] = 7
    edited["instructions"]["mma"]["operands"]["in"]["bits"] = [28, 31]
    edited["instructions"]["mma"]["operands"]["acc"]["bits"] = [25, 27]
    decoder = Decoder(EncodingTable(edited))
    word = decoder.encode("mma", ad=1, ta=2, tb=3, **{"in": FP32, "acc": FP32})
    assert (word >> 12) & 7 == 7
    assert word >> 28 == FP32
    assert DECODER.decode(word) is None  # the shipped table has nothing there

    core, _ = make_core([])
    unit = attach(core, _unit(decoder=decoder))
    unit.tile_view(2, "a", formats.get_format("fp32"))[...] = 1
    unit.tile_view(3, "b", formats.get_format("fp32"))[...] = 2
    run(word, core=core)
    assert (unit.acc_view(1, formats.get_format("fp32")) == 2 * K).all()


def _table_with(change):
    raw = copy.deepcopy(yaml.safe_load(ENCODINGS_PATH.read_text()))
    change(raw)
    return EncodingTable(raw)


def test_the_table_is_validated():
    def clash(raw):
        raw["instructions"]["mma"]["funct3"] = 3

    with pytest.raises(EncodingTableError, match="overlapping encodings"):
        _table_with(clash)

    def overlap(raw):
        raw["instructions"]["mma"]["operands"]["acc"]["bits"] = [28, 30]

    with pytest.raises(EncodingTableError, match="overlap"):
        _table_with(overlap)

    def alias(raw):
        raw["formats"]["bfloat16"] = raw["formats"].pop("bf16")

    with pytest.raises(EncodingTableError, match="canonical"):
        _table_with(alias)

    def bad_field(raw):
        raw["instructions"]["mcfg"]["operands"]["sel"] = {
            "field": "rs2",
            "kind": "config",
        }

    with pytest.raises(EncodingTableError, match="fields"):
        _table_with(bad_field)


# -- semantics vs numpy -----------------------------------------------------


def _ints(shape, lo, hi, dtype, seed):
    return np.random.default_rng(seed).integers(lo, hi, shape).astype(dtype)


def test_tile_load_and_store_with_a_row_stride():
    core, memory, unit = _core()
    stride = 4 * K + 12  # padded rows
    a = np.random.default_rng(1).standard_normal((M, K)).astype(np.float32)
    for r in range(M):
        memory.write(DATA + r * stride, a[r].tobytes())
    unit.tiles[0].fill(0xAB)
    _exec(
        core,
        "tload.a",
        _mem_ops(DATA, stride),
        td=0,
        base=X_BASE,
        stride=X_STRIDE,
        fmt=FP32,
    )
    np.testing.assert_array_equal(unit.tile_view(0, "a", formats.get_format("fp32")), a)
    # Outside the loaded shape the tile is zero, not what was there before
    assert unit.tiles[0].sum() == np.frombuffer(a.tobytes(), np.uint8).sum(
        dtype=np.int64
    )

    out = DATA + 0x800
    _exec(
        core,
        "tstore.a",
        _mem_ops(out, 4 * K),
        ts=0,
        base=X_BASE,
        stride=X_STRIDE,
        fmt=FP32,
    )
    back = np.frombuffer(memory.read(out, 4 * M * K), np.float32).reshape(M, K)
    np.testing.assert_array_equal(back, a)
    assert unit.stats["mem_to_tile_bytes"] == 4 * M * K
    assert unit.stats["tile_to_mem_bytes"] == 4 * M * K


def test_b_tiles_are_k_by_n_and_int8_is_one_byte_per_element():
    core, memory, unit = _core()
    b = _ints((K, N), -128, 128, np.int8, 2)
    memory.write(DATA, b.tobytes())
    _exec(
        core, "tload.b", _mem_ops(DATA, N), td=1, base=X_BASE, stride=X_STRIDE, fmt=INT8
    )
    np.testing.assert_array_equal(unit.tile_view(1, "b", formats.get_format("int8")), b)
    _exec(
        core,
        "tstore.b",
        _mem_ops(DATA + 0x400, N),
        ts=1,
        base=X_BASE,
        stride=X_STRIDE,
        fmt=INT8,
    )
    assert memory.read(DATA + 0x400, K * N) == b.tobytes()


@pytest.mark.parametrize("role", ["a", "b"])
def test_transposed_loads(role):
    core, memory, unit = _core()
    rows, cols = (M, K) if role == "a" else (K, N)
    tile = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    memory.write(DATA, np.ascontiguousarray(tile.T).tobytes())
    name = f"tload.{role}"
    _exec(
        core,
        name,
        _mem_ops(DATA, 4 * rows),
        td=2,
        base=X_BASE,
        stride=X_STRIDE,
        fmt=FP32,
        t=1,
    )
    np.testing.assert_array_equal(
        unit.tile_view(2, role, formats.get_format("fp32")), tile
    )


def test_accumulator_load_store_and_zero():
    core, memory, unit = _core()
    c = _ints((M, N), -(1 << 31), 1 << 31, np.int32, 3)
    stride = 4 * N + 4
    for r in range(M):
        memory.write(DATA + r * stride, c[r].tobytes())
    _exec(
        core,
        "aload",
        _mem_ops(DATA, stride),
        ad=1,
        base=X_BASE,
        stride=X_STRIDE,
        fmt=INT32,
    )
    np.testing.assert_array_equal(unit.acc_view(1, formats.get_format("int32")), c)
    _exec(
        core,
        "astore",
        _mem_ops(DATA + 0x800, 4 * N),
        base=X_BASE,
        stride=X_STRIDE,
        fmt=INT32,
        **{"as": 1},
    )
    assert memory.read(DATA + 0x800, 4 * M * N) == c.tobytes()
    _exec(core, "azero", ad=1)
    assert not unit.accs[1].any()
    assert unit.stats["mem_to_acc_bytes"] == unit.stats["acc_to_mem_bytes"] == 4 * M * N


def _load(unit, a, b, fmt, c=None, acc_fmt=None):
    f = formats.get_format(fmt)
    unit.tile_view(0, "a", f)[...] = a
    unit.tile_view(1, "b", f)[...] = b
    if c is not None:
        unit.acc_view(0, formats.get_format(acc_fmt))[...] = c


def test_mma_fp32_matches_tile_mac():
    core, _, unit = _core()
    rng = np.random.default_rng(4)
    a, b = (rng.standard_normal(s).astype(np.float32) for s in ((M, K), (K, N)))
    c = rng.standard_normal((M, N)).astype(np.float32)
    _load(unit, a, b, "fp32", c, "fp32")
    _exec(core, "mma", ad=0, ta=0, tb=1, **{"in": FP32, "acc": FP32})
    np.testing.assert_array_equal(
        unit.acc_view(0, formats.get_format("fp32")), tile_mac(c, a, b, "fp32", "fp32")
    )
    np.testing.assert_allclose(
        unit.acc_view(0, formats.get_format("fp32")),
        c + a.astype(np.float64) @ b,
        rtol=1e-5,
    )
    assert unit.stats["mma"] == 1
    assert unit.stats["macs"] == M * N * K


def test_mma_int8_accumulates_in_int32_with_wraparound():
    core, _, unit = _core()
    a = np.full((M, K), 127, np.int8)
    b = np.full((K, N), -128, np.int8)
    c = np.full((M, N), -(1 << 31), np.int32)
    _load(unit, a, b, "int8", c, "int32")
    _exec(core, "mma", ad=0, ta=0, tb=1, **{"in": INT8, "acc": INT32})
    want = (
        (c.astype(np.int64) + a.astype(np.int64) @ b.astype(np.int64) + (1 << 31))
        % (1 << 32)
        - (1 << 31)
    ).astype(np.int32)
    np.testing.assert_array_equal(unit.acc_view(0, formats.get_format("int32")), want)


@pytest.mark.parametrize(
    "name,semiring", [("mma.minplus", "min_plus"), ("mma.maxplus", "max_plus")]
)
def test_tropical_mma(name, semiring):
    core, _, unit = _core()
    a = _ints((M, K), -50, 50, np.float32, 5)
    b = _ints((K, N), -50, 50, np.float32, 6)
    c = _ints((M, N), -60, 60, np.float32, 7)
    _load(unit, a, b, "fp32", c, "fp32")
    _exec(core, name, ad=0, ta=0, tb=1, **{"in": FP32, "acc": FP32})
    reduce = np.min if semiring == "min_plus" else np.max
    want = (np.minimum if semiring == "min_plus" else np.maximum)(
        c, reduce(a[:, :, None] + b[None, :, :], axis=1)
    )
    np.testing.assert_array_equal(unit.acc_view(0, formats.get_format("fp32")), want)
    np.testing.assert_array_equal(want, tropical_accumulate(c, a, b, semiring))


@pytest.mark.parametrize("amount", [2, -1, 0, 9])
@pytest.mark.parametrize("cols", [0, 1])
@pytest.mark.parametrize("b", [0, 1])
def test_tshift(amount, cols, b):
    core, _, unit = _core()
    role = "b" if b else "a"
    f = formats.get_format("fp32")
    src = unit.tile_view(1, role, f)
    src[...] = np.arange(1, src.size + 1, dtype=np.float32).reshape(src.shape)
    want = np.zeros_like(src)
    axis = 1 if cols else 0
    n = src.shape[axis]
    for i in range(n):
        j = i - amount
        if 0 <= j < n:
            if axis == 0:
                want[i] = src[j]
            else:
                want[:, i] = src[:, j]
    _exec(
        core,
        "tshift",
        {X_VAL: amount},
        td=2,
        ts=1,
        amount=X_VAL,
        fmt=FP32,
        cols=cols,
        b=b,
    )
    np.testing.assert_array_equal(unit.tile_view(2, role, f), want)
    # In place
    _exec(
        core,
        "tshift",
        {X_VAL: amount},
        td=1,
        ts=1,
        amount=X_VAL,
        fmt=FP32,
        cols=cols,
        b=b,
    )
    np.testing.assert_array_equal(unit.tile_view(1, role, f), want)


def test_epilogue_scale_and_add():
    core, _, unit = _core()
    f32, i32 = formats.get_format("fp32"), formats.get_format("int32")
    c = np.linspace(-3, 3, M * N, dtype=np.float32).reshape(M, N)
    unit.acc_view(0, f32)[...] = c
    scale = np.float32(0.1)
    _exec(
        core, "ascal", {X_VAL: int(scale.view(np.uint32))}, ad=0, value=X_VAL, fmt=FP32
    )
    np.testing.assert_array_equal(unit.acc_view(0, f32), c * scale)
    _exec(
        core,
        "aadds",
        {X_VAL: int(np.float32(2.5).view(np.uint32))},
        ad=0,
        value=X_VAL,
        fmt=FP32,
    )
    np.testing.assert_array_equal(unit.acc_view(0, f32), c * scale + np.float32(2.5))

    unit.acc_view(1, i32)[...] = (1 << 30) + 5
    _exec(core, "ascal", {X_VAL: 4}, ad=1, value=X_VAL, fmt=INT32)  # wraps
    np.testing.assert_array_equal(unit.acc_view(1, i32), np.int32(20))
    _exec(
        core, "aadds", {X_VAL: (1 << 64) - 30}, ad=1, value=X_VAL, fmt=INT32
    )  # -30 in the low 32 bits
    np.testing.assert_array_equal(unit.acc_view(1, i32), np.int32(-10))


def test_mcfg_reads_shape_formats_and_features():
    core, _, unit = _core(features=("shift", "semiring"))
    want = {
        "M": M,
        "N": N,
        "K": K,
        "TILES": 4,
        "ACCS": 2,
        "IN_FORMATS": (1 << FP32) | (1 << INT8),
        "ACC_FORMATS": (1 << FP32) | (1 << INT32),
        "FEATURES": (1 << TABLE.features["shift"]) | (1 << TABLE.features["semiring"]),
    }
    for name, value in want.items():
        _exec(core, "mcfg", rd=X_RD, sel=TABLE.config[name])
        assert reg(core, X_RD) == value, name
    with pytest.raises(IllegalMatrixInstruction, match="selector"):
        _exec(core, "mcfg", rd=X_RD, sel=99)


def test_reset_clears_state_and_counters():
    core, _, unit = _core()
    unit.tiles.fill(1)
    unit.stats["mma"] = 3
    unit.reset()
    assert not unit.tiles.any()
    assert unit.stats["mma"] == 0


# -- gating: what the design leaves out is illegal --------------------------


def test_an_unlisted_input_format_is_illegal():
    core, _, _ = _core()
    with pytest.raises(
        IllegalMatrixInstruction, match="fp64 is not an implemented input format"
    ):
        _exec(core, "tload.a", _mem_ops(), td=0, base=X_BASE, stride=X_STRIDE, fmt=FP64)
    with pytest.raises(IllegalMatrixInstruction, match="input format"):
        _exec(core, "mma", ad=0, ta=0, tb=1, **{"in": FP64, "acc": FP32})


def test_an_unlisted_accumulator_format_is_illegal():
    core, _, _ = _core(acc_formats=("fp32",), in_formats=("fp32",))
    with pytest.raises(
        IllegalMatrixInstruction, match="int32 is not an implemented accumulator"
    ):
        _exec(core, "aload", _mem_ops(), ad=0, base=X_BASE, stride=X_STRIDE, fmt=INT32)
    with pytest.raises(IllegalMatrixInstruction, match="code 15"):
        _exec(
            core,
            "astore",
            _mem_ops(),
            base=X_BASE,
            stride=X_STRIDE,
            fmt=15,
            **{"as": 0},
        )


def test_mixing_float_and_int_formats_is_illegal():
    core, _, _ = _core()
    with pytest.raises(IllegalMatrixInstruction, match="cannot pair"):
        _exec(core, "mma", ad=0, ta=0, tb=1, **{"in": INT8, "acc": FP32})


def test_indices_past_the_register_files_are_illegal():
    core, _, _ = _core(tiles=2, accs=1)
    with pytest.raises(IllegalMatrixInstruction, match="tile 2"):
        _exec(core, "mma", ad=0, ta=0, tb=2, **{"in": FP32, "acc": FP32})
    with pytest.raises(IllegalMatrixInstruction, match="accumulator 1"):
        _exec(core, "azero", ad=1)


@pytest.mark.parametrize(
    "feature,name,ops",
    [
        (
            "transpose",
            "tload.a",
            dict(td=0, base=X_BASE, stride=X_STRIDE, fmt=FP32, t=1),
        ),
        ("shift", "tshift", dict(td=0, ts=1, amount=0, fmt=FP32)),
        ("epilogue", "ascal", dict(ad=0, value=0, fmt=FP32)),
        ("epilogue", "aadds", dict(ad=0, value=0, fmt=FP32)),
        (
            "semiring",
            "mma.minplus",
            {"ad": 0, "ta": 0, "tb": 1, "in": FP32, "acc": FP32},
        ),
        (
            "semiring",
            "mma.maxplus",
            {"ad": 0, "ta": 0, "tb": 1, "in": FP32, "acc": FP32},
        ),
    ],
)
def test_optional_features_are_illegal_unless_listed(feature, name, ops):
    others = tuple(f for f in ALL_FEATURES if f != feature)
    core, _, _ = _core(features=others)
    with pytest.raises(IllegalMatrixInstruction, match=f"'{feature}' feature"):
        _exec(core, name, _mem_ops(), **ops)
    core, _, _ = _core(features=(feature,))
    _exec(core, name, _mem_ops(), **ops)


def test_a_plain_load_needs_no_transpose_feature():
    core, _, _ = _core(features=())
    _exec(
        core,
        "tload.a",
        _mem_ops(DATA, 4 * K),
        td=0,
        base=X_BASE,
        stride=X_STRIDE,
        fmt=FP32,
    )


def _hide_ml_dtypes(monkeypatch):
    monkeypatch.setitem(sys.modules, "ml_dtypes", None)
    monkeypatch.setattr(formats, "_ML_DTYPES", None)


def test_wide_and_integer_formats_need_no_ml_dtypes(monkeypatch):
    _hide_ml_dtypes(monkeypatch)
    unit = _unit(
        in_formats=("fp64", "fp32", "fp16", "int8"),
        acc_formats=("fp64", "fp32", "int32"),
    )
    assert unit.tiles.shape == (4, max(M, K), max(K, N) * 8)


def test_narrow_float_formats_need_ml_dtypes(monkeypatch):
    _hide_ml_dtypes(monkeypatch)
    with pytest.raises(ImportError, match="ml_dtypes"):
        _unit(in_formats=("bf16",), acc_formats=("fp32",))


def test_bf16_inputs_with_ml_dtypes():
    pytest.importorskip("ml_dtypes")
    core, _, unit = _core(in_formats=("bf16",), acc_formats=("fp32",))
    bf16 = formats.get_format("bf16")
    a = bf16.round(np.random.default_rng(8).standard_normal((M, K)))
    b = bf16.round(np.random.default_rng(9).standard_normal((K, N)))
    _load(unit, a, b, "bf16")
    _exec(core, "mma", ad=0, ta=0, tb=1, **{"in": TABLE.formats["bf16"], "acc": FP32})
    np.testing.assert_array_equal(
        unit.acc_view(0, formats.get_format("fp32")),
        tile_mac(None, a, b, "bf16", "fp32"),
    )
