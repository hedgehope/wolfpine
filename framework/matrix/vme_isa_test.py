"""VME (Zvt v0.4): encodings, decode, configuration, every instruction against a numpy / rational golden.

The core is a real ``RV64I`` with F/D/Zicsr/RVV and the VME unit attached (``VCore`` plus
:func:`framework.matrix.vme_isa.attach`); every instruction is executed by placing its word at
the PC and clocking the core. Spike has no Zvt, so the goldens are written here: plain numpy
(fp32/fp64: a product rounded to TEW, then an add, per k; integers: an int64 sum wrapped),
and an independent per-element RVBNA (Python integers, the pseudocode's steps) with a
rational round-to-odd for 16- and 8-bit inputs.
"""

import importlib.util
import math
from fractions import Fraction

import numpy as np
import pytest

from framework.matrix import vme
from framework.matrix.formats import get_format
from framework.matrix.unit import IllegalMatrixInstruction, MatrixAccessFault
from framework.matrix.vme import TABLE, VMEError, VMESpec, encode
from framework.matrix.vme_isa import RV64_VME_ISA, VMEUnit, attach
from framework.pe.rv.isa.isa64_testing import csr
from framework.pe.rv.isa.v import isa as visa
from framework.pe.rv.isa.v.vtesting import DATA, VCore, vsetvli, vtype

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)

RNG = np.random.default_rng(11)
ALL = tuple(vme.EXTENSIONS)
T1, T2, T3, BASE = 6, 7, 28, 10
SEW_CODE = {8: 0, 16: 1, 32: 2, 64: 3}
MTW = {1: 1, 2: 2, 4: 3}


class ZCore(VCore):
    """``VCore`` with the VME unit attached after RVV."""

    def __init__(self, te=8, vlen=256, extensions=ALL):
        super().__init__(vlen=vlen)
        self.spec = VMESpec(vlen, 64, te, tuple(extensions))
        self.unit = attach(self.core, VMEUnit(self.spec, self.vs))

    def mset(self, sew, tw, alt=0, tm=None, tk=None, tn=None):
        """``msetmtypei`` then the msett* the caller asks for; returns (tm, tk, tn) granted."""
        self.exec(encode("msetmtypei", altfmt=alt, vsew=SEW_CODE[sew], mtwiden=MTW[tw]))
        out = []
        for name, v in (("msettm", tm), ("msettk", tk), ("msettn", tn)):
            if v is not None:
                self.x(T1, v)
                self.exec(encode(name, rd=T2, rs1=T1))
                out.append(self.x(T2))
        return out

    def tss(self, tile, index, col=0):
        self.x(T3, vme.tss(tile, index, col))

    def tile(self, tile, tew, dtype):
        return self.unit.tile(tile, tew).view(dtype)


# -- encodings ----------------------------------------------------------------------------------------

#: riscv-opcodes rv_zvt* @ bf6a8a4 match words (operands zero), from the files' bit fields.
OPCODES_MATCH = {
    "msettn": 0x88007057,
    "msettm": 0x88107057,
    "msettk": 0x88207057,
    "msetmtypei": 0x88407057,
    "msetmtype": 0x84007057,
    "vtle": 0x12007007,  # vtle8 (31..25 = 0x09)
    "vtse": 0x12007027,
    "vtmv.v.t": 0x43F06057,
    "vtmv.t.v": 0x5E006057,
    "vtzero": 0x43E06057,
    "vtdiscard": 0x43C06057,
    "vtfmm.tvv": 0xF2001077,
    "vtfmm.alt.tvv": 0xF20010F7,
    "vtmmu.tvv": 0xF2000077,
    "vtmms.tvv": 0xF20000F7,
}


def test_table_matches_riscv_opcodes():
    assert {n: i.match for n, i in TABLE.items()} == OPCODES_MATCH
    # vtle16/32/64 are nf = 1/2/3 of the same word (31..25 = 0x19 / 0x29 / 0x39).
    assert (
        encode("vtle", eee=2, rs1=10, rs2=11)
        == (0x29 << 25) | (11 << 20) | (10 << 15) | (7 << 12) | 0x07
    )


def test_clang_assembled_words_decode():
    # .insn r 0x77, 1, 0x79, x4, x8, x12 (vtfmm.tvv mt2, v12, v8) and a msettn from clang 20.
    assert vme.decode(0xF2C41277).name == "vtfmm.tvv"
    assert TABLE["vtfmm.tvv"].field(0xF2C41277, "mtd") == 1  # t[3:1] = 1: tile 2
    assert vme.decode(0x88057557).name == "msettn"


@pytest.mark.parametrize("name", sorted(TABLE))
def test_rvv_declines_every_zvt_word(name):
    insn = TABLE[name]
    for _ in range(64):
        ops = {
            op: int(RNG.integers(0, 1 << (hi - lo + 1)))
            for op, (lo, hi) in insn.operands.items()
        }
        word = encode(name, **ops)
        assert vme.decode(word) is insn
        assert visa.decode(word) is None, f"RVV claims {name} {word:#010x}"


def test_the_only_ime_collision_is_the_tile_load_store():
    assert sorted(vme.collisions()) == [("vtle", "vmtl.v"), ("vtse", "vmts.v")]


def test_dispatch_puts_vme_after_rvv():
    z = ZCore()
    names = [i.__name__ for i in z.core.isas]
    assert names.index("RV64_VME_ISA") > names.index("RV64_V_ISA")
    assert RV64_VME_ISA.OPCODES == {0x07, 0x27, 0x57, 0x77}


# -- configuration ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sew,tw,lmul,kmax",
    [
        (32, 1, 4, 1),
        (16, 2, 2, 2),
        (64, 1, 4, 1),
        (8, 4, 1, 4),
        (16, 1, 2, 2),
        (8, 1, 1, 4),
    ],
)
def test_msetmtypei_derives_lmul_and_policies(sew, tw, lmul, kmax):
    z = ZCore(te=32)
    z.x(T1, 99)
    z.exec(vsetvli(T2, T1, vtype(8, 1)))
    assert z.vs.vl > 0
    z.exec(encode("msetmtypei", altfmt=0, vsew=SEW_CODE[sew], mtwiden=MTW[tw]))
    vs = z.vs
    assert not vs.vill
    assert (vs.sew, 1 << vs.lmul_log2, vs.vta, vs.vma, vs.vl) == (sew, lmul, 1, 1, 0)
    assert z.unit.mtype == MTW[tw]  # tm = tk = 0
    ete = 32 if sew * tw < 64 else 16
    assert z.mset(sew, tw, tm=1000, tk=7, tn=1000) == [
        min(ete, lmul * 256 // sew),
        kmax,
        min(ete, lmul * 256 // sew),
    ]
    z.exec(csr(2, vme.CSR_MTYPE, rd=T2, rs1=0))
    tm = min(ete, lmul * 256 // sew)
    assert z.x(T2) == (tm << 10) | (kmax << 5) | MTW[tw]


def test_vset_writes_mtype_zero_and_msettm_then_traps():
    z = ZCore()
    z.mset(32, 1, tm=8, tk=1, tn=8)
    assert z.unit.mtype
    z.exec(vsetvli(T2, 0, vtype(32, 1)))
    assert z.unit.mtype == 0
    z.x(T1, 4)
    with pytest.raises(IllegalMatrixInstruction, match="mtwiden = 0"):
        z.exec(encode("msettm", rd=T2, rs1=T1))
    # msettn degenerates to vsetvl semantics under the current vtype.
    z.x(T1, 1000)
    z.exec(encode("msettn", rd=T2, rs1=T1))
    assert z.x(T2) == z.vs.vlmax == 8


def test_msetmtype_from_registers_clamps_and_vills():
    z = ZCore(te=8)
    z.x(T1, (100 << 10) | (7 << 5) | 1)  # tm 100, tk 7, mtwiden 1
    z.x(T2, vtype(32, 1))
    z.exec(encode("msetmtype", rs1=T1, rs2=T2))
    assert (z.unit.tm, z.unit.tk, z.unit.mtwiden) == (8, 1, 1)
    # TEW 128 (> ELEN), or TEW 8 without Zvt8e: vill, mtype = 0.
    z.x(T1, 3)
    z.exec(encode("msetmtype", rs1=T1, rs2=T2))
    assert z.vs.vill
    assert z.unit.mtype == 0
    z2 = ZCore(extensions=("Zvtbase", "Zvtfmm"))
    z2.exec(encode("msetmtypei", altfmt=0, vsew=0, mtwiden=1))
    assert z2.vs.vill
    assert z2.unit.mtype == 0


def test_altfmt_is_legal_only_with_an_extension_that_needs_it():
    z = ZCore(extensions=("Zvtbase", "Zvtfmm"))
    z.exec(encode("msetmtypei", altfmt=1, vsew=2, mtwiden=1))
    assert z.vs.vill
    z = ZCore()
    z.exec(encode("msetmtypei", altfmt=1, vsew=1, mtwiden=2))
    assert not z.vs.vill
    assert (z.vs.vtype >> 8) & 1


# -- tile state ------------------------------------------------------------------------------------------


def test_punning_matches_the_draft_function_and_is_a_bijection():
    for te in (4, 8, 16):
        for tew in vme.TEWS:
            maps = vme.tile_byte_index(te, tew)
            flat = np.concatenate([m.ravel() for m in maps.values()])
            assert sorted(flat.tolist()) == list(range(16 * te * te))
            e = vme.ete(te, tew)
            for t, m in maps.items():
                assert all(
                    m[r, c, 0] == vme.punning(te, t, r, c, tew)
                    for r in range(e)
                    for c in range(e)
                )


def test_cross_tew_view_is_the_punned_bytes():
    z = ZCore(te=8)
    vals = RNG.integers(0, 2**32, (8, 8), dtype=np.uint32)
    z.unit.set_tile(4, 32, vals, 8, 8)
    for r in range(8):
        for c in range(8):
            off = vme.punning(8, 4, r, c, 32)
            assert z.unit.buf[off : off + 4].view(np.uint32)[0] == vals[r, c]
    # TEW 8 tile 5 is part of TEW 32 tile 4's storage (ptiles 4..7).
    assert set(np.unique(z.unit.maps[8][5] // 64).tolist()) == {5}


def test_vtle_rows_vtse_columns_transpose():
    z = ZCore(te=8)
    z.mset(32, 1, tm=8, tk=1, tn=8)
    a = RNG.standard_normal((8, 8)).astype(np.float32)
    z.mem_write(DATA, a)
    z.x(BASE, DATA)
    for r in range(8):
        z.x(BASE, DATA + 32 * r)
        z.tss(8, r)
        z.exec(encode("vtle", eee=2, rs1=BASE, rs2=T3))
    assert np.array_equal(z.tile(8, 32, np.float32), a)
    for c in range(8):
        z.x(BASE, DATA + 0x1000 + 32 * c)
        z.tss(8, c, col=1)
        z.exec(encode("vtse", eee=2, rs1=BASE, rs2=T3))
    assert np.array_equal(z.mem_read(DATA + 0x1000, 64, np.float32).reshape(8, 8), a.T)
    assert z.unit.counters()["mem_to_tile_bytes"] == 8 * 32
    assert z.unit.counters()["tile_to_mem_bytes"] == 8 * 32


def test_vtle_body_vstart_tail_and_alignment():
    z = ZCore(te=8)
    z.mset(32, 1, tm=8, tk=1, tn=5)
    before = z.tile(0, 32, np.uint32).copy()
    z.mem_write(DATA, np.arange(1, 9, dtype=np.uint32))
    z.x(BASE, DATA)
    z.tss(0, 2)
    z.vs.vstart = 1
    z.exec(encode("vtle", eee=2, rs1=BASE, rs2=T3))
    after = z.tile(0, 32, np.uint32)
    expect = before.copy()
    expect[2, 1:5] = [
        2,
        3,
        4,
        5,
    ]  # body [vstart, min(vl, ETE)); memory from base + vstart * EEW
    assert np.array_equal(after, expect)
    assert z.vs.vstart == 0
    z.x(BASE, DATA + 2)
    with pytest.raises(MatrixAccessFault, match="aligned"):
        z.exec(encode("vtle", eee=2, rs1=BASE, rs2=T3))


def test_tile_moves_need_their_tew_extension_and_a_configuration():
    z = ZCore(extensions=("Zvtbase", "Zvtfmm"))
    z.x(BASE, DATA)
    z.tss(0, 0)
    with pytest.raises(IllegalMatrixInstruction, match="vill"):
        z.exec(encode("vtle", eee=2, rs1=BASE, rs2=T3))  # after reset vtype is vill
    z.exec(vsetvli(T2, 0, vtype(32, 1)))
    with pytest.raises(IllegalMatrixInstruction, match="mtwiden = 0"):
        z.exec(encode("vtle", eee=2, rs1=BASE, rs2=T3))
    z.mset(32, 1, tn=8)
    with pytest.raises(IllegalMatrixInstruction, match="Zvt16e"):
        z.exec(encode("vtle", eee=1, rs1=BASE, rs2=T3))
    with pytest.raises(IllegalMatrixInstruction, match="reserved EEW"):
        z.exec(encode("vtle", eee=4, rs1=BASE, rs2=T3))


def test_vtmv_round_trip_by_row_and_column():
    z = ZCore(te=16)
    z.mset(32, 1, tm=16, tk=1, tn=16)  # LMUL 2
    t = RNG.integers(0, 2**32, (16, 16), dtype=np.uint32)
    z.unit.set_tile(12, 32, t, 16, 16)
    z.tss(12, 3, col=1)
    z.exec(encode("vtmv.v.t", vd=4, rs1=T3))
    assert np.array_equal(z.v(4, np.uint32, 2)[:16], t[:, 3])
    z.v(8, np.uint32, 2)[:16] = np.arange(16, dtype=np.uint32)
    z.tss(12, 5)
    z.exec(encode("vtmv.t.v", rs1=T3, vs2=8))
    assert np.array_equal(z.tile(12, 32, np.uint32)[5], np.arange(16))
    c = z.unit.counters()
    assert c["tile_to_vec_bytes"] == c["vec_to_tile_bytes"] == 64
    z.tss(12, 0)
    with pytest.raises(IllegalMatrixInstruction, match="LMUL"):
        z.exec(encode("vtmv.v.t", vd=5, rs1=T3))


def test_vtmv_tail_is_left_undisturbed():
    z = ZCore(te=8)
    z.mset(32, 1, tm=8, tk=1, tn=3)
    z.v(16, np.uint32)[:] = 7
    z.unit.set_tile(0, 32, np.full((8, 8), 9, np.uint32), 8, 8)
    z.tss(0, 0)
    z.exec(encode("vtmv.v.t", vd=16, rs1=T3))
    assert z.v(16, np.uint32).tolist() == [9, 9, 9, 7, 7, 7, 7, 7]


def test_vtzero_clears_tm_by_tn_at_sew():
    z = ZCore(te=8)
    z.unit.set_tile(4, 32, np.full((8, 8), 5, np.uint32), 8, 8)
    z.mset(32, 1, tm=3, tk=1, tn=6)
    z.exec(encode("vtzero", mtd=4))
    t = z.tile(4, 32, np.uint32)
    assert (t[:3, :6] == 0).all()
    assert (t[3:] == 5).all()
    assert (t[:, 6:] == 5).all()
    z.mset(32, 1, tm=8, tn=8)
    with pytest.raises(IllegalMatrixInstruction, match="not a TEW-32 tile"):
        z.exec(encode("vtzero", mtd=2))


def test_vtdiscard_is_a_counted_no_op_and_reset_clears_state():
    z = ZCore()
    z.mset(32, 1, tm=8, tk=1, tn=8)
    z.unit.set_tile(0, 32, np.ones((8, 8), np.uint32), 8, 8)
    z.exec(encode("vtdiscard"))
    assert z.unit.counters()["discard"] == 1
    assert z.unit.buf.any()
    z.unit.reset()
    assert not z.unit.buf.any()
    assert z.unit.mtype == 0
    assert z.unit.counters()["discard"] == 0


# -- multiplies ------------------------------------------------------------------------------------------


def _operands(z, base, rows, stride, lmul, dtype):
    for r, row in enumerate(rows):
        z.v(base + r * stride, dtype, lmul)[: len(row)] = row


def _golden_ieee(C, A, B, dt):
    out = C.astype(dt).copy()
    for k in range(A.shape[0]):
        out = out + A[k].astype(dt)[:, None] * B[k].astype(dt)[None, :]
    return out


@pytest.mark.parametrize(
    "fmt,sew,te", [("fp32", 32, 8), ("fp32", 32, 32), ("fp64", 64, 8), ("fp64", 64, 32)]
)
def test_vtfmm_wide_is_round_product_then_add(fmt, sew, te):
    z = ZCore(te=te)
    tew_e = te if sew < 64 else te // 2
    tm, _, tn = z.mset(sew, 1, tm=tew_e, tk=1, tn=tew_e)
    lmul = 1 << z.vs.lmul_log2
    dt = get_format(fmt).dtype
    # Wide exponent spread so the two roundings differ from a fused multiply-add.
    A = (RNG.standard_normal((1, tm)) * 2.0 ** RNG.integers(-30, 30, (1, tm))).astype(
        dt
    )
    B = (RNG.standard_normal((1, tn)) * 2.0 ** RNG.integers(-30, 30, (1, tn))).astype(
        dt
    )
    C0 = (RNG.standard_normal((tm, tn)) * 1e-3).astype(dt)
    tile = 4 if sew == 32 else 6
    z.unit.set_tile(tile, sew, C0.view(f"<u{sew // 8}"), tm, tn)
    _operands(z, 8, A, 8, lmul, dt)
    _operands(z, 16, B, 8, lmul, dt)
    z.exec(encode("vtfmm.tvv", mtd=tile >> 1, vs2=8, vs1=16))
    got = z.tile(tile, sew, dt)[:tm, :tn]
    assert np.array_equal(
        got.view(f"<u{sew // 8}"), _golden_ieee(C0, A, B, dt).view(f"<u{sew // 8}")
    )
    st = z.unit.counters()
    assert (st["mma"], st["macs"]) == (1, tm * tn)
    assert st["vrf_a_read_bytes"] == tm * sew // 8
    assert st["vrf_b_read_bytes"] == tn * sew // 8


def test_vtfmm_tail_untouched_and_zero_dims_are_no_ops():
    z = ZCore(te=8)
    z.mset(32, 1, tm=3, tk=1, tn=5)
    C0 = np.full((8, 8), 2.0, np.float32)
    z.unit.set_tile(0, 32, C0.view(np.uint32), 8, 8)
    _operands(z, 8, [np.ones(8, np.float32)], 8, 1, np.float32)
    _operands(z, 16, [np.ones(8, np.float32)], 8, 1, np.float32)
    z.exec(encode("vtfmm.tvv", mtd=0, vs2=8, vs1=16))
    t = z.tile(0, 32, np.float32)
    assert (t[:3, :5] == 3).all()
    assert (t[3:] == 2).all()
    assert (t[:, 5:] == 2).all()
    z.x(T1, 0)
    z.exec(encode("msettk", rd=T2, rs1=T1))
    z.exec(encode("vtfmm.tvv", mtd=0, vs2=8, vs1=16))
    assert z.unit.counters()["mma"] == 1  # tk = 0: nothing computed


# -- an independent RVBNA reference (per element, Python integers) ----------------------------------------

_F32_MAX = Fraction(float(np.finfo(np.float32).max))


def _rto32(q):
    """A rational rounded to odd at fp32, with RVBNA's overflow to infinity."""
    if q == 0:
        return 0.0
    sign = -1 if q < 0 else 1
    a = abs(q)
    if a > _F32_MAX:
        return sign * math.inf
    e = math.floor(math.log2(a.numerator) - math.log2(a.denominator))
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    ulp = Fraction(2) ** (max(e, -126) - 23)
    t = math.floor(a / ulp)
    v = t * ulp if t * ulp == a else (t | 1) * ulp
    return sign * float(v)


def _fields(x, f):
    bits = int(np.asarray([x]).view(f"<u{f.bits // 8}")[0])
    m, e = f.mantissa_bits, f.exponent_bits
    exp, mant, sign = (
        (bits >> m) & ((1 << e) - 1),
        bits & ((1 << m) - 1),
        bits >> (m + e),
    )
    sig = mant if exp == 0 else mant | (1 << m)
    return sign, sig, (1 if exp == 0 else exp)


def ref_rvbna(a_row, b_col, fa, fb, n):
    """The pseudocode of rvbna.adoc for one element (``K <= n`` products), rationally finished."""
    fa, fb = get_format(fa), get_format(fb)
    vals = [float(x) * float(y) for x, y in zip(a_row, b_col)]
    if any(math.isnan(v) for v in vals) or (
        any(math.isinf(v) for v in vals) and math.isnan(sum(vals))
    ):
        return math.nan
    if any(math.isinf(v) for v in vals):
        return sum(vals)
    g = o = (n - 1).bit_length()
    q = 24
    prods = []
    for x, y in zip(a_row, b_col):
        sa, ga, ea = _fields(x, fa)
        sb, gb, eb = _fields(y, fb)
        zero = ga == 0 or gb == 0
        prods.append((sa ^ sb, ga * gb, 0 if zero else ea + eb))
    max_exp = max(p[2] for p in prods)
    pl, pr = fa.mantissa_bits + 1, fb.mantissa_bits + 1
    acc = 0
    for s, sig, ref in prods:
        shift = max_exp - ref
        v = sig << (q + 1 + g - (pl + pr))
        aligned = v >> shift
        if v & ((1 << shift) - 1):
            aligned |= 1  # round to odd: jam every discarded bit
        acc += -aligned if s else aligned
    if acc == 0:
        return 0.0
    bias = (1 << (fa.exponent_bits - 1)) - 1 + (1 << (fb.exponent_bits - 1)) - 1
    scale = max_exp - bias - (pl - 1) - (pr - 1) - (q + 1 + g - (pl + pr))
    assert abs(acc) < 1 << (q + 1 + g + o)
    return _rto32(Fraction(acc) * Fraction(2) ** scale)


def ref_zvt_narrow(C, A, B, fa, fb, n):
    out = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            s = ref_rvbna(A[:, i], B[:, j], fa, fb, n)
            c = float(C[i, j])
            if not (math.isfinite(s) and math.isfinite(c)):
                out[i, j] = np.float32(c) + np.float32(s)
            else:
                out[i, j] = _rto32(Fraction(c) + Fraction(s))
    return out.astype(np.float32)


def _narrow_values(fmt, shape, spread):
    f = get_format(fmt)
    x = RNG.standard_normal(shape) * 2.0 ** RNG.integers(-spread, spread + 1, shape)
    x[RNG.random(shape) < 0.1] = 0.0
    return f.round(x)


@pytest.mark.parametrize(
    "name,sew,alt,fa,fb",
    [
        ("vtfmm.tvv", 16, 0, "fp16", "fp16"),
        ("vtfmm.alt.tvv", 16, 1, "bf16", "bf16"),
        ("vtfmm.tvv", 8, 0, "fp8_e4m3", "fp8_e4m3"),
        ("vtfmm.tvv", 8, 1, "fp8_e4m3", "fp8_e5m2"),
        ("vtfmm.alt.tvv", 8, 0, "fp8_e5m2", "fp8_e4m3"),
        ("vtfmm.alt.tvv", 8, 1, "fp8_e5m2", "fp8_e5m2"),
    ],
)
@pytest.mark.parametrize("tk", [None, 1])
def test_vtfmm_narrow_is_rvbna_then_rto_add(name, sew, alt, fa, fb, tk):
    z = ZCore(te=8)
    tw = 32 // sew
    kmax = vme.KMAX[(sew, tw)]
    tm, tk_g, tn = z.mset(sew, tw, alt=alt, tm=8, tk=kmax if tk is None else tk, tn=8)
    lmul, stride = 1 << z.vs.lmul_log2, 8 // kmax
    spread = 12 if sew == 16 and fa == "fp16" else (40 if sew == 16 else 4)
    A = _narrow_values(fa, (tk_g, tm), spread)
    B = _narrow_values(fb, (tk_g, tn), spread)
    C0 = (RNG.standard_normal((tm, tn)) * 2.0 ** RNG.integers(-8, 8, (tm, tn))).astype(
        np.float32
    )
    z.unit.set_tile(8, 32, C0.view(np.uint32), tm, tn)
    _operands(z, 0, A, stride, lmul, get_format(fa).dtype)
    _operands(z, 16, B, stride, lmul, get_format(fb).dtype)
    z.exec(encode(name, mtd=4, vs2=0, vs1=16))
    got = z.tile(8, 32, np.float32)[:tm, :tn]
    want = ref_zvt_narrow(C0, A, B, fa, fb, kmax)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_rvbna_overflow_is_infinity_and_specials():
    from framework.matrix.vme_numerics import rto_fp32, rvbna_dot

    big = get_format("bf16").round(np.array([[2.0**127, 2.0**127]]))
    assert np.isposinf(rvbna_dot(big, big.T.copy(), "bf16", "bf16", 2))[0, 0]
    assert np.isposinf(rto_fp32(np.array([3.5e38]))[0])
    inf = get_format("bf16").round(np.array([[np.inf, 1.0]]))
    zero = get_format("bf16").round(np.array([[0.0], [1.0]]))
    assert np.isnan(rvbna_dot(inf, zero, "bf16", "bf16", 2))[0, 0]  # inf * 0
    one = get_format("bf16").round(np.array([[1.0], [-1.0]]))
    assert np.isposinf(rvbna_dot(inf, one, "bf16", "bf16", 2))[0, 0]
    # The cancelling sum is +0 (never -0).
    a = get_format("fp16").round(np.array([[1.0, -1.0]]))
    b = get_format("fp16").round(np.array([[1.0], [1.0]]))
    r = rvbna_dot(a, b, "fp16", "fp16", 2)
    assert r[0, 0] == 0
    assert not np.signbit(r[0, 0])


@pytest.mark.parametrize(
    "name,alt", [("vtmmu.tvv", 0), ("vtmmu.tvv", 1), ("vtmms.tvv", 0), ("vtmms.tvv", 1)]
)
def test_integer_multiplies_wrap_and_take_signedness(name, alt):
    z = ZCore(te=8)
    tm, tk, tn = z.mset(8, 4, alt=alt, tm=8, tk=4, tn=8)
    assert (1 << z.vs.lmul_log2, tk) == (1, 4)
    sa = name == "vtmms.tvv"
    da, db = np.dtype("<i1" if sa else "<u1"), np.dtype("<i1" if alt else "<u1")
    A = RNG.integers(-128 if sa else 0, 128 if sa else 256, (4, tm)).astype(da)
    B = RNG.integers(-128 if alt else 0, 128 if alt else 256, (4, tn)).astype(db)
    C0 = RNG.integers(2**31 - 2000, 2**31 - 1, (tm, tn)).astype(
        np.int32
    )  # near the top: wraps
    z.unit.set_tile(12, 32, C0.view(np.uint32), tm, tn)
    _operands(z, 8, A, 2, 1, da)
    _operands(z, 24, B, 2, 1, db)
    z.exec(encode(name, mtd=6, vs2=8, vs1=24))
    want = (C0.astype(np.int64) + A.astype(np.int64).T @ B.astype(np.int64)).astype(
        np.uint64
    ) % 2**32
    assert np.array_equal(z.tile(12, 32, np.uint32)[:tm, :tn], want.astype(np.uint32))


# -- legality -------------------------------------------------------------------------------------------


def test_multiply_legality():
    z = ZCore(te=8)
    z.mset(16, 2, alt=0, tm=8, tk=2, tn=8)  # KMAX 2: vs % 8 < 4, LMUL 1
    with pytest.raises(IllegalMatrixInstruction, match="vs % 8 < 8/KMAX"):
        z.exec(encode("vtfmm.tvv", mtd=0, vs2=4, vs1=16))
    with pytest.raises(IllegalMatrixInstruction, match="reserved type row"):
        z.exec(encode("vtfmm.alt.tvv", mtd=0, vs2=0, vs1=16))  # bf16 needs altfmt = 1
    with pytest.raises(IllegalMatrixInstruction, match="not a TEW-32 tile"):
        z.exec(encode("vtfmm.tvv", mtd=1, vs2=0, vs1=16))  # tile 2 at TEW 32
    z.vs.vstart = 1
    with pytest.raises(IllegalMatrixInstruction, match="vstart"):
        z.exec(encode("vtfmm.tvv", mtd=0, vs2=0, vs1=16))
    z.vs.vstart = 0
    z2 = ZCore(te=8, extensions=("Zvtbase", "Zvtfmm"))
    z2.exec(encode("msetmtypei", altfmt=0, vsew=3, mtwiden=1))
    assert z2.vs.vill  # no Zvt64e: TEW 64 unsupported


def test_extension_gating_and_frm():
    z = ZCore(te=8, extensions=("Zvtbase", "Zvt64e", "Zvtdmm"))
    z.mset(32, 1, tm=8, tk=1, tn=8)
    with pytest.raises(IllegalMatrixInstruction, match="needs Zvtfmm"):
        z.exec(encode("vtfmm.tvv", mtd=0, vs2=0, vs1=8))
    z = ZCore(te=8)
    z.mset(32, 1, tm=8, tk=1, tn=8)
    z.set_frm(1)
    with pytest.raises(VMEError, match="round-to-nearest-even"):
        z.exec(encode("vtfmm.tvv", mtd=0, vs2=0, vs1=8))


def test_rvv_after_vme_is_unchanged():
    """An ordinary RVV instruction still runs on a VME hart (and vsetvli resets mtype)."""
    z = ZCore()
    z.mset(32, 1, tm=8, tk=1, tn=8)
    z.vset(32, 1)
    z.v(1, np.uint32)[:] = 3
    z.v(2, np.uint32)[:] = 4
    from framework.pe.rv.isa.v.vtesting import OPIVV, opv

    z.exec(opv(OPIVV, 0x00, 3, 1, 2))
    assert (z.v(3, np.uint32) == 7).all()
    assert z.unit.mtype == 0
