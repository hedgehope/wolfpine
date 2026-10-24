"""IME (Zvvm @ 6effbcae): decode, layout, every instruction against an exact reference, vtype/WARL, gating, RVV interplay.

The core is a real ``RV64I`` with F/D/Zicsr/RVV and the IME unit attached
(``VCore`` plus :func:`framework.matrix.ime_isa.attach`); every instruction is
executed by placing its word at the PC and clocking the core. FP results are
checked bit for bit against a reference built from Python rationals with its
own correct rounding (nearest of the three neighbouring format values), not
against the numerics module.
"""

import importlib.util
from fractions import Fraction

import numpy as np
import pytest

from framework.matrix import ime
from framework.matrix.formats import get_format
from framework.matrix.ime import IMESpec, Reduction, encode, permissible_lambdas
from framework.matrix.ime_isa import RV64_IME_ISA, IMEUnit, attach
from framework.matrix.unit import IllegalMatrixInstruction
from framework.pe.rv.isa.isa64_testing import csr
from framework.pe.rv.isa.v import isa as visa
from framework.pe.rv.isa.v.vtesting import DATA, VCore, vmem, vsetvl, vsetvli, vtype

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)

RNG = np.random.default_rng(7)
ALL = tuple(e for e in ime.EXTENSIONS)
T_LAMBDA, T_VL, T_VT, T_LD, T_BASE = 5, 6, 7, 11, 10


class ICore(VCore):
    """``VCore`` with the IME unit attached after RVV."""

    def __init__(self, vlen=256, extensions=ALL, lambdas=None, reductions=None):
        super().__init__(vlen=vlen)
        if lambdas is None:
            lambdas = {s: permissible_lambdas(vlen, s) for s in (8, 16, 32, 64)}
        self.spec = IMESpec(vlen, lambdas, tuple(extensions), reductions or {})
        self.unit = attach(self.core, IMEUnit(self.spec, self.vs))

    def vsetl(self, sew, lmul, lam, vl=None, altfmt=0, afa=0, afb=0, bs=0):
        """``vsetvl`` with the IME fields (AVL = VLMAX when ``vl`` is None); returns the new vl."""
        vt = (
            vtype(sew, lmul)
            | (ime.encode_lambda(lam) << ime.LAMBDA_SHIFT if lam else 0)
            | (altfmt << ime.ALTFMT_SHIFT)
            | (afa << ime.ALTFMT_A_SHIFT)
            | (afb << ime.ALTFMT_B_SHIFT)
            | (bs << ime.BS_SHIFT)
        )
        self.x(T_VT, vt)
        if vl is None:
            self.exec(vsetvl(T_LAMBDA, 0, T_VT))
        else:
            self.x(T_VL, vl)
            self.exec(vsetvl(T_LAMBDA, T_VL, T_VT))
        return self.x(T_LAMBDA)

    def put_tile(self, base, nregs, tile, width, dtype):
        span = self.vs.vreg_bytes(base, nregs).view(dtype)
        ime.write_tile(span, np.asarray(tile, dtype=dtype), nregs, width)

    def get_tile(self, base, nregs, rows, width, dtype):
        return ime.tile_view(
            self.vs.vreg_bytes(base, nregs).view(dtype), nregs, rows, width
        )


# -- an independent exact reference ----------------------------------------------------------------


def _ref_round(q, fmt, odd=False):
    """Rational ``q`` rounded to ``fmt``: RNE (ties to even) or to odd, by checking the neighbours."""
    f = get_format(fmt)
    dt = f.dtype
    ut = np.dtype(f"<u{dt.itemsize}")
    guess = np.array([float(q)]).astype(dt)
    bits = int(guess.view(ut)[0])
    cands = []
    for b in (bits - 1, bits, bits + 1):
        if 0 <= b < 1 << (8 * dt.itemsize):
            v = np.array([b], dtype=ut).view(dt).astype(np.float64)[0]
            if np.isfinite(v):
                cands.append((b, v))
    if odd:
        below = [
            (b, v)
            for b, v in cands
            if abs(Fraction(v)) <= abs(q) and (v == 0 or (v > 0) == (q > 0))
        ]
        b, v = max(below, key=lambda bv: abs(bv[1]))
        if Fraction(v) == q:
            return v
        return np.array([b | 1], dtype=ut).view(dt).astype(np.float64)[0]
    best = min(cands, key=lambda bv: (abs(Fraction(bv[1]) - q), bv[0] & 1))
    return best[1]


def ref_fp(C, A, B, fmt_c, g=1, rnd="frm"):
    """``C + A . B^T`` (B is N x K) by the draft's model, in rationals; returns float64."""
    M, K = A.shape
    out = np.zeros(C.shape)
    for i in range(M):
        for j in range(B.shape[0]):
            c = Fraction(float(C[i, j]))
            for q in range(0, K, g):
                S = sum(
                    (
                        Fraction(float(A[i, k])) * Fraction(float(B[j, k]))
                        for k in range(q, q + g)
                    ),
                    Fraction(0),
                )
                if rnd != "xct":
                    S = Fraction(_ref_round(S, fmt_c, odd=rnd == "rto"))
                c = Fraction(_ref_round(c + S, fmt_c))
            out[i, j] = float(c)
    return out


def _rand(fmt, shape, scale=1.0):
    return (RNG.standard_normal(shape) * scale).astype(get_format(fmt).dtype)


# -- decode ---------------------------------------------------------------------------------------


def test_encodings_are_the_drafts():
    """The wavedrom encodings: funct6 << 1 | vm in funct7, vs1 in bits 19:15, vs2 in 24:20."""
    assert ime.INSTRUCTIONS["vmmacc.vv"].funct7 == 0x71
    assert ime.INSTRUCTIONS["vwmmacc.vv"].funct7 == 0x73
    assert ime.INSTRUCTIONS["vqmmacc.vv"].funct7 == 0x75
    assert ime.INSTRUCTIONS["v8wmmacc.vv"].funct7 == 0x77
    assert ime.INSTRUCTIONS["vfmmacc.vv"].funct7 == 0x29
    assert ime.INSTRUCTIONS["vfwmmacc.vv"].funct7 == 0x2B
    assert ime.INSTRUCTIONS["vfqmmacc.vv"].funct7 == 0x2D
    assert ime.INSTRUCTIONS["vf8wmmacc.vv"].funct7 == 0x2F
    assert (
        ime.INSTRUCTIONS["vmtl.v"].funct7 == ime.INSTRUCTIONS["vmts.v"].funct7 == 0x09
    )
    assert (
        ime.INSTRUCTIONS["vmttl.v"].funct7 == ime.INSTRUCTIONS["vmtts.v"].funct7 == 0x0B
    )
    # .insn r 0x57, 1, 0x29, v8, v2, v4 assembles (clang 20) to 0x52411457 with rd = v8.
    word = (0x29 << 25) | (4 << 20) | (2 << 15) | (1 << 12) | (8 << 7) | 0x57
    assert word == encode("vfmmacc.vv", 8, 2, 4)
    d = ime.decode(word)
    assert (d.insn.name, d.vd, d.vs1, d.vs2, d.vm) == ("vfmmacc.vv", 8, 2, 4, 1)
    # vmtl.v v8, (a0), a1, L4: nf = 3 (lambda 4), mew 1, mop 00, width 111.
    word = encode("vmtl.v", 8, 10, 11, lam_imm=3)
    assert word >> 29 == 3
    assert (word >> 28) & 1
    assert (word >> 12) & 7 == 7
    assert word & 0x7F == 0x07
    d = ime.decode(word)
    assert (d.insn.name, d.vd, d.rs1, d.rs2, d.lam_imm, d.transposing) == (
        "vmtl.v",
        8,
        10,
        11,
        3,
        False,
    )
    assert ime.decode(encode("vmtts.v", 4, 10, 11, vm=0)).transposing


def test_rvv_declines_and_ime_claims_exactly_the_zvvm_space():
    for name in ime.INSTRUCTIONS:
        word = encode(name, 8, 2, 4)
        assert visa.decode(word) is None
        assert ime.in_space(word)
        assert ime.decode(word).insn.name == name
    # Not Zvvm: an RVV vfmacc, a unit-stride vle32, mew = 1 with mop = 10 / 11, a width-110 mew load.
    others = [
        (0x2C << 26) | (1 << 25) | (1 << 12) | 0x57,
        vmem(True, 32, 8, 10),
        vmem(True, 64, 8, 10, mop=2, mew=1),
        vmem(True, 64, 8, 10, mop=3, mew=1),
        vmem(True, 32, 8, 10, mew=1),
    ]
    for word in others:
        assert ime.decode(word) is None
        assert not ime.in_space(word)
    assert RV64_IME_ISA.OPCODES == {0x57, 0x07, 0x27}


def test_ime_follows_rvv_in_the_chain():
    c = ICore()
    names = [i.__name__ for i in c.core.isas]
    assert names.index("RV64_IME_ISA") > names.index("RV64_V_ISA")
    assert c.core.matrix is c.unit is c.rf.matrix
    assert RV64_IME_ISA.INSTR_CLASS == "matrix"


# -- geometry and layout ---------------------------------------------------------------------------


def test_permissible_lambdas_match_the_drafts_examples():
    # VLEN 512, SEW 32: lambda 1, 2, 4 -> M 16, 8, 4 and EMUL_C 16, 4, 1.
    assert permissible_lambdas(512, 32) == (1, 2, 4)
    assert [512 // (32 * lam) for lam in (1, 2, 4)] == [16, 8, 4]
    assert [512 // (32 * lam * lam) for lam in (1, 2, 4)] == [16, 4, 1]
    # VLEN 1024, SEW 32: lambda 2 and 4 (lambda 1 would need EMUL_C = 32).
    assert permissible_lambdas(1024, 32) == (2, 4)
    # L = VLEN / SEW a perfect square: {sqrt L, /2, /4}; twice one: {sqrt(L/2), /2}.
    assert permissible_lambdas(256, 16) == (1, 2, 4)  # L = 16
    assert permissible_lambdas(256, 8) == (2, 4)  # L = 32
    assert permissible_lambdas(256, 64) == (1, 2)  # L = 4
    assert (
        permissible_lambdas(64, 128) == ()
    )  # VLEN < SEW: outside the IME-legal domain


def test_k1_geometry():
    """SEW 32, lambda 2, LMUL 1, int8 (W = 4) at VLEN 256 is the SpacemiT K1's 4 x 8 x 4."""
    g = ime.gemm_geometry(256, 32, 1, 2, 8, 4)
    assert (g.m, g.k, g.n, g.emul_c) == (4, 8, 4, 2)


def test_tile_reg_idx_against_the_sail():
    # LMUL = 1: the tile is the register in order.
    assert [ime.tile_reg_idx(i, 1, 2, 8) for i in range(8)] == list(range(8))
    # LMUL = 2, lambda = 2, 8 elements per register: a row is 4 elements, 2 per register.
    got = [ime.tile_reg_idx(i, 2, 2, 8) for i in range(16)]
    assert got == [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
    for lmul, lam, epr in ((1, 1, 8), (2, 4, 16), (4, 2, 8), (8, 1, 4)):
        n = lmul * epr
        ref = [ime.tile_reg_idx(i, lmul, lam, epr) for i in range(n)]
        assert list(ime.tile_index_map(lmul, lam, epr, n)) == ref
        assert sorted(ref) == list(range(n))  # a permutation
        # tile_view is the same map: the element at flat index ref[i] is tile element i.
        span = np.arange(n)
        tile = ime.tile_view(span, lmul, epr // lam, lam)
        assert list(tile.reshape(-1)) == ref


def test_tile_view_round_trips_and_aliases():
    c = ICore()
    tile = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)  # lambda 2 at SEW 32: M = 4
    c.put_tile(8, 4, tile, 2, np.float32)
    assert np.array_equal(c.get_tile(8, 4, 4, 2, np.float32), tile)
    # Register r holds columns [2r, 2r + 2) of every row.
    assert np.array_equal(c.v(9, np.float32)[:4], [2, 3, 10, 11])


# -- tile loads and stores ---------------------------------------------------------------------------


def test_vmtl_loads_a_row_major_panel_into_tile_layout():
    c = ICore()
    mat = np.arange(8 * 20, dtype=np.float32).reshape(8, 20)
    c.mem_write(DATA, mat)
    vl = c.vsetl(32, 4, 1)  # lambda 1, LMUL 4: M = 8, linesize 4
    assert vl == 32
    c.x(T_BASE, DATA + 4 * 4)  # columns 4..7
    c.x(T_LD, 20)
    c.exec(encode("vmtl.v", 8, T_BASE, T_LD))
    assert np.array_equal(c.get_tile(8, 4, 8, 1, np.float32), mat[:, 4:8])
    assert c.unit.stats["mem_to_tile_bytes"] == 128
    # The transposing load of the same panel gives the transpose (B row-major -> B_tile).
    c.x(T_BASE, DATA)
    c.exec(encode("vmttl.v", 16, T_BASE, T_LD))
    assert np.array_equal(c.get_tile(16, 4, 8, 1, np.float32), mat[:4, :8].T)


def test_default_leading_dimensions_and_store_round_trip():
    c = ICore()
    vl = c.vsetl(32, 2, 2)  # lambda 2, LMUL 2: M = 4, linesize 4, VLMAX 16
    assert vl == 16
    dense = np.arange(16, dtype=np.uint32) + 100
    c.mem_write(DATA, dense)
    c.x(T_BASE, DATA)
    c.exec(encode("vmtl.v", 8, T_BASE, 0))  # LD = 0: lambda * LMUL, i.e. dense rows
    assert np.array_equal(c.get_tile(8, 2, 4, 2, np.uint32), dense.reshape(4, 4))
    c.x(T_BASE, DATA + 0x400)
    c.exec(encode("vmts.v", 8, T_BASE, 0))
    assert np.array_equal(c.mem_read(DATA + 0x400, 16, np.uint32), dense)
    # vmttl's default LD is M = VLEN / (SEW * lambda) = 4.
    c.x(T_BASE, DATA)
    c.exec(encode("vmttl.v", 12, T_BASE, 0))
    assert np.array_equal(c.get_tile(12, 2, 4, 2, np.uint32), dense.reshape(4, 4).T)
    c.x(T_BASE, DATA + 0x800)
    c.exec(encode("vmtts.v", 12, T_BASE, 0))
    assert np.array_equal(c.mem_read(DATA + 0x800, 16, np.uint32), dense)


def test_immediate_lambda_is_exact_and_leaves_vtype():
    c = ICore()
    c.vsetl(32, 1, 2)
    vt = c.vs.vtype
    mat = np.arange(64, dtype=np.uint32)
    c.mem_write(DATA, mat)
    c.x(T_BASE, DATA)
    c.exec(encode("vmtl.v", 8, T_BASE, 0, lam_imm=1))  # L1 overrides lambda 2
    assert c.vs.vtype == vt
    assert np.array_equal(c.get_tile(8, 1, 8, 1, np.uint32), mat[:8].reshape(8, 1))
    with pytest.raises(IllegalMatrixInstruction, match="immediate lambda 4"):
        c.exec(
            encode("vmtl.v", 8, T_BASE, 0, lam_imm=3)
        )  # lambda 4: not legal at SEW 32


def test_masked_c_load_store_and_vstart():
    c = ICore()
    c.vsetl(32, 2, 2)  # C at lambda 2: M = 4, EMUL_C 2; LMUL = EMUL_C for C transfers
    mat = np.arange(16, dtype=np.uint32).reshape(4, 4) + 1
    c.mem_write(DATA, mat)
    cols = np.arange(16) % 4 < 3  # the draft's column mask for N_tile = 3
    c.set_mask(cols)
    c.put_tile(8, 2, np.full((4, 4), 7, np.uint32), 2, np.uint32)
    c.x(T_BASE, DATA)
    c.x(T_LD, 4)
    c.exec(encode("vmtl.v", 8, T_BASE, T_LD, vm=0))
    want = np.where(cols.reshape(4, 4), mat, 7)
    assert np.array_equal(c.get_tile(8, 2, 4, 2, np.uint32), want)
    # A masked store writes only the active columns.
    c.mem_write(DATA + 0x100, np.zeros(16, np.uint32))
    c.x(T_BASE, DATA + 0x100)
    c.exec(encode("vmts.v", 8, T_BASE, T_LD, vm=0))
    assert np.array_equal(
        c.mem_read(DATA + 0x100, 16, np.uint32).reshape(4, 4),
        np.where(cols.reshape(4, 4), want, 0),
    )
    # vstart: the prestart elements are untouched, vstart ends 0.
    c.put_tile(8, 2, np.zeros((4, 4), np.uint32), 2, np.uint32)
    c.x(T_BASE, DATA)
    c.vs.vstart = 5
    c.exec(encode("vmtl.v", 8, T_BASE, T_LD))
    got = c.get_tile(8, 2, 4, 2, np.uint32).reshape(-1)
    assert np.array_equal(got[:5], np.zeros(5))
    assert np.array_equal(got[5:], mat.reshape(-1)[5:])
    assert c.vs.vstart == 0
    with pytest.raises(IllegalMatrixInstruction, match="masked load cannot write v0"):
        c.exec(encode("vmtl.v", 0, T_BASE, T_LD, vm=0))


# -- multiply-accumulates against the reference ------------------------------------------------------


CASES = [
    # (name, sew, lmul, lam, (afa, afb, altfmt), fmt_a, fmt_b, fmt_c)
    ("vfmmacc.vv", 32, 2, 1, (0, 0, 0), "fp32", "fp32", "fp32"),
    ("vfmmacc.vv", 32, 1, 2, (0, 0, 0), "fp32", "fp32", "fp32"),
    ("vfmmacc.vv", 64, 4, 1, (0, 0, 0), "fp64", "fp64", "fp64"),
    ("vfmmacc.vv", 16, 1, 2, (0, 0, 0), "fp16", "fp16", "fp16"),
    ("vfmmacc.vv", 16, 2, 4, (1, 1, 1), "bf16", "bf16", "bf16"),
    ("vfmmacc.vv", 16, 1, 2, (0, 1, 0), "fp16", "bf16", "fp16"),
    ("vfwmmacc.vv", 32, 2, 1, (1, 1, 0), "bf16", "bf16", "fp32"),
    ("vfwmmacc.vv", 32, 1, 2, (0, 0, 0), "fp16", "fp16", "fp32"),
    ("vfwmmacc.vv", 64, 2, 1, (0, 0, 0), "fp32", "fp32", "fp64"),
    ("vfwmmacc.vv", 16, 1, 2, (0, 1, 1), "fp8_e4m3", "fp8_e5m2", "bf16"),
    ("vfqmmacc.vv", 32, 1, 2, (0, 1, 0), "fp8_e4m3", "fp8_e5m2", "fp32"),
    ("vfqmmacc.vv", 64, 2, 2, (1, 1, 0), "bf16", "bf16", "fp64"),
    ("vf8wmmacc.vv", 64, 1, 1, (0, 0, 0), "fp8_e4m3", "fp8_e4m3", "fp64"),
]


def _load_operands(c, sew, lmul, lam, w, fmt_a, fmt_b, fmt_c, n=None, scale=1.0):
    m = c.vs.vlen // (sew * lam)
    k = lam * w * lmul
    emul = m // lam
    n = m if n is None else n
    A = _rand(fmt_a, (m, k), scale)
    B = _rand(fmt_b, (m, k), scale)
    C = _rand(fmt_c, (m, m), 4 * scale)
    c.put_tile(8, lmul, A, lam * w, A.dtype)
    c.put_tile(16, lmul, B, lam * w, B.dtype)
    assert emul <= 8
    c.put_tile(24, emul, C, lam, C.dtype)
    return A, B, C, m, k, emul


@pytest.mark.parametrize(
    "name,sew,lmul,lam,alt,fmt_a,fmt_b,fmt_c",
    CASES,
    ids=[f"{c[0]}:{c[5]}x{c[6]}->{c[7]}@L{c[3]}m{c[2]}" for c in CASES],
)
def test_fp_mma_matches_reference(name, sew, lmul, lam, alt, fmt_a, fmt_b, fmt_c):
    c = ICore()
    w = ime.INSTRUCTIONS[name].w
    afa, afb, af = alt
    c.vsetl(sew, lmul, lam, afa=afa, afb=afb, altfmt=af)
    scale = 0.5 if "fp8" in fmt_a else 1.0
    A, B, C, m, k, emul = _load_operands(
        c, sew, lmul, lam, w, fmt_a, fmt_b, fmt_c, scale=scale
    )
    c.exec(encode(name, 24, 8, 16))
    got = c.get_tile(24, emul, m, lam, get_format(fmt_c).dtype).astype(np.float64)
    want = ref_fp(C, A, B, fmt_c, g=w)  # a group is one sub-dot-product of W products
    assert np.array_equal(got, want)
    st = c.unit.stats
    assert (st["mma"], st["macs"]) == (1, m * m * k)
    assert st["vrf_a_read_bytes"] == st["vrf_b_read_bytes"] == m * lam * lmul * sew // 8
    assert st["vrf_c_read_bytes"] == st["vrf_c_write_bytes"] == m * m * sew // 8


@pytest.mark.parametrize(
    "fmt_a,fmt_c,w,g,rnd",
    [
        ("fp32", "fp32", 1, 2, "frm"),
        ("fp32", "fp32", 1, 2, "rto"),
        ("fp32", "fp32", 1, 1, "xct"),
        ("bf16", "fp32", 2, 2, "frm"),
        ("bf16", "fp32", 2, 2, "rto"),
        ("bf16", "fp32", 2, 2, "xct"),
        ("fp64", "fp64", 1, 2, "frm"),
    ],
)
def test_disclosed_grouping_and_rounding(fmt_a, fmt_c, w, g, rnd):
    sew = get_format(fmt_c).bits
    red = {(sew, w, 2): Reduction(g, 0, rnd)}
    c = ICore(reductions=red)
    alt = 1 if fmt_a == "bf16" else 0
    c.vsetl(sew, 2, 2, afa=alt, afb=alt)
    name = "vfmmacc.vv" if w == 1 else "vfwmmacc.vv"
    # Spread magnitudes so the roundings differ between the modes.
    A, B, C, m, k, emul = _load_operands(
        c, sew, 2, 2, w, fmt_a, fmt_a, fmt_c, scale=1.0
    )
    A = (A.astype(np.float64) * 2.0 ** RNG.integers(-12, 12, A.shape)).astype(A.dtype)
    c.put_tile(8, 2, A, 2 * w, A.dtype)
    c.exec(encode(name, 24, 8, 16))
    got = c.get_tile(24, emul, m, 2, get_format(fmt_c).dtype).astype(np.float64)
    assert np.array_equal(got, ref_fp(C, A, B, fmt_c, g=g * w, rnd=rnd))


def test_partial_n_leaves_tail_columns():
    c = ICore()
    vl = c.vsetl(32, 2, 1, vl=2 * 5)  # N = VL / (lambda * LMUL) = 5 of M = 8
    assert vl == 10
    A, B, C, m, k, emul = _load_operands(c, 32, 2, 1, 1, "fp32", "fp32", "fp32")
    c.exec(encode("vfmmacc.vv", 24, 8, 16))
    got = c.get_tile(24, emul, m, 1, np.float32).astype(np.float64)
    assert np.array_equal(got[:, :5], ref_fp(C[:, :5], A, B[:5], "fp32"))
    assert np.array_equal(got[:, 5:], C[:, 5:])
    assert c.unit.stats["macs"] == 8 * 5 * 2


def test_specials_and_canonical_nan():
    c = ICore()
    c.vsetl(32, 1, 2)
    m, k = 4, 2
    A = np.ones((m, k), np.float32)
    B = np.ones((m, k), np.float32)
    A[0, 0] = np.inf
    B[0, 0] = 0.0  # inf * 0: NaN in C[0, 0]
    A[1, 1] = np.inf  # C[1, :] = inf
    C = np.zeros((m, m), np.float32)
    C[2, 2] = -np.inf
    C[3, 3] = np.float32(np.nan)
    c.put_tile(8, 1, A, 2, np.float32)
    c.put_tile(16, 1, B, 2, np.float32)
    c.put_tile(24, 2, C, 2, np.float32)
    c.exec(encode("vfmmacc.vv", 24, 8, 16))
    got = c.get_tile(24, 2, m, 2, np.float32)
    bits = got.view(np.uint32)
    assert bits[0, 0] == 0x7FC00000
    assert bits[3, 3] == 0x7FC00000
    assert np.isinf(got[1, 1:]).all()
    assert got[2, 2] == -np.inf
    assert got[2, 0] == 1.0


INT_CASES = [
    # (name, sew, lmul, lam, afa, afb)
    ("vmmacc.vv", 32, 2, 1, 0, 0),
    ("vmmacc.vv", 8, 1, 2, 1, 0),
    ("vmmacc.vv", 16, 1, 4, 0, 1),
    ("vwmmacc.vv", 32, 1, 2, 0, 0),
    ("vqmmacc.vv", 32, 1, 2, 0, 1),
    ("v8wmmacc.vv", 64, 1, 1, 1, 1),
]


@pytest.mark.parametrize(
    "name,sew,lmul,lam,afa,afb", INT_CASES, ids=[f"{c[0]}@e{c[1]}" for c in INT_CASES]
)
def test_int_mma_wraps_like_python_ints(name, sew, lmul, lam, afa, afb):
    exts = (
        "Zvvi8mm",
        "Zvvi16mm",
        "Zvvi32mm",
        "Zvvi16i32mm",
        "Zvvi8i32mm",
        "Zvvi8i64mm",
        "Zvvmtls",
    )
    c = ICore(extensions=exts)
    w = ime.INSTRUCTIONS[name].w
    c.vsetl(sew, lmul, lam, afa=afa, afb=afb)
    m = c.vs.vlen // (sew * lam)
    k, emul, ib = lam * w * lmul, m // lam, sew // w
    dt_a = np.dtype(f"<{'u' if afa else 'i'}{ib // 8}")
    dt_b = np.dtype(f"<{'u' if afb else 'i'}{ib // 8}")
    lo = lambda dt: np.iinfo(dt).min  # noqa: E731
    A = RNG.integers(lo(dt_a), np.iinfo(dt_a).max, (m, k), endpoint=True).astype(dt_a)
    B = RNG.integers(lo(dt_b), np.iinfo(dt_b).max, (m, k), endpoint=True).astype(dt_b)
    dt_c = np.dtype(f"<i{sew // 8}")
    C = RNG.integers(
        np.iinfo(dt_c).min, np.iinfo(dt_c).max, (m, m), endpoint=True
    ).astype(dt_c)
    c.put_tile(8, lmul, A, lam * w, dt_a)
    c.put_tile(16, lmul, B, lam * w, dt_b)
    c.put_tile(24, emul, C, lam, dt_c)
    c.exec(encode(name, 24, 8, 16))
    got = c.get_tile(24, emul, m, lam, dt_c)
    for i in range(m):
        for j in range(m):
            v = int(C[i, j]) + sum(int(A[i, q]) * int(B[j, q]) for q in range(k))
            v &= (1 << sew) - 1
            v -= (v >> (sew - 1)) << sew
            assert int(got[i, j]) == v


# -- vtype and lambda --------------------------------------------------------------------------------


def test_warl_lambda_rules():
    sup = (1, 2)
    assert ime.warl_lambda(2, 0, sup) == 2  # supported request: kept
    assert (
        ime.warl_lambda(3, 0, sup) == 2
    )  # lambda 4 asked: the largest supported <= it
    assert ime.warl_lambda(1, 0, (2, 4)) == 2  # nothing <= 1: the smallest supported
    assert ime.warl_lambda(0, 1, sup) == 1  # 000: preserve a supported current value
    assert (
        ime.warl_lambda(0, 3, sup) == 2
    )  # ... else initialise to the largest supported
    assert ime.warl_lambda(0, 0, sup) == 2
    assert ime.warl_lambda(2, 2, ()) == 0  # outside the IME-legal domain: 000


def test_vsetvl_writes_and_vsetvli_preserves_the_ime_fields():
    c = ICore(lambdas={8: (2, 4), 16: (1, 2, 4), 32: (1, 2), 64: (1,)})
    assert ime.lambda_of(c.vs.vtype) is None  # reset: vill, lambda 000
    c.vsetl(32, 1, 1, afa=1, afb=0, bs=1)
    vt = c.vs.vtype
    assert ime.lambda_of(vt) == 1
    assert (vt >> ime.ALTFMT_A_SHIFT) & 1
    assert (vt >> ime.BS_SHIFT) & 1
    c.exec(csr(2, 0xC21, rd=5, rs1=0))  # csrr t0, vtype: the discovery read-back
    assert c.x(5) == vt
    # vsetvli to SEW 16: lambda 1 still supported there, kept; bs / altfmt_A kept.
    c.vset(16, 2)
    vt = c.vs.vtype
    assert ime.lambda_of(vt) == 1
    assert (vt >> ime.ALTFMT_A_SHIFT) & 1
    assert (vt >> ime.BS_SHIFT) & 1
    # vsetvli to SEW 64, where only lambda 1 is supported (kept); then to SEW 8 (re-initialised to 4).
    c.vset(64, 1)
    assert ime.lambda_of(c.vs.vtype) == 1
    c.vset(8, 1)
    assert ime.lambda_of(c.vs.vtype) == 4
    # vsetvl asking lambda 8 at SEW 32: clamped to 2; asking 000 keeps a supported current value.
    c.vsetl(32, 1, 8)
    assert ime.lambda_of(c.vs.vtype) == 2
    c.vsetl(32, 2, None)
    assert ime.lambda_of(c.vs.vtype) == 2
    # vsetvl clears bs / altfmt_A that it does not supply.
    assert not (c.vs.vtype >> ime.ALTFMT_A_SHIFT) & 1
    # A vill configuration still leaves lambda 000.
    c.x(T_VT, (1 << 62) | 0x4)  # vlmul 4 is reserved
    c.exec(vsetvl(5, 0, T_VT))
    assert c.vs.vill
    assert ime.lambda_of(c.vs.vtype) is None


def test_bits_above_rvv_are_reserved_without_ime():
    c = VCore()
    c.x(T_VT, vtype(32, 1) | (2 << ime.LAMBDA_SHIFT))
    c.exec(vsetvl(5, 0, T_VT))
    assert c.vs.vill


@pytest.mark.parametrize(
    "setup,word,match",
    [
        # Reset leaves vill (and lambda 000); a vset* in the IME-legal domain never selects 000.
        (lambda c: None, None, "vill"),
        (lambda c: c.vsetl(32, 1, 2, vl=5), None, "multiple of lambda"),
        (
            lambda c: c.vs.__setattr__("vstart", 1) or c.vsetl(32, 1, 2),
            "vstart",
            "vstart",
        ),
        (lambda c: c.vsetl(32, 1, 2), encode("vfmmacc.vv", 25, 8, 16), "aligned"),
        (lambda c: c.vsetl(32, 2, 2), encode("vfmmacc.vv", 24, 9, 16), "aligned"),
        (lambda c: c.vsetl(32, 1, 2), encode("vfmmacc.vv", 24, 8, 16, vm=0), "vm = 0"),
        (lambda c: c.vsetl(32, 1, 2, altfmt=1), None, "reserved"),
        (lambda c: c.vsetl(16, 1, 2, afa=0, afb=0, altfmt=1), None, "reserved"),
    ],
)
def test_illegal_configurations(setup, word, match):
    c = ICore()
    setup(c)
    if word == "vstart":
        c.vs.vstart = 1
        word = None
    if word is None:
        word = encode("vfmmacc.vv", 24, 8, 16)
    with pytest.raises(IllegalMatrixInstruction, match=match):
        c.exec(word)


def test_fractional_lmul_is_illegal():
    c = ICore()
    c.vsetl(32, 1, 2)
    c.x(T_VT, vtype(32, 0.5) | (2 << ime.LAMBDA_SHIFT))
    c.exec(vsetvl(5, 0, T_VT))
    with pytest.raises(IllegalMatrixInstruction, match="fractional"):
        c.exec(encode("vfmmacc.vv", 24, 8, 16))


# -- format and feature gating ---------------------------------------------------------------------------


def test_unlisted_extensions_are_illegal():
    c = ICore(extensions=("Zvvfp32mm", "Zvvbf16fp32mm", "Zvvmtls"))
    c.vsetl(32, 1, 2, afa=0, afb=0)  # fp16 x fp16 -> fp32 needs Zvvfp16fp32mm
    with pytest.raises(IllegalMatrixInstruction, match="Zvvfp16fp32mm"):
        c.exec(encode("vfwmmacc.vv", 24, 8, 16))
    c.vsetl(32, 1, 2, afa=0, afb=1)  # mixed fp16 x bf16 needs both
    with pytest.raises(IllegalMatrixInstruction, match="Zvvfp16fp32mm"):
        c.exec(encode("vfwmmacc.vv", 24, 8, 16))
    c.vsetl(32, 1, 2, afa=1, afb=1)
    c.exec(encode("vfwmmacc.vv", 24, 8, 16))  # bf16 x bf16 -> fp32: listed
    c.vsetl(64, 1, 1)
    with pytest.raises(IllegalMatrixInstruction, match="Zvvfp64mm"):
        c.exec(encode("vfmmacc.vv", 24, 8, 16))
    c.vsetl(32, 1, 2)
    with pytest.raises(IllegalMatrixInstruction, match="Zvvi32mm"):
        c.exec(encode("vmmacc.vv", 24, 8, 16))
    c.x(T_BASE, DATA)
    with pytest.raises(IllegalMatrixInstruction, match="Zvvmttls"):
        c.exec(encode("vmttl.v", 8, T_BASE, 0))


def test_unsupported_lambda_in_vtype_is_illegal():
    # The design supports only lambda 2 at SEW 32: vsetvl asking lambda 1 is clamped to 2 (WARL).
    c = ICore(lambdas={8: (4,), 16: (4,), 32: (2,), 64: (2,)})
    c.vsetl(32, 1, 1)
    assert ime.lambda_of(c.vs.vtype) == 2
    c.x(T_BASE, DATA)
    with pytest.raises(IllegalMatrixInstruction, match="immediate lambda 1"):
        c.exec(encode("vmtl.v", 8, T_BASE, 0, lam_imm=1))


# -- interplay with RVV on the same registers ----------------------------------------------------------------


def test_rvv_and_ime_share_the_register_file():
    """RVV loads build A and B, the mma accumulates, RVV reads C back and keeps working on it."""
    c = ICore()
    c.vsetl(32, 1, 2)  # M = 4, K = 2, EMUL_C = 2, VLMAX 8
    A = _rand("fp32", (4, 2))
    B = _rand("fp32", (4, 2))
    c.mem_write(DATA, A)  # row-major A is exactly the LMUL = 1 tile layout
    c.mem_write(DATA + 0x100, B)
    c.x(T_BASE, DATA)
    c.exec(vmem(True, 32, 8, T_BASE))  # vle32.v v8, (a0)
    c.x(T_BASE, DATA + 0x100)
    c.exec(vmem(True, 32, 16, T_BASE))
    # Zero C with RVV (vmv.v.i at LMUL 2 over the C group), keeping lambda.
    c.vset(32, 2)
    c.exec((0x17 << 26) | (1 << 25) | (3 << 12) | (24 << 7) | 0x57)  # vmv.v.i v24, 0
    c.vset(32, 1)
    assert ime.lambda_of(c.vs.vtype) == 2
    c.exec(encode("vfmmacc.vv", 24, 8, 16))
    want = ref_fp(np.zeros((4, 4)), A, B, "fp32")
    # Read C with an RVV store at LMUL 2 = EMUL_C: lambda 2 rows of 4 are contiguous per register pair.
    c.vset(32, 2)
    c.x(T_BASE, DATA + 0x200)
    c.exec(vmem(False, 32, 24, T_BASE))
    flat = c.mem_read(DATA + 0x200, 16, np.float32)
    tile = ime.tile_view(flat, 2, 4, 2)
    assert np.array_equal(tile.astype(np.float64), want)
    # An RVV op on the C group (vfadd.vv v24, v24, v24) doubles it.
    c.exec(
        (0x00 << 26)
        | (1 << 25)
        | (24 << 20)
        | (24 << 15)
        | (1 << 12)
        | (24 << 7)
        | 0x57
    )
    assert np.array_equal(
        c.get_tile(24, 2, 4, 2, np.float32), (2 * want).astype(np.float32)
    )


def test_vsetvli_from_c_code_keeps_lambda_through_rvv_work():
    """What compiled code does: vsetvl once with lambda, then vsetvli for ordinary RVV, then IME again."""
    c = ICore()
    c.vsetl(32, 1, 1)
    c.exec(vsetvli(5, 0, vtype(16, 1)))  # SEW 16: lambda 1 still supported there
    c.exec(vsetvli(5, 0, vtype(32, 4)))
    assert ime.lambda_of(c.vs.vtype) == 1
    A, B, C, m, k, emul = _load_operands(c, 32, 4, 1, 1, "fp32", "fp32", "fp32")
    c.exec(encode("vfmmacc.vv", 24, 8, 16))
    assert np.array_equal(
        c.get_tile(24, emul, m, 1, np.float32).astype(np.float64),
        ref_fp(C, A, B, "fp32"),
    )
