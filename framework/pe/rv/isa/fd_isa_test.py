"""F / D on the RV64 core: every instruction, NaN-boxing, rounding modes, fflags.

Each test runs real instruction words through an ``RV64I`` (see
``isa64_testing``), so decode, the rm field / ``frm`` resolution and the fcsr
accrual are exercised with the arithmetic. Expected values are literal bit
patterns.
"""

import struct

import pytest

from framework.pe.rv.isa import fd_softfloat as sf
from framework.pe.rv.isa.fd_isa import RV64_D_ISA, RV64_F_ISA, read_f, write_f
from framework.pe.rv.isa.isa64_testing import (
    DATA,
    M64,
    csr,
    i_type,
    make_core,
    reg,
    run,
    s_type,
    set_reg,
)
from framework.pe.rv.isa.zicsr64_isa import RV64_ZICSR_ISA
from framework.pe.rv.rv32 import FCSR_INDEX

EXT = [RV64_F_ISA, RV64_D_ISA, RV64_ZICSR_ISA]
S, D = 0, 1
RNE, RTZ, RDN, RUP, RMM, DYN = 0, 1, 2, 3, 4, 7
NX, UF, OF, DZ, NV = sf.NX, sf.UF, sf.OF, sf.DZ, sf.NV

BOX = 0xFFFFFFFF00000000
QNAN_S = BOX | 0x7FC00000
QNAN_D = 0x7FF8000000000000
SNAN_S = BOX | 0x7F800001
SNAN_D = 0x7FF0000000000001
MAX_S, MAX_D = 0x7F7FFFFF, 0x7FEFFFFFFFFFFFFF
INF_S, INF_D = 0x7F800000, 0x7FF0000000000000


def s(x):
    """A boxed single from a Python float, or from raw 32 bits."""
    bits = x if isinstance(x, int) else struct.unpack("<I", struct.pack("<f", x))[0]
    return BOX | bits


def d(x):
    return x if isinstance(x, int) else struct.unpack("<Q", struct.pack("<d", x))[0]


def opfp(funct5, fmt, rm=RNE, rd=1, rs1=2, rs2=3):
    return (
        (funct5 << 27)
        | (fmt << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (rm << 12)
        | (rd << 7)
        | 0x53
    )


def fma_op(opcode, fmt, rm=RNE, rd=1, rs1=2, rs2=3, rs3=4):
    return (
        (rs3 << 27)
        | (fmt << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (rm << 12)
        | (rd << 7)
        | opcode
    )


def fs_type(funct3, imm, rs1, rs2):
    """STORE-FP (0x27): ``s_type`` with the opcode swapped."""
    return (s_type(funct3, imm, rs1, rs2) & ~0x7F) | 0x27


def fexec(instrs, f=None, x=None, frm=0, extensions=EXT):
    """Run with f-registers ``f`` ({idx: raw 64 bits}) and x-registers ``x``."""
    core, _ = make_core(extensions)
    for idx, value in (f or {}).items():
        write_f(core.register_file, idx, value)
    core.register_file[FCSR_INDEX].write((frm << 5).to_bytes(4, "little"))
    return run(instrs, x, core=core)


def result(core, idx=1):
    return read_f(core.register_file, idx)


def fflags(core):
    return core.register_file[FCSR_INDEX].read_uint() & 0x1F


def binop(funct5, fmt, a, b, rm=RNE, frm=0):
    core = fexec(opfp(funct5, fmt, rm), {2: a, 3: b}, frm=frm)
    return result(core), fflags(core)


ADD, SUB, MUL, DIV = 0x00, 0x01, 0x02, 0x03


# -- arithmetic ----------------------------------------------------------------


def test_basic_arithmetic():
    assert binop(ADD, S, s(1.0), s(2.0)) == (s(3.0), 0)
    assert binop(SUB, S, s(1.0), s(2.0)) == (s(-1.0), 0)
    assert binop(MUL, D, d(1.5), d(-4.0)) == (d(-6.0), 0)
    assert binop(DIV, D, d(1.0), d(3.0)) == (0x3FD5555555555555, NX)
    assert binop(DIV, S, s(1.0), s(3.0)) == (BOX | 0x3EAAAAAB, NX)
    assert binop(ADD, D, d(0.1), d(0.2)) == (0x3FD3333333333334, NX)


def test_sqrt():
    core = fexec(opfp(0x0B, D, rs2=0), {2: d(2.0)})
    assert (result(core), fflags(core)) == (0x3FF6A09E667F3BCD, NX)
    core = fexec(opfp(0x0B, S, rs2=0), {2: s(16.0)})
    assert (result(core), fflags(core)) == (s(4.0), 0)
    core = fexec(opfp(0x0B, S, rs2=0), {2: s(-1.0)})
    assert (result(core), fflags(core)) == (QNAN_S, NV)
    core = fexec(opfp(0x0B, D, rs2=0), {2: d(-0.0)})
    assert (result(core), fflags(core)) == (d(-0.0), 0)


def test_divide_by_zero_and_invalid():
    assert binop(DIV, S, s(1.0), s(0.0)) == (BOX | INF_S, DZ)
    assert binop(DIV, D, d(-1.0), d(0.0)) == (0xFFF0000000000000, DZ)
    assert binop(DIV, D, d(0.0), d(0.0)) == (QNAN_D, NV)
    assert binop(SUB, D, INF_D, INF_D) == (QNAN_D, NV)
    assert binop(MUL, S, BOX | INF_S, s(0.0)) == (QNAN_S, NV)


@pytest.mark.parametrize(
    "rm, positive, negative",
    [
        (RNE, INF_D, INF_D),
        (RMM, INF_D, INF_D),
        (RTZ, MAX_D, MAX_D),
        (RDN, MAX_D, INF_D),
        (RUP, INF_D, MAX_D),
    ],
)
def test_overflow_to_inf_or_max_finite(rm, positive, negative):
    neg = 1 << 63
    assert binop(MUL, D, MAX_D, d(2.0), rm) == (positive, OF | NX)
    assert binop(MUL, D, MAX_D, d(-2.0), rm) == (neg | negative, OF | NX)
    # Single precision, NaN-boxed, through the same path.
    single = INF_S if positive == INF_D else MAX_S
    assert binop(ADD, S, s(MAX_S), s(MAX_S), rm) == (BOX | single, OF | NX)


def test_ties_under_each_mode():
    one_plus_half_ulp = s(2.0**-24)  # 1 + this is exactly halfway
    odd = s(1.0 + 2.0**-23)
    for rm, want in (
        (RNE, s(1.0)),
        (RTZ, s(1.0)),
        (RDN, s(1.0)),
        (RUP, odd),
        (RMM, odd),
    ):
        assert binop(ADD, S, s(1.0), one_plus_half_ulp, rm) == (want, NX)
    # Ties to even goes up from an odd mantissa.
    assert binop(ADD, S, odd, one_plus_half_ulp, RNE) == (s(1.0 + 2.0**-22), NX)
    # Negative ties: RDN and RMM go away from zero, RUP and RTZ towards it.
    for rm, want in (
        (RNE, s(-1.0)),
        (RTZ, s(-1.0)),
        (RUP, s(-1.0)),
        (RDN, s(-1.0 - 2.0**-23)),
        (RMM, s(-1.0 - 2.0**-23)),
    ):
        assert binop(SUB, S, s(-1.0), one_plus_half_ulp, rm) == (want, NX)


def test_subnormals_and_underflow():
    min_normal = s(0x00800000)
    # Exact subnormal results raise nothing.
    assert binop(MUL, S, min_normal, s(0.5)) == (s(0x00400000), 0)
    assert binop(ADD, S, s(0x00000001), s(0x00000003)) == (s(0x00000004), 0)
    # 2**-126 * (1 - 2**-24) is a tie in the subnormal grid and rounds (to
    # even) up to the smallest normal, but with an unbounded exponent it is an
    # exact 24-bit value below 2**-126: tiny after rounding, so UF.
    assert binop(MUL, S, min_normal, s(0x3F7FFFFF)) == (min_normal, UF | NX)
    assert binop(MUL, S, min_normal, s(0x3F7FFFFF), RTZ) == (s(0x007FFFFF), UF | NX)
    # Smallest subnormal, halved: a tie to zero, and underflow.
    assert binop(MUL, S, s(0x00000001), s(0.5)) == (s(0.0), UF | NX)
    assert binop(MUL, S, s(0x00000001), s(0.5), RUP) == (s(0x00000001), UF | NX)
    assert binop(MUL, D, d(5e-324), d(-0.25)) == (d(-0.0), UF | NX)


def test_tininess_is_detected_after_rounding():
    # 2**-126 - 2**-152 (a double) rounds to 2**-126 at 24 bits even with an
    # unbounded exponent, so it is not tiny after rounding: NX without UF.
    v = d(2.0**-126 - 2.0**-152)
    core = fexec(opfp(0x08, S, RNE, rs2=1), {2: v})  # fcvt.s.d
    assert (result(core), fflags(core)) == (s(0x00800000), NX)
    core = fexec(opfp(0x08, S, RTZ, rs2=1), {2: v})
    assert (result(core), fflags(core)) == (s(0x007FFFFF), UF | NX)


def test_exact_zero_signs():
    assert binop(SUB, D, d(1.0), d(1.0)) == (d(0.0), 0)
    assert binop(SUB, D, d(1.0), d(1.0), RDN) == (d(-0.0), 0)
    assert binop(ADD, S, s(-0.0), s(-0.0)) == (s(-0.0), 0)
    assert binop(ADD, S, s(-0.0), s(0.0)) == (s(0.0), 0)
    assert binop(ADD, S, s(-0.0), s(0.0), RDN) == (s(-0.0), 0)


# -- fused multiply-add --------------------------------------------------------


def fma3(opcode, fmt, a, b, c, rm=RNE):
    core = fexec(fma_op(opcode, fmt, rm), {2: a, 3: b, 4: c})
    return result(core), fflags(core)


def test_fma_rounds_once():
    # (1 + 2**-30)(1 - 2**-30) - 1 = -2**-60 exactly; an unfused multiply
    # would round the product to 1 and return 0.
    a, b = d(1.0 + 2.0**-30), d(1.0 - 2.0**-30)
    assert fma3(0x47, D, a, b, d(1.0)) == (d(-(2.0**-60)), 0)  # fmsub
    assert fma3(0x43, D, a, b, d(-1.0)) == (d(-(2.0**-60)), 0)  # fmadd
    assert fma3(0x4B, D, a, b, d(1.0)) == (d(2.0**-60), 0)  # fnmsub: -(ab) + c
    assert fma3(0x4F, D, a, b, d(-1.0)) == (d(2.0**-60), 0)  # fnmadd: -(ab) - c
    a, b = s(1.0 + 2.0**-12), s(1.0 - 2.0**-12)
    assert fma3(0x47, S, a, b, s(1.0)) == (s(-(2.0**-24)), 0)
    assert fma3(0x43, S, s(2.0), s(3.0), s(1.0)) == (s(7.0), 0)
    assert fma3(0x43, S, s(1.0), s(1.0), s(2.0**-30)) == (s(1.0), NX)
    assert fma3(0x43, S, s(1.0), s(1.0), s(2.0**-30), RUP) == (s(1.0 + 2.0**-23), NX)


def test_fma_special_cases():
    # inf * 0 is invalid even when the addend is a quiet NaN.
    assert fma3(0x43, D, INF_D, d(0.0), QNAN_D) == (QNAN_D, NV)
    assert fma3(0x43, D, d(1.0), d(1.0), QNAN_D) == (QNAN_D, 0)
    assert fma3(0x43, S, s(1.0), s(1.0), SNAN_S) == (QNAN_S, NV)
    assert fma3(0x43, D, INF_D, d(1.0), d(float("-inf"))) == (QNAN_D, NV)  # inf - inf
    assert fma3(0x43, D, INF_D, d(1.0), INF_D) == (INF_D, 0)
    # Exact zeros: the product's sign against the addend's.
    assert fma3(0x43, D, d(2.0), d(3.0), d(-6.0)) == (d(0.0), 0)
    assert fma3(0x43, D, d(2.0), d(3.0), d(-6.0), RDN) == (d(-0.0), 0)
    assert fma3(0x43, D, d(-0.0), d(1.0), d(-0.0)) == (d(-0.0), 0)
    assert fma3(0x4F, D, d(0.0), d(1.0), d(0.0)) == (d(-0.0), 0)  # -(+0) - (+0)


# -- NaNs, NaN-boxing, sign injection ------------------------------------------


def test_nan_inputs_give_canonical_nan():
    assert binop(ADD, S, SNAN_S, s(1.0)) == (QNAN_S, NV)
    assert binop(ADD, S, BOX | 0x7FC12345, s(1.0)) == (
        QNAN_S,
        0,
    )  # quiet payload dropped
    assert binop(MUL, D, SNAN_D, d(1.0)) == (QNAN_D, NV)
    assert binop(DIV, D, d(1.0), 0x7FF8DEAD00000000) == (QNAN_D, 0)


def test_improperly_boxed_single_reads_as_canonical_nan():
    unboxed_one = 0x3F800000  # upper half zero
    assert binop(ADD, S, unboxed_one, s(1.0)) == (QNAN_S, 0)
    assert binop(ADD, S, 0x7FFFFFFF3F800000, s(1.0)) == (QNAN_S, 0)
    # fsgnjn (fneg) is not a transfer: it negates the canonical NaN.
    core = fexec(opfp(0x04, S, 1, rs2=2), {2: unboxed_one})
    assert result(core) == BOX | 0xFFC00000
    # A double write is not a boxed single.
    assert binop(ADD, S, d(1.0), s(1.0)) == (QNAN_S, 0)


def test_sign_injection():
    for funct3, a, b, want in (
        (0, s(1.5), s(-2.0), s(-1.5)),
        (1, s(1.5), s(-2.0), s(1.5)),
        (2, s(-1.5), s(-2.0), s(1.5)),
        (2, s(-1.5), s(2.0), s(-1.5)),
    ):
        core = fexec(opfp(0x04, S, funct3), {2: a, 3: b})
        assert (result(core), fflags(core)) == (want, 0)
    core = fexec(opfp(0x04, D, 2), {2: d(-3.0), 3: d(-1.0)})  # fabs
    assert result(core) == d(3.0)
    core = fexec(opfp(0x04, D, 0), {2: SNAN_D, 3: d(-1.0)})  # no NaN handling
    assert (result(core), fflags(core)) == (SNAN_D | (1 << 63), 0)


def test_min_max():
    def mm(fmt, a, b, is_max):
        core = fexec(opfp(0x05, fmt, int(is_max)), {2: a, 3: b})
        return result(core), fflags(core)

    assert mm(S, s(1.0), s(2.0), False) == (s(1.0), 0)
    assert mm(S, s(1.0), s(2.0), True) == (s(2.0), 0)
    assert mm(D, d(-0.0), d(0.0), False) == (d(-0.0), 0)
    assert mm(D, d(0.0), d(-0.0), False) == (d(-0.0), 0)
    assert mm(D, d(-0.0), d(0.0), True) == (d(0.0), 0)
    assert mm(S, QNAN_S, s(3.0), False) == (s(3.0), 0)
    assert mm(S, s(3.0), SNAN_S, True) == (s(3.0), NV)
    assert mm(D, QNAN_D, SNAN_D, True) == (QNAN_D, NV)
    assert mm(D, d(-1e300), d(-1e299), True) == (d(-1e299), 0)


def test_compare():
    def cmp(funct3, fmt, a, b):
        core = fexec(opfp(0x14, fmt, funct3), {2: a, 3: b})
        return reg(core, 1), fflags(core)

    FLE, FLT, FEQ = 0, 1, 2
    assert cmp(FEQ, S, s(1.0), s(1.0)) == (1, 0)
    assert cmp(FEQ, D, d(-0.0), d(0.0)) == (1, 0)
    assert cmp(FLT, D, d(-0.0), d(0.0)) == (0, 0)
    assert cmp(FLE, D, d(-0.0), d(0.0)) == (1, 0)
    assert cmp(FLT, S, s(-2.0), s(1.0)) == (1, 0)
    # feq is quiet: NV only for a signalling NaN; flt / fle signal on any NaN.
    assert cmp(FEQ, S, QNAN_S, s(1.0)) == (0, 0)
    assert cmp(FEQ, S, SNAN_S, s(1.0)) == (0, NV)
    assert cmp(FLT, D, QNAN_D, d(1.0)) == (0, NV)
    assert cmp(FLE, D, d(1.0), QNAN_D) == (0, NV)


def test_fclass():
    cases = [
        (d(float("-inf")), 0),
        (d(-1.0), 1),
        (d(-5e-324), 2),
        (d(-0.0), 3),
        (d(0.0), 4),
        (d(5e-324), 5),
        (d(1.0), 6),
        (INF_D, 7),
        (SNAN_D, 8),
        (QNAN_D, 9),
    ]
    for value, bit in cases:
        core = fexec(opfp(0x1C, D, 1, rs2=0), {2: value})
        assert reg(core, 1) == 1 << bit
    core = fexec(opfp(0x1C, S, 1, rs2=0), {2: s(0x00000001)})
    assert reg(core, 1) == 1 << 5
    core = fexec(opfp(0x1C, S, 1, rs2=0), {2: 0x3F800000})  # unboxed -> qNaN
    assert reg(core, 1) == 1 << 9


# -- conversions ---------------------------------------------------------------


def to_int(fmt, kind, value, rm=RNE):
    core = fexec(opfp(0x18, fmt, rm, rs2=kind), {2: value})
    return reg(core, 1), fflags(core)


W, WU, L, LU = 0, 1, 2, 3


def test_float_to_int_rounding():
    assert to_int(D, W, d(2.5)) == (2, NX)
    assert to_int(D, W, d(3.5)) == (4, NX)
    assert to_int(D, W, d(2.5), RMM) == (3, NX)
    assert to_int(D, W, d(-2.5), RDN) == (-3 & M64, NX)
    assert to_int(D, W, d(-2.5), RUP) == (-2 & M64, NX)
    assert to_int(D, W, d(-2.7), RTZ) == (-2 & M64, NX)
    assert to_int(S, L, s(-7.0)) == (-7 & M64, 0)
    assert to_int(D, LU, d(2.0**63)) == (1 << 63, 0)
    # A negative value that rounds to zero is in range for unsigned: NX only.
    assert to_int(D, WU, d(-0.4)) == (0, NX)


def test_float_to_int_saturates():
    neg_inf = d(float("-inf"))
    cases = [
        (D, W, d(3e9), 0x7FFFFFFF),
        (D, W, d(-3e9), 0xFFFFFFFF80000000),
        (D, W, QNAN_D, 0x7FFFFFFF),
        (D, W, neg_inf, 0xFFFFFFFF80000000),
        (S, WU, s(-1.0), 0),
        (S, WU, QNAN_S, M64),  # 2**32 - 1, sign-extended
        (D, WU, d(5e9), M64),
        (D, L, d(2.0**63), 0x7FFFFFFFFFFFFFFF),
        (D, L, neg_inf, 1 << 63),
        (S, L, SNAN_S, 0x7FFFFFFFFFFFFFFF),
        (D, LU, d(-1.0), 0),
        (D, LU, QNAN_D, M64),
        (D, LU, INF_D, M64),
    ]
    for fmt, kind, value, want in cases:
        assert to_int(fmt, kind, value) == (want, NV), (fmt, kind, hex(value))
    # The W forms sign-extend an unsigned result with bit 31 set.
    assert to_int(D, WU, d(3e9)) == (0xFFFFFFFFB2D05E00, 0)
    assert to_int(D, L, d(-(2.0**63))) == (1 << 63, 0)


def from_int(fmt, kind, value, rm=RNE):
    core = fexec(opfp(0x1A, fmt, rm, rs2=kind), x={2: value})
    return result(core), fflags(core)


def test_int_to_float():
    assert from_int(S, W, 2**24 + 1) == (s(2.0**24), NX)
    assert from_int(S, W, 2**24 + 1, RUP) == (s(2.0**24 + 2), NX)
    assert from_int(S, W, -3 & M64) == (s(-3.0), 0)
    # W reads only the low 32 bits, as a signed value.
    assert from_int(D, W, 0x12345678FFFFFFFF) == (d(-1.0), 0)
    assert from_int(D, WU, M64) == (d(4294967295.0), 0)
    assert from_int(D, L, M64) == (d(-1.0), 0)
    assert from_int(D, L, 2**53 + 1) == (d(2.0**53), NX)
    assert from_int(D, LU, M64) == (d(2.0**64), NX)
    assert from_int(D, LU, M64, RTZ) == (0x43EFFFFFFFFFFFFF, NX)
    assert from_int(S, LU, 0) == (s(0.0), 0)


def test_single_double_conversions():
    def cvt(dst, src_code, value, rm=RNE):
        core = fexec(opfp(0x08, dst, rm, rs2=src_code), {2: value})
        return result(core), fflags(core)

    assert cvt(S, 1, d(1.0 / 3.0)) == (s(0x3EAAAAAB), NX)
    assert cvt(S, 1, d(1.0 / 3.0), RTZ) == (s(0x3EAAAAAA), NX)
    assert cvt(S, 1, d(1e300)) == (BOX | INF_S, OF | NX)
    assert cvt(S, 1, d(1e300), RTZ) == (s(MAX_S), OF | NX)
    assert cvt(S, 1, d(1e-50)) == (s(0.0), UF | NX)
    assert cvt(S, 1, SNAN_D) == (QNAN_S, NV)
    assert cvt(S, 1, d(-1.5)) == (s(-1.5), 0)
    assert cvt(D, 0, s(0x00000001)) == (d(2.0**-149), 0)  # subnormal single, exact
    assert cvt(D, 0, s(-0.1)) == (
        d(float(struct.unpack("<f", struct.pack("<f", -0.1))[0])),
        0,
    )
    assert cvt(D, 0, SNAN_S) == (QNAN_D, NV)
    assert cvt(D, 0, 0x3F800000) == (QNAN_D, 0)  # unboxed


# -- moves, loads and stores ---------------------------------------------------


def test_moves():
    core = fexec(opfp(0x1C, S, 0, rs2=0), {2: s(-1.0)})  # fmv.x.w sign-extends
    assert reg(core, 1) == 0xFFFFFFFFBF800000
    core = fexec(opfp(0x1C, S, 0, rs2=0), {2: 0x123456783F800000})  # unchecked
    assert reg(core, 1) == 0x3F800000
    core = fexec(opfp(0x1E, S, 0, rs2=0), x={2: 0xDEADBEEF40490FDB})  # fmv.w.x boxes
    assert result(core) == BOX | 0x40490FDB
    core = fexec(opfp(0x1C, D, 0, rs2=0), {2: SNAN_D})  # fmv.x.d
    assert (reg(core, 1), fflags(core)) == (SNAN_D, 0)
    core = fexec(opfp(0x1E, D, 0, rs2=0), x={2: 0x400921FB54442D18})
    assert result(core) == 0x400921FB54442D18


def test_loads_and_stores():
    flw = i_type(0x07, 2, 8, rd=1, rs1=2)
    fld = i_type(0x07, 3, 16, rd=3, rs1=2)
    fsw = fs_type(2, 0x20, rs1=2, rs2=1)
    fsd = fs_type(3, 0x28, rs1=2, rs2=3)
    core, memory = make_core(EXT)
    memory.write(DATA + 8, (0x40490FDB).to_bytes(4, "little"))
    memory.write(DATA + 16, (0x400921FB54442D18).to_bytes(8, "little"))
    set_reg(core, 2, DATA)
    run([flw, fld, fsw, fsd], core=core)
    assert read_f(core.register_file, 1) == BOX | 0x40490FDB
    assert read_f(core.register_file, 3) == 0x400921FB54442D18
    assert memory.read(DATA + 0x20, 4) == (0x40490FDB).to_bytes(4, "little")
    assert memory.read(DATA + 0x28, 8) == (0x400921FB54442D18).to_bytes(8, "little")
    # fsw is a transfer: an unboxed register's low half is stored as is.
    core, memory = make_core(EXT)
    write_f(core.register_file, 1, 0x0000000012345678)
    set_reg(core, 2, DATA)
    run(fs_type(2, 0, rs1=2, rs2=1), core=core)
    assert memory.read(DATA, 4) == (0x12345678).to_bytes(4, "little")


# -- rounding mode resolution, flags, decode ------------------------------------


def test_dynamic_rounding_mode_reads_frm():
    third = d(1.0 / 3.0)
    assert binop(ADD, D, third, d(1.0), DYN, frm=RNE) == (0x3FF5555555555555, NX)
    assert binop(ADD, D, third, d(1.0), DYN, frm=RUP) == (0x3FF5555555555556, NX)
    assert binop(ADD, D, third, d(1.0), DYN, frm=RTZ) == (0x3FF5555555555555, NX)


@pytest.mark.parametrize("rm, frm", [(5, 0), (6, 0), (DYN, 5), (DYN, 6), (DYN, 7)])
def test_illegal_rounding_mode_is_an_illegal_instruction(rm, frm):
    with pytest.raises(Exception, match="Unknown instruction"):
        binop(ADD, D, d(1.0), d(1.0), rm, frm=frm)
    with pytest.raises(Exception, match="Unknown instruction"):
        fexec(opfp(0x18, S, rm, rs2=W), {2: s(1.0)}, frm=frm)


def test_rm_is_ignored_where_the_spec_has_none():
    # fsgnj / fmin / feq / fmv use funct3 as an opcode; there is no rm to check.
    core = fexec(opfp(0x05, D, 1), {2: d(1.0), 3: d(2.0)}, frm=7)
    assert result(core) == d(2.0)


def test_flags_accrue_and_are_visible_as_csrs():
    instrs = [
        opfp(DIV, D, RNE, rd=1, rs1=2, rs2=3),  # 1 / 0: DZ
        opfp(DIV, D, RNE, rd=4, rs1=2, rs2=5),  # 1 / 3: NX
        csr(2, 0x001, rd=6, rs1=0),  # csrr fflags
        csr(2, 0x003, rd=7, rs1=0),  # csrr fcsr
    ]
    core = fexec(instrs, {2: d(1.0), 3: d(0.0), 5: d(3.0)}, frm=RUP)
    assert reg(core, 6) == DZ | NX
    assert reg(core, 7) == (RUP << 5) | DZ | NX
    # csrw fflags, x0 clears them; a later op accrues afresh.
    run([csr(1, 0x001, rd=0, rs1=0), opfp(SUB, D, RNE, rd=8, rs1=2, rs2=2)], core=core)
    assert fflags(core) == 0


def test_csrw_frm_steers_dynamic_rounding():
    instrs = [
        csr(5, 0x002, rd=0, rs1=RTZ),  # csrwi frm, rtz
        opfp(ADD, D, DYN, rd=1, rs1=2, rs2=3),
    ]
    core = fexec(instrs, {2: d(1.0 / 3.0), 3: d(1.0)})
    assert result(core) == 0x3FF5555555555555


def test_f_only_core_does_not_execute_d():
    with pytest.raises(Exception, match="Unknown instruction"):
        fexec(opfp(ADD, D), {2: d(1.0), 3: d(1.0)}, extensions=[RV64_F_ISA])
    with pytest.raises(Exception, match="Unknown instruction"):
        fexec(i_type(0x07, 3, 0, rd=1, rs1=0), extensions=[RV64_F_ISA])  # fld
    core = fexec(opfp(ADD, S), {2: s(1.0), 3: s(1.0)}, extensions=[RV64_F_ISA])
    assert result(core) == s(2.0)


def test_comparison_into_x0_is_discarded():
    core = fexec(opfp(0x14, D, 2, rd=0), {2: d(1.0), 3: d(1.0)})
    assert reg(core, 0) == 0


def test_reserved_encodings_are_not_claimed():
    for instr in (
        opfp(0x0B, D, rs2=1),  # fsqrt with rs2 != 0
        opfp(0x04, S, 3),  # fsgnj funct3 3
        opfp(0x18, D, rs2=4),  # fcvt int kind 4
        opfp(0x08, D, rs2=2),  # fcvt.d.h: Zfh, not modelled
    ):
        with pytest.raises(Exception, match="Unknown instruction"):
            fexec(instr, {2: d(1.0), 3: d(1.0)})


def test_snoop_disassembles(capsys):
    from framework.pe.rv.isa.zfbfmin_isa import RV64_ZFBFMIN_ISA

    instrs = [
        opfp(ADD, D, DYN),
        opfp(0x0B, S, RTZ, rs2=0),
        opfp(0x04, D, 1),
        opfp(0x05, S, 0),
        opfp(0x14, D, 2),
        opfp(0x18, D, RNE, rs2=LU),
        opfp(0x1A, S, RNE, rs2=W),
        opfp(0x1C, D, 0, rs2=0),
        opfp(0x1C, S, 1, rs2=0),
        opfp(0x1E, S, 0, rs2=0),
        opfp(0x08, S, RNE, rs2=1),
        opfp(0x08, D, RNE, rs2=0),
        opfp(0x08, 2, RNE, rs2=8),
        opfp(0x08, 0, RNE, rs2=6),
        fma_op(0x4F, D, RMM),
        i_type(0x07, 3, 0, rd=1, rs1=2),
        fs_type(2, 0, rs1=2, rs2=1),
        i_type(0x07, 1, 0, rd=1, rs1=2),
    ]
    core, _ = make_core(EXT + [RV64_ZFBFMIN_ISA], snoop=True)
    set_reg(core, 2, DATA)
    run(instrs, core=core)
    out = capsys.readouterr().out
    for text in (
        "fadd.d f1, f2, f3",
        "fsqrt.s f1, f2, rtz",
        "fcvt.lu.d ra, f2",
        "fnmadd.d f1, f2, f3, f4, rmm",
    ):
        assert text in out
    for text in (
        "fcvt.s.d",
        "fcvt.d.s",
        "fcvt.bf16.s",
        "fcvt.s.bf16",
        "fld",
        "fsw",
        "flh",
        "fclass.s",
    ):
        assert text in out
