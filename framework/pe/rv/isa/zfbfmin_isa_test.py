"""Zfbfmin on the RV64 core: the BF16 conversions and the 16-bit transfers."""

import pytest

from framework.pe.rv.isa import fd_softfloat as sf
from framework.pe.rv.isa.fd_isa import RV64_D_ISA, RV64_F_ISA, read_f, write_f
from framework.pe.rv.isa.fd_isa_test import (
    QNAN_S,
    SNAN_S,
    fexec,
    fflags,
    fs_type,
    opfp,
    result,
    s,
)
from framework.pe.rv.isa.isa64_testing import DATA, i_type, make_core, reg, run, set_reg
from framework.pe.rv.isa.zfbfmin_isa import RV64_ZFBFMIN_ISA

EXT = [RV64_F_ISA, RV64_D_ISA, RV64_ZFBFMIN_ISA]
BOX16 = 0xFFFFFFFFFFFF0000
RNE, RTZ, RUP, RMM = 0, 1, 3, 4


def to_bf16(value, rm=RNE):
    core = fexec(opfp(0x08, 2, rm, rs2=8), {2: value}, extensions=EXT)
    return result(core), fflags(core)


def to_s(value):
    core = fexec(opfp(0x08, 0, RNE, rs2=6), {2: value}, extensions=EXT)
    return result(core), fflags(core)


def test_fcvt_bf16_s_rounds():
    assert to_bf16(s(1.0)) == (BOX16 | 0x3F80, 0)
    assert to_bf16(s(-2.5)) == (BOX16 | 0xC020, 0)
    # 1 + 2**-8 is halfway between bf16 1.0 and 1 + 2**-7: ties to even.
    assert to_bf16(s(1.0 + 2.0**-8)) == (BOX16 | 0x3F80, sf.NX)
    assert to_bf16(s(1.0 + 2.0**-8), RMM) == (BOX16 | 0x3F81, sf.NX)
    assert to_bf16(s(1.0 + 2.0**-8 + 2.0**-20)) == (BOX16 | 0x3F81, sf.NX)
    assert to_bf16(s(0x7F7FFFFF)) == (BOX16 | 0x7F80, sf.OF | sf.NX)
    # Truncated, fp32's largest value is bf16's largest: in range, so no OF.
    assert to_bf16(s(0x7F7FFFFF), RTZ) == (BOX16 | 0x7F7F, sf.NX)
    assert to_bf16(s(float("-inf")), RTZ) == (BOX16 | 0xFF80, 0)
    assert to_bf16(s(0x00000001)) == (BOX16 | 0x0000, sf.UF | sf.NX)
    assert to_bf16(s(0x00000001), RUP) == (BOX16 | 0x0001, sf.UF | sf.NX)
    assert to_bf16(SNAN_S) == (BOX16 | 0x7FC0, sf.NV)
    assert to_bf16(QNAN_S) == (BOX16 | 0x7FC0, 0)
    assert to_bf16(0x3F800000) == (BOX16 | 0x7FC0, 0)  # unboxed single


def test_fcvt_s_bf16_is_exact():
    assert to_s(BOX16 | 0x3F80) == (s(1.0), 0)
    assert to_s(BOX16 | 0x0001) == (s(0x00010000), 0)  # bf16 subnormal
    assert to_s(BOX16 | 0xFF80) == (s(float("-inf")), 0)
    assert to_s(BOX16 | 0x7F81) == (QNAN_S, sf.NV)  # signalling
    assert to_s(BOX16 | 0x7FC1) == (QNAN_S, 0)
    # A single-boxed value is not a boxed bf16: it reads as the canonical NaN.
    assert to_s(s(1.0)) == (QNAN_S, 0)


def test_transfers():
    core = fexec(
        opfp(0x1C, 2, 0, rs2=0), {2: BOX16 | 0xBF80}, extensions=EXT
    )  # fmv.x.h
    assert reg(core, 1) == 0xFFFFFFFFFFFFBF80
    core = fexec(opfp(0x1C, 2, 0, rs2=0), {2: 0x3F80}, extensions=EXT)  # unchecked
    assert reg(core, 1) == 0x3F80
    core = fexec(opfp(0x1E, 2, 0, rs2=0), x={2: 0x12344049}, extensions=EXT)  # fmv.h.x
    assert result(core) == BOX16 | 0x4049

    core, memory = make_core(EXT)
    memory.write(DATA + 2, (0x4049).to_bytes(2, "little"))
    set_reg(core, 2, DATA)
    write_f(core.register_file, 3, 0x1234)  # unboxed: fsh stores the low bits
    run([i_type(0x07, 1, 2, rd=1, rs1=2), fs_type(1, 6, rs1=2, rs2=3)], core=core)
    assert read_f(core.register_file, 1) == BOX16 | 0x4049
    assert memory.read(DATA + 6, 2) == (0x1234).to_bytes(2, "little")


def test_round_trip_through_bf16():
    instrs = [
        opfp(0x08, 2, RNE, rd=1, rs1=2, rs2=8),
        opfp(0x08, 0, RNE, rd=3, rs1=1, rs2=6),
    ]
    core = fexec(instrs, {2: s(3.140625)}, extensions=EXT)
    assert read_f(core.register_file, 3) == s(3.140625)


def test_needs_zfbfmin():
    with pytest.raises(Exception, match="Unknown instruction"):
        fexec(
            opfp(0x08, 2, RNE, rs2=8), {2: s(1.0)}, extensions=[RV64_F_ISA, RV64_D_ISA]
        )
