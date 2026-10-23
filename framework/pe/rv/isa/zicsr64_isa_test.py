"""RV64 Zicsr / Zicntr unit tests: 64-bit CSR values, the counters, no ``h``
halves, read-only refusals, and the side-effect suppression rules."""

import pytest

from framework.pe.rv.isa import zicsr_isa
from framework.pe.rv.isa.isa64_testing import (
    M64,
    PC,
    RD,
    RS1,
    csr,
    make_core,
    reg,
    run,
    set_reg,
)
from framework.pe.rv.isa.m64_isa import RV64_M_ISA
from framework.pe.rv.isa.zicsr64_isa import (
    RV64_ZICSR_ISA,
    misa_value,
)

RW, RS, RC, RWI, RSI, RCI = 1, 2, 3, 5, 6, 7
CYCLE, TIME, INSTRET = 0xC00, 0xC01, 0xC02
MCYCLE, MINSTRET, MSCRATCH = 0xB00, 0xB02, 0x340
NOP = 0x00000013


def _csrr(addr, rd=RD):
    return csr(RS, addr, rd=rd, rs1=0)


def test_cycle_and_instret_count():
    core = run([NOP, NOP, _csrr(CYCLE, rd=2), _csrr(INSTRET, rd=3)])
    assert reg(core, 2) == 3  # read on the third tick
    assert reg(core, 3) == 3  # three instructions retired before it
    assert core.csrs.read(CYCLE) == 4
    assert core.csrs.read(INSTRET) == 4


def test_time_follows_cycle_unless_a_source_is_bound():
    core = run([NOP, _csrr(TIME)])
    assert reg(core, RD) == 2
    core.csrs.time_source = lambda: 1 << 40
    assert core.csrs.read(TIME) == 1 << 40


def test_counters_are_64_bit_and_mcycle_is_writable():
    core = run(csr(RW, MCYCLE, rd=0), regs={RS1: (1 << 32) - 1})
    core = run(_csrr(CYCLE), core=core)
    # Written to 2**32-1 on the first tick, read one tick later.
    assert reg(core, RD) == 1 << 32


def test_minstret_writable():
    core, _ = make_core()
    core.csrs.write(MINSTRET, 100)
    run([NOP, _csrr(MINSTRET)], core=core)
    assert reg(core, RD) == 101


@pytest.mark.parametrize("addr", [0xC80, 0xC81, 0xC82, 0xB80, 0xB82])
def test_no_h_halves_on_rv64(addr):
    with pytest.raises(zicsr_isa.UnknownCSRError):
        run(_csrr(addr))


@pytest.mark.parametrize("addr", [CYCLE, TIME, INSTRET, 0xF14])
def test_write_to_read_only_raises(addr):
    with pytest.raises(zicsr_isa.UnknownCSRError, match="read-only"):
        run(csr(RW, addr, rd=0), regs={RS1: 1})


def test_csrrs_with_x0_on_read_only_is_a_plain_read():
    core = run(csr(RS, CYCLE, rs1=0))
    assert reg(core, RD) == 1
    core = run(csr(RCI, INSTRET, rs1=0))  # uimm 0: no write either
    assert reg(core, RD) == 0


def test_scratch_holds_64_bits():
    value = 0xDEADBEEF_CAFEF00D
    core = run([csr(RW, MSCRATCH, rd=0), _csrr(MSCRATCH, rd=2)], regs={RS1: value})
    assert reg(core, 2) == value


def test_csrrs_csrrc_and_immediates():
    core, _ = make_core()
    core.csrs.write(MSCRATCH, 1 << 63)
    run(
        [csr(RS, MSCRATCH, rd=2), csr(RCI, MSCRATCH, rd=3, rs1=0x1F)],
        regs={RS1: 0xF0},
        core=core,
    )
    assert reg(core, 2) == 1 << 63
    assert reg(core, 3) == (1 << 63) | 0xF0
    assert core.csrs.read(MSCRATCH) == (1 << 63) | 0xE0
    run(csr(RWI, MSCRATCH, rd=4, rs1=7), core=core)
    assert reg(core, 4) == (1 << 63) | 0xE0
    assert core.csrs.read(MSCRATCH) == 7
    run(csr(RSI, MSCRATCH, rd=0, rs1=8), core=core)
    assert core.csrs.read(MSCRATCH) == 15


def test_csrrw_with_rd_x0_does_not_read():
    core, _ = make_core()
    reads = []
    original = core.csrs.read
    core.csrs.read = lambda addr, pc=None: reads.append(addr) or original(addr, pc)
    run(csr(RW, MSCRATCH, rd=0), regs={RS1: 5}, core=core)
    assert reads == []
    assert original(MSCRATCH) == 5


def test_mhartid_and_misa():
    core, _ = make_core([RV64_M_ISA, RV64_ZICSR_ISA], core_id=3)
    run([_csrr(0xF14, rd=2), _csrr(0x301, rd=3)], core=core)
    assert reg(core, 2) == 3
    assert reg(core, 3) == misa_value("IM")
    assert misa_value("IM") == (2 << 62) | (1 << 8) | (1 << 12)


def test_misa_of_full_core_has_a():
    core = run(_csrr(0x301))
    assert reg(core, RD) == misa_value("AIM")


def test_unknown_csr_raises():
    with pytest.raises(zicsr_isa.UnknownCSRError):
        run(_csrr(0x7C0))


def test_fp_csrs_alias_fcsr():
    core, _ = make_core(fp_registers=True)
    core.csrs.write(0x002, 0x3)  # frm
    core.csrs.write(0x001, 0x1F)  # fflags
    assert core.csrs.read(0x003) == (0x3 << 5) | 0x1F
    assert core.register_file["fcsr"].read_uint() == 0x7F
    # f-registers are FLEN = 64.
    assert len(core.register_file["f0"].read()) == 8


def test_fp_csr_without_fp_registers_raises():
    with pytest.raises(zicsr_isa.CSRError):
        run(_csrr(0x003))


def test_reset_restarts_counters():
    core = run([NOP, NOP])
    core.reset()
    assert core.csrs.read(CYCLE) == 0
    assert core.csrs.read(INSTRET) == 0
    assert reg(core, "pc") == PC


def test_csr_sees_full_64_bit_source():
    core, _ = make_core()
    set_reg(core, RS1, M64)
    run(csr(RW, MSCRATCH, rd=0), core=core)
    assert core.csrs.read(MSCRATCH) == M64
