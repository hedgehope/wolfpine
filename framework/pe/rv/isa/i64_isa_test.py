"""RV64I unit tests: the places where 64-bit XLEN changes the answer.

Sign extension of ``lui`` / ``auipc`` / ``lw`` / the W-ops, the 6-bit shamt and
funct6 encoding of the immediate shifts, the new ``ld`` / ``sd`` / ``lwu``, and
64-bit comparisons that a 32-bit implementation would get backwards.
"""

import pytest

from framework.pe.rv.isa import zicsr_isa
from framework.pe.rv.isa.isa64_testing import (
    DATA,
    M64,
    PC,
    RD,
    RS1,
    RS2,
    b_type,
    csr,
    i_type,
    j_type,
    r_type,
    reg,
    run,
    run1,
    s_type,
    u_type,
)

NEG32 = 0xFFFFFFFF80000000  # sext32(0x80000000)


def _op(funct3, funct7=0):
    return r_type(0x33, funct3, funct7)


def _opw(funct3, funct7=0):
    return r_type(0x3B, funct3, funct7)


def _opimm(funct3, imm):
    return i_type(0x13, funct3, imm)


def _shift_imm(funct3, funct6, shamt):
    return i_type(0x13, funct3, (funct6 << 6) | shamt)


def _shift_imm_w(funct3, funct7, shamt):
    return i_type(0x1B, funct3, (funct7 << 5) | shamt)


# -- upper immediates ------------------------------------------------------


def test_lui_sign_extends_to_64():
    assert reg(run(u_type(0x37, 0x80000)), RD) == NEG32
    assert reg(run(u_type(0x37, 0x7FFFF)), RD) == 0x7FFFF000


def test_auipc_adds_sign_extended_immediate():
    assert reg(run(u_type(0x17, 0x80000)), RD) == (PC + NEG32) & M64
    assert reg(run(u_type(0x17, 0x1)), RD) == PC + 0x1000


# -- OP-IMM ----------------------------------------------------------------


def test_addi_wraps_at_64_not_32():
    assert run1(_opimm(0, 1), a=0xFFFFFFFF) == 0x100000000
    assert run1(_opimm(0, 1), a=M64) == 0
    assert run1(_opimm(0, -1), a=0) == M64


def test_slti_sltiu_compare_64_bit():
    # Negative only in 64 bits: 0x0000_0000_8000_0000 is positive.
    assert run1(_opimm(2, 0), a=0x80000000) == 0
    assert run1(_opimm(2, 0), a=NEG32) == 1
    # sltiu sign-extends the immediate to 64 bits, then compares unsigned.
    assert run1(_opimm(3, -1), a=0xFFFFFFFF) == 1
    assert run1(_opimm(3, -1), a=M64) == 0


def test_xori_ori_andi_with_negative_immediate():
    assert run1(_opimm(4, -1), a=0x1234) == 0x1234 ^ M64  # not
    assert run1(_opimm(6, -16), a=0x5) == (M64 & ~0xF) | 0x5
    assert run1(_opimm(7, -1), a=0xDEADBEEF00000000) == 0xDEADBEEF00000000


def test_slli_srli_srai_take_a_6_bit_shamt():
    assert run1(_shift_imm(1, 0x00, 63), a=1) == 1 << 63
    assert run1(_shift_imm(1, 0x00, 32), a=0xFFFFFFFF) == 0xFFFFFFFF00000000
    assert run1(_shift_imm(5, 0x00, 40), a=1 << 63) == 1 << 23
    assert run1(_shift_imm(5, 0x10, 63), a=1 << 63) == M64
    assert run1(_shift_imm(5, 0x10, 32), a=0x8000000000000000) == 0xFFFFFFFF80000000
    # srai of a value negative only in 32 bits shifts zeros in.
    assert run1(_shift_imm(5, 0x10, 4), a=0x80000000) == 0x08000000


@pytest.mark.parametrize("funct3,funct6", [(1, 0x18), (5, 0x18), (5, 0x1A), (1, 0x01)])
def test_shift_imm_other_funct6_is_declined(funct3, funct6):
    # rori / rev8 / ... belong to Zbb, and a core without it must not run them
    # as shifts.
    with pytest.raises(Exception, match="Unknown instruction"):
        run(_shift_imm(funct3, funct6, 3), extensions=[])


# -- OP --------------------------------------------------------------------


def test_add_sub_64_bit():
    assert run1(_op(0), a=0xFFFFFFFF, b=1) == 0x100000000
    assert run1(_op(0, 0x20), a=0, b=1) == M64
    assert run1(_op(0), a=M64, b=2) == 1


def test_register_shifts_use_six_bits():
    assert run1(_op(1), a=1, b=63) == 1 << 63
    assert run1(_op(1), a=1, b=64 + 1) == 2
    assert run1(_op(5), a=1 << 63, b=63) == 1
    assert run1(_op(5, 0x20), a=1 << 63, b=60) == 0xFFFFFFFFFFFFFFF8
    assert run1(_op(5, 0x20), a=0x80000000, b=31) == 1


def test_slt_sltu_64_bit():
    assert run1(_op(2), a=NEG32, b=0x80000000) == 1
    assert run1(_op(2), a=0x80000000, b=NEG32) == 0
    assert run1(_op(3), a=0x80000000, b=NEG32) == 1
    assert run1(_op(3), a=0x100000000, b=0xFFFFFFFF) == 0


def test_logic_ops_keep_upper_word():
    a, b = 0xF0F0F0F0_00000000, 0xFF00FF00_0000FFFF
    assert run1(_op(4), a=a, b=b) == a ^ b
    assert run1(_op(6), a=a, b=b) == a | b
    assert run1(_op(7), a=a, b=b) == a & b


@pytest.mark.parametrize("funct3,funct7", [(4, 0x04), (7, 0x20), (0, 0x01), (2, 0x10)])
def test_extension_funct7_is_declined(funct3, funct7):
    with pytest.raises(Exception, match="Unknown instruction"):
        run(_op(funct3, funct7), extensions=[])


# -- W forms ---------------------------------------------------------------


def test_addw_subw_sign_extend_and_ignore_upper_bits():
    assert run1(_opw(0), a=0x7FFFFFFF, b=1) == NEG32
    assert run1(_opw(0), a=0x1_00000001, b=0xFFFF_0000_0000_0000) == 1
    assert run1(_opw(0, 0x20), a=0, b=1) == M64
    assert run1(_opw(0, 0x20), a=0x80000000, b=1) == 0x7FFFFFFF


def test_addiw_is_sext_w():
    assert run1(i_type(0x1B, 0, 0), a=0x80000000) == NEG32
    assert run1(i_type(0x1B, 0, 0), a=0xDEADBEEF_7FFFFFFF) == 0x7FFFFFFF
    assert run1(i_type(0x1B, 0, -1), a=0) == M64
    assert run1(i_type(0x1B, 0, 1), a=0x7FFFFFFF) == NEG32


def test_shift_imm_w():
    assert run1(_shift_imm_w(1, 0x00, 31), a=1) == NEG32
    assert run1(_shift_imm_w(1, 0x00, 1), a=0xFFFFFFFF_40000000) == NEG32
    assert run1(_shift_imm_w(5, 0x00, 4), a=NEG32) == 0x08000000
    assert run1(_shift_imm_w(5, 0x00, 0), a=0x80000000) == NEG32
    assert run1(_shift_imm_w(5, 0x20, 4), a=0x80000000) == 0xFFFFFFFFF8000000
    assert run1(_shift_imm_w(5, 0x20, 0), a=0x1_7FFFFFFF) == 0x7FFFFFFF


@pytest.mark.parametrize("funct3,funct7", [(1, 0x01), (5, 0x01), (5, 0x21), (1, 0x30)])
def test_shift_imm_w_reserved_or_extension_is_declined(funct3, funct7):
    # imm[25] set is reserved on the W shifts; 0x30 is Zbb (clzw / roriw).
    with pytest.raises(Exception, match="Unknown instruction"):
        run(_shift_imm_w(funct3, funct7, 3), extensions=[])


def test_register_shifts_w_use_five_bits():
    assert run1(_opw(1), a=1, b=31) == NEG32
    assert run1(_opw(1), a=1, b=33) == 2
    assert run1(_opw(5), a=NEG32, b=4) == 0x08000000
    assert run1(_opw(5), a=0x80000000, b=32) == NEG32  # shift by 0
    assert run1(_opw(5, 0x20), a=0x80000000, b=4) == 0xFFFFFFFFF8000000
    assert run1(_opw(5, 0x20), a=0x1_40000000, b=1) == 0x20000000


# -- loads / stores --------------------------------------------------------

_WORDS = bytes.fromhex("f0debc9a78563412") + bytes.fromhex("8180ff7f00000080")


@pytest.mark.parametrize(
    "funct3,offset,expected",
    [
        (0, 0, 0xFFFFFFFFFFFFFFF0),  # lb
        (4, 0, 0xF0),  # lbu
        (1, 0, 0xFFFFFFFFFFFFDEF0),  # lh
        (5, 0, 0xDEF0),  # lhu
        (2, 0, 0xFFFFFFFF9ABCDEF0),  # lw sign-extends to 64
        (6, 0, 0x9ABCDEF0),  # lwu
        (2, 4, 0x12345678),
        (3, 0, 0x123456789ABCDEF0),  # ld
        (3, 8, 0x800000007FFF8081),
        (1, 10, 0x7FFF),
    ],
)
def test_loads(funct3, offset, expected):
    core = run(i_type(0x03, funct3, offset), regs={RS1: DATA}, mem={DATA: _WORDS})
    assert reg(core, RD) == expected


def test_load_funct3_7_is_declined():
    with pytest.raises(Exception, match="Unknown instruction"):
        run(i_type(0x03, 7, 0), regs={RS1: DATA}, extensions=[])


@pytest.mark.parametrize("funct3,size", [(0, 1), (1, 2), (2, 4), (3, 8)])
def test_stores_write_exactly_their_width(funct3, size):
    value = 0x1122334455667788
    core = run(
        s_type(funct3, 8), regs={RS1: DATA - 8, RS2: value}, mem={DATA: b"\xee" * 16}
    )
    written = core.visible_memory.read(DATA, 16)
    assert written[:size] == value.to_bytes(8, "little")[:size]
    assert written[size:] == b"\xee" * (16 - size)


def test_negative_store_offset():
    core = run(s_type(3, -8), regs={RS1: DATA + 8, RS2: M64})
    assert core.visible_memory.read(DATA, 8) == b"\xff" * 8


# -- control flow ----------------------------------------------------------


def _pc_after(instr, regs):
    return reg(run(instr, regs), "pc")


@pytest.mark.parametrize(
    "funct3,a,b,taken",
    [
        (0, 0x1_00000000, 0, False),  # beq: equal low words only
        (0, NEG32, NEG32, True),
        (1, 0x1_00000000, 0, True),  # bne
        (4, NEG32, 0x80000000, True),  # blt: negative < positive in 64 bits
        (4, 0x80000000, NEG32, False),
        (5, 0x80000000, NEG32, True),  # bge
        (6, 0x80000000, NEG32, True),  # bltu
        (6, M64, 0, False),
        (7, M64, 0x1_00000000, True),  # bgeu
    ],
)
def test_branches_compare_64_bit(funct3, a, b, taken):
    pc = _pc_after(b_type(funct3, 64), {RS1: a, RS2: b})
    assert pc == (PC + 64 if taken else PC + 4)


def test_backward_branch():
    assert _pc_after(b_type(0, -16), {}) == PC - 16


def test_jal_links_and_jumps():
    core = run(j_type(-0x800))
    assert reg(core, RD) == PC + 4
    assert reg(core, "pc") == PC - 0x800


def test_jalr_clears_lsb_and_reaches_above_4g():
    core = run(i_type(0x67, 0, 3), regs={RS1: 0x1_0000_2000})
    assert reg(core, "pc") == 0x1_0000_2002
    assert reg(core, RD) == PC + 4


def test_jalr_rd_equal_rs1_uses_old_value():
    core = run(i_type(0x67, 0, 8, rd=RS1), regs={RS1: 0x3000})
    assert reg(core, "pc") == 0x3008
    assert reg(core, RS1) == PC + 4


def test_x0_stays_zero():
    core = run(_opimm(0, 5) & ~(0x1F << 7))  # addi x0, rs1, 5
    assert reg(core, 0) == 0


def test_registers_and_pc_are_eight_bytes():
    core = run(_opimm(0, 0))
    assert all(len(core.register_file[i].read()) == 8 for i in range(34))


def test_csr_instruction_without_zicsr_raises():
    with pytest.raises(zicsr_isa.NoCSRsError):
        run(csr(2, 0xC00, rs1=0), extensions=[])


def test_snoop_prints_disassembly(capsys):
    run(_opw(0), {RS1: 1, RS2: 2}, snoop=True)
    out = capsys.readouterr().out
    assert "addw ra, a5, s4" in out


def test_fence_and_ecall_are_claimed():
    core = run([i_type(0x0F, 0, 0, rd=0, rs1=0), 0x00000073])
    assert reg(core, "pc") == PC + 8
