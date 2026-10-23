"""RV64 Zba / Zbb unit tests: 64-bit widths, the ``.uw`` zero extension and
the word forms' sign extension, and the RV64-specific encodings (``rev8`` 0x6B8,
``zext.h`` in OP-32, 6-bit ``rori``)."""

import pytest

from framework.pe.rv.isa.isa64_testing import M64, i_type, r_type, run, run1

NEG32 = 0xFFFFFFFF80000000


def _imm(opcode, funct3, imm12):
    return i_type(opcode, funct3, imm12)


# -- Zba -------------------------------------------------------------------


@pytest.mark.parametrize(
    "instr,a,b,expected",
    [
        (r_type(0x3B, 0, 0x04), M64, 1, 0x1_00000000),  # add.uw
        (r_type(0x3B, 0, 0x04, rs2=0), 0xDEADBEEF_80000000, 0, 0x80000000),  # zext.w
        (r_type(0x3B, 2, 0x10), 0xFFFF_FFFF_8000_0000, 1, 0x1_00000001),  # sh1add.uw
        (r_type(0x3B, 4, 0x10), 0x1_00000001, 0x10, 0x14),  # sh2add.uw
        (r_type(0x3B, 6, 0x10), 0xFFFFFFFF, 0, 0x7_FFFFFFF8),  # sh3add.uw
        (r_type(0x33, 2, 0x10), 1 << 62, 1, (1 << 63) + 1),  # sh1add
        (r_type(0x33, 6, 0x10), 0x1_00000000, 8, 0x8_00000008),  # sh3add, 64-bit
        (r_type(0x33, 4, 0x10), M64, 0, M64 - 3),  # sh2add wraps at 64
    ],
)
def test_zba(instr, a, b, expected):
    assert run1(instr, a=a, b=b) == expected


def test_slli_uw():
    assert run1(_imm(0x1B, 1, (0x02 << 6) | 4), a=NEG32) == 0x8_00000000
    assert (
        run1(_imm(0x1B, 1, (0x02 << 6) | 32), a=0xFFFFFFFF_FFFFFFFF)
        == 0xFFFFFFFF_00000000
    )
    assert run1(_imm(0x1B, 1, (0x02 << 6) | 0), a=M64) == 0xFFFFFFFF


# -- Zbb: logic, min/max, rotate --------------------------------------------


@pytest.mark.parametrize(
    "funct3,funct7,a,b,expected",
    [
        (7, 0x20, M64, 0xFF << 56, M64 >> 8),  # andn
        (6, 0x20, 0, 0xFFFFFFFF, 0xFFFFFFFF_00000000),  # orn
        (4, 0x20, 0x0F, 0xF0, M64 ^ 0xFF),  # xnor
        (4, 0x05, NEG32, 0x80000000, NEG32),  # min
        (6, 0x05, NEG32, 0x80000000, 0x80000000),  # max
        (5, 0x05, NEG32, 0x80000000, 0x80000000),  # minu
        (7, 0x05, NEG32, 0x80000000, NEG32),  # maxu
        (1, 0x30, 1 << 63, 1, 1),  # rol
        (1, 0x30, 0x1, 64 + 4, 0x10),  # rol uses rs2[5:0]
        (5, 0x30, 1, 1, 1 << 63),  # ror
        (5, 0x30, 0x12345678_9ABCDEF0, 32, 0x9ABCDEF0_12345678),
        (5, 0x30, 0xABC, 0, 0xABC),
    ],
)
def test_zbb_r(funct3, funct7, a, b, expected):
    assert run1(r_type(0x33, funct3, funct7), a=a, b=b) == expected


@pytest.mark.parametrize(
    "funct3,a,b,expected",
    [
        (1, 0x80000001, 1, 3),  # rolw
        (1, 0x40000000, 1, NEG32),  # rolw result sign-extends
        (1, 0xFFFF_FFFF_0000_0001, 33, 2),  # upper bits and rs2[5] ignored
        (5, 1, 1, NEG32),  # rorw
        (5, 0x1_00000002, 1, 1),
    ],
)
def test_rolw_rorw(funct3, a, b, expected):
    assert run1(r_type(0x3B, funct3, 0x30), a=a, b=b) == expected


# -- Zbb: unary and immediate -----------------------------------------------


@pytest.mark.parametrize(
    "sel,a,expected",
    [
        (0, 1, 63),  # clz
        (0, 0, 64),
        (0, 1 << 63, 0),
        (0, 0x80000000, 32),
        (1, 0, 64),  # ctz
        (1, 1 << 40, 40),
        (2, M64, 64),  # cpop
        (2, 0xF0000000_0000000F, 8),
        (4, 0x80, M64 ^ 0x7F),  # sext.b to 64
        (4, 0xFFFF_FF7F, 0x7F),
        (5, 0x8000, M64 ^ 0x7FFF),  # sext.h to 64
        (5, 0x1_7FFF, 0x7FFF),
    ],
)
def test_count_and_sext(sel, a, expected):
    assert run1(_imm(0x13, 1, (0x30 << 5) | sel), a=a) == expected


@pytest.mark.parametrize(
    "sel,a,expected",
    [
        (0, 1, 31),  # clzw
        (0, 0xFFFF_FFFF_0000_0000, 32),
        (0, NEG32, 0),
        (1, 0x1_00000000, 32),  # ctzw
        (1, 0x8, 3),
        (2, M64, 32),  # cpopw
        (2, 0xFF_0000_0003, 2),
    ],
)
def test_count_w(sel, a, expected):
    assert run1(_imm(0x1B, 1, (0x30 << 5) | sel), a=a) == expected


def test_rori_six_bit_shamt():
    rori = (0x18 << 6) | 32
    assert run1(_imm(0x13, 5, rori), a=0x12345678_9ABCDEF0) == 0x9ABCDEF0_12345678
    assert run1(_imm(0x13, 5, (0x18 << 6) | 63), a=1) == 2
    assert run1(_imm(0x13, 5, (0x18 << 6) | 0), a=0x55) == 0x55


def test_roriw_sign_extends():
    assert run1(_imm(0x1B, 5, (0x30 << 5) | 1), a=1) == NEG32
    assert run1(_imm(0x1B, 5, (0x30 << 5) | 4), a=0xFFFF_0000_0000_0012) == 0x20000001
    assert run1(_imm(0x1B, 5, (0x30 << 5) | 0), a=0x9_00000005) == 5


def test_rev8_is_64_bit():
    assert run1(_imm(0x13, 5, 0x6B8), a=0x0102030405060708) == 0x0807060504030201


def test_rv32_rev8_encoding_is_not_rv64():
    with pytest.raises(Exception, match="Unknown instruction"):
        run(_imm(0x13, 5, 0x698))


def test_orc_b_64_bit():
    assert run1(_imm(0x13, 5, 0x287), a=0x0100_0000_0000_8000) == 0xFF00_0000_0000_FF00


def test_zext_h_lives_in_op32():
    assert run1(r_type(0x3B, 4, 0x04, rs2=0), a=0xFFFF_FFFF_FFFF_8001) == 0x8001
    # RV32's zext.h encoding (OP, 0x33) is not one on RV64.
    with pytest.raises(Exception, match="Unknown instruction"):
        run(r_type(0x33, 4, 0x04, rs2=0))


def test_zbb_not_decoded_without_the_extension():
    with pytest.raises(Exception, match="Unknown instruction"):
        run(_imm(0x13, 1, 0x600), extensions=[])
