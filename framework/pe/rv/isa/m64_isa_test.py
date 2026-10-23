"""RV64M unit tests: 64x64->128 high products, the W forms' sign extension,
and the spec's division-by-zero / overflow results (no trap)."""

import pytest

from framework.pe.rv.isa.isa64_testing import M64, r_type, run, run1

MIN64 = 1 << 63
NEG32 = 0xFFFFFFFF80000000


def _m(funct3):
    return r_type(0x33, funct3, 0x01)


def _mw(funct3):
    return r_type(0x3B, funct3, 0x01)


def _neg(v):
    return v & M64


@pytest.mark.parametrize(
    "funct3,a,b,expected",
    [
        # mul: low 64 bits
        (0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE00000001),
        (0, M64, 3, _neg(-3)),
        # mulh: signed x signed, high 64
        (1, M64, M64, 0),
        (1, MIN64, MIN64, 1 << 62),
        (1, MIN64, 2, M64),
        # mulhsu: signed rs1 x unsigned rs2
        (2, M64, M64, M64),
        (2, 2, M64, 1),
        # mulhu
        (3, M64, M64, M64 - 1),
        (3, 1 << 32, 1 << 32, 1),
        # div: rounds towards zero
        (4, _neg(-7), 2, _neg(-3)),
        (4, 7, _neg(-2), _neg(-3)),
        (4, _neg(-7), _neg(-2), 3),
        (4, 5, 0, M64),  # by zero -> -1
        (4, MIN64, M64, MIN64),  # overflow -> dividend
        # divu
        (5, M64, 2, M64 >> 1),
        (5, 5, 0, M64),
        # rem: sign of the dividend
        (6, _neg(-7), 2, _neg(-1)),
        (6, 7, _neg(-2), 1),
        (6, _neg(-7), 0, _neg(-7)),  # by zero -> dividend
        (6, MIN64, M64, 0),  # overflow -> 0
        # remu
        (7, M64, 10, M64 % 10),
        (7, 1234, 0, 1234),
    ],
)
def test_m_64(funct3, a, b, expected):
    assert run1(_m(funct3), a=a, b=b) == expected


def test_exact_division_of_large_values():
    # float division would lose these low bits.
    a, b = 0x7FFFFFFFFFFFFFFF, 3
    assert run1(_m(4), a=a, b=b) == a // b
    assert run1(_m(5), a=M64, b=7) == M64 // 7
    assert run1(_m(6), a=a, b=b) == a % b


@pytest.mark.parametrize(
    "funct3,a,b,expected",
    [
        # mulw: low 32 of the product, sign-extended; upper input bits ignored
        (0, 0x10000, 0x8000, NEG32),
        (0, 0xABCD_0000_0003, 0xFFFF_0000_0005, 15),
        (0, 0xFFFFFFFF, 0xFFFFFFFF, 1),
        # divw
        (4, _neg(-7), 2, _neg(-3)),
        (4, 0x1_00000007, 2, 3),
        (4, 5, 0x1_00000000, M64),  # rs2's low word is zero -> by zero
        (4, 0x80000000, 0xFFFFFFFF, NEG32),  # overflow -> dividend
        # divuw: unsigned on the low words, result sign-extended
        (5, 0xFFFFFFFF, 1, M64),
        (5, 0x80000000, 2, 0x40000000),
        (5, 5, 0, M64),
        # remw
        (6, _neg(-7), 2, _neg(-1)),
        (6, 0x80000000, 0xFFFFFFFF, 0),  # overflow -> 0
        (6, 0x1_80000000, 0, NEG32),  # by zero -> sext(low word of dividend)
        # remuw
        (7, 0xFFFFFFFF, 0x10, 0xF),
        (7, 0x9_80000001, 0, 0xFFFFFFFF80000001),
        (7, 0x80000005, 0x80000000, 5),
    ],
)
def test_m_w(funct3, a, b, expected):
    assert run1(_mw(funct3), a=a, b=b) == expected


@pytest.mark.parametrize("funct3", [1, 2, 3])
def test_mulh_has_no_w_form(funct3):
    with pytest.raises(Exception, match="Unknown instruction"):
        run(_mw(funct3))


def test_m_not_decoded_without_the_extension():
    with pytest.raises(Exception, match="Unknown instruction"):
        run(_m(0), extensions=[])
