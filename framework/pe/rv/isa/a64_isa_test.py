"""RV64 Zaamo / Zalrsc unit tests: word vs doubleword widths, sign extension of
the returned word, 32-bit compares for ``.w`` min/max, and lr/sc reservation
semantics on a single hart."""

import pytest

from framework.pe.rv.isa.isa64_testing import (
    DATA,
    M64,
    RD,
    RS1,
    RS2,
    amo,
    reg,
    run,
    s_type,
)

W, D = 0x2, 0x3
NEG32 = 0xFFFFFFFF80000000


def _amo(funct5, width, old, src, size):
    core = run(
        amo(funct5, width),
        regs={RS1: DATA, RS2: src},
        mem={DATA: old.to_bytes(size, "little") + b"\xaa" * 8},
    )
    new = int.from_bytes(core.visible_memory.read(DATA, size), "little")
    assert core.visible_memory.read(DATA + size, 8) == b"\xaa" * 8
    return reg(core, RD), new


@pytest.mark.parametrize(
    "funct5,old,src,new",
    [
        (0x00, M64, 2, 1),  # amoadd.d wraps at 64
        (0x01, 5, 0x1234_5678_9ABC_DEF0, 0x1234_5678_9ABC_DEF0),  # amoswap.d
        (0x04, 0xFF00, 0x0FF0, 0xF0F0),  # amoxor.d
        (0x08, 1 << 63, 1, (1 << 63) | 1),  # amoor.d
        (0x0C, M64, 0xF0F0 << 40, 0xF0F0 << 40),  # amoand.d
        (0x10, 0x80000000, NEG32, NEG32),  # amomin.d: 64-bit signed
        (0x14, 0x80000000, NEG32, 0x80000000),  # amomax.d
        (0x18, 0x80000000, NEG32, 0x80000000),  # amominu.d
        (0x1C, 0x80000000, NEG32, NEG32),  # amomaxu.d
    ],
)
def test_amo_d(funct5, old, src, new):
    rd, mem = _amo(funct5, D, old, src, 8)
    assert rd == old
    assert mem == new


@pytest.mark.parametrize(
    "funct5,old,src,new",
    [
        (0x00, 0xFFFFFFFF, 2, 1),  # amoadd.w wraps at 32, next word untouched
        (0x01, 0x80000000, 0xDEAD_0000_0007, 7),  # amoswap.w stores the low word
        (0x10, 0x80000000, 0xFFFF_FFFF_7FFF_FFFF, 0x80000000),  # min: 32-bit signed
        (0x14, 0x80000000, 0x1_0000_0001, 1),  # max: 32-bit signed
        (0x18, 0x80000000, 0x1_0000_0001, 1),  # minu: low 32 bits only
        (0x1C, 0x7FFFFFFF, 0x1_8000_0000, 0x80000000),
    ],
)
def test_amo_w(funct5, old, src, new):
    rd, mem = _amo(funct5, W, old, src, 4)
    # The old word comes back sign-extended to 64.
    assert rd == (old | 0xFFFFFFFF00000000 if old & 0x80000000 else old)
    assert mem == new


def test_amo_other_width_is_declined():
    with pytest.raises(Exception, match="Unknown instruction"):
        run(amo(0x00, 0x0), regs={RS1: DATA})


_LR, _SC = 0x02, 0x03


def _lr(width, rd=RD, rs1=RS1):
    return amo(_LR, width, rd=rd, rs1=rs1, rs2=0)


def _sc(width, rd=2, rs1=RS1, rs2=RS2):
    return amo(_SC, width, rd=rd, rs1=rs1, rs2=rs2)


def _mem(core, addr=DATA, size=8):
    return int.from_bytes(core.visible_memory.read(addr, size), "little")


def test_lr_sc_d_succeeds():
    core = run(
        [_lr(D), _sc(D)],
        regs={RS1: DATA, RS2: 0x1122334455667788},
        mem={DATA: (42).to_bytes(8, "little")},
    )
    assert reg(core, RD) == 42
    assert reg(core, 2) == 0
    assert _mem(core) == 0x1122334455667788
    assert core.register_file.reservation is None


def test_lr_w_sign_extends_and_sc_w_writes_a_word():
    core = run(
        [_lr(W), _sc(W)],
        regs={RS1: DATA, RS2: 0xFFFF_FFFF_1234_5678},
        mem={DATA: bytes.fromhex("00000080") + b"\xbb" * 4},
    )
    assert reg(core, RD) == NEG32
    assert reg(core, 2) == 0
    assert _mem(core) == 0xBBBBBBBB_12345678


def test_sc_without_lr_fails_and_writes_nothing():
    core = run(_sc(D), regs={RS1: DATA, RS2: M64})
    assert reg(core, 2) == 1
    assert _mem(core) == 0


def test_second_sc_fails():
    core = run([_lr(D), _sc(D), _sc(D, rd=3)], regs={RS1: DATA, RS2: 7})
    assert reg(core, 2) == 0
    assert reg(core, 3) == 1


def test_store_to_reserved_address_breaks_reservation():
    sw = s_type(0x2, 4, rs1=RS1, rs2=0)  # sw zero, 4(rs1): overlaps the .d
    core = run([_lr(D), sw, _sc(D)], regs={RS1: DATA, RS2: 9})
    assert reg(core, 2) == 1
    assert _mem(core) == 0


def test_store_elsewhere_keeps_reservation():
    sd = s_type(0x3, 8, rs1=RS1, rs2=0)  # sd zero, 8(rs1): the next doubleword
    core = run([_lr(D), sd, _sc(D)], regs={RS1: DATA, RS2: 9})
    assert reg(core, 2) == 0
    assert _mem(core) == 9


def test_amo_to_reserved_address_breaks_reservation():
    core = run([_lr(D), amo(0x00, D, rd=0, rs2=0), _sc(D)], regs={RS1: DATA, RS2: 9})
    assert reg(core, 2) == 1


def test_sc_to_other_address_fails():
    core = run([_lr(D), _sc(D, rs1=5)], regs={RS1: DATA, 5: DATA + 64, RS2: 9})
    assert reg(core, 2) == 1
    assert _mem(core, DATA + 64) == 0


def test_sc_w_inside_lr_d_reservation_succeeds():
    core = run([_lr(D), _sc(W, rs1=5)], regs={RS1: DATA, 5: DATA + 4, RS2: 9})
    assert reg(core, 2) == 0
    assert _mem(core, DATA + 4, 4) == 9


def test_lr_with_nonzero_rs2_is_declined():
    with pytest.raises(Exception, match="Unknown instruction"):
        run(amo(_LR, D, rs2=3), regs={RS1: DATA})


def test_reset_clears_reservation():
    core = run(_lr(D), regs={RS1: DATA})
    assert core.register_file.reservation == (DATA, DATA + 8)
    core.reset()
    assert core.register_file.reservation is None
