"""RV64M: multiply/divide on 64-bit registers, plus the W forms.

OP (0x33) with funct7 0x01 is the 64-bit ``mul`` / ``mulh`` / ``mulhsu`` /
``mulhu`` / ``div`` / ``divu`` / ``rem`` / ``remu``; OP-32 (0x3B) with funct7 0x01
is ``mulw`` / ``divw`` / ``divuw`` / ``remw`` / ``remuw``, which operate on the low
32 bits and sign-extend the 32-bit result (``divuw`` / ``remuw`` included).

Division is exact integer arithmetic rounding towards zero (Python's ``//``
floors, so the quotient is built from magnitudes), with the spec's two special
cases rather than a trap:

* divide by zero: quotient all ones (-1, or 2**XLEN-1 unsigned), remainder the
  dividend;
* signed overflow (most-negative / -1): quotient the dividend, remainder 0.
"""

from framework.pe.rv.isa.i64_isa import sext32, to_signed64, write_reg
from framework.pe.rv.isa.rv_isa import RV_ISA

_M64 = 0xFFFFFFFFFFFFFFFF
_M32 = 0xFFFFFFFF


def _s32(value):
    value &= _M32
    return value - (1 << 32) if value & 0x80000000 else value


def _div(a, b):
    """Signed division rounding towards zero; ``b`` is non-zero."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _rem(a, b):
    """Remainder with the sign of the dividend; ``b`` is non-zero."""
    return a - b * _div(a, b)


def _divs(a, b, bits):
    if b == 0:
        return -1
    if a == -(1 << (bits - 1)) and b == -1:
        return a
    return _div(a, b)


def _rems(a, b, bits):
    if b == 0:
        return a
    if a == -(1 << (bits - 1)) and b == -1:
        return 0
    return _rem(a, b)


# funct3 -> mnemonic, for the snoop line.
_NAMES = ("mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu")
_OPS = ("*", "*", "*", "*", "/", "/", "%", "%")


class RV64_M_ISA(RV_ISA):
    #: Major opcodes (``instr & 0x7F``) this ISA may claim; ``RV64I``'s dispatch table offers it no others.
    OPCODES = frozenset({0x33, 0x3B})

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        opcode = instr & 0x7F
        if (opcode != 0x33 and opcode != 0x3B) or instr >> 25 != 0x01:
            return False

        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rd = (instr >> 7) & 0x1F
        a = register_file[rs1].read_uint()
        b = register_file[rs2].read_uint()

        if opcode == 0x33:
            suffix = ""
            match funct3:
                case 0x0:
                    result = a * b
                case 0x1:
                    result = (to_signed64(a) * to_signed64(b)) >> 64
                case 0x2:
                    result = (to_signed64(a) * b) >> 64
                case 0x3:
                    result = (a * b) >> 64
                case 0x4:
                    result = _divs(to_signed64(a), to_signed64(b), 64)
                case 0x5:
                    result = a // b if b else _M64
                case 0x6:
                    result = _rems(to_signed64(a), to_signed64(b), 64)
                case _:
                    result = a % b if b else a
        else:
            suffix = "w"
            match funct3:
                case 0x0:
                    result = a * b
                case 0x4:
                    result = _divs(_s32(a), _s32(b), 32)
                case 0x5:
                    a &= _M32
                    b &= _M32
                    result = a // b if b else _M32
                case 0x6:
                    result = _rems(_s32(a), _s32(b), 32)
                case 0x7:
                    a &= _M32
                    b &= _M32
                    result = a % b if b else a
                case _:
                    # mulh* have no W form.
                    return False
            result = sext32(result)

        write_reg(register_file, rd, result)
        if snoop:
            name = _NAMES[funct3] + suffix
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}",
                f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} {_OPS[funct3]} {cls.get_reg_name(rs2)}",
            )
        return True
