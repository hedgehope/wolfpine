"""RV64 forms of the Zba and Zbb bit-manipulation extensions.

The RV32 classes (``b_isa.py``) are written for the Blackhole baby cores and
mask to 32 bits; these are their RV64 counterparts, per the Bitmanip spec:

- **Zba**: ``sh1add`` / ``sh2add`` / ``sh3add`` (64-bit), and the ``.uw`` forms
  that zero-extend rs1's low word first: ``add.uw``, ``sh1add.uw`` /
  ``sh2add.uw`` / ``sh3add.uw`` (OP-32, 0x3B) and ``slli.uw`` (OP-IMM-32, 0x1B,
  funct6 0x02, 6-bit shamt). ``zext.w`` is ``add.uw rd, rs, zero``.
- **Zbb**: ``andn`` / ``orn`` / ``xnor``, ``min`` / ``max`` / ``minu`` / ``maxu``,
  ``rol`` / ``ror`` (6-bit amount), ``rori`` (funct6 0x18, 6-bit shamt),
  ``clz`` / ``ctz`` / ``cpop`` / ``sext.b`` / ``sext.h`` on 64 bits, ``rev8``
  (imm 0x6B8 on RV64; 0x698 is RV32's), ``orc.b`` (imm 0x287), and ``zext.h``,
  which on RV64 is in OP-32 (0x3B), not OP. The word forms ``clzw`` / ``ctzw`` /
  ``cpopw`` / ``rolw`` / ``rorw`` / ``roriw`` operate on the low 32 bits and,
  where the result is a value, sign-extend it.

``RV64_I_ISA`` decodes funct7 / funct6 strictly, so every encoding here falls
through to these classes rather than being mis-run as a base op.
"""

from framework.pe.rv.isa.i64_isa import sext32, to_signed64, write_reg
from framework.pe.rv.isa.rv_isa import RV_ISA

_M64 = 0xFFFFFFFFFFFFFFFF
_M32 = 0xFFFFFFFF


def _sext(value, bits):
    sign = 1 << (bits - 1)
    return ((value & ((1 << bits) - 1)) ^ sign) - sign


def _clz(value, bits):
    return bits - value.bit_length()


def _ctz(value, bits):
    return bits if value == 0 else (value & -value).bit_length() - 1


def _ror(value, amount, bits, mask):
    if amount == 0:
        return value
    return ((value >> amount) | (value << (bits - amount))) & mask


def _orc_b(value):
    out = 0
    for i in range(0, 64, 8):
        if (value >> i) & 0xFF:
            out |= 0xFF << i
    return out


def _snoop_r(cls, name, rd, rs1, rs2):
    RV_ISA.print_snoop(
        True,
        f"{name} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}",
        None,
    )


def _snoop_i(cls, name, rd, rs1, imm=None):
    tail = "" if imm is None else f", {hex(imm)}"
    RV_ISA.print_snoop(
        True, f"{name} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}{tail}", None
    )


class RV64_ZBA_ISA(RV_ISA):
    """Zba address generation, RV64."""

    # funct3 -> shift amount for the shNadd / shNadd.uw family (funct7 0x10).
    _SHADD = {0x2: 1, 0x4: 2, 0x6: 3}

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        opcode = instr & 0x7F
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rd = (instr >> 7) & 0x1F
        funct7 = instr >> 25

        if opcode == 0x33 or opcode == 0x3B:
            uw = opcode == 0x3B
            if funct7 == 0x10 and funct3 in cls._SHADD:
                shift = cls._SHADD[funct3]
                name = f"sh{shift}add.uw" if uw else f"sh{shift}add"
            elif uw and funct7 == 0x04 and funct3 == 0x0:
                shift = 0
                name = "add.uw"
            else:
                return False
            a = register_file[rs1].read_uint()
            if uw:
                a &= _M32
            write_reg(register_file, rd, (a << shift) + register_file[rs2].read_uint())
            if snoop:
                _snoop_r(cls, name, rd, rs1, rs2)
            return True

        if opcode == 0x1B and funct3 == 0x1 and instr >> 26 == 0x02:
            shamt = (instr >> 20) & 0x3F
            write_reg(
                register_file, rd, (register_file[rs1].read_uint() & _M32) << shamt
            )
            if snoop:
                _snoop_i(cls, "slli.uw", rd, rs1, shamt)
            return True
        return False


class RV64_ZBB_ISA(RV_ISA):
    """Zbb basic bit manipulation, RV64."""

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        opcode = instr & 0x7F
        if opcode == 0x33:
            return cls._run_r(instr, register_file, snoop)
        if opcode == 0x13:
            return cls._run_i(instr, register_file, snoop)
        if opcode == 0x3B:
            return cls._run_r_w(instr, register_file, snoop)
        if opcode == 0x1B:
            return cls._run_i_w(instr, register_file, snoop)
        return False

    @classmethod
    def _run_r(cls, instr, register_file, snoop):
        funct3 = (instr >> 12) & 0x7
        funct7 = instr >> 25
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rd = (instr >> 7) & 0x1F
        a = register_file[rs1].read_uint()
        b = register_file[rs2].read_uint()

        name = None
        if funct7 == 0x20:  # logic-with-negate (funct7 shared with sub/sra)
            if funct3 == 0x7:
                name, result = "andn", a & ~b
            elif funct3 == 0x6:
                name, result = "orn", a | (~b & _M64)
            elif funct3 == 0x4:
                name, result = "xnor", ~(a ^ b)
        elif funct7 == 0x05:
            if funct3 == 0x4:
                name, result = "min", a if to_signed64(a) < to_signed64(b) else b
            elif funct3 == 0x5:
                name, result = "minu", a if a < b else b
            elif funct3 == 0x6:
                name, result = "max", a if to_signed64(a) > to_signed64(b) else b
            elif funct3 == 0x7:
                name, result = "maxu", a if a > b else b
        elif funct7 == 0x30:
            if funct3 == 0x1:
                name, result = "rol", _ror(a, (64 - (b & 0x3F)) & 0x3F, 64, _M64)
            elif funct3 == 0x5:
                name, result = "ror", _ror(a, b & 0x3F, 64, _M64)

        if name is None:
            return False
        write_reg(register_file, rd, result)
        if snoop:
            _snoop_r(cls, name, rd, rs1, rs2)
        return True

    @classmethod
    def _run_i(cls, instr, register_file, snoop):
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rd = (instr >> 7) & 0x1F
        imm12 = instr >> 20
        a = register_file[rs1].read_uint()

        name = None
        shamt = None
        if funct3 == 0x1 and instr >> 25 == 0x30:
            # Unary count / sign-extend ops; the rs2 field selects which.
            sel = (instr >> 20) & 0x1F
            if sel == 0x0:
                name, result = "clz", _clz(a, 64)
            elif sel == 0x1:
                name, result = "ctz", _ctz(a, 64)
            elif sel == 0x2:
                name, result = "cpop", a.bit_count()
            elif sel == 0x4:
                name, result = "sext.b", _sext(a, 8)
            elif sel == 0x5:
                name, result = "sext.h", _sext(a, 16)
        elif funct3 == 0x5:
            if instr >> 26 == 0x18:
                shamt = (instr >> 20) & 0x3F
                name, result = "rori", _ror(a, shamt, 64, _M64)
            elif imm12 == 0x6B8:
                name, result = "rev8", int.from_bytes(a.to_bytes(8, "little"), "big")
            elif imm12 == 0x287:
                name, result = "orc.b", _orc_b(a)

        if name is None:
            return False
        write_reg(register_file, rd, result)
        if snoop:
            _snoop_i(cls, name, rd, rs1, shamt)
        return True

    @classmethod
    def _run_r_w(cls, instr, register_file, snoop):
        funct3 = (instr >> 12) & 0x7
        funct7 = instr >> 25
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rd = (instr >> 7) & 0x1F
        a = register_file[rs1].read_uint() & _M32

        if funct7 == 0x30 and (funct3 == 0x1 or funct3 == 0x5):
            amount = register_file[rs2].read_uint() & 0x1F
            if funct3 == 0x1:
                name = "rolw"
                amount = (32 - amount) & 0x1F
            else:
                name = "rorw"
            write_reg(register_file, rd, sext32(_ror(a, amount, 32, _M32)))
            if snoop:
                _snoop_r(cls, name, rd, rs1, rs2)
            return True
        if funct7 == 0x04 and funct3 == 0x4 and rs2 == 0:
            write_reg(register_file, rd, a & 0xFFFF)
            if snoop:
                _snoop_i(cls, "zext.h", rd, rs1)
            return True
        return False

    @classmethod
    def _run_i_w(cls, instr, register_file, snoop):
        funct3 = (instr >> 12) & 0x7
        if instr >> 25 != 0x30:
            return False
        rs1 = (instr >> 15) & 0x1F
        rd = (instr >> 7) & 0x1F
        sel = (instr >> 20) & 0x1F
        a = register_file[rs1].read_uint() & _M32

        shamt = None
        if funct3 == 0x1:
            if sel == 0x0:
                name, result = "clzw", _clz(a, 32)
            elif sel == 0x1:
                name, result = "ctzw", _ctz(a, 32)
            elif sel == 0x2:
                name, result = "cpopw", a.bit_count()
            else:
                return False
        elif funct3 == 0x5:
            shamt = sel
            name, result = "roriw", sext32(_ror(a, shamt, 32, _M32))
        else:
            return False
        write_reg(register_file, rd, result)
        if snoop:
            _snoop_i(cls, name, rd, rs1, shamt)
        return True
