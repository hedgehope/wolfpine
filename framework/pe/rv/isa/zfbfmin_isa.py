"""Zfbfmin: scalar BF16 conversions for the RV64 core.

Zfbfmin depends on F. It adds the two conversions ``fcvt.bf16.s`` (round a
single to BF16 under the rm field, with flags) and ``fcvt.s.bf16`` (exact), and,
because BF16 values have to get in and out of the f-registers somehow, it also
includes Zfhmin's 16-bit transfers ``flh`` / ``fsh`` / ``fmv.x.h`` / ``fmv.h.x``.
Those move bits only, so they are the same instructions whether the 16 bits are
FP16 or BF16; Zfh / Zfhmin proper (FP16 arithmetic and conversions) is not here.

A 16-bit value in a 64-bit f-register is NaN-boxed (upper 48 bits all ones).
``fcvt.s.bf16`` reads an improperly boxed operand as the canonical NaN;
``fsh`` and ``fmv.x.h`` are transfers and take the low 16 bits unchecked.

Encodings (OP-FP, funct5 = 01000): ``fcvt.bf16.s`` is fmt = H with rs2 = 01000,
``fcvt.s.bf16`` fmt = S with rs2 = 00110.
"""

from framework.pe.rv.isa import fd_softfloat as sf
from framework.pe.rv.isa.fd_isa import (
    fname,
    raise_flags,
    read_f,
    resolve_rm,
    unbox32,
    write_f,
)
from framework.pe.rv.isa.i64_isa import write_reg
from framework.pe.rv.isa.i_isa import _imm_i, _imm_s
from framework.pe.rv.isa.rv_isa import RV_ISA

_M64 = 0xFFFFFFFFFFFFFFFF
_BOX16 = 0xFFFFFFFFFFFF0000
_BOX32 = 0xFFFFFFFF00000000


def unbox16(raw):
    """A NaN-boxed 16-bit value, or the canonical BF16 NaN if it is not boxed."""
    return raw & 0xFFFF if raw >> 16 == 0xFFFFFFFFFFFF else sf.BF16.canonical_nan


class RV64_ZFBFMIN_ISA(RV_ISA):
    #: Major opcodes (``instr & 0x7F``) this ISA may claim; ``RV64I``'s dispatch table offers it no others.
    OPCODES = frozenset({0x07, 0x27, 0x53})

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        opcode = instr & 0x7F
        if opcode == 0x53:
            return cls._op_fp(instr, register_file, snoop)
        if (opcode == 0x07 or opcode == 0x27) and (instr >> 12) & 0x7 == 1:
            rs1 = (instr >> 15) & 0x1F
            if opcode == 0x07:  # flh
                rd = (instr >> 7) & 0x1F
                imm = _imm_i(instr)
                addr = (register_file[rs1].read_uint() + imm) & _M64
                bits = int.from_bytes(memory_space.read(addr, 2), "little")
                write_f(register_file, rd, _BOX16 | bits)
                text = f"flh {fname(rd)}, {imm}({cls.get_reg_name(rs1)})"
            else:  # fsh
                rs2 = (instr >> 20) & 0x1F
                imm = _imm_s(instr)
                addr = (register_file[rs1].read_uint() + imm) & _M64
                bits = read_f(register_file, rs2) & 0xFFFF
                memory_space.write(addr, bits.to_bytes(2, "little"))
                text = f"fsh {fname(rs2)}, {imm}({cls.get_reg_name(rs1)})"
            if snoop:
                RV_ISA.print_snoop(snoop, text, hex(addr))
            return True
        return False

    @classmethod
    def _op_fp(cls, instr, rf, snoop):
        funct7 = instr >> 25
        funct3 = (instr >> 12) & 0x7
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        flags = 0
        if (funct7 == 0x22 and rs2 == 0x08) or (funct7 == 0x20 and rs2 == 0x06):
            rm = resolve_rm(rf, funct3)
            if rm is None:
                return False
            if funct7 == 0x22:  # fcvt.bf16.s
                bits, flags = sf.convert(unbox32(read_f(rf, rs1)), sf.F32, sf.BF16, rm)
                write_f(rf, rd, _BOX16 | bits)
                text = f"fcvt.bf16.s {fname(rd)}, {fname(rs1)}"
            else:  # fcvt.s.bf16
                bits, flags = sf.convert(unbox16(read_f(rf, rs1)), sf.BF16, sf.F32, rm)
                write_f(rf, rd, _BOX32 | bits)
                text = f"fcvt.s.bf16 {fname(rd)}, {fname(rs1)}"
        elif funct7 == 0x72 and rs2 == 0 and funct3 == 0:  # fmv.x.h
            bits = read_f(rf, rs1) & 0xFFFF
            if rd != 0:
                write_reg(rf, rd, bits - 0x10000 if bits & 0x8000 else bits)
            text = f"fmv.x.h {cls.get_reg_name(rd)}, {fname(rs1)}"
        elif funct7 == 0x7A and rs2 == 0 and funct3 == 0:  # fmv.h.x
            bits = rf[rs1].read_uint() & 0xFFFF
            write_f(rf, rd, _BOX16 | bits)
            text = f"fmv.h.x {fname(rd)}, {cls.get_reg_name(rs1)}"
        else:
            return False
        raise_flags(rf, flags)
        if snoop:
            RV_ISA.print_snoop(snoop, text, hex(bits))
        return True
