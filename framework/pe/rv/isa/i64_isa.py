"""RV64I base integer ISA.

A whole base ISA rather than a set of additions to ``RV_I_ISA``: every RV32
handler masks to 32 bits and writes 4-byte register values, and the RV32 baby
cores' interpreter is the simulator's hottest path, so XLEN is not threaded
through those handlers as a parameter. This module is the RV64 counterpart,
written against 8-byte registers throughout. The immediate decoders are shared.

What differs from RV32I, per the unprivileged spec's RV64I chapter:

* x-registers and the PC are 64 bits. ``lui`` / ``auipc`` sign-extend their
  32-bit immediate to 64; ``slt`` / ``sltu`` / the branches compare 64-bit values.
* ``slli`` / ``srli`` / ``srai`` take a 6-bit shamt (imm[25:20]), with the
  high field narrowed to funct6 (imm[31:26]): 0x00 for ``slli`` / ``srli``, 0x10
  for ``srai``. Any other funct6 is a Zba/Zbb op (``rori``, ``rev8``, ``clz``,
  ...) and is declined. ``sll`` / ``srl`` / ``sra`` use rs2[5:0].
* New loads/stores: ``ld``, ``sd``, ``lwu``. ``lw`` sign-extends to 64.
* The OP-IMM-32 (0x1B) and OP-32 (0x3B) "W" forms: ``addiw``, ``slliw``,
  ``srliw``, ``sraiw``, ``addw``, ``subw``, ``sllw``, ``srlw``, ``sraw``.
  Each computes on the low 32 bits and sign-extends the 32-bit result to 64.
  ``slliw`` / ``srliw`` / ``sraiw`` with imm[25] set are reserved and declined.

Register values are written as ``(value & _M64).to_bytes(8, "little")``: the
``Register`` coerces a short write by keeping its old upper bytes, so every
write here is a full 8 bytes.
"""

from framework.memory.memory import MemoryStall
from framework.pe.pe import ProcessingElement
from framework.pe.rv import breakpoint as breakpoint_trap
from framework.pe.rv.isa import zicsr_isa as zicsr
from framework.pe.rv.isa.i_isa import _imm_b, _imm_i, _imm_j, _imm_s
from framework.pe.rv.isa.rv_isa import RV_ISA

_M64 = 0xFFFFFFFFFFFFFFFF
_M32 = 0xFFFFFFFF
#: ``wfi`` (SYSTEM, funct12 0x105).
_WFI = 0x10500073
# pc / nextpc register indices (``REGISTER_NAME_MAPPING``): an integer index
# is a list lookup, a name goes through RegisterFile.get's fallback (~0.5 us).
_PC, _NEXTPC = 32, 33


def sext32(value):
    """The low 32 bits of ``value``, sign-extended to 64 (as an unsigned int)."""
    value &= _M32
    return value | 0xFFFFFFFF00000000 if value & 0x80000000 else value


def to_signed64(value):
    """An unsigned 64-bit value as a Python signed int."""
    return value - (1 << 64) if value & 0x8000000000000000 else value


def write_reg(register_file, rd, value):
    """Write ``value`` (any int; reduced mod 2**64) to x-register ``rd``."""
    register_file[rd].write((value & _M64).to_bytes(8, "little"))


def store_clears_reservation(register_file, addr, size):
    """Drop an ``lr`` reservation that a store to ``[addr, addr+size)`` overlaps.

    The reservation (see ``a64_isa``) lives on the register file. This covers
    the hart's own stores; on a multi-hart Naiad SoC the bus clears every
    hart's reservation that any store overlaps (``MulticoreMemory``).
    """
    res = register_file.reservation
    if res is not None and addr < res[1] and res[0] < addr + size:
        register_file.reservation = None


# funct3 -> (bytes, signed, mnemonic) for LOAD (0x03).
_LOADS = {
    0x0: (1, True, "lb"),
    0x1: (2, True, "lh"),
    0x2: (4, True, "lw"),
    0x3: (8, True, "ld"),
    0x4: (1, False, "lbu"),
    0x5: (2, False, "lhu"),
    0x6: (4, False, "lwu"),
}

# funct3 -> (bytes, mnemonic) for STORE (0x23).
_STORES = {0x0: (1, "sb"), 0x1: (2, "sh"), 0x2: (4, "sw"), 0x3: (8, "sd")}


class RV64_I_ISA(RV_ISA):
    #: Major opcodes (``instr & 0x7F``) this ISA may claim; ``RV64I``'s dispatch table offers it no others.
    OPCODES = frozenset(
        {0x03, 0x0F, 0x13, 0x17, 0x1B, 0x23, 0x33, 0x37, 0x3B, 0x63, 0x67, 0x6F, 0x73}
    )

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)

        match instr & 0x7F:
            case 0x13:
                return cls.handle_i_arith(instr, register_file, snoop)
            case 0x33:
                return cls.handle_r_arith(instr, register_file, snoop)
            case 0x3:
                return cls.handle_i_load(instr, register_file, memory_space, snoop)
            case 0x23:
                return cls.handle_s_store(instr, register_file, memory_space, snoop)
            case 0x63:
                return cls.handle_b_branch(instr, register_file, snoop)
            case 0x1B:
                return cls.handle_i_arith_w(instr, register_file, snoop)
            case 0x3B:
                return cls.handle_r_arith_w(instr, register_file, snoop)
            case 0x37:
                return cls.handle_u_lui(instr, register_file, snoop)
            case 0x17:
                return cls.handle_u_auipc(instr, register_file, snoop)
            case 0x6F:
                return cls.handle_j_jal(instr, register_file, snoop)
            case 0x67:
                return cls.handle_i_jalr(instr, register_file, snoop)
            case 0x0F:
                return cls.handle_i_fence(instr, snoop)
            case 0x73:
                return cls.handle_i_misc(instr, register_file, memory_space, snoop)
            case _:
                return False

    @classmethod
    def handle_u_lui(cls, instr, register_file, snoop):
        rd = (instr >> 7) & 0x1F
        immediate = sext32(instr & 0xFFFFF000)
        write_reg(register_file, rd, immediate)
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"lui {cls.get_reg_name(rd)}, {hex((instr >> 12) & 0xFFFFF)}",
                f"{cls.get_reg_name(rd)} = {hex(immediate)}",
            )
        return True

    @classmethod
    def handle_u_auipc(cls, instr, register_file, snoop):
        rd = (instr >> 7) & 0x1F
        immediate = sext32(instr & 0xFFFFF000)
        pc_val = register_file[_PC].read_uint()
        write_reg(register_file, rd, pc_val + immediate)
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"auipc {cls.get_reg_name(rd)}, {hex((instr >> 12) & 0xFFFFF)}",
                f"{cls.get_reg_name(rd)} = pc + {hex(to_signed64(immediate))}",
            )
        return True

    @classmethod
    def handle_j_jal(cls, instr, register_file, snoop):
        rd = (instr >> 7) & 0x1F
        pc_val = register_file[_PC].read_uint()
        if rd > 0:
            write_reg(register_file, rd, pc_val + 4)
        offset = _imm_j(instr)
        new_pc_val = (pc_val + offset) & _M64
        register_file[_NEXTPC].write(new_pc_val.to_bytes(8, "little"))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"jal {cls.get_reg_name(rd)}, {hex(offset)}",
                f"jump to {hex(new_pc_val)}",
            )
        return True

    @classmethod
    def handle_i_jalr(cls, instr, register_file, snoop):
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F
        # Read rs1 before writing rd: they may be the same register.
        rs1_val = register_file[rs1].read_uint()
        pc_val = register_file[_PC].read_uint()
        if rd > 0:
            write_reg(register_file, rd, pc_val + 4)
        offset = _imm_i(instr)
        new_pc_val = (rs1_val + offset) & (_M64 - 1)
        register_file[_NEXTPC].write(new_pc_val.to_bytes(8, "little"))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"jalr {cls.get_reg_name(rd)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                f"jump to {hex(new_pc_val)}",
            )
        return True

    @classmethod
    def handle_b_branch(cls, instr, register_file, snoop):
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        if funct3 == 0x0:
            taken = register_file[rs1].read() == register_file[rs2].read()
            name = "beq"
        elif funct3 == 0x1:
            taken = register_file[rs1].read() != register_file[rs2].read()
            name = "bne"
        elif funct3 == 0x4:
            taken = register_file[rs1].read_int() < register_file[rs2].read_int()
            name = "blt"
        elif funct3 == 0x5:
            taken = register_file[rs1].read_int() >= register_file[rs2].read_int()
            name = "bge"
        elif funct3 == 0x6:
            taken = register_file[rs1].read_uint() < register_file[rs2].read_uint()
            name = "bltu"
        elif funct3 == 0x7:
            taken = register_file[rs1].read_uint() >= register_file[rs2].read_uint()
            name = "bgeu"
        else:
            return False

        offset = _imm_b(instr)
        if taken:
            new_pc_val = (register_file[_PC].read_uint() + offset) & _M64
            register_file[_NEXTPC].write(new_pc_val.to_bytes(8, "little"))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}, {hex(offset)}",
                f"taken to {hex(new_pc_val)}" if taken else "false",
            )
        return True

    @classmethod
    def handle_i_load(cls, instr, register_file, memory_space, snoop):
        load = _LOADS.get((instr >> 12) & 0x7)
        if load is None:
            return False
        size, signed, name = load
        rs1 = (instr >> 15) & 0x1F
        rd = (instr >> 7) & 0x1F
        offset = _imm_i(instr)
        addr = (register_file[rs1].read_uint() + offset) & _M64
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rd)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                f"{cls.get_reg_name(rd)} = mem[{hex(addr)}]",
            )
        data = memory_space.read(addr, size)
        if data is MemoryStall:
            return ProcessingElement.PEStall
        write_reg(register_file, rd, int.from_bytes(data, "little", signed=signed))
        return True

    @classmethod
    def handle_s_store(cls, instr, register_file, memory_space, snoop):
        store = _STORES.get((instr >> 12) & 0x7)
        if store is None:
            return False
        size, name = store
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        offset = _imm_s(instr)
        addr = (register_file[rs1].read_uint() + offset) & _M64
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rs2)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                f"mem[{hex(addr)}] = {cls.get_reg_name(rs2)}",
            )
        if memory_space.write(addr, register_file[rs2].read()[:size]) is MemoryStall:
            return ProcessingElement.PEStall
        store_clears_reservation(register_file, addr, size)
        return True

    @classmethod
    def handle_i_arith(cls, instr, register_file, snoop):
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rd = (instr >> 7) & 0x1F
        a = register_file[rs1].read_uint()

        if funct3 == 0x1 or funct3 == 0x5:
            # 6-bit shamt in imm[25:20]; funct6 (imm[31:26]) selects the op.
            # Anything but slli/srli (0x00) or srai (0x10) is Zba/Zbb.
            funct6 = instr >> 26
            shamt = (instr >> 20) & 0x3F
            if funct3 == 0x1:
                if funct6 != 0x00:
                    return False
                name, op, result = "slli", "<<", a << shamt
            elif funct6 == 0x00:
                name, op, result = "srli", ">>", a >> shamt
            elif funct6 == 0x10:
                name, op, result = "srai", ">>", to_signed64(a) >> shamt
            else:
                return False
            write_reg(register_file, rd, result)
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"{name} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, {hex(shamt)}",
                    f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} {op} {hex(shamt)}",
                )
            return True

        immediate = _imm_i(instr)
        imm_u = immediate & _M64
        if funct3 == 0x0:
            name, op, result = "addi", "+", a + immediate
        elif funct3 == 0x2:
            name, op = "slti", "<s"
            result = 1 if to_signed64(a) < immediate else 0
        elif funct3 == 0x3:
            name, op, result = "sltiu", "<u", 1 if a < imm_u else 0
        elif funct3 == 0x4:
            name, op, result = "xori", "^", a ^ imm_u
        elif funct3 == 0x6:
            name, op, result = "ori", "|", a | imm_u
        else:
            name, op, result = "andi", "&", a & imm_u
        write_reg(register_file, rd, result)
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, {hex(immediate)}",
                f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} {op} {hex(immediate)}",
            )
        return True

    @classmethod
    def handle_r_arith(cls, instr, register_file, snoop):
        funct3 = (instr >> 12) & 0x7
        # As in RV32I: funct7 0x00 for every funct3, 0x20 only for sub / sra.
        # Every other funct7 (M's 0x01, Zba's 0x10, Zbb's 0x05/0x20/0x30) is
        # left for the extension that owns it.
        funct7 = instr >> 25
        if funct7 == 0x20:
            if funct3 != 0x0 and funct3 != 0x5:
                return False
        elif funct7 != 0x00:
            return False

        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rd = (instr >> 7) & 0x1F
        a = register_file[rs1].read_uint()
        b = register_file[rs2].read_uint()
        if funct3 == 0x0:
            if funct7:
                name, op, result = "sub", "-", a - b
            else:
                name, op, result = "add", "+", a + b
        elif funct3 == 0x1:
            name, op, result = "sll", "<<", a << (b & 0x3F)
        elif funct3 == 0x2:
            name, op = "slt", "<s"
            result = 1 if to_signed64(a) < to_signed64(b) else 0
        elif funct3 == 0x3:
            name, op, result = "sltu", "<u", 1 if a < b else 0
        elif funct3 == 0x4:
            name, op, result = "xor", "^", a ^ b
        elif funct3 == 0x5:
            if funct7:
                name, op, result = "sra", ">>", to_signed64(a) >> (b & 0x3F)
            else:
                name, op, result = "srl", ">>", a >> (b & 0x3F)
        elif funct3 == 0x6:
            name, op, result = "or", "|", a | b
        else:
            name, op, result = "and", "&", a & b
        write_reg(register_file, rd, result)
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}",
                f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} {op} {cls.get_reg_name(rs2)}",
            )
        return True

    @classmethod
    def handle_i_arith_w(cls, instr, register_file, snoop):
        """OP-IMM-32: ``addiw``, ``slliw``, ``srliw``, ``sraiw``."""
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rd = (instr >> 7) & 0x1F
        if funct3 == 0x0:
            immediate = _imm_i(instr)
            result = register_file[rs1].read_uint() + immediate
            name, op, operand = "addiw", "+", immediate
        elif funct3 == 0x1 or funct3 == 0x5:
            # 5-bit shamt; funct7 must be exactly 0x00 (or 0x20 for sraiw).
            # imm[25] set is reserved, and other funct7s are Zba/Zbb
            # (slli.uw, clzw/ctzw/cpopw, roriw).
            funct7 = instr >> 25
            shamt = (instr >> 20) & 0x1F
            a = register_file[rs1].read_uint() & _M32
            operand = shamt
            if funct3 == 0x1 and funct7 == 0x00:
                name, op, result = "slliw", "<<", a << shamt
            elif funct3 == 0x5 and funct7 == 0x00:
                name, op, result = "srliw", ">>", a >> shamt
            elif funct3 == 0x5 and funct7 == 0x20:
                name, op, result = "sraiw", ">>", sext32(a) >> shamt
            else:
                return False
        else:
            return False
        write_reg(register_file, rd, sext32(result))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, {hex(operand)}",
                f"{cls.get_reg_name(rd)} = sext32({cls.get_reg_name(rs1)} {op} {hex(operand)})",
            )
        return True

    @classmethod
    def handle_r_arith_w(cls, instr, register_file, snoop):
        """OP-32: ``addw``, ``subw``, ``sllw``, ``srlw``, ``sraw``."""
        funct3 = (instr >> 12) & 0x7
        funct7 = instr >> 25
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rd = (instr >> 7) & 0x1F
        if funct7 == 0x00:
            if funct3 == 0x0:
                name, op = "addw", "+"
                result = register_file[rs1].read_uint() + register_file[rs2].read_uint()
            elif funct3 == 0x1:
                name, op = "sllw", "<<"
                shamt = register_file[rs2].read_uint() & 0x1F
                result = register_file[rs1].read_uint() << shamt
            elif funct3 == 0x5:
                name, op = "srlw", ">>"
                shamt = register_file[rs2].read_uint() & 0x1F
                result = (register_file[rs1].read_uint() & _M32) >> shamt
            else:
                return False
        elif funct7 == 0x20:
            if funct3 == 0x0:
                name, op = "subw", "-"
                result = register_file[rs1].read_uint() - register_file[rs2].read_uint()
            elif funct3 == 0x5:
                name, op = "sraw", ">>"
                shamt = register_file[rs2].read_uint() & 0x1F
                result = to_signed64(sext32(register_file[rs1].read_uint())) >> shamt
            else:
                return False
        else:
            return False
        write_reg(register_file, rd, sext32(result))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}",
                f"{cls.get_reg_name(rd)} = sext32({cls.get_reg_name(rs1)} {op} {cls.get_reg_name(rs2)})",
            )
        return True

    @classmethod
    def handle_i_fence(cls, instr, snoop):
        if snoop:
            name = "fence.i" if (instr >> 12) & 0x1 else "fence"
            RV_ISA.print_snoop(snoop, name, "ignored")
        return True

    @classmethod
    def handle_i_misc(cls, instr, register_file, memory_space, snoop):
        funct3 = (instr >> 12) & 0x7
        if funct3 != 0x0:
            # Zicsr (funct3 1-3, 5-7) belongs to RV64_ZICSR_ISA, which attaches
            # the CSR file. Without one there is no honest value to put in rd,
            # so refuse rather than fall through to the unknown-instruction
            # path; funct3 4 is not an encoding and is declined.
            if funct3 != 0x4 and getattr(register_file, "csrs", None) is None:
                raise zicsr.NoCSRsError(
                    f"CSR instruction {hex(instr)} at PC "
                    f"{hex(register_file['pc'].read_uint())} on an RV64 core "
                    f"built without Zicsr. Add RV64_ZICSR_ISA ('zicsr') to the "
                    f"core's extensions; see framework/pe/rv/isa/zicsr64_isa.py."
                )
            return False
        if instr == _WFI:
            # A hint with no interrupts modelled: a nop, unless the platform
            # routes it (a multi-hart Naiad SoC's CLINT puts the hart to sleep).
            hook = getattr(register_file, "wfi", None)
            if hook is not None:
                hook()
            if snoop:
                RV_ISA.print_snoop(snoop, "wfi", "sleep" if hook else "nop")
            return True
        is_ebreak = ((instr >> 20) & 0x1) == 1
        if is_ebreak and breakpoint_trap.trapping_enabled():
            if snoop:
                RV_ISA.print_snoop(snoop, "ebreak", "trap")
            raise breakpoint_trap.RiscvBreakpoint(memory_space)
        if snoop:
            RV_ISA.print_snoop(snoop, "ebreak" if is_ebreak else "ecall", "ignored")
        return True
