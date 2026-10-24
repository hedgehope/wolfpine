"""RV64 atomics: Zaamo (``amo*.w`` / ``amo*.d``) and Zalrsc (``lr`` / ``sc``).

AMO (0x2F) with funct3 010 is word width, 011 doubleword. A word AMO operates
on the low 32 bits of rs2, compares signed or unsigned *as 32-bit values* for
``min`` / ``max`` / ``minu`` / ``maxu``, and returns the old memory word
sign-extended to 64 in rd. The simulator is single-stepped, so each
read-modify-write is naturally atomic; aq / rl are accepted and ignored.

Zalrsc on a single hart. ``lr`` loads (sign-extending a word) and registers a
reservation for the bytes it read, held on the register file as
``reservation = (start, end)`` (``RV64I`` initialises it to ``None``). ``sc``
succeeds, writing memory and putting 0 in rd, only if a reservation is held and
covers every byte it would write; otherwise it writes nothing and puts 1 in rd.
Either way the reservation is then gone. Every store this hart makes (``sb`` /
``sh`` / ``sw`` / ``sd`` in ``RV64_I_ISA`` and each AMO here) clears a
reservation it overlaps. Stores by other bus masters (a DMA engine, another
hart) are not observed: the reservation is per-core state with no snoop on the
memory map, which is exact for the one-hart designs this core is built for.
"""

from framework.memory.memory import MemoryStall
from framework.pe.pe import ProcessingElement
from framework.pe.rv.isa.i64_isa import (
    sext32,
    store_clears_reservation,
    to_signed64,
    write_reg,
)
from framework.pe.rv.isa.rv_isa import RV_ISA

_M64 = 0xFFFFFFFFFFFFFFFF
_M32 = 0xFFFFFFFF


def _s32(value):
    return value - (1 << 32) if value & 0x80000000 else value


# funct5 -> (name, op(old, src) for word, op(old, src) for doubleword). Operands
# arrive as unsigned values of the access width.
_OPS = {
    0x00: ("amoadd", lambda o, s: o + s, lambda o, s: o + s),
    0x01: ("amoswap", lambda o, s: s, lambda o, s: s),
    0x04: ("amoxor", lambda o, s: o ^ s, lambda o, s: o ^ s),
    0x08: ("amoor", lambda o, s: o | s, lambda o, s: o | s),
    0x0C: ("amoand", lambda o, s: o & s, lambda o, s: o & s),
    0x10: (
        "amomin",
        lambda o, s: o if _s32(o) < _s32(s) else s,
        lambda o, s: o if to_signed64(o) < to_signed64(s) else s,
    ),
    0x14: (
        "amomax",
        lambda o, s: o if _s32(o) > _s32(s) else s,
        lambda o, s: o if to_signed64(o) > to_signed64(s) else s,
    ),
    0x18: ("amominu", lambda o, s: min(o, s), lambda o, s: min(o, s)),
    0x1C: ("amomaxu", lambda o, s: max(o, s), lambda o, s: max(o, s)),
}

# funct3 -> (bytes, mask, suffix).
_WIDTHS = {0x2: (4, _M32, "w"), 0x3: (8, _M64, "d")}


def _decode(instr):
    """``(size, mask, suffix, funct5, rs1, rs2, rd)``, or None if not 0x2F .w/.d."""
    if instr & 0x7F != 0x2F:
        return None
    width = _WIDTHS.get((instr >> 12) & 0x7)
    if width is None:
        return None
    size, mask, suffix = width
    return (
        size,
        mask,
        suffix,
        instr >> 27,
        (instr >> 15) & 0x1F,
        (instr >> 20) & 0x1F,
        (instr >> 7) & 0x1F,
    )


def _loaded(data, size):
    """The value rd receives from a ``size``-byte atomic load."""
    value = int.from_bytes(data, "little")
    return sext32(value) if size == 4 else value


class RV64_ZAAMO_ISA(RV_ISA):
    #: Major opcodes (``instr & 0x7F``) this ISA may claim; ``RV64I``'s dispatch table offers it no others.
    OPCODES = frozenset({0x2F})

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        decoded = _decode(instr)
        if decoded is None:
            return False
        size, mask, suffix, funct5, rs1, rs2, rd = decoded
        op = _OPS.get(funct5)
        if op is None:  # lr / sc belong to Zalrsc
            return False
        name, fn_w, fn_d = op
        addr = register_file[rs1].read_uint()
        data = memory_space.read(addr, size)
        if data is MemoryStall:
            return ProcessingElement.PEStall
        old = int.from_bytes(data, "little")
        src = register_file[rs2].read_uint() & mask
        new = (fn_w if size == 4 else fn_d)(old, src) & mask
        if memory_space.write(addr, new.to_bytes(size, "little")) is MemoryStall:
            return ProcessingElement.PEStall
        store_clears_reservation(register_file, addr, size)
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name}.{suffix} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs2)}, "
                f"({cls.get_reg_name(rs1)})",
                f"{cls.get_reg_name(rd)} = [{hex(addr)}] (={hex(old)}); "
                f"[{hex(addr)}] = {name}(old, {cls.get_reg_name(rs2)})",
            )
        write_reg(register_file, rd, _loaded(data, size))
        return True


class RV64_ZALRSC_ISA(RV_ISA):
    #: Major opcodes (``instr & 0x7F``) this ISA may claim; ``RV64I``'s dispatch table offers it no others.
    OPCODES = frozenset({0x2F})

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        decoded = _decode(instr)
        if decoded is None:
            return False
        size, _mask, suffix, funct5, rs1, rs2, rd = decoded
        addr = register_file[rs1].read_uint()

        if funct5 == 0x02:
            if rs2 != 0:  # lr's rs2 field is reserved as zero
                return False
            data = memory_space.read(addr, size)
            if data is MemoryStall:
                return ProcessingElement.PEStall
            register_file.reservation = (addr, addr + size)
            write_reg(register_file, rd, _loaded(data, size))
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"lr.{suffix} {cls.get_reg_name(rd)}, ({cls.get_reg_name(rs1)})",
                    f"{cls.get_reg_name(rd)} = [{hex(addr)}]; reserve",
                )
            return True

        if funct5 == 0x03:
            res = register_file.reservation
            ok = res is not None and res[0] <= addr and addr + size <= res[1]
            if ok:
                value = register_file[rs2].read()[:size]
                if memory_space.write(addr, value) is MemoryStall:
                    return ProcessingElement.PEStall
            register_file.reservation = None
            write_reg(register_file, rd, 0 if ok else 1)
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"sc.{suffix} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs2)}, "
                    f"({cls.get_reg_name(rs1)})",
                    f"[{hex(addr)}] = {cls.get_reg_name(rs2)}" if ok else "failed",
                )
            return True
        return False
