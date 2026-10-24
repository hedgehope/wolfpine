"""The compressed ISA (Zca + Zcd) for the RV64 core: 16-bit encodings expanded to 32-bit ones.

**Naiad's RV64 hart only.** The Tenstorrent baby cores use the compressed
opcode space for ``.ttinsn`` (``tt_isa.py``), so nothing here is ever attached
to an RV32 core.

A compressed instruction is executed as the 32-bit instruction it stands for
(the unprivileged spec's "Zc*" expansion tables): :func:`expand` maps a 16-bit
parcel to that word, and the core dispatches the word to the ordinary RV64 ISAs.
Executing the expansion keeps one implementation of every operation, so the
compressed forms cannot drift from the uncompressed ones. What the expansion
cannot express the core does itself (see ``RV64I.clock_tick``): the next PC is
``pc + 2``, and ``c.jalr`` links ``pc + 2`` rather than the ``pc + 4`` a 32-bit
``jalr`` writes. (``c.jal`` is RV32-only; on RV64 its encoding is ``c.addiw``.)

``RV64_ZCA_ISA`` and ``RV64_ZCD_ISA`` claim no 32-bit encoding (``OPCODES`` is
empty); listing them in a core's extensions is what turns on 16-bit fetch and
expansion. Zcd's four forms (``c.fld``, ``c.fsd``, ``c.fldsp``, ``c.fsdsp``)
expand only when Zcd is present, and execute only when D is too.

Reserved encodings (the all-zero parcel, ``c.addi4spn`` with a zero immediate,
``c.addiw`` / ``c.lwsp`` / ``c.ldsp`` with rd = x0, ``c.jr`` with rs1 = x0,
``c.lui`` / ``c.addi16sp`` with a zero immediate, quadrant 0's funct3 = 4, and
quadrant 1's ``11 1 1x`` row) expand to ``None``: an illegal instruction, which
the core reports as unknown. HINTs (``c.nop`` with an immediate, ``c.addi`` with
a zero immediate, ``c.li`` / ``c.lui`` / ``c.mv`` / ``c.add`` / ``c.slli`` with
rd = x0, zero-shift ``c.slli`` / ``c.srli`` / ``c.srai``) expand to the
instruction they encode, which writes x0 or leaves rd unchanged: a no-op, as the
spec requires.
"""

from framework.pe.rv.isa.rv_isa import RV_ISA

_LOAD, _STORE, _LOAD_FP, _STORE_FP = 0x03, 0x23, 0x07, 0x27
_OP_IMM, _OP_IMM_32, _OP, _OP_32 = 0x13, 0x1B, 0x33, 0x3B
_LUI, _BRANCH, _JALR, _JAL = 0x37, 0x63, 0x67, 0x6F
_EBREAK = 0x00100073


def _i(op, rd, f3, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def _s(op, f3, rs1, rs2, imm):
    return (
        (((imm >> 5) & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | ((imm & 0x1F) << 7)
        | op
    )


def _r(op, rd, f3, rs1, rs2, f7=0):
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def _b(f3, rs1, rs2, imm):
    return (
        (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 1) << 7)
        | _BRANCH
    )


def _j(rd, imm):
    return (
        (((imm >> 20) & 1) << 31)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 1) << 20)
        | (((imm >> 12) & 0xFF) << 12)
        | (rd << 7)
        | _JAL
    )


def _sext(value, bits):
    return value - (1 << bits) if value >> (bits - 1) else value


def _bit(h, n):
    return (h >> n) & 1


def _q0(h, zcd):
    f3 = h >> 13
    rd_ = 8 + ((h >> 2) & 7)  # rd' / rs2'
    rs1_ = 8 + ((h >> 7) & 7)
    # uimm[5:3] in bits 12:10 for every load/store here; the low bits differ.
    uimm_d = (((h >> 10) & 7) << 3) | (
        ((h >> 5) & 3) << 6
    )  # c.ld / c.sd / c.fld / c.fsd
    if f3 == 0:  # c.addi4spn: nzuimm[5:4|9:6|2|3]
        nzuimm = (
            (((h >> 11) & 3) << 4)
            | (((h >> 7) & 0xF) << 6)
            | (_bit(h, 6) << 2)
            | (_bit(h, 5) << 3)
        )
        if nzuimm == 0:
            return None  # reserved (includes the all-zero parcel)
        return _i(_OP_IMM, rd_, 0, 2, nzuimm)
    if f3 == 1:  # c.fld
        return _i(_LOAD_FP, rd_, 3, rs1_, uimm_d) if zcd else None
    if f3 == 2:  # c.lw: uimm[5:3|2|6]
        uimm = (((h >> 10) & 7) << 3) | (_bit(h, 6) << 2) | (_bit(h, 5) << 6)
        return _i(_LOAD, rd_, 2, rs1_, uimm)
    if f3 == 3:  # c.ld
        return _i(_LOAD, rd_, 3, rs1_, uimm_d)
    if f3 == 4:
        return None  # reserved (Zcb)
    if f3 == 5:  # c.fsd
        return _s(_STORE_FP, 3, rs1_, rd_, uimm_d) if zcd else None
    if f3 == 6:  # c.sw
        uimm = (((h >> 10) & 7) << 3) | (_bit(h, 6) << 2) | (_bit(h, 5) << 6)
        return _s(_STORE, 2, rs1_, rd_, uimm)
    return _s(_STORE, 3, rs1_, rd_, uimm_d)  # c.sd


def _q1(h):
    f3 = h >> 13
    rd = (h >> 7) & 0x1F
    imm6 = _sext((_bit(h, 12) << 5) | ((h >> 2) & 0x1F), 6)
    if f3 == 0:  # c.addi (c.nop, HINTs)
        return _i(_OP_IMM, rd, 0, rd, imm6)
    if f3 == 1:  # c.addiw (RV64; c.jal is RV32-only)
        return None if rd == 0 else _i(_OP_IMM_32, rd, 0, rd, imm6)
    if f3 == 2:  # c.li
        return _i(_OP_IMM, rd, 0, 0, imm6)
    if f3 == 3:
        if rd == 2:  # c.addi16sp: nzimm[9|4|6|8:7|5]
            nzimm = _sext(
                (_bit(h, 12) << 9)
                | (_bit(h, 6) << 4)
                | (_bit(h, 5) << 6)
                | (((h >> 3) & 3) << 7)
                | (_bit(h, 2) << 5),
                10,
            )
            return None if nzimm == 0 else _i(_OP_IMM, 2, 0, 2, nzimm)
        # c.lui: nzimm[17|16:12]
        if imm6 == 0:
            return None
        return ((imm6 & 0xFFFFF) << 12) | (rd << 7) | _LUI
    if f3 == 4:
        rd_ = 8 + ((h >> 7) & 7)
        f2 = (h >> 10) & 3
        shamt = (_bit(h, 12) << 5) | ((h >> 2) & 0x1F)
        if f2 == 0:  # c.srli
            return _i(_OP_IMM, rd_, 5, rd_, shamt)
        if f2 == 1:  # c.srai
            return _i(_OP_IMM, rd_, 5, rd_, 0x400 | shamt)
        if f2 == 2:  # c.andi
            return _i(_OP_IMM, rd_, 7, rd_, imm6)
        rs2_ = 8 + ((h >> 2) & 7)
        sel = (h >> 5) & 3
        if not _bit(h, 12):
            # c.sub / c.xor / c.or / c.and
            f3_, f7 = ((0, 0x20), (4, 0), (6, 0), (7, 0))[sel]
            return _r(_OP, rd_, f3_, rd_, rs2_, f7)
        if sel == 0:  # c.subw
            return _r(_OP_32, rd_, 0, rd_, rs2_, 0x20)
        if sel == 1:  # c.addw
            return _r(_OP_32, rd_, 0, rd_, rs2_)
        return None  # reserved (Zcb's c.mul and friends)
    if f3 == 5:  # c.j: offset[11|4|9:8|10|6|7|3:1|5]
        off = _sext(
            (_bit(h, 12) << 11)
            | (_bit(h, 11) << 4)
            | (((h >> 9) & 3) << 8)
            | (_bit(h, 8) << 10)
            | (_bit(h, 7) << 6)
            | (_bit(h, 6) << 7)
            | (((h >> 3) & 7) << 1)
            | (_bit(h, 2) << 5),
            12,
        )
        return _j(0, off)
    # c.beqz / c.bnez: offset[8|4:3] in 12:10, [7:6|2:1|5] in 6:2
    rs1_ = 8 + ((h >> 7) & 7)
    off = _sext(
        (_bit(h, 12) << 8)
        | (((h >> 10) & 3) << 3)
        | (((h >> 5) & 3) << 6)
        | (((h >> 3) & 3) << 1)
        | (_bit(h, 2) << 5),
        9,
    )
    return _b(0 if f3 == 6 else 1, rs1_, 0, off)


def _q2(h, zcd):
    f3 = h >> 13
    rd = (h >> 7) & 0x1F  # rd / rs1
    rs2 = (h >> 2) & 0x1F
    if f3 == 0:  # c.slli (HINT for rd = x0 or shamt = 0)
        return _i(_OP_IMM, rd, 1, rd, (_bit(h, 12) << 5) | rs2)
    # The sp-relative doubleword loads: uimm[5] in 12, [4:3|8:6] in 6:2.
    uimm_d = (_bit(h, 12) << 5) | (((h >> 5) & 3) << 3) | (((h >> 2) & 7) << 6)
    if f3 == 1:  # c.fldsp
        return _i(_LOAD_FP, rd, 3, 2, uimm_d) if zcd else None
    if f3 == 2:  # c.lwsp: uimm[5] in 12, [4:2|7:6] in 6:2
        if rd == 0:
            return None
        uimm = (_bit(h, 12) << 5) | (((h >> 4) & 7) << 2) | (((h >> 2) & 3) << 6)
        return _i(_LOAD, rd, 2, 2, uimm)
    if f3 == 3:  # c.ldsp
        return None if rd == 0 else _i(_LOAD, rd, 3, 2, uimm_d)
    if f3 == 4:
        if not _bit(h, 12):
            if rs2 == 0:  # c.jr
                return None if rd == 0 else _i(_JALR, 0, 0, rd, 0)
            return _r(_OP, rd, 0, 0, rs2)  # c.mv
        if rs2 == 0:
            if rd == 0:
                return _EBREAK  # c.ebreak
            return _i(_JALR, 1, 0, rd, 0)  # c.jalr (the core links pc + 2)
        return _r(_OP, rd, 0, rd, rs2)  # c.add
    # The sp-relative stores: uimm in 12:7.
    if f3 == 6:  # c.swsp: uimm[5:2|7:6]
        uimm = (((h >> 9) & 0xF) << 2) | (((h >> 7) & 3) << 6)
        return _s(_STORE, 2, 2, rs2, uimm)
    uimm = (((h >> 10) & 7) << 3) | (((h >> 7) & 7) << 6)  # [5:3|8:6]
    if f3 == 5:  # c.fsdsp
        return _s(_STORE_FP, 3, 2, rs2, uimm) if zcd else None
    return _s(_STORE, 3, 2, rs2, uimm)  # c.sdsp


def expand(parcel, zcd=True):
    """The 32-bit RV64 instruction a 16-bit compressed ``parcel`` stands for, or ``None``.

    ``None`` means a reserved (illegal) encoding, or a Zcd form when ``zcd`` is
    false. ``parcel``'s low two bits must not be ``0b11`` (that is a 32-bit
    instruction); only its low 16 bits are read.
    """
    h = parcel & 0xFFFF
    quadrant = h & 3
    if quadrant == 0:
        return _q0(h, zcd)
    if quadrant == 1:
        return _q1(h)
    if quadrant == 2:
        return _q2(h, zcd)
    raise ValueError(f"{h:#06x} is not a compressed instruction (low bits are 0b11)")


#: ``jalr x1, 0(rs1)`` with rs1 masked out: the expansion of ``c.jalr``, which
#: must link ``pc + 2``. Compare ``word & C_JALR_MASK == C_JALR_MATCH``.
C_JALR_MASK = 0xFFF07FFF
C_JALR_MATCH = (1 << 7) | _JALR


class RV64_ZCA_ISA(RV_ISA):
    """Zca: the compressed integer forms. Claims no 32-bit encoding; see the module docstring."""

    OPCODES = frozenset()

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        return False


class RV64_ZCD_ISA(RV_ISA):
    """Zcd: the compressed double-precision loads and stores (needs Zca and D)."""

    OPCODES = frozenset()

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        return False
