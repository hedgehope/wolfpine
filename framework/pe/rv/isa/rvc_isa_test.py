"""Tests for the compressed-ISA expansion (``rvc_isa.expand``): every opcode, the reserved encodings and the HINTs.

The table below is three parcels per mnemonic (first, middle and last of its
encodings), with the 32-bit instruction LLVM's assembler produces for the
expansion. ``test_every_parcel_matches_llvm`` repeats that for all 49,152
parcels when ``llvm-mc`` is installed.
"""

import re
import subprocess
from pathlib import Path

import pytest

from framework.pe.rv.isa.rvc_isa import C_JALR_MASK, C_JALR_MATCH, expand

LLVM_MC = Path("/opt/llvm/20.1.7/bin/llvm-mc")

#: (parcel, expansion): llvm-mc --disassemble -M no-aliases, then the expansion
#: assembled with -mattr=-c.
EXPANSIONS = [
    (0x9006, 0x100033),  # c.add zero, ra -> add zero, zero, ra
    (0x9806, 0x180833),  # c.add a6, ra -> add a6, a6, ra
    (0x9FFE, 0x1FF8FB3),  # c.add t6, t6 -> add t6, t6, t6
    (0x0081, 0x8093),  # c.addi ra, 0 -> addi ra, ra, 0
    (0x1081, 0xFE008093),  # c.addi ra, -32 -> addi ra, ra, -32
    (0x1FFD, 0xFFFF8F93),  # c.addi t6, -1 -> addi t6, t6, -1
    (0x6105, 0x2010113),  # c.addi16sp sp, 32 -> addi sp, sp, 32
    (0x7101, 0xE0010113),  # c.addi16sp sp, -512 -> addi sp, sp, -512
    (0x717D, 0xFF010113),  # c.addi16sp sp, -16 -> addi sp, sp, -16
    (0x0020, 0x810413),  # c.addi4spn s0, sp, 8 -> addi s0, sp, 8
    (0x1010, 0x2010613),  # c.addi4spn a2, sp, 32 -> addi a2, sp, 32
    (0x1FFC, 0x3FC10793),  # c.addi4spn a5, sp, 1020 -> addi a5, sp, 1020
    (0x2081, 0x809B),  # c.addiw ra, 0 -> addiw ra, ra, 0
    (0x3081, 0xFE00809B),  # c.addiw ra, -32 -> addiw ra, ra, -32
    (0x3FFD, 0xFFFF8F9B),  # c.addiw t6, -1 -> addiw t6, t6, -1
    (0x9C21, 0x84043B),  # c.addw s0, s0 -> addw s0, s0, s0
    (0x9E21, 0x86063B),  # c.addw a2, s0 -> addw a2, a2, s0
    (0x9FBD, 0xF787BB),  # c.addw a5, a5 -> addw a5, a5, a5
    (0x8C61, 0x847433),  # c.and s0, s0 -> and s0, s0, s0
    (0x8E61, 0x867633),  # c.and a2, s0 -> and a2, a2, s0
    (0x8FFD, 0xF7F7B3),  # c.and a5, a5 -> and a5, a5, a5
    (0x8801, 0x47413),  # c.andi s0, 0 -> andi s0, s0, 0
    (0x9801, 0xFE047413),  # c.andi s0, -32 -> andi s0, s0, -32
    (0x9BFD, 0xFFF7F793),  # c.andi a5, -1 -> andi a5, a5, -1
    (0xC001, 0x40063),  # c.beqz s0, 0 -> beq s0, zero, 0
    (0xD001, 0xF00400E3),  # c.beqz s0, -256 -> beq s0, zero, -256
    (0xDFFD, 0xFE078FE3),  # c.beqz a5, -2 -> beq a5, zero, -2
    (0xE001, 0x41063),  # c.bnez s0, 0 -> bne s0, zero, 0
    (0xF001, 0xF00410E3),  # c.bnez s0, -256 -> bne s0, zero, -256
    (0xFFFD, 0xFE079FE3),  # c.bnez a5, -2 -> bne a5, zero, -2
    (0x9002, 0x100073),  # c.ebreak -> ebreak
    (0x2000, 0x43407),  # c.fld fs0, 0(s0) -> fld fs0, 0(s0)
    (0x3000, 0x2043407),  # c.fld fs0, 32(s0) -> fld fs0, 32(s0)
    (0x3FFC, 0xF87B787),  # c.fld fa5, 248(a5) -> fld fa5, 248(a5)
    (0x2002, 0x13007),  # c.fldsp ft0, 0(sp) -> fld ft0, 0(sp)
    (0x3002, 0x2013007),  # c.fldsp ft0, 32(sp) -> fld ft0, 32(sp)
    (0x3FFE, 0x1F813F87),  # c.fldsp ft11, 504(sp) -> fld ft11, 504(sp)
    (0xA000, 0x843027),  # c.fsd fs0, 0(s0) -> fsd fs0, 0(s0)
    (0xB000, 0x2843027),  # c.fsd fs0, 32(s0) -> fsd fs0, 32(s0)
    (0xBFFC, 0xEF7BC27),  # c.fsd fa5, 248(a5) -> fsd fa5, 248(a5)
    (0xA002, 0x13027),  # c.fsdsp ft0, 0(sp) -> fsd ft0, 0(sp)
    (0xB002, 0x2013027),  # c.fsdsp ft0, 32(sp) -> fsd ft0, 32(sp)
    (0xBFFE, 0x1FF13C27),  # c.fsdsp ft11, 504(sp) -> fsd ft11, 504(sp)
    (0xA001, 0x6F),  # c.j 0 -> jal zero, 0
    (0xB001, 0x801FF06F),  # c.j -2048 -> jal zero, -2048
    (0xBFFD, 0xFFFFF06F),  # c.j -2 -> jal zero, -2
    (0x9082, 0x80E7),  # c.jalr ra -> jalr ra, 0(ra)
    (0x9802, 0x800E7),  # c.jalr a6 -> jalr ra, 0(a6)
    (0x9F82, 0xF80E7),  # c.jalr t6 -> jalr ra, 0(t6)
    (0x8082, 0x8067),  # c.jr ra -> jalr zero, 0(ra)
    (0x8802, 0x80067),  # c.jr a6 -> jalr zero, 0(a6)
    (0x8F82, 0xF8067),  # c.jr t6 -> jalr zero, 0(t6)
    (0x6000, 0x43403),  # c.ld s0, 0(s0) -> ld s0, 0(s0)
    (0x7000, 0x2043403),  # c.ld s0, 32(s0) -> ld s0, 32(s0)
    (0x7FFC, 0xF87B783),  # c.ld a5, 248(a5) -> ld a5, 248(a5)
    (0x6082, 0x13083),  # c.ldsp ra, 0(sp) -> ld ra, 0(sp)
    (0x7082, 0x2013083),  # c.ldsp ra, 32(sp) -> ld ra, 32(sp)
    (0x7FFE, 0x1F813F83),  # c.ldsp t6, 504(sp) -> ld t6, 504(sp)
    (0x4001, 0x13),  # c.li zero, 0 -> addi zero, zero, 0
    (0x5001, 0xFE000013),  # c.li zero, -32 -> addi zero, zero, -32
    (0x5FFD, 0xFFF00F93),  # c.li t6, -1 -> addi t6, zero, -1
    (0x7001, 0xFFFE0037),  # c.lui zero, -32 -> lui zero, 1048544
    (0x7FFD, 0xFFFFFFB7),  # c.lui t6, 1048575 -> lui t6, 1048575
    (0x4000, 0x42403),  # c.lw s0, 0(s0) -> lw s0, 0(s0)
    (0x5000, 0x2042403),  # c.lw s0, 32(s0) -> lw s0, 32(s0)
    (0x5FFC, 0x7C7A783),  # c.lw a5, 124(a5) -> lw a5, 124(a5)
    (0x4082, 0x12083),  # c.lwsp ra, 0(sp) -> lw ra, 0(sp)
    (0x5082, 0x2012083),  # c.lwsp ra, 32(sp) -> lw ra, 32(sp)
    (0x5FFE, 0xFC12F83),  # c.lwsp t6, 252(sp) -> lw t6, 252(sp)
    (0x8006, 0x100033),  # c.mv zero, ra -> add zero, zero, ra
    (0x8806, 0x100833),  # c.mv a6, ra -> add a6, zero, ra
    (0x8FFE, 0x1F00FB3),  # c.mv t6, t6 -> add t6, zero, t6
    (0x0001, 0x13),  # c.nop -> addi zero, zero, 0
    (0x1001, 0xFE000013),  # c.nop -32 -> addi zero, zero, -32
    (0x107D, 0xFFF00013),  # c.nop -1 -> addi zero, zero, -1
    (0x8C41, 0x846433),  # c.or s0, s0 -> or s0, s0, s0
    (0x8E41, 0x866633),  # c.or a2, s0 -> or a2, a2, s0
    (0x8FDD, 0xF7E7B3),  # c.or a5, a5 -> or a5, a5, a5
    (0xE000, 0x843023),  # c.sd s0, 0(s0) -> sd s0, 0(s0)
    (0xF000, 0x2843023),  # c.sd s0, 32(s0) -> sd s0, 32(s0)
    (0xFFFC, 0xEF7BC23),  # c.sd a5, 248(a5) -> sd a5, 248(a5)
    (0xE002, 0x13023),  # c.sdsp zero, 0(sp) -> sd zero, 0(sp)
    (0xF002, 0x2013023),  # c.sdsp zero, 32(sp) -> sd zero, 32(sp)
    (0xFFFE, 0x1FF13C23),  # c.sdsp t6, 504(sp) -> sd t6, 504(sp)
    (0x0006, 0x101013),  # c.slli zero, 1 -> slli zero, zero, 1
    (0x1042, 0x3001013),  # c.slli zero, 48 -> slli zero, zero, 48
    (0x1FFE, 0x3FF9F93),  # c.slli t6, 63 -> slli t6, t6, 63
    (0x0002, 0x1013),  # c.slli64 zero -> slli zero, zero, 0
    (0x0802, 0x81813),  # c.slli64 a6 -> slli a6, a6, 0
    (0x0F82, 0xF9F93),  # c.slli64 t6 -> slli t6, t6, 0
    (0x8405, 0x40145413),  # c.srai s0, 1 -> srai s0, s0, 1
    (0x9411, 0x42445413),  # c.srai s0, 36 -> srai s0, s0, 36
    (0x97FD, 0x43F7D793),  # c.srai a5, 63 -> srai a5, a5, 63
    (0x8401, 0x40045413),  # c.srai64 s0 -> srai s0, s0, 0
    (0x8601, 0x40065613),  # c.srai64 a2 -> srai a2, a2, 0
    (0x8781, 0x4007D793),  # c.srai64 a5 -> srai a5, a5, 0
    (0x8005, 0x145413),  # c.srli s0, 1 -> srli s0, s0, 1
    (0x9011, 0x2445413),  # c.srli s0, 36 -> srli s0, s0, 36
    (0x93FD, 0x3F7D793),  # c.srli a5, 63 -> srli a5, a5, 63
    (0x8001, 0x45413),  # c.srli64 s0 -> srli s0, s0, 0
    (0x8201, 0x65613),  # c.srli64 a2 -> srli a2, a2, 0
    (0x8381, 0x7D793),  # c.srli64 a5 -> srli a5, a5, 0
    (0x8C01, 0x40840433),  # c.sub s0, s0 -> sub s0, s0, s0
    (0x8E01, 0x40860633),  # c.sub a2, s0 -> sub a2, a2, s0
    (0x8F9D, 0x40F787B3),  # c.sub a5, a5 -> sub a5, a5, a5
    (0x9C01, 0x4084043B),  # c.subw s0, s0 -> subw s0, s0, s0
    (0x9E01, 0x4086063B),  # c.subw a2, s0 -> subw a2, a2, s0
    (0x9F9D, 0x40F787BB),  # c.subw a5, a5 -> subw a5, a5, a5
    (0xC000, 0x842023),  # c.sw s0, 0(s0) -> sw s0, 0(s0)
    (0xD000, 0x2842023),  # c.sw s0, 32(s0) -> sw s0, 32(s0)
    (0xDFFC, 0x6F7AE23),  # c.sw a5, 124(a5) -> sw a5, 124(a5)
    (0xC002, 0x12023),  # c.swsp zero, 0(sp) -> sw zero, 0(sp)
    (0xD002, 0x2012023),  # c.swsp zero, 32(sp) -> sw zero, 32(sp)
    (0xDFFE, 0xFF12E23),  # c.swsp t6, 252(sp) -> sw t6, 252(sp)
    (0x8C21, 0x844433),  # c.xor s0, s0 -> xor s0, s0, s0
    (0x8E21, 0x864633),  # c.xor a2, s0 -> xor a2, a2, s0
    (0x8FBD, 0xF7C7B3),  # c.xor a5, a5 -> xor a5, a5, a5
]

#: Reserved encodings: illegal instructions, so ``expand`` gives ``None``.
RESERVED = [
    (0x0000, "the all-zero parcel (c.addi4spn, nzuimm = 0)"),
    (0x0004, "c.addi4spn a1, sp, 0"),
    (0x001C, "c.addi4spn a5, sp, 0"),
    (0x8000, "quadrant 0, funct3 = 4 (Zcb)"),
    (0x9FFC, "quadrant 0, funct3 = 4 (Zcb)"),
    (0x2001, "c.addiw x0, 0"),
    (0x3005, "c.addiw x0, -31"),
    (0x6101, "c.addi16sp sp, 0"),
    (0x6001, "c.lui x0, 0"),
    (0x6501, "c.lui a0, 0"),
    (0x9C41, "quadrant 1 row 100 1 11 10 (Zcb c.mul)"),
    (0x9C61, "quadrant 1 row 100 1 11 11 (Zcb)"),
    (0x4002, "c.lwsp x0"),
    (0x6002, "c.ldsp x0"),
    (0x8002, "c.jr x0"),
]


@pytest.mark.parametrize(
    "parcel,word", EXPANSIONS, ids=[f"{p:#06x}" for p, _ in EXPANSIONS]
)
def test_expansion(parcel, word):
    assert expand(parcel) == word


@pytest.mark.parametrize(
    "parcel,what", RESERVED, ids=[f"{p:#06x}" for p, _ in RESERVED]
)
def test_reserved_encodings_are_illegal(parcel, what):
    assert expand(parcel) is None, what


def test_zcd_forms_need_zcd():
    for parcel in (0x2000, 0x3FFE, 0xA000, 0xBFFE):  # c.fld, c.fldsp, c.fsd, c.fsdsp
        assert expand(parcel) is not None
        assert expand(parcel, zcd=False) is None
    # The integer doubleword forms are Zca.
    for parcel in (0x6000, 0x7FFE, 0xE000, 0xFFFE):
        assert expand(parcel, zcd=False) == expand(parcel)


def test_hints_expand_to_no_ops():
    nop_like = {
        0x0001: 0x00000013,  # c.nop
        0x107D: 0xFFF00013,  # c.nop -1 (HINT): addi x0, x0, -1
        0x0081: 0x00008093,  # c.addi ra, 0 (HINT): addi ra, ra, 0
        0x4005: 0x00100013,  # c.li x0, 1 (HINT)
        0x6005: 0x00001037,  # c.lui x0, 1 (HINT)
        0x8006: 0x00100033,  # c.mv x0, ra (HINT)
        0x9006: 0x00100033,  # c.add x0, ra (HINT)
        0x0006: 0x00101013,  # c.slli x0, 1 (HINT)
        0x0802: 0x00081813,  # c.slli a6, 0 (HINT)
        0x8001: 0x00045413,  # c.srli s0, 0 (HINT)
        0x8401: 0x40045413,  # c.srai s0, 0 (HINT)
    }
    for parcel, word in nop_like.items():
        assert expand(parcel) == word, hex(parcel)
        rd = (word >> 7) & 0x1F
        rs1 = (word >> 15) & 0x1F
        imm = word >> 20
        # Either rd is x0, or the op is rd = rd (+ 0 | << 0 | >> 0).
        assert rd == 0 or (rs1 == rd and imm & 0x3F == 0), hex(parcel)


def test_32_bit_parcel_is_refused():
    with pytest.raises(ValueError, match="not a compressed instruction"):
        expand(0x0013)


def test_every_expansion_is_a_32_bit_word():
    for parcel in range(0x10000):
        if parcel & 3 == 3:
            continue
        word = expand(parcel)
        if word is not None:
            assert word & 3 == 3, hex(parcel)
            assert word >> 32 == 0, hex(parcel)


def test_c_jalr_match_picks_out_c_jalr_alone():
    jalrs = [
        p
        for p in range(0x10000)
        if p & 3 != 3
        and (w := expand(p)) is not None
        and w & C_JALR_MASK == C_JALR_MATCH
    ]
    assert jalrs == [0x9002 | (rs1 << 7) for rs1 in range(1, 32)]


@pytest.mark.skipif(not LLVM_MC.is_file(), reason="needs llvm-mc 20.1.7")
def test_every_parcel_matches_llvm():
    """LLVM is the oracle for all 49,152 parcels, except ``c.lui rd, 0``.

    LLVM decodes ``c.lui`` with nzimm = 0 as a valid instruction; the spec
    reserves it, and Spike traps on it.
    """
    parcels = [h for h in range(0x10000) if h & 3 != 3]
    text = "".join(f"0x{h & 0xFF:02x} 0x{h >> 8:02x}\n" for h in parcels)
    dis = subprocess.run(
        [
            str(LLVM_MC),
            "--disassemble",
            "-triple=riscv64",
            "-mattr=+c,+d",
            "-M",
            "no-aliases",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    bad = {
        int(m.group(1)) - 1
        for m in re.finditer(r"<stdin>:(\d+):\d+: warning", dis.stderr)
    }
    valid = [h for i, h in enumerate(parcels) if i not in bad]
    lines = [
        ln.strip()
        for ln in dis.stdout.splitlines()
        if ln.strip() and not ln.strip().startswith(".")
    ]
    assert len(lines) == len(valid)
    asm = [_uncompressed(line) for line in lines]
    enc = subprocess.run(
        [
            str(LLVM_MC),
            "-triple=riscv64",
            "-mattr=+d,-c",
            "-show-encoding",
            "-M",
            "no-aliases",
        ],
        input="".join((a or "nop") + "\n" for a in asm),
        capture_output=True,
        text=True,
        check=True,
    )
    words = [
        int.from_bytes(bytes(int(b, 16) for b in m.group(1).split(",")), "little")
        for m in re.finditer(r"encoding: \[([^\]]+)\]", enc.stdout)
    ]
    assert len(words) == len(asm)
    llvm = dict.fromkeys(parcels)
    for parcel, a, word in zip(valid, asm, words):
        llvm[parcel] = None if a is None else word
    mismatches = [
        (hex(p), lines[valid.index(p)] if p in valid else "invalid")
        for p in parcels
        if expand(p) != llvm[p]
        and not (expand(p) is None and llvm[p] & 0xFFFFF07F == 0x37)
    ]
    assert mismatches == []
    reserved_lui = [p for p in parcels if expand(p) is None and llvm[p] is not None]
    assert len(reserved_lui) == 31  # c.lui rd, 0 for rd != x2 (x0 included)


def _uncompressed(line):
    """The 32-bit assembly for one ``llvm-mc -M no-aliases`` compressed line (None for c.unimp)."""
    op, _, args = line.partition("\t")
    a = [x.strip() for x in args.split(",")] if args else []
    m = op[2:]
    if m in ("lw", "ld", "fld", "sw", "sd", "fsd"):
        return f"{m} {a[0]}, {a[1]}"
    if m in ("lwsp", "ldsp", "fldsp", "swsp", "sdsp", "fsdsp"):
        return f"{m[:-2]} {a[0]}, {a[1]}"
    if m in ("addi", "addiw", "andi", "srli", "srai", "slli"):
        return f"{m} {a[0]}, {a[0]}, {a[1]}"
    if m in ("sub", "xor", "or", "and", "subw", "addw", "add"):
        return f"{m} {a[0]}, {a[0]}, {a[1]}"
    simple = {
        "addi4spn": lambda: f"addi {a[0]}, sp, {a[2]}",
        "nop": lambda: f"addi zero, zero, {a[0] if a else 0}",
        "li": lambda: f"addi {a[0]}, zero, {a[1]}",
        "addi16sp": lambda: f"addi sp, sp, {a[1]}",
        "lui": lambda: f"lui {a[0]}, {int(a[1]) % (1 << 20)}",
        "mv": lambda: f"add {a[0]}, zero, {a[1]}",
        "j": lambda: f"jal zero, {a[0]}",
        "beqz": lambda: f"beq {a[0]}, zero, {a[1]}",
        "bnez": lambda: f"bne {a[0]}, zero, {a[1]}",
        "jr": lambda: f"jalr zero, 0({a[0]})",
        "jalr": lambda: f"jalr ra, 0({a[0]})",
        "ebreak": lambda: "ebreak",
        "slli64": lambda: f"slli {a[0]}, {a[0]}, 0",
        "srli64": lambda: f"srli {a[0]}, {a[0]}, 0",
        "srai64": lambda: f"srai {a[0]}, {a[0]}, 0",
        "unimp": lambda: None,
    }
    return simple[m]()
