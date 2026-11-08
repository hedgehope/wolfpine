"""Ztt v0.6 definitions: the imported encodings, the decoder, datatype descriptors, the design section.

The expected words and descriptors are the draft's own (``INSTRUCTION_ENCODINGS.md`` and the
examples in ``src/examples.adoc`` / ``src/datatypes.adoc``), written out here so a re-import
that changed them would fail.
"""

import random

import pytest

from framework.matrix import ztt
from framework.matrix.ztt import (
    TABLE,
    ZttSpec,
    classify,
    decode,
    descriptor,
    encode,
    parse_ztt,
)
from tools.naiad import ztt_import


class _Error(Exception):
    pass


class _Cores:
    xlen = 64


def _parse(**raw):
    return parse_ztt({"mode": "ztt", **raw}, _Cores(), _Error)


def test_the_table_is_v06_with_all_138_instructions():
    assert ztt.SOURCE["tag"] == "v0.6"
    assert ztt.SOURCE["commit"].startswith("8d27acd")
    assert len(TABLE) == 138
    assert all(i.match & 0x707F == 0x2B for i in TABLE.values())  # custom-1, funct3 000


@pytest.mark.parametrize(
    "name, match",
    [
        ("ame.acquire", 0xA490002B),
        ("ame.release", 0xA4A0002B),
        ("msettyp", 0xA230002B),
        ("mls.rm", 0xA380002B),
        ("mss.tst", 0xAE00002B),
        ("mmulacc.2d", 0x9400002B),
        ("mmulbtaccneg.2d", 0xA000002B),
        ("mzero.2d.m", 0xA600802B),
        ("mmov.m.a", 0xA3C0002B),
        ("mabsdiff.ew", 0x0000002B),
    ],
)
def test_match_words_are_the_drafts(name, match):
    assert TABLE[name].match == match


def test_every_instruction_decodes_to_itself_with_any_operands():
    rnd = random.Random(7)
    for insn in TABLE.values():
        for _ in range(20):
            ops = {
                op: rnd.randrange(1 << (hi - lo + 1))
                for op, (lo, hi) in insn.operands.items()
            }
            word = encode(insn.name, **ops)
            assert decode(word) is insn
            assert {op: insn.field(word, op) for op in insn.operands} == ops


def test_words_outside_the_table_decode_to_nothing():
    assert decode(0x0000102B) is None  # funct3 = 1
    assert decode(0x9200002B) is None  # funct7 0x49: reserved
    assert (
        decode(TABLE["ame.release"].match | (1 << 7)) is None
    )  # a fixed field of ame.release
    assert decode(0xC800002B) is None  # funct7 0x64: free


def test_encode_refuses_bad_operands():
    with pytest.raises(ValueError, match="does not fit"):
        encode("mls.rm", md=32)
    with pytest.raises(ValueError, match="unknown operand"):
        encode("mls.rm", ms2=1)


def test_the_importer_reads_the_table_format_and_finds_overlaps():
    row = (
        "| 46 | R2 | `mls.rm` | `mls.rm md, rs1` | `rs1[19:15]`; `md[11:7]` | `[31:25]=0x51`; `[24:20]=0x18`; "
        "`[14:12]=0x0` (`ame-enc-funct3`); `[6:0]=0x2b` | `0xfff0707f` | `0xa380002b` |"
    )
    rows = ztt_import.parse(row)
    assert rows == [
        (
            "mls.rm",
            "R2",
            "mls.rm md, rs1",
            {"rs1": (15, 19), "md": (7, 11)},
            0xFFF0707F,
            0xA380002B,
        )
    ]
    assert ztt_import.check(rows) == []
    twin = ("other", "R2", "", rows[0][3], 0xFFF0707F, 0xA380002B)
    assert any("overlap" in p for p in ztt_import.check(rows + [twin]))


# -- datatypes -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, desc",
    [
        ("int4", 0x40000004),  # src/examples.adoc's probes
        ("int8", 0x40000008),
        ("int32", 0x40000020),
        ("uint8", 0x00000008),
        ("bf16", 0x20300110),
        ("fp16", 0x14300110),
        ("fp32", 0x20300120),
        ("fp64", 0x2C300140),
        ("fp8_e4m3", 0x10100108),
        ("fp8_e5m2", 0x14300108),
    ],
)
def test_standard_descriptors(name, desc):
    assert descriptor(name) == desc
    dt = classify(desc)
    assert dt.name == name
    assert dt.bits == desc & 0xFF


def test_descriptor_fields():
    sat = classify(descriptor("int16", sat=True, rnd=ztt.INT_RNE))
    assert (sat.signed, sat.sat, sat.rnd, sat.bits) == (True, True, ztt.INT_RNE, 16)
    assert classify(descriptor("fp32", rnd=1)).rnd == 1


@pytest.mark.parametrize(
    "desc",
    [
        0,
        0x80000008,
        0x40000208,
        0x40000009,
        0x20300120 | (1 << 9),
        0x20300120 | (6 << 22),
        0x24300120,
    ],
    ids=[
        "zero",
        "custom",
        "int reserved bit",
        "int size 9",
        "fp reserved bit",
        "fp rounding 6",
        "fp unknown shape",
    ],
)
def test_unknown_descriptors_classify_as_none(desc):
    assert classify(desc) is None


# -- the design section ------------------------------------------------------------------------


def test_defaults():
    spec = _parse()
    assert (spec.nelem, spec.n, spec.unit_bits, spec.m_regs, spec.acc_regs) == (
        256,
        16,
        8,
        16,
        8,
    )
    assert spec.mreg_bytes == 256
    assert ("int8", "uint8", "int32") in spec.matmul
    assert ("bf16", "bf16", "fp32") in spec.matmul


@pytest.mark.parametrize(
    "raw, why",
    [
        ({"nelem": 128}, "power of 4"),
        ({"unit_bits": 12}, "power of two"),
        ({"m_regs": 8}, "16 or 32"),
        ({"acc_regs": 3}, "1, 2, 4, 8, 16"),
        ({"unit_bits": 16, "datatypes": ["int8", "int32"]}, "packed"),
        ({"datatypes": ["int7"]}, "unknown Ztt datatype"),
        (
            {"datatypes": ["int8"], "matmul": [["int8", "int8", "int32"]]},
            "not in matrix.datatypes",
        ),
        ({"datatypes": ["int8", "fp32"], "matmul": [["int8", "int8", "fp32"]]}, "mix"),
        ({"backend": "pooled"}, "dedicated or shared"),
        ({"ms_shift": 23}, "WPRI"),
        ({"tiles": 8}, "unknown key"),
    ],
)
def test_design_errors(raw, why):
    with pytest.raises(_Error, match=why):
        _parse(**raw)


def test_explicit_matmul_list():
    spec = _parse(
        datatypes=["int8", "uint8", "int32"], matmul=[["uint8", "int8", "int32"]]
    )
    assert spec.matmul == (("uint8", "int8", "int32"),)


def test_header_lines():
    text = "\n".join(ztt.header_lines(ZttSpec()))
    assert "#define NAIAD_HAS_ZTT 1" in text
    assert "#define NAIAD_ZTT_N 16" in text
    assert "#define NAIAD_ZTT_DT_INT8 0x40000008u" in text
    assert "#define NAIAD_ZTT_MMULACC_2D_MATCH 0x9400002bu" in text
    assert "#define NAIAD_ZTT_CSR_AMESTATUS 0x800" in text
