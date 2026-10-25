"""VME (Zvt v0.4): the vector-coupled matrix mode's definitions, independent of execution.

Pinned to the VME TG's "Zvt Family of Vector Matrix Extensions, Version 0.4"
(``zvt-20260831.pdf``, ISA-manual pages 564-579), with the encodings of
``riscv/riscv-opcodes`` ``extensions/unratified/rv_zvt*`` @ ``bf6a8a4``
(2026-08-19). Everything here is that draft's, by section:

* **State** (12.1.1.1): one buffer of ``16 * TE * TE`` bytes, viewed as 16 / 8 /
  4 / 8 tiles of ``ETE x ETE`` elements at TEW = 8 / 16 / 32 / 64 (``ETE = TE``,
  or ``TE / 2`` at TEW 64) through the draft's fixed ``punning()`` function
  (:func:`punning`; :func:`tile_byte_index` is its vectorised form). TE is a
  design parameter, a power of two with ``4 <= TE <= VLEN / 4``.
* **mtype** (12.1.1.2): CSR 0xC23, read-only: ``tm[23:10]``, ``tk[7:5]``,
  ``mtwiden[1:0]`` (``TWIDEN = 1 << (mtwiden - 1)``; 0 = not configured).
* **Operation** (12.1.1.3): ``C[tm, tn] += A[tk, tm]^T . B[tk, tn]`` with
  ``tn = min(vl, ETE)``. A (``vs2``) and B (``vs1``) are vector register groups,
  one LMUL group per K-row, rows ``8 / KMAX`` registers apart; ``vs % LMUL == 0``
  and ``vs % 8 < 8 / KMAX``. KMAX by (SEW, TWIDEN): :data:`KMAX`.
* **Configuration** (12.1.1.4): ``msetmtype[i]`` (set mtype and vtype, derive
  LMUL, ``vta = vma = 1``), ``msettn/tm/tk``; ``vset{i}vl{i}`` write ``mtype = 0``.
* **TSS** (12.1.1.5): ``tile[30:27]`` (low ``log2(16 / NTiles)`` bits ignored),
  ``pattern[26:24]`` (0 row, 1 column; taken mod 2, the draft's expected
  implementation), ``index[23:0]`` (taken mod ETE).
* **Moves, loads, stores** (12.1.1.6-7): one row or column per instruction,
  ``vtle/vtse`` at TEW = EEW (unmasked, contiguous, naturally aligned),
  ``vtmv.v.t`` / ``vtmv.t.v`` at TEW = SEW; body ``[vstart, min(vl, ETE))``.
  All reserved when ``mtwiden = 0``.
* **Arithmetic** (12.1.1.8): the type table is :func:`arith_row`; rounding is
  :mod:`framework.matrix.vme_numerics` (``"zvt"``).
* **Encodings**: the ``vme:`` section of ``encodings.yaml`` (:data:`TABLE`).

**Interpretations** (each where it lives): ``vtzero`` is at TEW = SEW (the
draft says only "supports TEW values corresponding to all supported SEW
settings", the phrase it uses for the moves), and reserved when ``mtwiden =
0``; ``msetmtype[i]`` write no ``rd`` (the PDF's ``00000``; riscv-opcodes
leaves ``rd`` free); a TEW without its ``Zvt<TEW>e`` extension is "unsupported"
for ``msetmtype``'s vill rule. ``mstatus.MS`` is not modelled (always Dirty);
``vtdiscard`` is a counted no-op.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

XLEN = 64
CSR_MTYPE = 0xC23
OP_V, LOAD_FP, STORE_FP, OP_VE = 0x57, 0x07, 0x27, 0x77
ALTFMT_SHIFT = 8  # Zvfbfa's vtype.altfmt: B's type (and bf16 for .alt)

ENCODINGS_PATH = Path(__file__).with_name("encodings.yaml")


class VMEError(Exception):
    """A VME configuration or instruction this model cannot honour (not an architectural trap)."""


# -- the encodings table -------------------------------------------------------------------


@dataclass(frozen=True)
class Instruction:
    name: str
    mask: int
    match: int
    #: operand name -> (lo, hi) bit range.
    operands: dict
    #: The draft extensions that provide it (any one of them).
    ext: tuple

    @property
    def opcode(self):
        return self.match & 0x7F

    @property
    def funct3(self):
        return (self.match >> 12) & 0x7

    def field(self, instr, name):
        lo, hi = self.operands[name]
        return (instr >> lo) & ((1 << (hi - lo + 1)) - 1)


def _range(key):
    text = str(key)
    hi, _, lo = text.partition("..")
    hi = int(hi)
    lo = int(lo) if lo else hi
    if not 0 <= lo <= hi <= 31:
        raise VMEError(f"encodings.yaml vme: bad bit range {key!r}")
    return lo, hi


def _build(raw):
    out = {}
    for name, spec in raw["instructions"].items():
        mask = match = 0
        for key, value in spec.get("fixed", {}).items():
            lo, hi = _range(key)
            m = ((1 << (hi - lo + 1)) - 1) << lo
            if mask & m or value >> (hi - lo + 1):
                raise VMEError(
                    f"encodings.yaml vme.{name}: fixed field {key} overlaps or overflows"
                )
            mask |= m
            match |= value << lo
        ops = {}
        for op, (lo, hi) in (spec.get("operands") or {}).items():
            m = ((1 << (hi - lo + 1)) - 1) << lo
            if mask & m:
                raise VMEError(
                    f"encodings.yaml vme.{name}: operand {op} overlaps a fixed field"
                )
            mask |= m
            ops[op] = (lo, hi)
        if mask != 0xFFFFFFFF:
            raise VMEError(
                f"encodings.yaml vme.{name}: bits {0xFFFFFFFF & ~mask:#010x} are neither fixed nor an operand"
            )
        op_mask = sum(((1 << (hi - lo + 1)) - 1) << lo for lo, hi in ops.values())
        ext = spec.get("ext", ())
        out[name] = Instruction(
            name,
            0xFFFFFFFF & ~op_mask,
            match,
            ops,
            (ext,) if isinstance(ext, str) else tuple(ext),
        )
    insns = list(out.values())
    for i, a in enumerate(insns):
        for b in insns[i + 1 :]:
            if not (a.match ^ b.match) & a.mask & b.mask:
                raise VMEError(f"encodings.yaml vme: {a.name} and {b.name} overlap")
    return out


def load_table(path=ENCODINGS_PATH):
    with open(path) as f:
        return _build(yaml.safe_load(f)["vme"])


#: The Zvt instructions, by mnemonic.
TABLE = load_table()
_BY_KEY = {}
for _insn in TABLE.values():
    _BY_KEY.setdefault(_insn.match & 0x707F, []).append(_insn)
OPCODES = frozenset(i.opcode for i in TABLE.values())


def decode(instr):
    """The :class:`Instruction` a Zvt word is, or None."""
    for insn in _BY_KEY.get(instr & 0x707F, ()):
        if instr & insn.mask == insn.match:
            return insn
    return None


def encode(name, **operands):
    """Assemble ``name`` from its operand values (missing ones 0)."""
    insn = TABLE[name]
    word = insn.match
    for op, (lo, hi) in insn.operands.items():
        value = operands.pop(op, 0)
        if not 0 <= value < (1 << (hi - lo + 1)):
            raise ValueError(f"{name}: {op}={value} does not fit bits {lo}-{hi}")
        word |= value << lo
    if operands:
        raise ValueError(f"{name}: unknown operand(s) {sorted(operands)}")
    return word


def ime_words():
    """``(name, mask, match)`` for every Zvvm (IME) instruction, from :data:`framework.matrix.ime.INSTRUCTIONS`."""
    from framework.matrix import ime

    out = []
    for insn in ime.INSTRUCTIONS.values():
        if insn.kind in ("int", "fp"):  # opcode, funct3, funct6; vm/vd/vs1/vs2 free
            mask = 0x7F | (0x7 << 12) | (0x3F << 26)
            match = insn.opcode | (insn.funct3 << 12) | (insn.code << 26)
        else:  # opcode, width 111, mew 1, mop; nf (lambda), vm and the registers free
            mask = 0x7F | (0x7 << 12) | (1 << 28) | (0x3 << 26)
            match = insn.opcode | (7 << 12) | (1 << 28) | (insn.code << 26)
        out.append((insn.name, mask, match))
    return out


def collisions():
    """``(zvt mnemonic, zvvm mnemonic)`` pairs that share at least one encoding."""
    return [
        (v.name, name)
        for v in TABLE.values()
        for name, mask, match in ime_words()
        if not (v.match ^ match) & v.mask & mask
    ]


# -- geometry ----------------------------------------------------------------------------------

#: KMAX by (SEW, TWIDEN) (12.1.1.4, "Supported SEW, TWIDEN, and KMAX values").
KMAX = {
    (8, 1): 4,
    (8, 2): 4,
    (16, 1): 2,
    (8, 4): 4,
    (16, 2): 2,
    (32, 1): 1,
    (16, 4): 2,
    (32, 2): 1,
    (64, 1): 1,
}
#: Tiles per TEW, and the tile-number step (16 / NTiles).
NTILES = {8: 16, 16: 8, 32: 4, 64: 8}
TEWS = (8, 16, 32, 64)


def twiden(mtwiden):
    return 1 << (mtwiden - 1) if mtwiden else 0


def ete(te, tew):
    """Tile edge in elements at ``tew``: TE, or TE / 2 at TEW 64."""
    return te if tew < 64 else te // 2


def tile_step(tew):
    return 16 // NTILES[tew]


def punning(te, tile, row, col, tew):
    """The draft's ``punning()``: the byte offset of element ``(row, col)`` of ``tile`` at ``tew``."""
    if tew == 8:
        ptile = tile
        minor = (row % 4) * 4 + (col % 4)
        major = (row // 4) * (te // 4) + (col // 4)
    elif tew == 16:
        ptile = tile + ((row & 2) >> 1)
        minor = (row % 2) * 4 + (col % 2) * 2 + ((col // 2) % 2) * 8
        major = (row // 4) * (te // 4) + (col // 4)
    elif tew == 32:
        ptile = tile + 2 * (row // (te // 2)) + ((col & 2) >> 1)
        minor = (row % 2) * 8 + (col % 2) * 4
        major = ((row // 2) % (te // 4)) * (te // 4) + (col // 4)
    elif tew == 64:
        ptile = tile + (row // (te // 4))
        minor = (col % 2) * 8
        major = (row % (te // 4)) * (te // 4) + (col // 2)
    else:
        return 0
    return ptile * te * te + major * 16 + minor


def tile_byte_index(te, tew):
    """``{tile: int64 (ETE, ETE, TEW/8)}``: every byte of every valid tile at ``tew``, by :func:`punning`."""
    e = ete(te, tew)
    eb = tew // 8
    r, c = np.meshgrid(np.arange(e), np.arange(e), indexing="ij")
    out = {}
    for tile in range(0, 16, tile_step(tew)):
        if tew == 8:
            ptile = tile + 0 * r
            minor = (r % 4) * 4 + (c % 4)
            major = (r // 4) * (te // 4) + (c // 4)
        elif tew == 16:
            ptile = tile + ((r & 2) >> 1)
            minor = (r % 2) * 4 + (c % 2) * 2 + ((c // 2) % 2) * 8
            major = (r // 4) * (te // 4) + (c // 4)
        elif tew == 32:
            ptile = tile + 2 * (r // (te // 2)) + ((c & 2) >> 1)
            minor = (r % 2) * 8 + (c % 2) * 4
            major = ((r // 2) % (te // 4)) * (te // 4) + (c // 4)
        else:
            ptile = tile + (r // (te // 4))
            minor = (c % 2) * 8
            major = (r % (te // 4)) * (te // 4) + (c // 2)
        start = ptile * te * te + major * 16 + minor
        out[tile] = (start[..., None] + np.arange(eb)).astype(np.int64)
    return out


def physical_tiles(tile, tew):
    """The TEW-8 storage tiles (``ptile`` values) a tile at ``tew`` covers, for hazard tracking."""
    return {
        8: (tile,),
        16: (tile, tile + 1),
        32: tuple(range(tile, tile + 4)),
        64: (tile, tile + 1),
    }[tew]


def config_lmul(sew, tw, te, vlen):
    """``msetmtype``'s ``vlmul``: ``log2(min(8/KMAX, 8/TWIDEN, ceil(ETE/EVE)))`` (as an LMUL)."""
    tew = sew * tw
    e = ete(te, tew)
    eve = vlen // sew
    kmax = KMAX.get((sew, tw), 1)
    lmul = min(8 // kmax, 8 // tw, max(1, -(-e // eve)))
    return 1 << (lmul.bit_length() - 1)  # a power of two (all three terms are)


def decode_tss(value, tew, te):
    """``(tile, column, index)`` from a Tile Subset Specifier, as an implementation may read it."""
    tile = (value >> 27) & 0xF
    tile -= tile % tile_step(tew)
    pattern = (value >> 24) & 0x7
    return tile, pattern & 1, (value & 0xFFFFFF) % ete(te, tew)


def tss(tile, index, column=False):
    """A Tile Subset Specifier value."""
    return (tile << 27) | (int(bool(column)) << 24) | index


# -- the arithmetic type table (12.1.1.8) ---------------------------------------------------------


@dataclass(frozen=True)
class Row:
    fmt_a: str  # vs2's type
    fmt_b: str  # vs1's type
    fmt_c: str
    ext: str


def arith_row(name, sew, tw, altfmt):
    """The type row an arithmetic instruction selects under (SEW, TWIDEN, altfmt), or None (reserved)."""
    if name == "vtfmm.tvv":
        table = {
            (8, 4, 0): Row("fp8_e4m3", "fp8_e4m3", "fp32", "Zvtofp8fmm"),
            (8, 4, 1): Row("fp8_e4m3", "fp8_e5m2", "fp32", "Zvtofp8fmm"),
            (16, 2, 0): Row("fp16", "fp16", "fp32", "Zvtfp16fmm"),
            (32, 1, 0): Row("fp32", "fp32", "fp32", "Zvtfmm"),
            (64, 1, 0): Row("fp64", "fp64", "fp64", "Zvtdmm"),
        }
    elif name == "vtfmm.alt.tvv":
        table = {
            (8, 4, 0): Row("fp8_e5m2", "fp8_e4m3", "fp32", "Zvtofp8fmm"),
            (8, 4, 1): Row("fp8_e5m2", "fp8_e5m2", "fp32", "Zvtofp8fmm"),
            (16, 2, 1): Row("bf16", "bf16", "fp32", "Zvtbf16fmm"),
        }
    elif name == "vtmmu.tvv":
        table = {
            (8, 4, 0): Row("uint8", "uint8", "int32", "Zvti8i32mm"),
            (8, 4, 1): Row("uint8", "int8", "int32", "Zvti8i32mm"),
        }
    elif name == "vtmms.tvv":
        table = {
            (8, 4, 0): Row("int8", "uint8", "int32", "Zvti8i32mm"),
            (8, 4, 1): Row("int8", "int8", "int32", "Zvti8i32mm"),
        }
    else:
        return None
    return table.get((sew, tw, altfmt))


# -- extensions and the design spec -----------------------------------------------------------------


@dataclass(frozen=True)
class Extension:
    name: str
    #: The other Zvt extensions it requires (12.1.1, the dependency tables).
    requires: tuple = ()
    #: ``(in, acc)`` format pairs it gives (computational extensions).
    pairs: tuple = ()
    #: The TEW whose moves/loads/stores it adds (Zvt8e/16e/64e).
    tew: int = 0


EXTENSIONS = {
    e.name: e
    for e in (
        Extension("Zvtbase", tew=32),
        Extension("Zvt8e", ("Zvtbase",), tew=8),
        Extension("Zvt16e", ("Zvtbase",), tew=16),
        Extension("Zvt64e", ("Zvtbase",), tew=64),
        Extension("Zvti8i32mm", ("Zvtbase", "Zvt8e"), (("int8", "int32"),)),
        Extension(
            "Zvtofp8fmm",
            ("Zvtbase", "Zvt8e"),
            (("fp8_e4m3", "fp32"), ("fp8_e5m2", "fp32")),
        ),
        Extension("Zvtfp16fmm", ("Zvtbase", "Zvt16e"), (("fp16", "fp32"),)),
        Extension("Zvtbf16fmm", ("Zvtbase", "Zvt16e"), (("bf16", "fp32"),)),
        Extension("Zvtfmm", ("Zvtbase",), (("fp32", "fp32"),)),
        Extension("Zvtdmm", ("Zvtbase", "Zvt64e"), (("fp64", "fp64"),)),
    )
}
_BY_LOWER = {name.lower(): name for name in EXTENSIONS}
#: The extensions that make Zvfbfa's altfmt (vtype bit 8) selectable (12.1.1.8).
ALTFMT_EXTENSIONS = ("Zvtbf16fmm", "Zvti8i32mm", "Zvtofp8fmm")


def canonical_extension(name):
    return _BY_LOWER.get(str(name).lower())


@dataclass(frozen=True)
class VMESpec:
    """A design's ``matrix:`` section in ``mode: vme``."""

    vlen: int
    elen: int
    te: int
    extensions: tuple = ("Zvtbase", "Zvtfmm")
    mode: str = "vme"

    def has(self, name):
        return name in self.extensions

    @property
    def tews(self):
        """The TEWs with moves, loads and stores (32 always; 8 / 16 / 64 by Zvt<n>e)."""
        return tuple(
            sorted({EXTENSIONS[e].tew for e in self.extensions if EXTENSIONS[e].tew})
        )

    @property
    def format_pairs(self):
        return tuple(p for e in self.extensions for p in EXTENSIONS[e].pairs)

    @property
    def storage_bytes(self):
        return 16 * self.te * self.te


def parse_vme(raw, cores, error):
    """Validate ``matrix: {mode: vme, ...}`` against the cores; returns a :class:`VMESpec`.

    Keys: ``te`` (power of two, ``4 <= TE <= VLEN / 4``; default ``min(32, VLEN / 4)``)
    and ``extensions`` (the draft's names; ``Zvtbase`` is implied; default
    ``[Zvtfmm]``). Each computational extension's dependencies (12.1.1) must be
    listed too, as the draft requires them to be implemented.
    """
    allowed = {"mode", "te", "extensions"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise error(
            f"matrix (mode vme): unknown key(s) {unknown}; expected some of {sorted(allowed)}"
        )
    if cores.xlen != 64 or not cores.has_vector:
        raise error(
            "matrix.mode vme needs an RV64 core with a vector unit ('v' or 'zve64d' in cores.isa)"
        )
    zvv = [e for e in cores.isa if str(e).lower().startswith("zvv")]
    if zvv:
        pairs = sorted({f"{a}/{b}" for a, b in collisions()})
        raise error(
            f"matrix.mode vme cannot share a hart with IME ({zvv}): Zvt and Zvvm share encodings ({', '.join(pairs)})"
        )
    vlen = cores.vlen
    te = raw.get("te", min(32, vlen // 4))
    if (
        not isinstance(te, int)
        or isinstance(te, bool)
        or te < 4
        or te & (te - 1)
        or te > vlen // 4
    ):
        raise error(
            f"matrix.te must be a power of two in 4..VLEN/4 = {vlen // 4}, got {te!r}"
        )
    names = raw.get("extensions", ["Zvtfmm"])
    if not isinstance(names, list) or not names:
        raise error("matrix.extensions must be a non-empty list of Zvt extension names")
    exts = ["Zvtbase"]
    for name in names:
        canon = canonical_extension(name)
        if canon is None:
            raise error(
                f"matrix.extensions: {name!r} is not a Zvt v0.4 extension ({sorted(EXTENSIONS)})"
            )
        if canon in exts and canon != "Zvtbase":
            raise error(f"matrix.extensions lists {canon} twice")
        if canon not in exts:
            exts.append(canon)
    if not any(EXTENSIONS[e].pairs for e in exts):
        raise error(
            "matrix.extensions names no computational extension (Zvti8i32mm, Zvtfmm, ...)"
        )
    for e in exts:
        missing = [r for r in EXTENSIONS[e].requires if r not in exts]
        if missing:
            raise error(
                f"matrix.extensions: {e} requires {missing} (the draft's dependency table)"
            )
        if EXTENSIONS[e].tew > cores.elen:
            raise error(f"matrix.extensions: {e} needs ELEN >= {EXTENSIONS[e].tew}")
    return VMESpec(vlen=vlen, elen=cores.elen, te=te, extensions=tuple(exts))


# -- naiad.h ----------------------------------------------------------------------------------------


def _ident(name):
    return name.upper().replace(".", "_")


def header_lines(spec):
    """The ``naiad.h`` lines for a VME design: mode, TE, extensions, the encodings as ``.insn`` fields."""
    lines = [
        "",
        "/* matrix unit: vme (Zvt v0.4, 2026-08-31), vector-coupled accumulator tiles */",
        "#define NAIAD_HAS_VME 1",
        "#define NAIAD_MATRIX_VME 1",
        '#define NAIAD_MATRIX_MODE "vme"',
        f"#define NAIAD_VME_TE {spec.te}",
        f"#define NAIAD_VME_CSR_MTYPE {CSR_MTYPE:#x}",
    ]
    lines += [f"#define NAIAD_VME_EXT_{e.upper()} 1" for e in spec.extensions]
    lines += [
        "",
        "/* Zvt encodings (framework/matrix/encodings.yaml, vme:): match words with operands zero */",
    ]
    for insn in TABLE.values():
        lines.append(f"#define NAIAD_VME_{_ident(insn.name)}_MATCH {insn.match:#010x}")
    return lines


def reference_geometry(te, vlen, sew, tw):
    """``(tm, tn, tk, lmul, macs)`` of a full-tile multiply: the shape an equal-resource comparison uses."""
    tew = sew * tw
    e = ete(te, tew)
    lmul = config_lmul(sew, tw, te, vlen)
    edge = min(e, lmul * vlen // sew)
    k = KMAX[(sew, tw)]
    return edge, edge, k, lmul, edge * edge * k


__all__ = [
    "CSR_MTYPE",
    "EXTENSIONS",
    "KMAX",
    "TABLE",
    "VMEError",
    "VMESpec",
    "arith_row",
    "collisions",
    "decode",
    "encode",
    "header_lines",
    "parse_vme",
    "punning",
    "tile_byte_index",
]
