"""Ztt (the RISC-V AME TG's Attached Matrix Extension): definitions, independent of execution.

Pinned to the TG's public draft, ``riscv/riscv-attached-matrix-extension`` tag ``v0.6``
(commit ``8d27acd``, 2026-08-16); the encodings are that tag's
``INSTRUCTION_ENCODINGS.md``, imported into ``ztt_encodings.yaml`` by
``tools/naiad/ztt_import.py``. This is the draft's own ISA: its instruction
words, its state and its datatype descriptors. The ``ame`` mode's
``profile: ztt`` (``framework/naiad/design.py``) is a different thing: Naiad's
own AME encodings restricted to Ztt's shapes.

What the draft defines, and where it lives here:

* **Parameters** (``src/parameters.adoc``): ``AME_NELEM`` (a power of 4; squares
  are ``N x N``, ``N = sqrt(AME_NELEM)``), ``AME_UNIT_DATATYPE_SIZE``,
  ``AME_NUM_M_REGS`` (16 or 32), ``AME_NUM_ACC_REGS`` (1-16): :class:`ZttSpec`.
* **Datatypes** (``src/datatypes.adoc``): 32-bit descriptors, integer and IEEE-like
  classes: :func:`descriptor` and :func:`classify`. An implementation supports a
  subset; anything else is an *unsupported tuple* (``amestatus.UN``), not a trap.
* **CSRs** (``src/state.adoc``, provisional addresses): :data:`CSRS`.
* **Encodings**: :data:`TABLE`, :func:`decode`, :func:`encode`.

Interpretations, each also noted where it is applied:

* ``mstatus.MS`` has no bit position in v0.6. Naiad puts it at bits
  ``[ms_shift+1:ms_shift]`` (default 25, among RV64's WPRI bits) and sets ``SD``
  (bit 63) when it goes Dirty.
* ``amestype``, ``amefflags`` and ``amexsat`` are writable (the draft says software
  writes them) though their provisional addresses (``0xCC2``, ``0xCC4``,
  ``0xCC5``) are in the read-only custom range (bits [11:10] = 0b11). The draft's
  text wins: Naiad accepts the writes.
* Packed datatypes (narrower than the unit size) are not modelled: a design's
  datatypes are all at least ``AME_UNIT_DATATYPE_SIZE`` wide, so ``N_sq = 1``.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

ENCODINGS_PATH = Path(__file__).with_name("ztt_encodings.yaml")
OPCODE = 0x2B
XLEN = 64

#: The CSRs (``src/state.adoc``): address -> name. Provisional allocations.
CSRS = {
    0xCC0: "amenlen",
    0xCC1: "ameudsz",
    0xCC2: "amestype",
    0xCC3: "ameown",
    0xCC4: "amefflags",
    0xCC5: "amexsat",
    0x800: "amestatus",
}
CSR_BY_NAME = {name: addr for addr, name in CSRS.items()}
#: Context CSRs software may write (the rest are capability or ownership reports).
WRITABLE_CSRS = ("amestype", "amefflags", "amexsat", "amestatus")

MS_OFF, MS_INITIAL, MS_CLEAN, MS_DIRTY = 0, 1, 2, 3
MSTATUS_SD = 1 << 63
DEFAULT_MS_SHIFT = 25

#: ``ame.acquire`` result STATUS values (``src/backend_management.adoc``).
ACQ_GRANTED, ACQ_BUSY, ACQ_UNSUPPORTED, ACQ_DISABLED, ACQ_BAD_DESC, ACQ_TIMEOUT = range(
    6
)


class ZttError(Exception):
    """A Ztt design or table this model cannot honour (not an architectural trap)."""


# -- the encodings table --------------------------------------------------------------------


@dataclass(frozen=True)
class Instruction:
    name: str
    format: str
    mnemonic: str
    mask: int
    match: int
    #: operand name -> (lo, hi) bit range, in the order the table lists them.
    operands: dict

    def field(self, instr, name):
        lo, hi = self.operands[name]
        return (instr >> lo) & ((1 << (hi - lo + 1)) - 1)


def load_table(path=ENCODINGS_PATH):
    """``(source, {name: Instruction})`` from the imported YAML; checks it as the importer does."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    table = {}
    for name, spec in raw["instructions"].items():
        ops = {op: (lo, hi) for op, (lo, hi) in (spec.get("operands") or {}).items()}
        insn = Instruction(
            name, spec["format"], spec["mnemonic"], spec["mask"], spec["match"], ops
        )
        op_bits = sum(((1 << (hi - lo + 1)) - 1) << lo for lo, hi in ops.values())
        if op_bits & insn.mask or (op_bits | insn.mask) != 0xFFFFFFFF:
            raise ZttError(
                f"{path}: {name}: operand fields and mask do not tile the word"
            )
        if insn.match & 0x7F != OPCODE:
            raise ZttError(f"{path}: {name} is not in custom-1")
        table[name] = insn
    insns = list(table.values())
    for i, a in enumerate(insns):
        for b in insns[i + 1 :]:
            if not (a.match ^ b.match) & a.mask & b.mask:
                raise ZttError(f"{path}: {a.name} and {b.name} overlap")
    return raw["source"], table


SOURCE, TABLE = load_table()
#: funct7 -> the instructions in that bank (every Ztt word has funct3 = 000).
_BY_F7 = {}
for _insn in TABLE.values():
    _BY_F7.setdefault(_insn.match >> 25, []).append(_insn)
del _insn


def decode(instr):
    """The :class:`Instruction` a word is, or None."""
    for insn in _BY_F7.get((instr >> 25) & 0x7F, ()):
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


# -- datatypes ----------------------------------------------------------------------------------

#: The standard IEEE-class types: name -> (size, exp_width, inf, denorm).
FP_TYPES = {
    "fp16": (16, 5, 1, 1),
    "bf16": (16, 8, 1, 1),
    "fp32": (32, 8, 1, 1),
    "fp64": (64, 11, 1, 1),
    "fp8_e4m3": (8, 4, 0, 1),
    "fp8_e5m2": (8, 5, 1, 1),
}
_FP_BY_FIELDS = {fields: name for name, fields in FP_TYPES.items()}
INT_SIZES = (4, 8, 16, 32, 64, 128)
#: The IEEE class's rounding-mode codes; Naiad implements RNE only.
FP_RNE = 0
#: The integer class's rounding-mode codes (``src/instruction_semantics.adoc``).
INT_RNU, INT_RNE, INT_RDN, INT_ROD = range(4)


@dataclass(frozen=True)
class Dtype:
    """A decoded standard descriptor. ``name`` is the base type (``int8``, ``uint8``, ``bf16`` ...)."""

    desc: int
    name: str
    bits: int
    is_float: bool
    signed: bool = True
    sat: bool = False
    rnd: int = 0

    @property
    def bytes(self):
        return self.bits // 8


def standard_names():
    return [*(f"{s}int{n}" for n in INT_SIZES for s in ("", "u")), *FP_TYPES]


def descriptor(name, *, sat=False, rnd=None):
    """The 32-bit descriptor of a standard type: ``int8``, ``uint32``, ``bf16``, ``fp8_e4m3`` ...

    ``sat`` makes an integer type saturating; ``rnd`` is the rounding-mode field
    (integers: RNU/RNE/RDN/ROD = 0-3, default RNU; floats: default RNE = 0).
    """
    if name in FP_TYPES:
        if sat:
            raise ValueError(f"{name}: saturation is an integer-class property")
        size, exp, inf, den = FP_TYPES[name]
        return (
            (exp << 26)
            | ((rnd or 0) << 22)
            | (inf << 21)
            | (den << 20)
            | (1 << 8)
            | size
        )
    signed = not name.startswith("u")
    try:
        size = int(name[3:] if signed else name[4:])
    except ValueError:
        raise ValueError(
            f"unknown Ztt datatype {name!r}; known: {standard_names()}"
        ) from None
    if size not in INT_SIZES or not name.startswith("int" if signed else "uint"):
        raise ValueError(f"unknown Ztt datatype {name!r}; known: {standard_names()}")
    return (int(signed) << 30) | (int(sat) << 29) | ((rnd or 0) << 27) | size


def classify(desc):
    """The :class:`Dtype` a descriptor names, or None (zero, custom, reserved bits set, or unknown)."""
    desc &= 0xFFFFFFFF
    if desc == 0 or desc >> 31:
        return None
    size = desc & 0xFF
    if desc & (1 << 8):  # IEEE floating-point class
        if desc & 0x000FFE00:  # reserved bits 19:9
            return None
        exp, rnd = (desc >> 26) & 0x1F, (desc >> 22) & 0xF
        inf, den = (desc >> 21) & 1, (desc >> 20) & 1
        if rnd > 5:
            return None
        name = _FP_BY_FIELDS.get((size, exp, inf, den))
        if name is None:
            return None
        return Dtype(desc, name, size, True, rnd=rnd)
    if desc & 0x07FFFE00:  # reserved bits 26:9
        return None
    if size not in INT_SIZES:
        return None
    signed, sat, rnd = bool((desc >> 30) & 1), bool((desc >> 29) & 1), (desc >> 27) & 3
    return Dtype(
        desc, f"{'' if signed else 'u'}int{size}", size, False, signed, sat, rnd
    )


# -- the design's matrix: section -------------------------------------------------------------


@dataclass(frozen=True)
class ZttSpec:
    """``matrix: {mode: ztt, ...}``: the implementation's parameters and supported tuples."""

    mode: str = "ztt"
    nelem: int = 256
    unit_bits: int = 8
    m_regs: int = 16
    acc_regs: int = 8
    #: Base type names accepted in M and Acc registers (integers in both their wrapping
    #: and saturating forms).
    datatypes: tuple = ("int8", "uint8", "int32", "fp16", "bf16", "fp32")
    #: ``(a, b, acc)`` base-type triples the matrix multiplies accept.
    matmul: tuple = ()
    #: ``dedicated``: one backend per hart; ``shared``: one backend for every hart.
    backend: str = "dedicated"
    ms_shift: int = DEFAULT_MS_SHIFT

    @property
    def n(self):
        return int(round(self.nelem**0.5))

    @property
    def mreg_bytes(self):
        return self.nelem * self.unit_bits // 8


def default_matmul(datatypes):
    """The multiplies a datatype list implies: int8/uint8 into int32, narrow floats into fp32, fp64."""
    have = set(datatypes)
    out = []
    for a in ("int8", "uint8"):
        for b in ("int8", "uint8"):
            if {a, b, "int32"} <= have:
                out.append((a, b, "int32"))
    for f in ("fp8_e4m3", "fp8_e5m2", "fp16", "bf16", "fp32"):
        if {f, "fp32"} <= have:
            out.append((f, f, "fp32"))
    if "fp64" in have:
        out.append(("fp64", "fp64", "fp64"))
    return tuple(out)


def _is_pow(n, base):
    while n > 1 and n % base == 0:
        n //= base
    return n == 1


def parse_ztt(raw, cores, error):
    """Validate ``matrix: {mode: ztt, ...}`` against the cores; returns a :class:`ZttSpec`.

    Keys: ``nelem`` (AME_NELEM, a power of 4, default 256), ``unit_bits``
    (AME_UNIT_DATATYPE_SIZE, a power of two >= 8, default 8), ``m_regs`` (16 or
    32), ``acc_regs`` (1, 2, 4, 8 or 16), ``datatypes`` (standard base names, each
    at least ``unit_bits`` wide: packed types are not modelled), ``matmul``
    (``[[a, b, acc], ...]``, default :func:`default_matmul`), ``backend``
    (``dedicated`` or ``shared``), ``ms_shift`` (where ``mstatus.MS`` sits).
    """
    allowed = {
        "mode",
        "nelem",
        "unit_bits",
        "m_regs",
        "acc_regs",
        "datatypes",
        "matmul",
        "backend",
        "ms_shift",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise error(
            f"matrix (mode ztt): unknown key(s) {unknown}; expected some of {sorted(allowed)}"
        )
    if cores.xlen != 64:
        raise error("matrix.mode ztt is modelled on RV64 cores only")
    d = ZttSpec()

    def num(key, default):
        v = raw.get(key, default)
        if not isinstance(v, int) or isinstance(v, bool):
            raise error(f"matrix.{key} must be an integer, got {v!r}")
        return v

    nelem = num("nelem", d.nelem)
    if nelem < 4 or not _is_pow(nelem, 4):
        raise error(
            f"matrix.nelem (AME_NELEM) must be a power of 4 (>= 4), got {nelem}"
        )
    unit = num("unit_bits", d.unit_bits)
    if unit < 8 or unit & (unit - 1):
        raise error(
            f"matrix.unit_bits (AME_UNIT_DATATYPE_SIZE) must be a power of two >= 8, got {unit}"
        )
    m_regs = num("m_regs", d.m_regs)
    if m_regs not in (16, 32):
        raise error(f"matrix.m_regs (AME_NUM_M_REGS) must be 16 or 32, got {m_regs}")
    acc_regs = num("acc_regs", d.acc_regs)
    if acc_regs not in (1, 2, 4, 8, 16):
        raise error(
            f"matrix.acc_regs (AME_NUM_ACC_REGS) must be one of 1, 2, 4, 8, 16, got {acc_regs}"
        )
    names = raw.get("datatypes", list(d.datatypes))
    if not isinstance(names, list) or not names:
        raise error(
            "matrix.datatypes must be a non-empty list of Ztt standard type names"
        )
    canon = []
    for name in names:
        name = str(name).lower()
        try:
            dt = classify(descriptor(name))
        except ValueError as e:
            raise error(f"matrix.datatypes: {e}") from None
        if dt.bits < unit:
            raise error(
                f"matrix.datatypes: {name} is narrower than unit_bits = {unit} (a packed type, not modelled)"
            )
        if dt.bits // unit > m_regs:
            raise error(
                f"matrix.datatypes: a {name} square needs {dt.bits // unit} M registers, more than m_regs"
            )
        if name in canon:
            raise error(f"matrix.datatypes lists {name} twice")
        if not dt.is_float and dt.bits == 128:
            raise error("matrix.datatypes: 128-bit integers are not modelled")
        canon.append(name)
    if raw.get("matmul") is None:
        matmul = default_matmul(canon)
    else:
        matmul = []
        for t in raw["matmul"]:
            if not isinstance(t, list) or len(t) != 3:
                raise error(f"matrix.matmul entries are [a, b, acc] lists, got {t!r}")
            t = tuple(str(x).lower() for x in t)
            missing = [x for x in t if x not in canon]
            if missing:
                raise error(
                    f"matrix.matmul {list(t)}: {missing} not in matrix.datatypes"
                )
            kinds = {classify(descriptor(x)).is_float for x in t}
            if len(kinds) != 1:
                raise error(
                    f"matrix.matmul {list(t)}: cannot mix integer and floating-point types"
                )
            if t[0] in FP_TYPES and t[0] != t[1]:
                raise error(
                    f"matrix.matmul {list(t)}: floating-point A and B must be the same type"
                )
            matmul.append(t)
        matmul = tuple(matmul)
    backend = str(raw.get("backend", d.backend)).lower()
    if backend not in ("dedicated", "shared"):
        raise error(f"matrix.backend must be dedicated or shared, got {backend!r}")
    ms_shift = num("ms_shift", d.ms_shift)
    if not (25 <= ms_shift <= 30 or 43 <= ms_shift <= 61):
        raise error(
            f"matrix.ms_shift must place MS in RV64 mstatus WPRI bits (25-31 or 43-62), got {ms_shift}"
        )
    return ZttSpec(
        "ztt", nelem, unit, m_regs, acc_regs, tuple(canon), matmul, backend, ms_shift
    )


def _ident(name):
    return name.upper().replace(".", "_")


def header_lines(spec):
    """The ``naiad.h`` lines for a Ztt design: parameters, CSRs, descriptors, match words."""
    lines = [
        "",
        f"/* matrix unit: ztt (RISC-V AME draft {SOURCE['tag']}, {SOURCE['commit'][:7]}) */",
        "#define NAIAD_HAS_ZTT 1",
        "#define NAIAD_MATRIX_ZTT 1",
        '#define NAIAD_MATRIX_MODE "ztt"',
        f"#define NAIAD_ZTT_NELEM {spec.nelem}",
        f"#define NAIAD_ZTT_N {spec.n}",
        f"#define NAIAD_ZTT_UNIT_BITS {spec.unit_bits}",
        f"#define NAIAD_ZTT_M_REGS {spec.m_regs}",
        f"#define NAIAD_ZTT_ACC_REGS {spec.acc_regs}",
        f"#define NAIAD_ZTT_SHARED_BACKEND {int(spec.backend == 'shared')}",
        f"#define NAIAD_ZTT_MS_SHIFT {spec.ms_shift}",
    ]
    lines += [
        f"#define NAIAD_ZTT_CSR_{name.upper()} {addr:#x}" for addr, name in CSRS.items()
    ]
    lines.append(
        "/* descriptors of the supported types (integers: RNU, wrapping; floats: RNE) */"
    )
    for name in spec.datatypes:
        lines.append(f"#define NAIAD_ZTT_HAS_{_ident(name)} 1")
    for name in standard_names():
        if name == "int128" or name == "uint128":
            continue
        lines.append(f"#define NAIAD_ZTT_DT_{_ident(name)} {descriptor(name):#010x}u")
    for a, b, acc in spec.matmul:
        lines.append(
            f"#define NAIAD_ZTT_MATMUL_{_ident(a)}_{_ident(b)}_{_ident(acc)} 1"
        )
    lines += [
        "",
        "/* Ztt encodings (framework/matrix/ztt_encodings.yaml): match words with operands zero */",
    ]
    for insn in TABLE.values():
        lines.append(f"#define NAIAD_ZTT_{_ident(insn.name)}_MATCH {insn.match:#010x}u")
    return lines


__all__ = [
    "CSRS",
    "Dtype",
    "Instruction",
    "SOURCE",
    "TABLE",
    "ZttError",
    "ZttSpec",
    "classify",
    "decode",
    "default_matmul",
    "descriptor",
    "encode",
    "header_lines",
    "parse_ztt",
]
