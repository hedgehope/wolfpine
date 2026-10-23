"""The matrix unit's instruction encodings, decoded from ``encodings.yaml``.

The YAML table is the source of truth (see its header for the layout rules);
this module validates it and turns it into a :class:`Decoder` at import, so an
encoding can move without a code change. Nothing here executes anything: a
decode is ``(Instruction, {operand: value})``, and the coupling-mode ISAs
(``ame_isa`` now, IME/VME later) give it meaning.

:func:`header_defines` is the C side of the same table: the opcodes, funct3s,
format codes, feature bits, mcfg selectors and funct7/imm field shifts that
``kernels/naiad/include/matrix.h`` assembles its ``.insn`` wrappers from.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from framework.matrix.formats import get_format

ENCODINGS_PATH = Path(__file__).with_name("encodings.yaml")

#: The standard fields an operand may name instead of giving bits.
STANDARD_FIELDS = {"rd": (7, 11), "rs1": (15, 19), "rs2": (20, 24), "imm": (20, 31)}
#: Which standard fields each form has, and where its free (funct7/imm) bits start.
_FORMS = {"r": ({"rd", "rs1", "rs2"}, 25), "i": ({"rd", "rs1", "imm"}, 20)}
OPERAND_KINDS = ("tile", "acc", "gpr", "in_fmt", "acc_fmt", "flag", "config")
_FUNCT3 = (12, 14)
#: opcode + funct3: every instruction fixes both, so this is the decode key.
_KEY_MASK = 0x707F


class EncodingTableError(ValueError):
    pass


@dataclass(frozen=True)
class Operand:
    name: str
    lo: int
    hi: int
    kind: str
    #: The standard field it occupies (``rd``/``rs1``/``rs2``/``imm``), or None.
    field: str | None = None
    feature: str | None = None

    @property
    def width(self):
        return self.hi - self.lo + 1

    @property
    def mask(self):
        return ((1 << self.width) - 1) << self.lo


@dataclass(frozen=True)
class Instruction:
    name: str
    opcode: int
    form: str
    funct3: int
    exec: str
    operands: tuple[Operand, ...]
    feature: str | None = None
    params: dict = field(default_factory=dict, hash=False)
    doc: str = ""
    #: Bits fixed by the encoding (opcode, funct3 and every reserved bit) ...
    mask: int = 0
    #: ... and their values.
    match: int = 0

    def operand(self, name):
        for op in self.operands:
            if op.name == name:
                return op
        raise KeyError(f"{self.name} has no operand '{name}'")


def _bits(value, what):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, int) for v in value)
        or not 0 <= value[0] <= value[1] <= 31
    ):
        raise EncodingTableError(
            f"{what}: bits must be [lo, hi] within 0-31, got {value!r}"
        )
    return value[0], value[1]


def _parse_instruction(name, raw, opcodes, features):
    what = f"instruction '{name}'"
    allowed = {
        "opcode",
        "form",
        "funct3",
        "exec",
        "params",
        "operands",
        "feature",
        "doc",
    }
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise EncodingTableError(
            f"{what}: expected a mapping with keys from {sorted(allowed)}"
        )
    if raw.get("opcode") not in opcodes:
        raise EncodingTableError(f"{what}: opcode must be one of {sorted(opcodes)}")
    form = raw.get("form")
    if form not in _FORMS:
        raise EncodingTableError(f"{what}: form must be one of {sorted(_FORMS)}")
    funct3 = raw.get("funct3")
    if not isinstance(funct3, int) or not 0 <= funct3 <= 7:
        raise EncodingTableError(f"{what}: funct3 must be 0-7")
    feature = raw.get("feature")
    if feature is not None and feature not in features:
        raise EncodingTableError(f"{what}: unknown feature '{feature}'")
    if not isinstance(raw.get("exec"), str):
        raise EncodingTableError(f"{what}: missing 'exec'")

    std_fields, free_lo = _FORMS[form]
    opcode = opcodes[raw["opcode"]]
    used = 0x7F | (0x7 << _FUNCT3[0])
    operands = []
    for op_name, spec in (raw.get("operands") or {}).items():
        op_what = f"{what} operand '{op_name}'"
        if not isinstance(spec, dict) or spec.get("kind") not in OPERAND_KINDS:
            raise EncodingTableError(f"{op_what}: kind must be one of {OPERAND_KINDS}")
        if ("field" in spec) == ("bits" in spec):
            raise EncodingTableError(
                f"{op_what}: give exactly one of 'field' and 'bits'"
            )
        if "field" in spec:
            if spec["field"] not in std_fields:
                raise EncodingTableError(
                    f"{op_what}: a {form}-form has fields {sorted(std_fields)}"
                )
            lo, hi = STANDARD_FIELDS[spec["field"]]
        else:
            lo, hi = _bits(spec["bits"], op_what)
            if lo < free_lo:
                raise EncodingTableError(
                    f"{op_what}: bits must lie in {free_lo}-31 for a {form}-form"
                )
        op_feature = spec.get("feature")
        if op_feature is not None and op_feature not in features:
            raise EncodingTableError(f"{op_what}: unknown feature '{op_feature}'")
        if spec["kind"] == "flag" and lo != hi:
            raise EncodingTableError(f"{op_what}: a flag is one bit")
        operand = Operand(op_name, lo, hi, spec["kind"], spec.get("field"), op_feature)
        if used & operand.mask:
            raise EncodingTableError(f"{op_what}: bits {lo}-{hi} overlap another field")
        used |= operand.mask
        operands.append(operand)

    mask = 0xFFFFFFFF & ~sum(op.mask for op in operands)
    return Instruction(
        name=name,
        opcode=opcode,
        form=form,
        funct3=funct3,
        exec=raw["exec"],
        operands=tuple(operands),
        feature=feature,
        params=dict(raw.get("params") or {}),
        doc=" ".join(str(raw.get("doc", "")).split()),
        mask=mask,
        match=opcode | (funct3 << _FUNCT3[0]),
    )


class EncodingTable:
    """A validated encodings table: formats, features, config selectors, instructions."""

    def __init__(self, raw, source="<table>"):
        if not isinstance(raw, dict):
            raise EncodingTableError(f"{source}: expected a mapping")
        self.source = str(source)
        opcodes = raw.get("opcodes") or {}
        for name, value in opcodes.items():
            if not isinstance(value, int) or value & 0x3 != 0x3 or value > 0x7F:
                raise EncodingTableError(
                    f"opcode '{name}': {value!r} is not a 32-bit major opcode"
                )
        self.opcodes = dict(opcodes)
        self.formats = self._codes(raw, "formats", 15)
        self.features = self._codes(raw, "features", 63)
        self.config = self._codes(raw, "config", 0x7FF)

        for name in self.formats:
            if get_format(name).name != name:
                raise EncodingTableError(
                    f"formats: use the canonical name, not the alias '{name}'"
                )
        self.format_names = {code: name for name, code in self.formats.items()}

        instructions = raw.get("instructions") or {}
        self.instructions = {
            name: _parse_instruction(name, spec, self.opcodes, self.features)
            for name, spec in instructions.items()
        }
        insns = list(self.instructions.values())
        for i, a in enumerate(insns):
            for b in insns[i + 1 :]:
                if not (a.match ^ b.match) & a.mask & b.mask:
                    raise EncodingTableError(
                        f"instructions '{a.name}' and '{b.name}' have overlapping encodings"
                    )

    @staticmethod
    def _codes(raw, key, limit):
        out = raw.get(key) or {}
        if not isinstance(out, dict):
            raise EncodingTableError(f"{key}: expected a mapping of name -> code")
        for name, code in out.items():
            if not isinstance(code, int) or not 0 <= code <= limit:
                raise EncodingTableError(
                    f"{key}: '{name}' must map to an integer in 0-{limit}"
                )
        if len(set(out.values())) != len(out):
            raise EncodingTableError(f"{key}: codes must be distinct")
        return dict(out)

    def max_code(self, kind):
        """The largest value every ``kind`` operand (``tile``, ``in_fmt`` ...) can encode."""
        widths = [
            op.width
            for insn in self.instructions.values()
            for op in insn.operands
            if op.kind == kind
        ]
        return (1 << min(widths)) - 1 if widths else None


def load_table(path=ENCODINGS_PATH):
    with open(path) as f:
        return EncodingTable(yaml.safe_load(f), path)


class Decoder:
    """Instruction word -> ``(Instruction, {operand: value})``, or None if not ours.

    Every instruction fixes its opcode and funct3, so the lookup is one dict
    probe on those bits, then a mask/match on the candidate(s) for the
    reserved bits.
    """

    def __init__(self, table):
        self.table = table
        self.opcodes = frozenset(table.opcodes.values())
        self._by_key = {}
        for insn in table.instructions.values():
            fields = tuple(
                (op.name, op.lo, (1 << op.width) - 1) for op in insn.operands
            )
            self._by_key.setdefault(insn.match & _KEY_MASK, []).append((insn, fields))

    def decode(self, instr):
        for insn, fields in self._by_key.get(instr & _KEY_MASK, ()):
            if instr & insn.mask == insn.match:
                return insn, {name: (instr >> lo) & m for name, lo, m in fields}
        return None

    def encode(self, name, **operands):
        """Assemble ``name`` with the given operand values (missing ones are 0)."""
        insn = self.table.instructions[name]
        word = insn.match
        for op in insn.operands:
            value = operands.pop(op.name, 0)
            if not 0 <= value < (1 << op.width):
                raise ValueError(
                    f"{name}: {op.name}={value} does not fit {op.width} bits"
                )
            word |= value << op.lo
        if operands:
            raise ValueError(f"{name}: unknown operand(s) {sorted(operands)}")
        return word


def _ident(name):
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def header_defines(table):
    """``(name, value)`` pairs for ``naiad.h``: the table, as the C wrappers need it."""
    out = [(f"NAIAD_MFMT_{_ident(n)}", c) for n, c in table.formats.items()]
    out += [(f"NAIAD_MFEAT_{_ident(n)}", 1 << b) for n, b in table.features.items()]
    out += [(f"NAIAD_MCFG_{_ident(n)}", s) for n, s in table.config.items()]
    for insn in table.instructions.values():
        prefix = f"NAIAD_MI_{_ident(insn.name)}"
        out += [(f"{prefix}_OPCODE", insn.opcode), (f"{prefix}_FUNCT3", insn.funct3)]
        free_lo = _FORMS[insn.form][1]
        out += [
            (f"{prefix}_{_ident(op.name)}_SHIFT", op.lo - free_lo)
            for op in insn.operands
            if op.field is None
        ]
    return out


#: The shipped table and its decoder, built once at import.
TABLE = load_table()
DECODER = Decoder(TABLE)
