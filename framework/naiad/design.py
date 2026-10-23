"""Naiad design files: YAML -> validated dataclasses.

A design names the cores, the IP blocks and, optionally, the matrix unit every
hart gets (a top-level ``matrix:`` section; see :func:`_parse_matrix`). Each block is
``{name, type, base, window, params}``: ``type`` is a key of
:data:`~framework.naiad.registry.IP_REGISTRY`, ``window`` is the address space
reserved for the block, and the block's *size* (what is actually mapped, from
``base`` up) is the IP class's answer for its ``params`` and must fit inside the
window. A block type that lives on ELF symbols rather than a fixed address (the
HTIF) takes neither ``base`` nor ``window``.

New IP is added by registering a class and naming its type in a design; the
builder never special-cases a block by name.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Single-letter extensions in the order ``-march`` wants them.
_SINGLE_LETTER_ORDER = "mafdqcbv"

_SIZE_SUFFIXES = {
    "": 1,
    "k": 1 << 10,
    "kib": 1 << 10,
    "m": 1 << 20,
    "mib": 1 << 20,
    "g": 1 << 30,
    "gib": 1 << 30,
}


class DesignError(ValueError):
    pass


def parse_int(value, what):
    """An int from YAML: a plain int, or a string like ``0x1000`` / ``64KiB``."""
    if isinstance(value, bool):
        raise DesignError(f"{what}: expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(0x[0-9a-fA-F_]+|\d[\d_]*)\s*([a-zA-Z]*)\s*", value)
        if m and m.group(2).lower() in _SIZE_SUFFIXES:
            return int(m.group(1), 0) * _SIZE_SUFFIXES[m.group(2).lower()]
    raise DesignError(f"{what}: expected an integer, got {value!r}")


@dataclass(frozen=True)
class CoreSpec:
    count: int = 1
    xlen: int = 32
    isa: tuple[str, ...] = ("i", "m")
    reset_pc: int = 0x1000

    @property
    def march(self):
        """The ``-march`` string for this ISA list, e.g. ``rv32im_zicsr``."""
        singles = sorted(
            (e for e in self.isa if len(e) == 1 and e != "i"),
            key=_SINGLE_LETTER_ORDER.index,
        )
        multis = [e for e in self.isa if len(e) > 1]
        return "_".join([f"rv{self.xlen}i" + "".join(singles)] + multis)

    @property
    def mabi(self):
        base = "ilp32" if self.xlen == 32 else "lp64"
        if "d" in self.isa:
            return base + "d"
        if "f" in self.isa:
            return base + "f"
        return base


@dataclass(frozen=True)
class BlockSpec:
    name: str
    type: str
    base: int | None = None
    window: int | None = None
    params: dict = field(default_factory=dict)

    @property
    def ip_class(self):
        from framework.naiad.registry import IP_REGISTRY

        return IP_REGISTRY[self.type]

    @property
    def size(self):
        """Bytes actually mapped from ``base`` (``None`` for unmapped blocks)."""
        if not self.ip_class.mapped:
            return None
        return self.ip_class.block_size(self.params, self.window)


@dataclass(frozen=True)
class ProgramSpec:
    """Where the generated linker script puts a program."""

    #: The block a program is linked into (default: the first ``dram`` block).
    memory: str | None = None
    stack_size: int = 64 * 1024


@dataclass(frozen=True)
class MatrixSpec:
    """The matrix unit every hart gets (the design's top-level ``matrix:`` section).

    Formats are canonical ``framework.matrix.formats`` names and fixed at
    instantiation: an instruction naming any other format is illegal, as is one
    needing a feature not listed here.
    """

    mode: str = "ame"
    m: int = 16
    n: int = 16
    k: int = 16
    tiles: int = 8
    accs: int = 4
    in_formats: tuple[str, ...] = ("fp32",)
    acc_formats: tuple[str, ...] = ("fp32",)
    features: tuple[str, ...] = ()
    #: ``tile_mac``'s bit-exact mode: round the accumulator every ``k_chunk`` of K.
    k_chunk: int | None = None


@dataclass
class Design:
    name: str
    cores: CoreSpec
    blocks: list[BlockSpec]
    program: ProgramSpec = field(default_factory=ProgramSpec)
    path: str | None = None
    matrix: MatrixSpec | None = None

    def block(self, name):
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"design '{self.name}' has no block named '{name}'")

    def blocks_of_type(self, type_name):
        return [b for b in self.blocks if b.type == type_name]

    @property
    def program_memory(self):
        """The :class:`BlockSpec` programs are linked into."""
        if self.program.memory is not None:
            return self.block(self.program.memory)
        drams = self.blocks_of_type("dram")
        if not drams:
            raise DesignError(
                f"design '{self.name}' has no dram block and names no program.memory"
            )
        return drams[0]


def _check_keys(mapping, allowed, what):
    if not isinstance(mapping, dict):
        raise DesignError(f"{what}: expected a mapping, got {type(mapping).__name__}")
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise DesignError(
            f"{what}: unknown key(s) {unknown}; expected some of {sorted(allowed)}"
        )


def _parse_cores(raw):
    from framework.naiad.registry import NAIAD_ISA_REGISTRY
    from framework.pe.rv.rv64 import RV64_ISA_REGISTRY

    raw = {} if raw is None else raw
    _check_keys(raw, {"count", "xlen", "isa", "reset_pc"}, "cores")
    count = parse_int(raw.get("count", 1), "cores.count")
    xlen = parse_int(raw.get("xlen", 32), "cores.xlen")
    isa = tuple(str(e).lower() for e in raw.get("isa", ("i", "m")))
    reset_pc = parse_int(raw.get("reset_pc", 0x1000), "cores.reset_pc")
    if count < 1:
        raise DesignError(f"cores.count must be >= 1, got {count}")
    if xlen not in (32, 64):
        raise DesignError(f"cores.xlen must be 32 or 64, got {xlen}")
    if "i" not in isa:
        raise DesignError(f"cores.isa must include the base 'i', got {list(isa)}")
    registry = RV64_ISA_REGISTRY if xlen == 64 else NAIAD_ISA_REGISTRY
    unknown = [e for e in isa if e not in registry]
    if unknown:
        raise DesignError(
            f"cores.isa: unknown rv{xlen} extension(s) {unknown}; known: {sorted(registry)}"
        )
    if len(set(isa)) != len(isa):
        raise DesignError(f"cores.isa lists an extension twice: {list(isa)}")
    needs_f = [e for e in ("d", "zfbfmin") if e in isa and "f" not in isa]
    if needs_f:
        raise DesignError(f"cores.isa: {needs_f} require 'f', got {list(isa)}")
    return CoreSpec(count=count, xlen=xlen, isa=isa, reset_pc=reset_pc)


#: Coupling modes the schema knows; only AME is built so far.
MATRIX_MODES = ("ame", "ime", "vme")
_MATRIX_MAX_DIM = 256


def _parse_matrix(raw):
    """The ``matrix:`` section, checked against the encodings table it will be decoded with.

    Top-level rather than under ``cores``: it is a block of its own with its own
    schema, every hart gets one instance, and keeping it out of ``CoreSpec``
    leaves the ISA list (and the ``-march`` derived from it) about the core.
    """
    from framework.matrix.encodings import TABLE
    from framework.matrix.formats import get_format

    if raw is None:
        return None
    keys = {"mode", "m", "n", "k", "tiles", "accs", "formats", "features", "k_chunk"}
    _check_keys(raw, keys, "matrix")
    mode = str(raw.get("mode", "ame")).lower()
    if mode not in MATRIX_MODES:
        raise DesignError(
            f"matrix.mode must be one of {list(MATRIX_MODES)}, got {mode!r}"
        )
    if mode != "ame":
        raise DesignError(
            f"matrix.mode '{mode}' is not implemented yet (only 'ame' is)"
        )
    dims = {}
    for key, default in (("m", 16), ("n", 16), ("k", 16)):
        dims[key] = parse_int(raw.get(key, default), f"matrix.{key}")
        if not 1 <= dims[key] <= _MATRIX_MAX_DIM:
            raise DesignError(
                f"matrix.{key} must be in 1-{_MATRIX_MAX_DIM}, got {dims[key]}"
            )
    counts = {}
    for key, kind, default in (("tiles", "tile", 8), ("accs", "acc", 4)):
        counts[key] = parse_int(raw.get(key, default), f"matrix.{key}")
        limit = TABLE.max_code(kind) + 1
        if not 1 <= counts[key] <= limit:
            raise DesignError(
                f"matrix.{key} must be in 1-{limit} (the encoding's {kind} field), got {counts[key]}"
            )

    formats = raw.get("formats") or {}
    _check_keys(formats, {"in", "acc"}, "matrix.formats")
    lists = {}
    for role, kind in (("in", "in_fmt"), ("acc", "acc_fmt")):
        names = formats.get(role, ["fp32"])
        if not isinstance(names, list) or not names:
            raise DesignError(f"matrix.formats.{role} must be a non-empty list")
        canonical = []
        for name in names:
            try:
                fmt = get_format(str(name).lower())
            except ValueError as e:
                raise DesignError(f"matrix.formats.{role}: {e}") from None
            if fmt.name not in TABLE.formats:
                raise DesignError(
                    f"matrix.formats.{role}: {fmt.name} has no code in the encodings table"
                )
            if TABLE.formats[fmt.name] > TABLE.max_code(kind):
                raise DesignError(
                    f"matrix.formats.{role}: {fmt.name} (code {TABLE.formats[fmt.name]}) does not fit the {kind} field"
                )
            canonical.append(fmt.name)
        if len(set(canonical)) != len(canonical):
            raise DesignError(
                f"matrix.formats.{role} lists a format twice: {canonical}"
            )
        lists[role] = tuple(canonical)
    for name in lists["acc"]:
        if name in ("int8", "fp8_e4m3", "fp8_e5m2"):
            raise DesignError(
                f"matrix.formats.acc: {name} is not an accumulator format"
            )
    kinds = {get_format(n).is_float for n in lists["acc"]}
    for name in lists["in"]:
        if get_format(name).is_float not in kinds:
            raise DesignError(
                f"matrix.formats.in: {name} has no accumulator of its kind in {list(lists['acc'])}"
            )

    features = tuple(str(f).lower() for f in raw.get("features") or ())
    unknown = [f for f in features if f not in TABLE.features]
    if unknown:
        raise DesignError(
            f"matrix.features: unknown {unknown}; known: {sorted(TABLE.features)}"
        )
    k_chunk = raw.get("k_chunk")
    if k_chunk is not None:
        k_chunk = parse_int(k_chunk, "matrix.k_chunk")
        if k_chunk < 1:
            raise DesignError(f"matrix.k_chunk must be >= 1, got {k_chunk}")
    return MatrixSpec(
        mode=mode,
        **dims,
        **counts,
        in_formats=lists["in"],
        acc_formats=lists["acc"],
        features=features,
        k_chunk=k_chunk,
    )


def _parse_block(raw, index):
    from framework.naiad.registry import IP_REGISTRY

    what = f"blocks[{index}]"
    _check_keys(raw, {"name", "type", "base", "window", "params"}, what)
    for key in ("name", "type"):
        if key not in raw:
            raise DesignError(f"{what}: missing '{key}'")
    name, type_name = str(raw["name"]), str(raw["type"])
    what = f"block '{name}'"
    if type_name not in IP_REGISTRY:
        raise DesignError(
            f"{what}: unknown type '{type_name}'; known types: {sorted(IP_REGISTRY)}"
        )
    ip_class = IP_REGISTRY[type_name]
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise DesignError(f"{what}: params must be a mapping")
    unknown = sorted(set(params) - set(ip_class.PARAMS))
    if unknown:
        raise DesignError(
            f"{what}: unknown param(s) {unknown} for type '{type_name}'; known: {sorted(ip_class.PARAMS)}"
        )
    if not ip_class.mapped:
        if "base" in raw or "window" in raw:
            raise DesignError(
                f"{what}: type '{type_name}' is not memory-mapped, so takes no base/window"
            )
        spec = BlockSpec(name=name, type=type_name, params=params)
    else:
        for key in ("base", "window"):
            if key not in raw:
                raise DesignError(f"{what}: missing '{key}'")
        base = parse_int(raw["base"], f"{what}.base")
        window = parse_int(raw["window"], f"{what}.window")
        if base < 0 or window <= 0:
            raise DesignError(f"{what}: base must be >= 0 and window > 0")
        spec = BlockSpec(
            name=name, type=type_name, base=base, window=window, params=params
        )
    ip_class.validate_params(spec, DesignError)
    if spec.size is not None and not 0 < spec.size <= spec.window:
        raise DesignError(
            f"{what}: size {hex(spec.size)} does not fit its window {hex(spec.window)}"
        )
    return spec


def validate(design):
    """Cross-block checks: unique names, disjoint windows, reachable reset PC."""
    names = [b.name for b in design.blocks]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise DesignError(f"duplicate block name(s): {dupes}")

    mapped = sorted(
        (b for b in design.blocks if b.base is not None), key=lambda b: b.base
    )
    for prev, cur in zip(mapped, mapped[1:]):
        if cur.base < prev.base + prev.window:
            raise DesignError(
                f"block '{cur.name}' window [{hex(cur.base)}, {hex(cur.base + cur.window)}) overlaps "
                f"block '{prev.name}' window [{hex(prev.base)}, {hex(prev.base + prev.window)})"
            )
    limit = 1 << design.cores.xlen
    for b in mapped:
        if b.base + b.window > limit:
            raise DesignError(
                f"block '{b.name}' window ends above the {design.cores.xlen}-bit address space"
            )

    for b in design.blocks:
        b.ip_class.validate_in_design(b, design, DesignError)

    pc = design.cores.reset_pc
    if not any(b.base <= pc < b.base + b.size for b in mapped):
        raise DesignError(f"cores.reset_pc {hex(pc)} is not inside any mapped block")

    prog = design.program_memory
    if prog.base is None:
        raise DesignError(f"program.memory '{prog.name}' is not a memory-mapped block")
    if design.program.stack_size >= prog.size:
        raise DesignError(
            f"program.stack_size {hex(design.program.stack_size)} does not fit block '{prog.name}'"
        )
    return design


def parse_design(raw, path=None):
    """Build and validate a :class:`Design` from an already-loaded mapping."""
    _check_keys(
        raw, {"name", "description", "cores", "blocks", "program", "matrix"}, "design"
    )
    blocks_raw = raw.get("blocks")
    if not isinstance(blocks_raw, list) or not blocks_raw:
        raise DesignError("design: 'blocks' must be a non-empty list")
    program_raw = raw.get("program") or {}
    _check_keys(program_raw, {"memory", "stack_size"}, "program")
    program = ProgramSpec(
        memory=program_raw.get("memory"),
        stack_size=parse_int(
            program_raw.get("stack_size", ProgramSpec.stack_size), "program.stack_size"
        ),
    )
    default_name = Path(path).stem if path else "naiad"
    design = Design(
        name=str(raw.get("name", default_name)),
        cores=_parse_cores(raw.get("cores")),
        blocks=[_parse_block(b, i) for i, b in enumerate(blocks_raw)],
        program=program,
        path=None if path is None else str(path),
        matrix=_parse_matrix(raw.get("matrix")),
    )
    return validate(design)


def load_design(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise DesignError(f"'{path}' is empty")
    return parse_design(raw, path)
