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

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from framework.util.schema import DesignError, parse_int
from framework.util.schema import check_keys as _check_keys

#: Single-letter extensions in the order ``-march`` wants them.
_SINGLE_LETTER_ORDER = "mafdqcbv"

#: The vector base extensions (either one gives the core a vector unit), and
#: the vector sub-extensions that need one.
VECTOR_EXTENSIONS = ("v", "zve64d")
VECTOR_SUBEXTENSIONS = ("zvfh", "zvfbfmin", "zvfbfwma")


@dataclass(frozen=True)
class CoreSpec:
    count: int = 1
    xlen: int = 32
    isa: tuple[str, ...] = ("i", "m")
    reset_pc: int = 0x1000
    #: Vector register width and largest element width in bits; None without a vector extension.
    vlen: int | None = None
    elen: int | None = None

    @property
    def has_vector(self):
        return any(e in VECTOR_EXTENSIONS for e in self.isa)

    @property
    def march(self):
        """The ``-march`` string for this ISA list, e.g. ``rv32im_zicsr``.

        A vector design also gets ``_zvl<VLEN>b``, so the compiler may assume
        the design's VLEN and Spike (whose ``--isa`` this string becomes) runs
        with it.
        """
        singles = sorted(
            (e for e in self.isa if len(e) == 1 and e != "i"),
            key=_SINGLE_LETTER_ORDER.index,
        )
        multis = [e for e in self.isa if len(e) > 1]
        if self.has_vector:
            multis.append(f"zvl{self.vlen}b")
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
    #: ``tile_mac``'s summation strategy: ``ieee``, or ``tensix_bh``/``tensix_wh``
    #: (the Tensix FPU bit for bit, at ``fidelity`` LoFi/HiFi2/HiFi3/HiFi4).
    numerics: str = "ieee"
    fidelity: str | None = None
    #: aload / astore (the ``acc_mem`` feature). Ztt has no memory <-> accumulator path.
    acc_mem: bool = True
    #: A draft the unit claims to model (``ztt``: Ztt v0.6), whose ranges are then enforced.
    profile: str | None = None


@dataclass(frozen=True)
class CountersSpec:
    """The design's ``counters:`` section: Zihpm bindings and on-demand event families.

    ``hpm`` is ``((n, event), ...)``: ``mhpmcounter<n>`` reads hub event ``event``
    (see ``framework.ip.counters``). The binding is fixed by the design, not
    chosen at run time through ``mhpmevent``, so ``naiad.h`` can give each event
    its CSR number as a constant (``csrr`` needs an immediate) and a kernel reads
    a counter with one instruction and no setup. Event names are checked against
    the built SoC's hub, since the hub discovers them from the IP.
    """

    hpm: tuple[tuple[int, str], ...] = ()
    #: On-demand families to count even when nothing binds them (for the report).
    enable: tuple[str, ...] = ()

    @property
    def events(self):
        return [event for _, event in self.hpm]


@dataclass
class Design:
    name: str
    cores: CoreSpec
    blocks: list[BlockSpec]
    program: ProgramSpec = field(default_factory=ProgramSpec)
    path: str | None = None
    matrix: MatrixSpec | None = None
    counters: CountersSpec = field(default_factory=CountersSpec)
    #: The cycle model (``framework.pe.rv.timing.core.TimingSpec``); None runs untimed.
    timing: object = None

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


def _parse_cores(raw):
    from framework.naiad.registry import NAIAD_ISA_REGISTRY
    from framework.pe.rv.rv64 import RV64_ISA_REGISTRY

    raw = {} if raw is None else raw
    _check_keys(raw, {"count", "xlen", "isa", "reset_pc", "vlen", "elen"}, "cores")
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
    if "zcd" in isa and ("d" not in isa or not {"c", "zca"} & set(isa)):
        raise DesignError(
            f"cores.isa: 'zcd' requires 'd' and 'zca' (or 'c'), got {list(isa)}"
        )
    vlen, elen = _parse_vector(raw, isa, xlen)
    return CoreSpec(
        count=count, xlen=xlen, isa=isa, reset_pc=reset_pc, vlen=vlen, elen=elen
    )


def _parse_vector(raw, isa, xlen):
    """``cores.vlen`` / ``cores.elen`` for a vector ISA: ``(vlen, elen)``, or ``(None, None)``."""
    bases = [e for e in isa if e in VECTOR_EXTENSIONS]
    subs = [e for e in isa if e in VECTOR_SUBEXTENSIONS]
    if not bases:
        if subs:
            raise DesignError(
                f"cores.isa: {subs} need a vector unit ('v' or 'zve64d'), got {list(isa)}"
            )
        given = [k for k in ("vlen", "elen") if k in raw]
        if given:
            raise DesignError(
                f"cores.{given[0]} given, but cores.isa has no vector extension ('v' or 'zve64d')"
            )
        return None, None
    if xlen != 64:
        raise DesignError("cores.isa: the vector unit is RV64 only")
    if "f" not in isa or "d" not in isa:
        raise DesignError(f"cores.isa: {bases} require 'f' and 'd', got {list(isa)}")
    if "zvfbfwma" in isa and ("zvfbfmin" not in isa or "zfbfmin" not in isa):
        raise DesignError("cores.isa: zvfbfwma requires zvfbfmin and zfbfmin")
    elen = parse_int(raw.get("elen", 64), "cores.elen")
    vlen = parse_int(raw.get("vlen", 128), "cores.vlen")
    if elen != 64:
        raise DesignError(f"cores.elen must be 64 (V and Zve64d), got {elen}")
    minimum = 128 if "v" in bases else 64
    if vlen < minimum or vlen > 65536 or vlen & (vlen - 1):
        raise DesignError(
            f"cores.vlen must be a power of two in {minimum}-65536 for {bases}, got {vlen}"
        )
    return vlen, elen


#: Coupling modes the schema knows; only AME is built so far.
MATRIX_MODES = ("ame", "ime", "vme")
_MATRIX_MAX_DIM = 256


def _parse_ime_matrix(raw, cores):
    """``matrix: {mode: ime, ...}``: Zvvm on the cores' vector unit (``framework.matrix.ime.parse_ime``).

    A Zvt (VME) extension on the same hart is refused: Zvt's ``vtle*``/``vtse*``
    are bit-identical to Zvvm's ``vmtl.v``/``vmts.v`` with an immediate lambda.
    """
    from framework.matrix.ime import parse_ime

    zvt = [e for e in cores.isa if e.startswith("zvt")]
    if zvt:
        raise DesignError(
            f"matrix.mode ime cannot share a hart with VME ({zvt}): Zvt's vtle*/vtse* and Zvvm's "
            f"vmtl.v/vmts.v occupy the same LOAD-FP/STORE-FP encodings (width 111, mew 1, mop 00)"
        )
    return parse_ime(raw, cores, DesignError)


def _parse_matrix(raw, cores=None):
    """The ``matrix:`` section, checked against the encodings table it will be decoded with.

    Top-level rather than under ``cores``: it is a block of its own with its own
    schema, every hart gets one instance, and keeping it out of ``CoreSpec``
    leaves the ISA list (and the ``-march`` derived from it) about the core.
    """
    from framework.matrix.encodings import TABLE
    from framework.matrix.formats import get_format

    if raw is None:
        return None
    if isinstance(raw, dict) and str(raw.get("mode", "ame")).lower() == "ime":
        return _parse_ime_matrix(raw, cores)
    if isinstance(raw, dict) and str(raw.get("mode", "ame")).lower() == "vme":
        # Zvt v0.4; refuses an IME (zvv*) hart.
        from framework.matrix.vme import parse_vme

        return parse_vme(raw, cores, DesignError)
    keys = {"mode", "m", "n", "k", "tiles", "accs", "formats", "features", "k_chunk"}
    _check_keys(raw, keys | {"numerics", "fidelity", "acc_mem", "profile"}, "matrix")
    mode = str(raw.get("mode", "ame")).lower()
    if mode not in MATRIX_MODES:
        raise DesignError(
            f"matrix.mode must be one of {list(MATRIX_MODES)}, got {mode!r}"
        )
    if mode != "ame":
        raise DesignError(
            f"matrix.mode '{mode}' is not implemented yet (only 'ame' and 'ime' are)"
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
    numerics, fidelity = _parse_matrix_numerics(raw, lists, k_chunk)
    acc_mem, profile = _parse_matrix_ame_options(raw, dims, features)
    spec = MatrixSpec(
        mode=mode,
        **dims,
        **counts,
        in_formats=lists["in"],
        acc_formats=lists["acc"],
        features=features,
        k_chunk=k_chunk,
        numerics=numerics,
        fidelity=fidelity,
        acc_mem=acc_mem,
        profile=profile,
    )
    if profile == "ztt":
        _check_ztt_profile(spec)
    return spec


#: Matrix profiles a design may claim (``matrix.profile``).
MATRIX_PROFILES = ("ztt",)
#: Ztt v0.6 ranges (riscv-attached-matrix-extension tag v0.6, ``src/parameters.adoc``):
#: AME_NUM_M_REGS and AME_NUM_ACC_REGS.
ZTT_TILES = (16, 32)
ZTT_ACCS = (1, 2, 4, 8, 16)
#: Our features with a Ztt v0.6 counterpart: transposed loads (mls/mss .tst/.cm),
#: tshift (mrowshift/mcolshift.ew.x), mmov.m.a/.a.m, mconv.ew, mmul{a,b}tacc.2d,
#: mreduce*/mprefix*. Not in Ztt: epilogue (ascal/aadds act on accumulators;
#: Ztt's element-wise ops act on M registers), semiring, shift2 and halo (proposals),
#: and acc_mem (Ztt has no memory <-> Acc path).
ZTT_FEATURES = ("transpose", "shift", "acc_move", "convert", "mma_transpose", "reduce")


def _parse_matrix_ame_options(raw, dims, features):
    """``matrix.acc_mem`` and ``matrix.profile``, and the shapes the newer features need."""
    acc_mem = raw.get("acc_mem", True)
    if not isinstance(acc_mem, bool):
        raise DesignError(f"matrix.acc_mem must be true or false, got {acc_mem!r}")
    if "acc_mem" in features:
        raise DesignError(
            "matrix.features: acc_mem is not listed; it is on unless matrix.acc_mem: false"
        )
    profile = raw.get("profile")
    if profile is not None:
        profile = str(profile).lower()
        if profile not in MATRIX_PROFILES:
            raise DesignError(
                f"matrix.profile must be one of {list(MATRIX_PROFILES)}, got {profile!r}"
            )
    m, n, k = dims["m"], dims["n"], dims["k"]
    if "shift2" in features and not m == n == k:
        raise DesignError(
            f"matrix.features: shift2 needs a square unit (m = n = k), got {m} x {n} x {k}"
        )
    if "acc_move" in features and k not in (m, n):
        raise DesignError(
            f"matrix.features: acc_move needs an M x N tile view (k = n or k = m), got {m} x {n} x {k}"
        )
    if "mma_transpose" in features and (m > max(k, n) or n > max(m, k)):
        raise DesignError(
            f"matrix.features: mma_transpose needs m <= max(k, n) and n <= max(m, k), got {m} x {n} x {k}"
        )
    return acc_mem, profile


def _check_ztt_profile(spec):
    """``profile: ztt``: the unit must be one Ztt v0.6 allows, within what Naiad models.

    Ztt squares are N x N with N = sqrt(AME_NELEM), AME_NELEM a power of 4, and
    the accumulator one square. A matmul's K is N times the operand's
    pack_factor, so it equals N for every format at least AME_UNIT_DATATYPE_SIZE
    wide; Naiad's unit has one K for every format, so the profile takes
    AME_UNIT_DATATYPE_SIZE = 16 bits and admits no narrower (packed) input
    format. Ztt requires a deterministic, accumulator-seeded fold one product at
    a time, which is ``k_chunk: 1`` (each product rounded to the compute format,
    then added: unfused). Each Naiad tile holds one square of the widest format,
    i.e. one Ztt register *group*, so ``tiles`` counts groups.
    """
    from framework.matrix.formats import get_format

    bad = []
    if not spec.m == spec.n == spec.k:
        bad.append(f"squares must be N x N x N, got {spec.m} x {spec.n} x {spec.k}")
    elif spec.m & (spec.m - 1):
        bad.append(f"N must be a power of two (AME_NELEM a power of 4), got {spec.m}")
    if spec.tiles not in ZTT_TILES:
        bad.append(
            f"tiles must be one of {list(ZTT_TILES)} (AME_NUM_M_REGS), got {spec.tiles}"
        )
    if spec.accs not in ZTT_ACCS:
        bad.append(
            f"accs must be one of {list(ZTT_ACCS)} (AME_NUM_ACC_REGS), got {spec.accs}"
        )
    if spec.acc_mem:
        bad.append("acc_mem must be false (Ztt has no memory <-> accumulator path)")
    if spec.k_chunk != 1:
        bad.append(
            f"k_chunk must be 1 (Ztt's one-product-at-a-time fold), got {spec.k_chunk}"
        )
    if spec.numerics != "ieee":
        bad.append(f"numerics must be ieee, got {spec.numerics}")
    extra = [f for f in spec.features if f not in ZTT_FEATURES]
    if extra:
        bad.append(f"features {extra} are not in Ztt (allowed: {list(ZTT_FEATURES)})")
    packed = [f for f in spec.in_formats if get_format(f).dtype.itemsize < 2]
    if packed:
        bad.append(
            f"input formats {packed} would be packed (K = 2N or more), which this unit does not model"
        )
    if bad:
        raise DesignError("matrix.profile ztt: " + "; ".join(bad))


def _parse_matrix_numerics(raw, lists, k_chunk):
    """``matrix.numerics`` / ``matrix.fidelity``: a Tensix mode only with the formats it models exactly."""
    from framework.matrix.tensix_numerics import (
        MVMUL_K,
        TENSIX_NUMERICS,
        check_formats,
        fidelity_phases,
    )

    numerics = str(raw.get("numerics", "ieee")).lower()
    fidelity = raw.get("fidelity")
    if numerics == "ieee":
        if fidelity is not None:
            raise DesignError("matrix.fidelity applies only to tensix numerics")
        return numerics, None
    if numerics not in TENSIX_NUMERICS:
        raise DesignError(
            f"matrix.numerics must be one of {['ieee', *TENSIX_NUMERICS]}, got {numerics!r}"
        )
    try:
        for in_fmt in lists["in"]:
            for acc_fmt in lists["acc"]:
                check_formats(numerics, in_fmt, acc_fmt)
        if fidelity is not None:
            fidelity = str(fidelity)
            fidelity_phases(fidelity)
    except ValueError as e:
        raise DesignError(f"matrix: {e}") from None
    if k_chunk not in (None, MVMUL_K):
        raise DesignError(
            f"matrix.k_chunk is fixed at {MVMUL_K} (one MVMUL) under {numerics} numerics, got {k_chunk}"
        )
    return numerics, fidelity


def _parse_counters(raw, cores):
    """The ``counters:`` section: ``hpm: {n: event}`` and ``enable: [family, ...]``."""
    from framework.ip.counters import FAMILIES
    from framework.pe.rv.isa.zicsr64_isa import HPM_COUNTERS

    if raw is None:
        return CountersSpec()
    _check_keys(raw, {"hpm", "enable"}, "counters")
    hpm_raw = raw.get("hpm") or {}
    if not isinstance(hpm_raw, dict):
        raise DesignError("counters.hpm: expected a mapping of counter number -> event")
    hpm = {}
    for key, event in hpm_raw.items():
        n = parse_int(key, "counters.hpm key")
        if n not in HPM_COUNTERS:
            raise DesignError(
                f"counters.hpm: no counter {key!r}; Zihpm counters are 3-31"
            )
        if n in hpm:
            raise DesignError(f"counters.hpm: counter {n} is bound twice")
        if not isinstance(event, str) or not event:
            raise DesignError(
                f"counters.hpm.{n}: expected an event name, got {event!r}"
            )
        hpm[n] = event
    events = list(hpm.values())
    dupes = sorted({e for e in events if events.count(e) > 1})
    if dupes:
        raise DesignError(
            f"counters.hpm: event(s) {dupes} bound to more than one counter (naiad.h names one per event)"
        )
    if hpm and (
        cores.xlen != 64 or "zicsr" not in cores.isa or "zihpm" not in cores.isa
    ):
        raise DesignError(
            "counters.hpm needs an RV64 core with zicsr and zihpm in cores.isa (the Zihpm CSRs live in CSRFile64)"
        )
    enable = tuple(str(f).lower() for f in raw.get("enable") or ())
    unknown = [f for f in enable if f not in (*FAMILIES, "all")]
    if unknown:
        raise DesignError(
            f"counters.enable: unknown {unknown}; known: {[*FAMILIES, 'all']}"
        )
    return CountersSpec(hpm=tuple(sorted(hpm.items())), enable=enable)


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
    _validate_multicore(design)
    return design


def _validate_multicore(design):
    """``cores.count > 1``: what the generated multi-hart crt0 and platform library rely on.

    crt0 reads ``mhartid`` (Zicsr, RV64's per-hart CSR file) to pick each
    hart's stack and releases the secondary harts through the CLINT's ``msip``;
    the platform's barrier and console lock need atomics (A or Zaamo).
    """
    cores = design.cores
    n = cores.count
    if n == 1:
        return
    what = f"cores.count is {n}: a multi-hart design needs"
    if cores.xlen != 64:
        raise DesignError(f"{what} xlen 64 (the RV32 CSR file has no per-hart mhartid)")
    if "zicsr" not in cores.isa:
        raise DesignError(f"{what} zicsr in cores.isa (crt0 reads mhartid)")
    if not {"a", "zaamo"} & set(cores.isa):
        raise DesignError(
            f"{what} a (or zaamo) in cores.isa (the barrier and console lock are AMOs)"
        )
    clints = design.blocks_of_type("clint")
    if not clints or clints[0].base is None:
        raise DesignError(
            f"{what} a mapped clint block (crt0 releases secondary harts through msip)"
        )
    if 4 * n > 0x4000:
        raise DesignError(
            f"{what} at most {0x4000 // 4} harts (the CLINT's msip array)"
        )
    if n * design.program.stack_size >= design.program_memory.size:
        raise DesignError(
            f"{what} {n} stacks of program.stack_size {hex(design.program.stack_size)}, "
            f"which do not fit block '{design.program_memory.name}'"
        )


def parse_design(raw, path=None):
    """Build and validate a :class:`Design` from an already-loaded mapping."""
    _check_keys(
        raw,
        {
            "name",
            "description",
            "cores",
            "blocks",
            "program",
            "matrix",
            "counters",
            "timing",
        },
        "design",
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
    cores = _parse_cores(raw.get("cores"))
    design = Design(
        name=str(raw.get("name", default_name)),
        cores=cores,
        blocks=[_parse_block(b, i) for i, b in enumerate(blocks_raw)],
        program=program,
        path=None if path is None else str(path),
        matrix=_parse_matrix(raw.get("matrix"), cores),
        counters=_parse_counters(raw.get("counters"), cores),
    )
    from framework.naiad.timing import parse_timing

    design.timing = parse_timing(raw.get("timing"), design)
    return validate(design)


class _UniqueKeyLoader(yaml.SafeLoader):
    """``SafeLoader`` that refuses a mapping with a repeated key (PyYAML keeps the last)."""

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise DesignError(
                    f"line {key_node.start_mark.line + 1}: key {key!r} appears twice"
                )
            keys.add(key)
        return super().construct_mapping(node, deep)


def load_design(path):
    with open(path) as f:
        raw = yaml.load(f, Loader=_UniqueKeyLoader)
    if raw is None:
        raise DesignError(f"'{path}' is empty")
    return parse_design(raw, path)
