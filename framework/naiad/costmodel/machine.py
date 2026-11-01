"""The design parameters the cost model reads, flattened out of a Naiad design.

:class:`Machine` is everything the per-dwarf formulas need to know about the
hardware, and nothing else: issue width (one instruction a cycle, or two at
``timing.scalar.issue_width: 2``), per-class
scalar latencies, the data region's latency and bandwidth, the vector unit's
datapath / LSU, and the matrix unit's shape, peak and transfer paths. Every
field comes from the design's ``timing:`` section (or that section's documented
default), so a compiler that knows a target's timing table can build one
without a Naiad design file (:meth:`Machine.from_dict` takes plain values).

Nothing here is fitted: a field is either read from the design or is a default
of the simulator's own spec classes.
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

#: Element widths (bytes) of the formats the dwarfs use.
FORMAT_BYTES = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int32": 4}


@dataclass(frozen=True)
class Scalar:
    """In-order core: result latencies per class (1 = back to back); ``issue_width`` 1 or 2."""

    #: ``timing.scalar.issue_width``. At 2 the engine pairs adjacent instructions by the simulator's
    #: rules (two ALU pipes; one each of memory, mul/div, FP and vector/matrix dispatch).
    issue_width: int = 1
    alu: int = 1
    mul: int = 3
    div: int = 20
    div_pipelined: bool = False
    load: int = 2
    store: int = 1
    branch_taken: int = 2
    csr: int = 1
    fp_add: int = 4
    fp_mul: int = 4
    fp_fma: int = 4
    fp_div: int = 20
    fp_sqrt: int = 20
    fp_div_pipelined: bool = False
    fp_misc: int = 2


@dataclass(frozen=True)
class Memory:
    """The region kernels keep their data in (the design's program memory)."""

    #: Cycles beyond ``Scalar.load`` (a tile/vector transfer adds it to its own latency).
    latency: int = 0
    #: Bytes per cycle through its port; None is unlimited.
    bandwidth: float | None = None


@dataclass(frozen=True)
class Vector:
    vlen: int = 256
    datapath_bits: int = 256
    startup: int = 4
    div_ii: int = 8
    mem_startup: int = 2
    mem_bytes_per_cycle: float = 32
    indexed_elements_per_cycle: float = 1
    #: Element-group chaining (``timing.vector.chaining``): the engine mirrors ``VectorTiming``'s rule.
    chaining: bool = False

    @property
    def vlenb(self):
        return self.vlen // 8


@dataclass(frozen=True)
class AME:
    """An attached matrix unit (Naiad ``mode: ame``) and its ``timing.matrix``."""

    m: int = 16
    n: int = 16
    k: int = 16
    tiles: int = 8
    accs: int = 4
    features: tuple = ()
    acc_mem: bool = True
    macs_per_cycle: int = 512
    mma_ii: int | None = None
    format_passes: dict = field(default_factory=dict, hash=False)
    pipeline_depth: int = 4
    acc_forwarding: bool = True
    overlap: bool = True
    tile_load_bandwidth: float | None = 64
    tile_load_latency: int = 1
    tile_store_bandwidth: float | None = 64
    tile_store_latency: int = 1
    azero: int = 1
    mcfg: int = 1
    tshift: int = 1
    epilogue: int = 1
    acc_to_tile_rows_per_cycle: int = 4
    acc_to_tile_latency: int = 2
    tile_to_acc_rows_per_cycle: int = 8
    tile_to_acc_latency: int = 4
    tconv: int | None = None
    treduce_rows_per_cycle: int = 8
    treduce_latency: int = 5
    tshift2: int | None = None
    ewise_bytes_per_cycle: float | None = 64
    ewise_latency: int = 4
    ewise_tx_factor: int = 4
    ewise_tx_latency: int = 12
    ewise_pipe: str = "ewise"

    def ewise_ii(self, fmt, tx=False):
        """Interval of one element-wise accumulator op (``aew.*`` / ``amacc.*``; ``aewu`` when ``tx``)."""
        nb = self.m * self.n * FORMAT_BYTES[fmt]
        ii = (
            1
            if self.ewise_bytes_per_cycle is None
            else max(1, math.ceil(nb / self.ewise_bytes_per_cycle))
        )
        return ii * (self.ewise_tx_factor if tx else 1)

    def ii(self, fmt, m=None, n=None, k=None):
        """Initiation interval of one ``m x n x k`` mma on in-format ``fmt`` (the full tile by default)."""
        if self.mma_ii is not None:
            return self.mma_ii
        m, n, k = m or self.m, n or self.n, k or self.k
        passes = self.format_passes.get(fmt, 1)
        return max(1, math.ceil(m * n * k * passes / self.macs_per_cycle))

    def peak_macs(self, fmt):
        """MACs per cycle a back-to-back stream of full-tile mmas sustains."""
        return self.m * self.n * self.k / self.ii(fmt)


@dataclass(frozen=True)
class IME:
    """Zvvm IME (``mode: ime``): the parameters of :class:`framework.matrix.ime_timing.IMETimingSpec`."""

    spec: object = None


@dataclass(frozen=True)
class VME:
    """Zvt VME (``mode: vme``): the parameters of :class:`framework.matrix.vme_timing.VMETimingSpec`."""

    te: int = 16
    spec: object = None


@dataclass(frozen=True)
class Machine:
    """One hart's cost-relevant parameters."""

    scalar: Scalar = Scalar()
    memory: Memory = Memory()
    vector: Vector | None = None
    #: ``"ame"``, ``"ime"``, ``"vme"`` or None.
    mode: str | None = None
    ame: AME | None = None
    ime: IME | None = None
    vme: VME | None = None
    harts: int = 1
    name: str = ""
    #: The parsed design this was read from (IME / VME formulas hand it to the emitters' own models).
    design: object = field(default=None, compare=False, repr=False)

    @property
    def load_latency(self):
        """Load-to-use cycles of a scalar load from the data region."""
        return self.scalar.load + self.memory.latency

    def with_(self, **kw):
        return replace(self, **kw)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_design(cls, design):
        """A :class:`Machine` from a Naiad ``Design`` (or a path to its YAML)."""
        if not hasattr(design, "cores"):
            from framework.naiad.design import load_design

            design = load_design(Path(design))
        t = design.timing
        if t is None:
            raise ValueError(
                f"design '{design.name}' has no timing: section; the cost model predicts timed runs"
            )
        s = t.scalar
        scalar = Scalar(**{f: getattr(s, f) for f in Scalar.__dataclass_fields__})
        pm = design.program_memory
        r = t.region(pm.name, pm.type)
        memory = Memory(latency=r.latency, bandwidth=r.bandwidth)
        vector = None
        if t.vector is not None:
            v = t.vector
            vector = Vector(
                vlen=design.cores.vlen,
                datapath_bits=v.datapath_bits,
                startup=v.startup,
                div_ii=v.div_ii,
                mem_startup=v.mem_startup,
                mem_bytes_per_cycle=v.mem_bytes_per_cycle,
                indexed_elements_per_cycle=v.indexed_elements_per_cycle,
                chaining=v.chaining,
            )
        mode = design.matrix.mode if design.matrix is not None else None
        ame = ime = vme = None
        if mode == "ame":
            mx, tm = design.matrix, t.matrix
            ame = AME(
                m=mx.m,
                n=mx.n,
                k=mx.k,
                tiles=mx.tiles,
                accs=mx.accs,
                features=tuple(mx.features),
                acc_mem=mx.acc_mem,
                **{
                    f: getattr(tm, f)
                    for f in AME.__dataclass_fields__
                    if hasattr(tm, f)
                },
            )
        elif mode == "ime":
            ime = IME(spec=t.matrix)
        elif mode == "vme":
            vme = VME(te=design.matrix.te, spec=t.matrix)
        return cls(
            scalar=scalar,
            memory=memory,
            vector=vector,
            mode=mode,
            ame=ame,
            ime=ime,
            vme=vme,
            harts=design.cores.count,
            name=design.name,
            design=design,
        )

    @classmethod
    def from_dict(cls, raw):
        """A :class:`Machine` from plain values: ``{scalar: {...}, memory: {...}, vector: {...}, ame: {...}}``."""
        ame = raw.get("ame")
        return cls(
            scalar=Scalar(**raw.get("scalar", {})),
            memory=Memory(**raw.get("memory", {})),
            vector=Vector(**raw["vector"]) if raw.get("vector") is not None else None,
            mode="ame" if ame is not None else None,
            ame=AME(**ame) if ame is not None else None,
            harts=raw.get("harts", 1),
            name=raw.get("name", ""),
        )
