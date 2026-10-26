"""The RV64 core cycle model: the ``timing:`` spec, the memory regions, and the cost state on each core.

Off by default. A design without a ``timing:`` section builds cores whose
``rv_cost`` is ``None``, so they run exactly the untimed interpreter (one
instruction per cycle) at its full speed. With one, the Naiad SoC
(:func:`framework.naiad.timing.install`, which also parses the section with
:func:`framework.naiad.timing.parse_timing`) gives every
hart a :class:`NaiadCostState`, the same hook the Tenstorrent baby cores' cost
model uses (``RV32I.clock_tick`` asks ``rv_cost.can_issue(instr, cycle,
register_file)`` before executing a word, and a False is a stall: the word is
re-offered next cycle). Functional semantics never change: an instruction still
executes completely in the tick it issues, and the model only decides which
tick that is and when its result may be read.

**The scalar model** is an in-order, single-issue core with a scoreboard:

* Every x- and f-register has a ready cycle. An instruction issues once all its
  source registers are ready (stall reason ``raw``, or ``mem`` when the
  producer was a load), and then sets its destination ready at
  ``issue + latency`` for its class. A latency of 1 is back-to-back.
* Divides (and FP divide/sqrt) are unpipelined unless configured otherwise: a
  second one waits for the unit (``structural``). Everything else pipelines.
* A taken branch or a jump holds the next instruction for ``branch_taken``
  cycles (``branch``). There is no predictor.
* Loads cost ``load`` plus the latency of the region the address falls in;
  stores are posted and hold the load/store unit for ``store`` cycles. Both hold
  the region's port for ``ceil(bytes / bandwidth)`` cycles, and a second access
  to a busy port waits (``mem``). AMOs are timed as loads.
* A load (or AMO) whose bytes an asynchronous store -- an AME tile/acc
  store, an RVV / IME / VME store on the vector LSU, on any hart -- is still
  writing waits until that store completes (``mem_raw``): the per-SoC table in
  :mod:`framework.pe.rv.timing.store_hazards` holds the policy.
* ``issue_width`` is 1 (the default) or 2: see **Dual issue** below.

**Dual issue** (``issue_width: 2``, opt-in) is an in-order two-wide front end
in the style of SiFive's U74 or SpacemiT's X60: two ALU pipes, and one each of
memory, multiply/divide, FP and vector/matrix dispatch. The interpreter still
executes one instruction per tick, so the SoC clock gives every core
``issue_width`` ticks (*issue slots*) per cycle, all with the same cycle number,
before it ticks the other blocks (DMA, CLINT wake-up, the Tensix tile) once,
and ``cycle`` / ``mcycle`` count cycles, not slots
(:meth:`framework.naiad.builder.NaiadClock.set_issue_width`). Slot 0 is the
single-issue model unchanged. The next instruction issues in slot 1 of the
same cycle only if:

* slot 0 issued an instruction this cycle, and it was not a branch or jump
  (control ends the issue group, taken or not);
* neither is a CSR access or a system word (``ecall``, ``fence``, ``wfi``,
  unknown words): those issue alone;
* they do not both need one of the single pipes: memory (scalar loads, stores
  and AMOs, RVV loads/stores, AME tile loads/stores, Zvvm/Zvt tile loads and
  stores), multiply/divide, FP, or the vector/matrix dispatch port (every RVV,
  Zvvm, Zvt or AME word, ``vset{i}vl{i}`` and ``mset*`` included). So a vector
  or matrix instruction pairs with scalar ALU work, a scalar multiply or a
  scalar FP op, never with another vector or matrix instruction, and a vector
  or tile memory access never pairs with a scalar one;
* it passes every check the single-issue model makes at that cycle: a RAW on
  slot 0's result fails it (every latency is >= 1), as do a busy unit, port or
  LSU and a pending penalty (a taken branch's, AME's ``overlap: false``).

A slot-1 refusal is not a stall: the stall counters count the cycles in which
slot 0 could not issue, and the refused instruction is offered again in slot 0
of the next cycle. ``timing.paired`` counts the instructions issued in slot 1.

**Memory regions** are the design's mapped blocks. Each takes its ``latency``
(cycles beyond ``load``) and ``bandwidth`` (bytes/cycle, ``null`` unlimited)
from ``timing.memory``, keyed by block name, else block type, else ``default``.
A region can have a **latency provider** in front of it: any object with
``access(addr, size, is_write) -> cycles``, whose answer replaces the region's
latency for that access. The builder installs every block that declares
``fronts: [block, ...]`` and has an ``access`` method (the ``cache`` IP) as
the provider of the blocks it fronts; :meth:`MemoryTiming.attach_provider` does
the same by hand.

**The matrix unit** is :mod:`framework.matrix.timing` (``timing.matrix``);
**the vector unit** is :mod:`framework.pe.rv.timing.vector` (``timing.vector``,
defaults when a vector design omits it; rejected without a vector unit).

The defaults below are a generic five-stage in-order core, documented per
field; ``naiad-ame-timed.yaml`` replaces them with numbers from the Tenstorrent
baby cores and the Tensix FPU and cites each one.
"""

import bisect
from dataclasses import dataclass, field, fields

from framework.matrix.timing import AMETiming, AMETimingSpec
from framework.pe.rv.timing.store_hazards import StoresInFlight
from framework.pe.rv.timing.vector import LOAD as VLOAD
from framework.pe.rv.timing.vector import STORE as VSTORE
from framework.pe.rv.timing.vector import VectorTiming, VectorTimingSpec
from framework.util.schema import DesignError, parse_int
from framework.util.schema import check_keys as _check_keys

#: Stall reasons, in report order.
STALL_REASONS = (
    "raw",
    "mem",
    "structural",
    "branch",
    "matrix_busy",
    "matrix_raw",
    "vector_busy",
    "vector_raw",
    "mem_raw",
)
(
    _RAW,
    _MEM,
    _STRUCTURAL,
    _CONTROL,
    _MATRIX_BUSY,
    _MATRIX_RAW,
    _VECTOR_BUSY,
    _VECTOR_RAW,
    _MEM_RAW,
) = range(len(STALL_REASONS))
_REASON_INDEX = {name: i for i, name in enumerate(STALL_REASONS)}

_M64 = (1 << 64) - 1


@dataclass(frozen=True)
class ScalarTimingSpec:
    """``timing.scalar``: per-class result latencies (cycles) and the scalar pipeline's shape."""

    #: Instructions issued per cycle: 1, or 2 (the in-order dual-issue front end; module docstring).
    issue_width: int = 1
    alu: int = 1
    #: Pipelined.
    mul: int = 3
    div: int = 20
    #: False: one divide at a time, so the next waits ``div`` cycles for the unit.
    div_pipelined: bool = False
    #: Load-to-use latency from a zero-latency region; the region's own latency adds to it.
    load: int = 2
    #: Cycles a store holds the load/store unit.
    store: int = 1
    #: Extra cycles before the instruction after a taken branch, ``jal`` or ``jalr``.
    branch_taken: int = 2
    csr: int = 1
    fp_add: int = 4
    fp_mul: int = 4
    fp_fma: int = 4
    fp_div: int = 20
    fp_sqrt: int = 20
    #: fdiv and fsqrt share one unit; False makes it unpipelined, as ``div_pipelined``.
    fp_div_pipelined: bool = False
    #: Sign injection, min/max, compares, classify, conversions and moves.
    fp_misc: int = 2


@dataclass(frozen=True)
class RegionTimingSpec:
    """One entry of ``timing.memory``."""

    #: Cycles a load from this region adds to ``scalar.load`` (and to a tile transfer's latency).
    latency: int = 0
    #: Bytes per cycle through the region's port; None is unlimited (never busy).
    bandwidth: float | None = None


#: Built-in region timing by block type, used when ``timing.memory`` names neither the block nor its type.
REGION_DEFAULTS = {
    "dram": RegionTimingSpec(latency=30, bandwidth=16),
    "scratchpad": RegionTimingSpec(latency=0, bandwidth=None),
}
DEFAULT_REGION = RegionTimingSpec()


@dataclass(frozen=True)
class TimingSpec:
    """A design's ``timing:`` section."""

    scalar: ScalarTimingSpec = field(default_factory=ScalarTimingSpec)
    #: Block name or type (or ``default``) -> region timing, as written in the design.
    memory: dict = field(default_factory=dict, hash=False)
    #: None when the design has no matrix unit.
    matrix: AMETimingSpec | None = None
    #: None when the design's cores have no vector unit.
    vector: VectorTimingSpec | None = None
    #: Multi-hart designs: True (the default) gives every region one port that
    #: all harts share, so a region's ``bandwidth`` caps the harts together;
    #: False gives each hart a private port. Ignored on one hart.
    shared_ports: bool = True

    def region(self, name, type_name):
        """The timing for block ``name`` of type ``type_name``.

        The first of: the design's entry for the name, its entry for the type,
        :data:`REGION_DEFAULTS` for the type, the design's ``default`` entry,
        and :data:`DEFAULT_REGION` (no latency, unlimited bandwidth).
        """
        for key in (name, type_name):
            if key in self.memory:
                return self.memory[key]
        if type_name in REGION_DEFAULTS:
            return REGION_DEFAULTS[type_name]
        return self.memory.get("default", DEFAULT_REGION)


# -- schema ----------------------------------------------------------------------


def _number(value, what, *, minimum, allow_none=False, allow_float=False):
    if value is None and allow_none:
        return None
    if allow_float and isinstance(value, float) and not isinstance(value, bool):
        number = value
    else:
        number = parse_int(value, what)
    if number < minimum:
        raise DesignError(f"{what} must be >= {minimum}, got {value!r}")
    return number


def _flag(value, what):
    if not isinstance(value, bool):
        raise DesignError(f"{what}: expected true or false, got {value!r}")
    return value


def _parse_fields(raw, cls, what, rules):
    """``cls(**raw)`` with every key checked against ``cls``'s fields and ``rules``."""
    raw = {} if raw is None else raw
    names = {f.name for f in fields(cls)}
    _check_keys(raw, names, what)
    values = {}
    for key, value in raw.items():
        values[key] = rules.get(key, lambda v, w: _number(v, w, minimum=1))(
            value, f"{what}.{key}"
        )
    return cls(**values)


def _bandwidth(v, w):
    return _number(v, w, minimum=1, allow_none=True, allow_float=True)


def _ewise_pipe(v, w):
    if v not in ("ewise", "mma"):
        raise DesignError(f"{w}: expected 'ewise' or 'mma', got {v!r}")
    return v


def _parse_scalar(raw):
    spec = _parse_fields(
        raw,
        ScalarTimingSpec,
        "timing.scalar",
        {
            "div_pipelined": _flag,
            "fp_div_pipelined": _flag,
            "branch_taken": lambda v, w: _number(v, w, minimum=0),
        },
    )
    if spec.issue_width not in (1, 2):
        raise DesignError(
            f"timing.scalar.issue_width is {spec.issue_width}, but only single (1) and dual (2) "
            f"issue are modelled"
        )
    return spec


def _parse_matrix(raw, design):
    from framework.matrix.formats import get_format

    if design.matrix is None:
        if raw is not None:
            raise DesignError(
                "timing.matrix: the design has no matrix: section to time"
            )
        return None
    if design.matrix.mode == "ime":
        from framework.matrix.ime_timing import parse_ime_timing

        return parse_ime_timing(raw, design, _parse_fields, _number, _flag, DesignError)
    if design.matrix.mode == "vme":
        from framework.matrix.vme_timing import parse_vme_timing

        return parse_vme_timing(raw, design, _parse_fields, _number, _flag, DesignError)
    zero_ok = lambda v, w: _number(v, w, minimum=0)  # noqa: E731
    rules = {
        "mma_ii": lambda v, w: _number(v, w, minimum=1, allow_none=True),
        "acc_forwarding": _flag,
        "overlap": _flag,
        "tile_load_bandwidth": _bandwidth,
        "tile_store_bandwidth": _bandwidth,
        "format_passes": lambda v, w: v,
        **{k: zero_ok for k in ("azero", "mcfg", "tshift", "epilogue")},
        **{
            k: lambda v, w: _number(v, w, minimum=0, allow_none=True)
            for k in ("tconv", "tshift2")
        },
        "ewise_bytes_per_cycle": _bandwidth,
        "ewise_pipe": _ewise_pipe,
    }
    spec = _parse_fields(raw, AMETimingSpec, "timing.matrix", rules)
    passes = spec.format_passes
    if not isinstance(passes, dict):
        raise DesignError(
            "timing.matrix.format_passes: expected a mapping of in-format -> passes"
        )
    canonical = {}
    for name, n in passes.items():
        try:
            fmt = get_format(str(name).lower()).name
        except ValueError as e:
            raise DesignError(f"timing.matrix.format_passes: {e}") from None
        if fmt not in design.matrix.in_formats:
            raise DesignError(
                f"timing.matrix.format_passes: {fmt} is not one of the unit's in formats {list(design.matrix.in_formats)}"
            )
        canonical[fmt] = _number(n, f"timing.matrix.format_passes.{fmt}", minimum=1)
    return AMETimingSpec(
        **{
            **{f.name: getattr(spec, f.name) for f in fields(spec)},
            "format_passes": canonical,
        }
    )


# -- memory regions --------------------------------------------------------------


class Region:
    """One mapped block's timing state: its port, its provider and its counts."""

    __slots__ = (
        "name",
        "low",
        "high",
        "latency",
        "bandwidth",
        "provider",
        "coherent",
        "port_free",
        "loads",
        "stores",
        "bytes",
    )

    def __init__(self, name, low, size, spec):
        self.name = name
        self.low, self.high = low, low + size - 1
        self.latency = spec.latency
        self.bandwidth = spec.bandwidth
        self.provider = None
        self.coherent = False
        self.reset()

    def reset(self):
        self.port_free = 0
        self.loads = self.stores = self.bytes = 0

    def access(self, addr, nbytes, is_write, cycle, t_cycles):
        """Commit an access: count it, hold the port, and return its latency.

        The port is held for ``ceil(nbytes / bandwidth)`` cycles, the port's own
        time for the bytes, at most ``t_cycles`` (the requester's transfer): a
        requester whose path is narrower than the port (a 64 B/cycle tile path
        on a 128 B/cycle port) takes longer than that, but leaves the rest of
        the port to the other harts, so ``bandwidth`` is the harts' combined cap.
        (Every shipped design's only finite region, DRAM, is narrower than every
        unit path, so there both are the same.) With a provider the provider's
        answer is the latency (and the provider's own state moves); otherwise
        the region's. Stores are posted, so callers use the return value only
        for loads.
        """
        if is_write:
            self.stores += 1
        else:
            self.loads += 1
        self.bytes += nbytes
        if self.bandwidth is not None:
            self.port_free = cycle + max(1, min(t_cycles, -(-nbytes // self.bandwidth)))
        if self.provider is not None:
            return self.provider.access(addr, nbytes, is_write)
        return self.latency


class SharedPort:
    """One region port that several harts' :class:`SharedPortRegion` s hold in common."""

    __slots__ = ("free",)

    def __init__(self):
        self.free = 0


class SharedPortRegion(Region):
    """A :class:`Region` whose port (``port_free``) is shared with the other harts'.

    Counts stay per hart. Harts are ticked in order, so on a tie the lower hart
    wins the port: a fixed-priority arbiter.
    """

    __slots__ = ("_port",)

    def __init__(self, name, low, size, spec, port):
        self._port = port
        super().__init__(name, low, size, spec)

    @property
    def port_free(self):
        return self._port.free

    @port_free.setter
    def port_free(self, value):
        self._port.free = value


class MemoryTiming:
    """Address -> :class:`Region`, over the design's mapped blocks (unmapped is a zero-cost region).

    ``ports`` (block name -> :class:`SharedPort`) shares each region's port with
    every other ``MemoryTiming`` built with the same dict (one per hart).
    ``inflight`` is the :class:`~framework.pe.rv.timing.store_hazards.StoresInFlight`
    the harts' async stores register with and their loads check (one per SoC;
    a fresh one when omitted).
    """

    def __init__(self, timing, blocks, ports=None, inflight=None):
        self.inflight = StoresInFlight() if inflight is None else inflight
        mapped = sorted((b for b in blocks.values() if b.mapped), key=lambda b: b.base)
        if ports is None:
            self.regions = [
                Region(b.name, b.base, b.size, timing.region(b.name, b.spec.type))
                for b in mapped
            ]
        else:
            self.regions = [
                SharedPortRegion(
                    b.name,
                    b.base,
                    b.size,
                    timing.region(b.name, b.spec.type),
                    ports.setdefault(b.name, SharedPort()),
                )
                for b in mapped
            ]
        self.by_name = {r.name: r for r in self.regions}
        self._lows = [r.low for r in self.regions]
        self.unmapped = Region("unmapped", 0, 1, DEFAULT_REGION)

    def reset(self):
        self.inflight.reset()
        for r in self.regions:
            r.reset()
        self.unmapped.reset()

    def lookup(self, addr):
        i = bisect.bisect_right(self._lows, addr) - 1
        if i >= 0 and addr <= self.regions[i].high:
            return self.regions[i]
        return self.unmapped

    def attach_provider(self, region_name, provider):
        """Put ``provider`` (``access(addr, size, is_write) -> cycles``) in front of a region."""
        if not callable(getattr(provider, "access", None)):
            raise TypeError(
                f"a latency provider needs an access(addr, size, is_write) method, got {provider!r}"
            )
        region = self.by_name[region_name]
        region.provider = provider
        # A coherent provider (a private MSI cache's per-hart view) also times
        # AMOs as writes and charges posted stores its bus cycles.
        region.coherent = bool(getattr(provider, "coherent", False))

    def counters(self):
        out = {}
        for r in self.regions:
            out[f"mem.{r.name}.loads"] = r.loads
            out[f"mem.{r.name}.stores"] = r.stores
            out[f"mem.{r.name}.bytes"] = r.bytes
        return out


# -- instruction classes -----------------------------------------------------------

(
    _ALU,
    _MUL,
    _DIV,
    _LOAD,
    _STORE,
    _BRANCH,
    _JUMP,
    _CSR,
    _FP_ADD,
    _FP_MUL,
    _FP_FMA,
    _FP_DIV,
    _FP_SQRT,
    _FP_MISC,
    _NONE,
    _MATRIX,
    _VECTOR,
) = range(17)

_F = 32  # f-register i is scoreboard entry 32 + i
_MATRIX_OPCODES = (0x0B, 0x2B, 0x5B, 0x7B)  # custom-0..3, as RV_AME_ISA.OPCODES


def _classify(instr):
    """``(kind, sources, destination, size)`` for a scalar word; entries index the 64-entry scoreboard.

    ``destination`` is -1 for none (x0 never counts as one); ``size`` is a
    memory access's byte count. Unknown words are ``_NONE`` and cost one cycle.
    """
    opcode = instr & 0x7F
    rd, f3 = (instr >> 7) & 0x1F, (instr >> 12) & 0x7
    rs1, rs2, rs3 = (instr >> 15) & 0x1F, (instr >> 20) & 0x1F, instr >> 27
    f7 = instr >> 25
    xd = rd if rd else -1
    if opcode in (0x37, 0x17):  # lui, auipc
        return _ALU, (), xd, 0
    if opcode == 0x6F:  # jal
        return _JUMP, (), xd, 0
    if opcode == 0x67:  # jalr
        return _JUMP, (rs1,), xd, 0
    if opcode == 0x63:
        return _BRANCH, (rs1, rs2), -1, 0
    if opcode == 0x03:
        return _LOAD, (rs1,), xd, 1 << (f3 & 3)
    if opcode == 0x07:  # flw / fld / flh
        return _LOAD, (rs1,), _F + rd, 1 << (f3 & 3)
    if opcode == 0x23:
        return _STORE, (rs1, rs2), -1, 1 << (f3 & 3)
    if opcode == 0x27:
        return _STORE, (rs1, _F + rs2), -1, 1 << (f3 & 3)
    if opcode in (0x13, 0x1B):
        return _ALU, (rs1,), xd, 0
    if opcode in (0x33, 0x3B):
        if f7 == 1:
            return (_MUL if f3 < 4 else _DIV), (rs1, rs2), xd, 0
        return _ALU, (rs1, rs2), xd, 0
    if opcode == 0x2F:  # AMO / LR / SC
        return _LOAD, (rs1, rs2), xd, 1 << (f3 & 3)
    if opcode == 0x73:
        if f3 == 0:
            return _NONE, (), -1, 0
        return _CSR, ((rs1,) if f3 < 4 else ()), xd, 0
    if opcode in (0x43, 0x47, 0x4B, 0x4F):  # fmadd / fmsub / fnmsub / fnmadd
        return _FP_FMA, (_F + rs1, _F + rs2, _F + rs3), _F + rd, 0
    if opcode == 0x53:
        f5 = f7 >> 2
        kind = {
            0x00: _FP_ADD,
            0x01: _FP_ADD,
            0x02: _FP_MUL,
            0x03: _FP_DIV,
            0x0B: _FP_SQRT,
        }.get(f5, _FP_MISC)
        x_src = f5 in (0x1A, 0x1E)  # fcvt.*.[w|l], fmv.*.x
        x_dst = f5 in (0x14, 0x18, 0x1C)  # compares, fcvt.[w|l].*, fmv.x.* / fclass
        srcs = (rs1,) if x_src else (_F + rs1,)
        if f5 in (0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x14):
            srcs += (_F + rs2,)
        return kind, srcs, (xd if x_dst else _F + rd), 0
    return _NONE, (), -1, 0


class NaiadCostState:
    """One hart's cycle model, installed as ``core.rv_cost``.

    :meth:`can_issue` is the whole interface the core sees. Its contract is the
    baby cores' one (``framework.pe.rv.cost.RiscvCostState``): False means
    stalled, and every state change happens after the last point that can
    return False, so a stalled cycle leaves the model as it found it.
    """

    #: The counter hub names this state's events ``hart<N>.timing.<key>``.
    counter_prefix = "timing"

    def __init__(self, timing, memory, xlen=64, matrix_unit=None, vector_state=None):
        self.timing = timing
        self.memory = memory
        self.inflight = memory.inflight
        self.xlen = xlen
        s = timing.scalar
        self._latency = {
            _ALU: s.alu,
            _MUL: s.mul,
            _DIV: s.div,
            _LOAD: s.load,
            _STORE: 0,
            _BRANCH: 0,
            _JUMP: s.alu,
            _CSR: s.csr,
            _FP_ADD: s.fp_add,
            _FP_MUL: s.fp_mul,
            _FP_FMA: s.fp_fma,
            _FP_DIV: s.fp_div,
            _FP_SQRT: s.fp_sqrt,
            _FP_MISC: s.fp_misc,
            _NONE: 1,
        }
        # Unpipelined units: kind -> unit slot.
        self._unit_of = {}
        if not s.div_pipelined:
            self._unit_of[_DIV] = 0
        if not s.fp_div_pipelined:
            self._unit_of[_FP_DIV] = self._unit_of[_FP_SQRT] = 1
        self.branch_taken = s.branch_taken
        self.store_occupancy = s.store
        self.matrix = None
        mode = getattr(matrix_unit, "mode", None)
        ime = mode in (
            "ime",
            "vme",
        )  # vector-coupled: the matrix model extends the vector one
        if matrix_unit is not None and timing.matrix is not None and not ime:
            self.matrix = AMETiming(timing.matrix, matrix_unit, memory)
        self.vector = None
        if vector_state is not None and timing.vector is not None:
            if mode == "vme":  # Zvt: vector operands, tile accumulators
                from framework.matrix.vme_timing import VMETiming, VMETimingSpec

                spec = timing.matrix or VMETimingSpec()
                self.vector = VMETiming(
                    timing.vector, vector_state, memory, spec, matrix_unit
                )
            elif ime:  # IME's operands are vector groups: it extends the vector model
                from framework.matrix.ime_timing import IMETiming

                self.vector = IMETiming(
                    timing.vector, vector_state, memory, timing.matrix, matrix_unit
                )
            else:
                self.vector = VectorTiming(timing.vector, vector_state, memory)
            self.vector.xlen = xlen
        self._decoded = {}
        self.reset()

    def reset(self):
        self._ready = [0] * 64
        self._ready_reason = [_RAW] * 64
        self._unit_free = [0, 0]
        self._lsu_free = 0
        self._stall_until = 0
        self._stall_reason = _RAW
        self.stall_by_reason = [0] * len(STALL_REASONS)
        self.pending_stall = 0
        self.memory.reset()
        if self.matrix is not None:
            self.matrix.reset()
        if self.vector is not None:
            self.vector.reset()

    # -- reporting ---------------------------------------------------------------

    def counters(self):
        out = {"stall_cycles": sum(self.stall_by_reason)}
        out.update(
            {
                f"stall_{name}": self.stall_by_reason[i]
                for i, name in enumerate(STALL_REASONS)
            }
        )
        if self.matrix is not None:
            out.update(self.matrix.counters())
        if self.vector is not None:
            out.update(self.vector.counters())
        out.update(self.memory.counters())
        return out

    def take_pending_stall(self):
        """``(cycles, reason)`` held before the instruction about to retire (the trace path's view)."""
        n = self.pending_stall
        if not n:
            return 0, ""
        self.pending_stall = 0
        return n, STALL_REASONS[self._stall_reason]

    # -- the hot path ------------------------------------------------------------

    def _stall(self, until, reason):
        self._stall_until = until
        self._stall_reason = reason
        self.stall_by_reason[reason] += 1
        self.pending_stall += 1
        return False

    def _decode(self, instr):
        opcode = instr & 0x7F
        if opcode in _MATRIX_OPCODES and self.matrix is not None:
            op = self.matrix.decode(instr)
            if op is None:
                # Not a word the unit implements; executing it raises, so no cost is owed.
                return _NONE, (), -1, 0
            return _MATRIX, op.srcs, op.dst, op
        if self.vector is not None and (
            opcode == 0x57
            or opcode == 0x77  # OP-VE (VME's multiplies; nothing else uses it)
            or (opcode in (0x07, 0x27) and (instr >> 12) & 0x7 in (0, 5, 6, 7))
        ):
            op = self.vector.decode(instr)
            if op is None:
                return _NONE, (), -1, 0
            return _VECTOR, op.ssrc, op.sdst, op
        return _classify(instr)

    def can_issue(self, instr, cycle_num, register_file):
        if cycle_num < self._stall_until:
            self.stall_by_reason[self._stall_reason] += 1
            self.pending_stall += 1
            return False
        if instr & 0x3 != 0x3:
            return True
        decoded = self._decoded.get(instr)
        if decoded is None:
            decoded = self._decoded[instr] = self._decode(instr)
        kind, srcs, dst, extra = decoded
        ready = self._ready
        for s in srcs:
            at = ready[s]
            if at > cycle_num:
                return self._stall(at, self._ready_reason[s])

        if kind == _MATRIX:
            matrix = self.matrix
            done = matrix.issue(extra, cycle_num, register_file)
            if done < 0:
                return self._stall(
                    matrix.stall_until, _REASON_INDEX[matrix.stall_reason]
                )
            if dst > 0:
                ready[dst] = done
                self._ready_reason[dst] = _RAW
            if not matrix.spec.overlap and done > cycle_num + 1:
                self._stall_until = done
                self._stall_reason = _MATRIX_BUSY
            return True

        if kind == _VECTOR:
            vector = self.vector
            done = vector.issue(extra, cycle_num, register_file)
            if done < 0:
                return self._stall(
                    vector.stall_until, _REASON_INDEX[vector.stall_reason]
                )
            if dst > 0:
                ready[dst] = done
                self._ready_reason[dst] = _RAW
            return True

        unit = self._unit_of.get(kind)
        if unit is not None and self._unit_free[unit] > cycle_num:
            return self._stall(self._unit_free[unit], _STRUCTURAL)

        latency = self._latency[kind]
        if kind == _LOAD or kind == _STORE:
            if self._lsu_free > cycle_num:
                return self._stall(self._lsu_free, _STRUCTURAL)
            imm = (
                instr >> 20
                if kind == _LOAD
                else ((instr >> 25) << 5) | ((instr >> 7) & 0x1F)
            )
            if instr & 0x7F == 0x2F:
                imm = 0
            elif imm & 0x800:
                imm -= 0x1000
            addr = (register_file[(instr >> 15) & 0x1F].read_uint() + imm) & _M64
            region = self.memory.lookup(addr)
            if region.port_free > cycle_num:
                return self._stall(region.port_free, _MEM)
            inflight = self.inflight
            if kind == _LOAD and inflight.until > cycle_num:
                at = inflight.wait_span(addr, extra, cycle_num)
                if at:  # an async store is still writing these bytes
                    return self._stall(at, _MEM_RAW)
            # -- commit
            t_cycles = (
                1 if region.bandwidth is None else max(1, -(-extra // region.bandwidth))
            )
            is_write = kind == _STORE or (
                region.coherent and instr & 0x7F == 0x2F and instr >> 27 != 0b00010
            )  # an AMO or sc takes the line for writing; lr reads it
            extra_latency = region.access(addr, extra, is_write, cycle_num, t_cycles)
            if kind == _STORE:
                self._lsu_free = cycle_num + self.store_occupancy
                if region.coherent:  # posted, but the bus upgrade holds the LSU
                    self._lsu_free += region.provider.coherence_cycles
                return True
            latency += extra_latency + t_cycles - 1
            if dst > 0:
                ready[dst] = cycle_num + latency
                self._ready_reason[dst] = _MEM
            return True

        # -- commit
        if unit is not None:
            self._unit_free[unit] = cycle_num + latency
        if dst > 0:
            ready[dst] = cycle_num + latency
            self._ready_reason[dst] = _RAW
        if self.branch_taken and (
            kind == _JUMP or (kind == _BRANCH and self._taken(instr, register_file))
        ):
            self._stall_until = cycle_num + 1 + self.branch_taken
            self._stall_reason = _CONTROL
        return True

    def _taken(self, instr, register_file):
        a = register_file[(instr >> 15) & 0x1F].read_uint()
        b = register_file[(instr >> 20) & 0x1F].read_uint()
        f3 = (instr >> 12) & 0x7
        if f3 == 0:
            return a == b
        if f3 == 1:
            return a != b
        if f3 in (4, 5):
            sign = 1 << (self.xlen - 1)
            a, b = (a ^ sign), (b ^ sign)  # signed order as unsigned order
        return a < b if f3 in (4, 6) else a >= b


# -- dual issue -------------------------------------------------------------------

# The pipes an instruction needs (a bit mask). The first four are single: two
# instructions that both need one of them do not issue in the same cycle.
_P_MEM, _P_MULDIV, _P_FP, _P_VEC = 1, 2, 4, 8
_P_SINGLE = _P_MEM | _P_MULDIV | _P_FP | _P_VEC
_P_CTRL = 16  # a branch or jump: nothing issues after it in its cycle
_P_ALONE = 32  # a CSR access or system word: issues alone

_KIND_PIPES = {
    _ALU: 0,
    _MUL: _P_MULDIV,
    _DIV: _P_MULDIV,
    _LOAD: _P_MEM,
    _STORE: _P_MEM,
    _BRANCH: _P_CTRL,
    _JUMP: _P_CTRL,
    _CSR: _P_ALONE,
    _FP_ADD: _P_FP,
    _FP_MUL: _P_FP,
    _FP_FMA: _P_FP,
    _FP_DIV: _P_FP,
    _FP_SQRT: _P_FP,
    _FP_MISC: _P_FP,
    _NONE: _P_ALONE,
}


def _vector_is_mem(op):
    """True for a vector-unit op that moves memory: an RVV, Zvvm (IME) or Zvt (VME) load or store."""
    insn = getattr(op, "insn", None)  # Zvt
    if insn is not None:
        return insn.name in ("vtle", "vtse")
    d = getattr(op, "d", None)  # Zvvm
    if d is not None:
        return d.insn.kind in ("load", "store")
    return op.kind in (VLOAD, VSTORE)


class DualIssueCostState(NaiadCostState):
    """:class:`NaiadCostState` for ``issue_width: 2``: the pairing rules of the module docstring.

    The SoC clock ticks the core twice per cycle with the same cycle number;
    the first tick of a cycle is slot 0 (exactly the single-issue model), the
    second slot 1, which issues only an instruction that pairs with slot 0's.
    """

    def __init__(self, *args, **kwargs):
        self._pipes_of = {}
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self._cycle = -1
        #: The pipes of the instruction slot 0 issued this cycle (None: it issued none,
        #: or slot 1 has been offered).
        self._slot0 = None
        self.paired = 0

    def counters(self):
        out = super().counters()
        out["paired"] = self.paired
        return out

    def _pipes(self, instr):
        pipes = self._pipes_of.get(instr)
        if pipes is None:
            if instr & 0x3 != 0x3:
                pipes = 0
            else:
                decoded = self._decoded.get(instr)
                if decoded is None:
                    decoded = self._decoded[instr] = self._decode(instr)
                kind, _, _, extra = decoded
                if kind == _MATRIX:
                    pipes = _P_VEC | (_P_MEM if extra.pipe in ("load", "store") else 0)
                elif kind == _VECTOR:
                    pipes = _P_VEC | (_P_MEM if _vector_is_mem(extra) else 0)
                else:
                    pipes = _KIND_PIPES[kind]
            self._pipes_of[instr] = pipes
        return pipes

    def can_issue(self, instr, cycle_num, register_file):
        if cycle_num != self._cycle:  # slot 0
            self._cycle = cycle_num
            if NaiadCostState.can_issue(self, instr, cycle_num, register_file):
                self._slot0 = self._pipes(instr)
                return True
            self._slot0 = None
            return False
        # slot 1
        first = self._slot0
        if first is None:
            return False
        self._slot0 = None
        pipes = self._pipes(instr)
        if (
            (first | pipes) & _P_ALONE
            or first & _P_CTRL
            or first & pipes & _P_SINGLE
            or cycle_num < self._stall_until
        ):
            return False
        if NaiadCostState.can_issue(self, instr, cycle_num, register_file):
            self.paired += 1
            return True
        # Not a stall cycle: undo the one count the refusal made (_stall). The
        # stall time it recorded stands: slot 0 of a later cycle waits for it.
        self.stall_by_reason[self._stall_reason] -= 1
        self.pending_stall -= 1
        return False
