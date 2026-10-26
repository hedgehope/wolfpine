"""The counter hub: one name for every event a Naiad run can count, and where to read it.

Everything a kernel, the run report or the experiment harness counts is a hub
*event*: a stable dotted name with a zero-argument reader that returns the live
value. The Zihpm CSRs (``mhpmcounter3..31``), the ``perfcounters`` MMIO block
and :meth:`NaiadSoC.counters` all read through the hub, so an event has the
same value whichever way it is read.

Sources are found generically, by the ``counters() -> dict`` convention (the
dict's keys must be the same on every call, zeros included):

* every IP block: ``<block>.<key>`` (``htif.chars``, ``dma0.bytes``);
* every public attribute of a core that has a ``counters()`` method:
  ``hart<N>.<attr>.<key>`` (``hart0.matrix.macs``, or a timing cost state's
  ``hart0.<attr>.stall_<reason>``). A source may set ``counter_prefix`` to
  choose its middle name instead of the attribute's;
* every public attribute of the SoC itself with one: ``<attr>.<key>`` (or
  ``<counter_prefix>.<key>``), for SoC-wide state such as a shared cost model.

New IP therefore adds events by implementing ``counters()``; the hub never
names a block or unit type. The hub adds the clock and the CSR files' own
counts (``cycle``, ``mtime``, ``hart<N>.cycle``, ``hart<N>.instret``), and, for
every per-hart event ``hart<N>.<rest>``, an aggregate ``<rest>`` summed over the
harts (``instret``, ``matrix.macs``). The flat names the perfcounters block took
before the hub (``matrix_macs``, ``htif_chars``) still resolve: a name with no
dot that is not an event is retried with its first ``_`` read as a ``.``.

Two families cost something on every instruction, so they exist only when
asked for (a Zihpm binding or a perfcounters entry naming one, the design's
``counters.enable``, or ``build(..., count_all=True)``):

* ``classes``: ``hart<N>.instret.{scalar,fp,vector,matrix}``, retired
  instructions by the ISA that claimed them. ``vector`` stays zero until an RVV
  ISA declares ``INSTR_CLASS = "vector"``.
* ``bus``: ``hart<N>.bus.<block>.{read,write}_bytes``, the bytes the core's
  scalar and FP loads, stores and AMOs moved to and from each mapped block.
  Instruction fetch is not counted, nor is matrix-unit traffic (the unit counts
  that itself, as ``matrix.mem_to_tile_bytes`` and the like).

Both are installed by wrapping the core's pre-bound ISA chain
(``core._isa_runs``) with a counting closure per ISA, so the core's
``clock_tick`` is untouched and a design that asks for neither runs exactly the
code it ran before.
"""

from framework.matrix.ame_isa import RV_AME_ISA
from framework.pe.pe import ProcessingElement
from framework.pe.rv.isa.fd_isa import RV64_D_ISA, RV64_F_ISA
from framework.pe.rv.isa.zfbfmin_isa import RV64_ZFBFMIN_ISA

_M64 = (1 << 64) - 1

#: Instruction classes, in report order.
INSTR_CLASSES = ("scalar", "fp", "vector", "matrix")

#: The on-demand event families (``counters.enable`` takes these names).
FAMILIES = ("classes", "bus")

#: ISA class -> instruction class, for ISAs that predate ``INSTR_CLASS``. An
#: ISA not listed here and without the attribute is ``scalar``.
_ISA_CLASS = {
    RV64_F_ISA: "fp",
    RV64_D_ISA: "fp",
    RV64_ZFBFMIN_ISA: "fp",
    RV_AME_ISA: "matrix",
}


class CounterError(KeyError):
    """An event name the hub does not have."""

    def __str__(self):
        return self.args[0] if self.args else ""


def instr_class(isa):
    return getattr(isa, "INSTR_CLASS", None) or _ISA_CLASS.get(isa, "scalar")


def _family(name):
    """The on-demand family a per-hart or aggregate event name belongs to, if any."""
    parts = name.split(".")
    if parts and parts[0].startswith("hart") and parts[0][4:].isdigit():
        parts = parts[1:]
    if len(parts) == 2 and parts[0] == "instret" and parts[1] in INSTR_CLASSES:
        return "classes"
    if len(parts) == 3 and parts[0] == "bus":
        return "bus"
    return None


# -- the on-demand probe --------------------------------------------------------


class _CountingBus:
    """The core's memory as its scalar ISAs see it while ``bus`` is counted.

    Tallies each access against the block that serves it, then forwards it.
    Anything other than ``read``/``write`` passes straight through.
    """

    def __init__(self, memory, reads, writes):
        self._memory = memory
        self._locate = memory.memory_map.locate
        self._reads = reads
        self._writes = writes

    def read(self, addr, size):
        block = self._locate(addr)[1]
        if block is not None:
            self._reads[block.name] += size
        return self._memory.read(addr, size)

    def write(self, addr, value, size=None):
        block = self._locate(addr)[1]
        if block is not None:
            self._writes[block.name] += len(value) if size is None else size
        return self._memory.write(addr, value, size)

    def __getattr__(self, name):
        return getattr(self._memory, name)


def _counted(run, tally, slot, memory):
    """``run`` with a retire tally; ``memory`` (if given) replaces the memory it is handed."""
    stall = ProcessingElement.PEStall
    if memory is None:

        def counted(register_file, memory_space, snoop, instr):
            result = run(register_file, memory_space, snoop, instr)
            if result and result is not stall:
                tally[slot] += 1
            return result

    else:

        def counted(register_file, memory_space, snoop, instr):
            result = run(register_file, memory, snoop, instr)
            if result and result is not stall:
                tally[slot] += 1
            return result

    return counted


class CoreProbe:
    """One hart's instruction-class tally, and optionally its scalar bus bytes.

    Installed by replacing each entry of ``core._isa_runs`` with a counting
    wrapper; :meth:`counters` answers for the ``classes`` family, and
    :attr:`bus` (when built with ``bus=True``) for the ``bus`` family.
    """

    def __init__(self, core, bus=False):
        self.tally = dict.fromkeys(INSTR_CLASSES, 0)
        self.bus = None
        counting_bus = None
        if bus:
            self.bus = _BusTally(core.visible_memory)
            counting_bus = _CountingBus(
                core.visible_memory, self.bus.reads, self.bus.writes
            )
        core._isa_runs = [
            _counted(
                run,
                self.tally,
                instr_class(isa),
                counting_bus if instr_class(isa) in ("scalar", "fp") else None,
            )
            for isa, run in zip(core.isas, core._isa_runs)
        ]

    def counters(self):
        return dict(self.tally)

    def reset(self):
        for key in self.tally:
            self.tally[key] = 0
        if self.bus is not None:
            self.bus.reset()


class _BusTally:
    def __init__(self, memory):
        names = [block.name for _, block in memory.memory_map.items()]
        self.reads = dict.fromkeys(names, 0)
        self.writes = dict.fromkeys(names, 0)

    def counters(self):
        out = {}
        for name in self.reads:
            out[f"{name}.read_bytes"] = self.reads[name]
            out[f"{name}.write_bytes"] = self.writes[name]
        return out

    def reset(self):
        for d in (self.reads, self.writes):
            for key in d:
                d[key] = 0


# -- the hub ----------------------------------------------------------------------


def _item(source, key):
    return lambda: source.counters()[key]


def _sum(readers):
    return lambda: sum(read() for read in readers)


def _has_counters(value):
    return not isinstance(value, type) and callable(getattr(value, "counters", None))


class CounterHub:
    """Event name -> live reader, discovered from a :class:`~framework.naiad.builder.NaiadSoC`.

    Build with :meth:`discover`, which also installs the on-demand families
    ``enable`` asks for (``"all"`` for every one).
    """

    def __init__(self):
        self._readers = {}
        self.probes = []

    # -- registration -------------------------------------------------------

    def add(self, name, reader):
        if name in self._readers:
            raise ValueError(f"counter event '{name}' is registered twice")
        self._readers[name] = reader

    def add_source(self, prefix, source):
        """Register every key of ``source.counters()`` as ``<prefix>.<key>``."""
        for key in source.counters():
            self.add(f"{prefix}.{key}" if prefix else key, _item(source, key))

    @classmethod
    def discover(cls, soc, enable=()):
        hub = cls()
        enable = set(FAMILIES) if "all" in enable else set(enable)
        clock = soc.clock
        hub.add("cycle", lambda: clock.current_cycle)
        clints = soc.blocks_of_type("clint")
        if clints:
            hub.add("mtime", clints[0].mtime)

        seen = {id(soc.memory)}
        per_hart = []
        for core in soc.cores:
            hart = f"hart{core.core_id}"
            csrs = core.csrs
            if hasattr(csrs, "cycles"):
                hub.add(f"{hart}.cycle", lambda c=csrs: c.cycles)
            hub.add(f"{hart}.instret", lambda c=csrs: c.retired)
            for attr, value in vars(core).items():
                if (
                    attr.startswith("_")
                    or id(value) in seen
                    or not _has_counters(value)
                ):
                    continue
                seen.add(id(value))
                prefix = getattr(value, "counter_prefix", attr)
                hub.add_source(f"{hart}.{prefix}", value)
            if enable & set(FAMILIES):
                probe = CoreProbe(core, bus="bus" in enable)
                hub.probes.append(probe)
                hub.add_source(f"{hart}.instret", probe)
                if probe.bus is not None:
                    hub.add_source(f"{hart}.bus", probe.bus)
            per_hart.append(hart)

        for block in soc.blocks.values():
            seen.add(id(block))
            hub.add_source(block.name, block)
        for attr, value in vars(soc).items():
            if attr.startswith("_") or id(value) in seen or not _has_counters(value):
                continue
            hub.add_source(getattr(value, "counter_prefix", attr), value)

        # Aggregates over harts, for every per-hart name not already an event.
        groups = {}
        for name, reader in list(hub._readers.items()):
            hart, _, rest = name.partition(".")
            if rest and hart in per_hart:
                groups.setdefault(rest, []).append(reader)
        aggregates = {
            rest: readers[0] if len(readers) == 1 else _sum(readers)
            for rest, readers in groups.items()
            if rest not in hub._readers
        }
        # Report order: the SoC-wide events, the aggregates, then everything else.
        front = {
            k: hub._readers.pop(k) for k in ("cycle", "mtime") if k in hub._readers
        }
        hub._readers = {**front, **aggregates, **hub._readers}
        return hub

    # -- lookup --------------------------------------------------------------

    def names(self):
        return list(self._readers)

    def __contains__(self, name):
        try:
            self.resolve(name)
        except CounterError:
            return False
        return True

    def resolve(self, name):
        """The canonical event ``name`` refers to (legacy flat names included)."""
        if name in self._readers:
            return name
        if "." not in name and "_" in name:
            dotted = name.replace("_", ".", 1)
            if dotted in self._readers:
                return dotted
        family = _family(name)
        hint = (
            f" ('{name}' is in the '{family}' family, which this design does not count: "
            f"bind it, or list '{family}' in counters.enable)"
            if family
            else ""
        )
        raise CounterError(
            f"unknown counter event '{name}'{hint}; known: {sorted(self._readers)}"
        )

    def reader(self, name):
        return self._readers[self.resolve(name)]

    def read(self, name):
        return self._readers[self.resolve(name)]() & _M64

    def snapshot(self):
        return {name: read() for name, read in self._readers.items()}

    def reset(self):
        for probe in self.probes:
            probe.reset()


def requested_families(names):
    """The on-demand families a list of event names needs."""
    return {f for f in map(_family, names) if f is not None}
