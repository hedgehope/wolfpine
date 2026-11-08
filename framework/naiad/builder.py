"""Design -> a runnable Naiad SoC (a Wolfpine ``Device`` plus the handles around it).

The output is the same shape as the standalone ``driver/simple`` examples: one
``MemoryMap`` of the design's blocks behind a ``DeviceMemory``, the cores, one
``Clock`` and one ``Reset``. :class:`NaiadSoC` adds what a program run needs:
loading an ELF, running to an HTIF exit, and a counter report.
"""

from dataclasses import dataclass, field

from framework.device.clock import Clock
from framework.device.device import Device, DeviceMemory
from framework.device.reset import Reset
from framework.ip.counters import CounterError, CounterHub, requested_families
from framework.matrix.ame_isa import attach as attach_matrix_unit
from framework.matrix.unit import MatrixUnit
from framework.memory.memory import MemorySpace
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.naiad.registry import IP_REGISTRY, NAIAD_ISA_REGISTRY
from framework.naiad.timing import install as install_timing
from framework.pe.rv.isa.v import RV64_V_ISA
from framework.pe.rv.isa.v import attach as attach_vector
from framework.pe.rv.isa.zicsr64_isa import CSRFile64
from framework.pe.rv.rv64 import rv64_extensions
from framework.pe.rv.trap import NaiadCSRFile32, NaiadRV32, NaiadRV32IM, NaiadRV64
from framework.util.elf_loader import load_elf


class NaiadBuildError(Exception):
    pass


class NaiadRunError(Exception):
    pass


class NaiadClock(Clock):
    """A ``Clock`` that knows which cycle it is ticking.

    ``current_cycle`` is what ``CSRFile`` (``mcycle``/``cycle``) and the CLINT
    (``mtime``) read, so every cycle number a program can observe comes from here.
    """

    def __init__(self, clockables, on_tick=None):
        super().__init__(clockables, on_tick)
        self.current_cycle = 0
        self.issue_width = 1
        self._cores = self._others = ()

    def set_issue_width(self, width, cores):
        """Tick each of ``cores`` ``width`` times per cycle (issue slots), the other items once.

        For the dual-issue timing model (``timing.scalar.issue_width: 2``,
        :mod:`framework.pe.rv.timing.core`): the interpreter executes at most one
        instruction per tick, so a cycle gives every core ``width`` ticks, all
        with the cycle's number (slot-major: every core's slot 0, then every
        core's slot 1), and then ticks the rest (DMA, CLINT wake-up, Tensix)
        once. A core's ``csrs.cycles`` (``cycle``/``mcycle``) counts cycles:
        the extra slots' increments are taken back.
        """
        self.issue_width = width
        self._cores = tuple(cores)
        self._others = tuple(c for c in self.clock_items if c not in self._cores)

    def clock_tick(self, cycle):
        self.current_cycle = cycle
        if self.issue_width == 1:
            super().clock_tick(cycle)
            return
        cores = self._cores
        for core in cores:
            core.clock_tick(cycle)
        for _ in range(1, self.issue_width):
            counted = [getattr(core.csrs, "cycles", None) for core in cores]
            for core in cores:
                core.clock_tick(cycle)
            for core, n in zip(cores, counted):
                if n is not None:
                    core.csrs.cycles = n
        for item in self._others:
            item.clock_tick(cycle)
        if self.on_tick is not None:
            self.on_tick(cycle)

    def reset(self):
        super().reset()
        self.current_cycle = 0


class NaiadMemory(DeviceMemory):
    """The SoC bus: a ``DeviceMemory`` that can call a block back on a store.

    Each intercept is ``(low, high, handler)``; ``handler(addr, size)`` runs after
    any write overlapping ``[low, high]`` has landed. The HTIF is the one user.
    With no intercepts a write costs one extra truthiness check.
    """

    def __init__(self, memory_map):
        super().__init__(memory_map)
        self.intercepts = []

    def add_write_intercept(self, addr, size, handler):
        self.intercepts.append((addr, addr + size - 1, handler))

    def write_raw(self, addr, value, size=None):
        """A write that does not trigger intercepts (for the handlers themselves)."""
        return MemorySpace.write(self, addr, value, size)

    def write(self, addr, value, size=None):
        result = MemorySpace.write(self, addr, value, size)
        if self.intercepts:
            n = len(value) if size is None else size
            for low, high, handler in self.intercepts:
                if addr <= high and addr + n > low:
                    handler(addr, n)
        return result


class MulticoreMemory(NaiadMemory):
    """The bus of a multi-hart SoC: every store also breaks the ``lr`` reservations it overlaps.

    ``reservation_holders`` are the harts' register files (the builder sets
    them); each holds its hart's Zalrsc reservation as ``(start, end)`` or None
    (``framework.pe.rv.isa.a64_isa``). Any write through the bus (a hart's
    scalar, FP, AMO, vector or matrix store, or a DMA write) clears every
    reservation it overlaps, so another hart's ``sc`` then fails. A single-hart
    SoC keeps the plain :class:`NaiadMemory`, whose stores cost nothing extra.
    """

    def __init__(self, memory_map):
        super().__init__(memory_map)
        self.reservation_holders = ()

    def write(self, addr, value, size=None):
        result = super().write(addr, value, size)
        n = len(value) if size is None else size
        end = addr + n
        for rf in self.reservation_holders:
            res = rf.reservation
            if res is not None and addr < res[1] and res[0] < end:
                rf.reservation = None
        return result


def make_core(xlen, isa, reset_pc, memory, core_id, vlen=None, elen=None):
    """Build one core.

    ``isa`` is the design's extension list; each name resolves through
    ``NAIAD_ISA_REGISTRY`` on RV32 and ``RV64_ISA_REGISTRY`` on RV64. Every core gets a ``CSRFile`` whether or not it lists
    Zicsr: the file is where retired instructions are counted for the run report,
    and without the Zicsr executor a CSR instruction is still an unknown one.
    """
    if xlen == 64:
        try:
            extensions = rv64_extensions(isa)
        except KeyError as e:
            raise NaiadBuildError(str(e)) from None
        core = NaiadRV64(
            reset_pc,
            [memory],
            extensions=extensions,
            unknown_instr_is_error=True,
            core_id=core_id,
            fp_registers="f" in isa or "d" in isa,
        )
        # RV64I only builds a CSR file when Zicsr is listed; without it the
        # file still counts cycles and retired instructions for the report.
        if core.csrs is None:
            core.csrs = CSRFile64(
                core.register_file, hart_id=core_id, core_label=f"hart{core_id}"
            )
            core.register_file.csrs = core.csrs
        if RV64_V_ISA in core.isas:
            attach_vector(core, vlen=vlen or 128, elen=elen or 64, extensions=isa)
        core.enable_traps()
        return core
    if xlen != 32:
        raise NaiadBuildError(f"unsupported xlen {xlen}")
    extensions = [
        NAIAD_ISA_REGISTRY[name] for name in isa if NAIAD_ISA_REGISTRY[name] is not None
    ]
    core_class = NaiadRV32IM if "m" in isa else NaiadRV32
    core = core_class(
        reset_pc,
        [memory],
        extensions=extensions,
        unknown_instr_is_error=True,
        core_id=core_id,
    )
    core.csrs = NaiadCSRFile32(core.register_file, core_label=f"hart{core_id}")
    core.register_file.csrs = core.csrs
    core.enable_traps()
    return core


def make_matrix_unit(spec, xlen):
    """Build one hart's matrix unit from the design's :class:`~framework.naiad.design.MatrixSpec`."""
    try:
        return MatrixUnit(
            m=spec.m,
            n=spec.n,
            k=spec.k,
            tiles=spec.tiles,
            accs=spec.accs,
            in_formats=spec.in_formats,
            acc_formats=spec.acc_formats,
            features=spec.features,
            acc_mem=spec.acc_mem,
            k_chunk=spec.k_chunk,
            numerics=spec.numerics,
            fidelity=spec.fidelity,
            xlen=xlen,
        )
    except ImportError as e:
        listed = ", ".join(spec.in_formats + spec.acc_formats)
        raise NaiadBuildError(
            f"the matrix unit's formats ({listed}) need a missing module: {e}"
        ) from None


@dataclass
class RunResult:
    exit_code: int | None
    cycles: int
    console: bytes
    counters: dict = field(default_factory=dict)

    @property
    def timed_out(self):
        return self.exit_code is None


class NaiadSoC:
    def __init__(self, design, count_all=False):
        self.design = design
        cores_spec = design.cores
        # Multi-hart rules (validated in design.py): RV64 with Zicsr, atomics and
        # a CLINT; see the module docstring of framework.naiad.gen for crt0's boot.
        multicore = cores_spec.count > 1

        self.blocks = {
            spec.name: IP_REGISTRY[spec.type](spec) for spec in design.blocks
        }
        mem_map = MemoryMap()
        for block in self.blocks.values():
            if block.mapped:
                mem_map[AddressRange(block.base, block.size)] = block
        self.memory = (MulticoreMemory if multicore else NaiadMemory)(mem_map)

        self.cores = [
            make_core(
                cores_spec.xlen,
                cores_spec.isa,
                cores_spec.reset_pc,
                self.memory,
                i,
                vlen=cores_spec.vlen,
                elen=cores_spec.elen,
            )
            for i in range(cores_spec.count)
        ]
        if multicore:
            self.memory.reservation_holders = tuple(c.register_file for c in self.cores)
        # The matrix unit is attached after construction, so the core factory
        # stays about the core: AME goes last in each hart's ISA chain.
        self.matrix_units = []
        if design.matrix is not None and design.matrix.mode == "ime":
            from framework.matrix.ime_isa import IMEUnit
            from framework.matrix.ime_isa import attach as attach_ime

            for core in self.cores:
                unit = IMEUnit(design.matrix, core.vector, cores_spec.xlen)
                self.matrix_units.append(attach_ime(core, unit))
        elif design.matrix is not None and design.matrix.mode == "vme":
            from framework.matrix.vme_isa import VMEUnit
            from framework.matrix.vme_isa import attach as attach_vme

            for core in self.cores:
                unit = VMEUnit(design.matrix, core.vector, cores_spec.xlen)
                self.matrix_units.append(attach_vme(core, unit))
        elif design.matrix is not None and design.matrix.mode == "ztt":
            from framework.matrix.ztt_isa import attach as attach_ztt
            from framework.matrix.ztt_isa import make_harts

            for core, hart in zip(self.cores, make_harts(design.matrix, self.cores)):
                self.matrix_units.append(attach_ztt(core, hart))
        elif design.matrix is not None:
            for core in self.cores:
                unit = make_matrix_unit(design.matrix, cores_spec.xlen)
                self.matrix_units.append(attach_matrix_unit(core, unit))
        # Blocks with a clock_tick (the DMA engine) tick after the cores.
        clocked = [b for b in self.blocks.values() if hasattr(b, "clock_tick")]
        clints = [b for b in self.blocks.values() if b.spec.type == "clint"]
        if clints:  # wfi's timer wake-up (mtime >= mtimecmp) is checked each cycle
            from framework.ip.clint import TimerWake

            clocked.append(TimerWake(clints[0]))
        self.clock = NaiadClock([*self.cores, *clocked])
        for core in self.cores:
            core.csrs.clock_owner = self.clock
            if isinstance(core.csrs, CSRFile64) and clints:
                core.csrs.time_source = clints[0].mtime
            if clints:
                core.csrs.mip_source = lambda h=core.core_id: clints[0].pending(h)
        self.reset_line = Reset(
            [self.clock, *self.cores, *self.blocks.values(), *self.matrix_units]
        )
        self.device = Device(self.memory, [self.clock], [self.reset_line])
        self.halted = False
        self.image = None
        for block in self.blocks.values():
            block.attach(self)
        if clints:
            for core in self.cores:
                clints[0].bind_wfi(core, always_sleeps=multicore)
        # The cycle model, before the counter hub so the hub finds each
        # ``rv_cost``. No ``timing:`` section leaves every core untimed.
        if design.timing is not None:
            install_timing(self)
        self._wire_counters(count_all)

    def _wire_counters(self, count_all):
        """Build the counter hub, then bind the Zihpm CSRs and every block that reads it.

        Last in construction, so the hub sees every unit and block already
        attached. The on-demand families are installed only when something
        names one of their events, ``counters.enable`` lists them, or ``count_all``.
        """
        spec = self.design.counters
        wanted = [*spec.events]
        for block in self.blocks.values():
            wanted += block.required_events()
        families = requested_families(wanted) | set(spec.enable)
        if count_all:
            families.add("all")
        self.counter_hub = hub = CounterHub.discover(self, families)
        try:
            for n, event in spec.hpm:
                reader = hub.reader(event)
                for core in self.cores:
                    # On a multi-hart SoC each hart counts its own share of a
                    # per-hart event (hart<N>.<event>); SoC-wide events are shared.
                    own = f"hart{core.core_id}.{event}"
                    if len(self.cores) > 1 and own in hub.names():
                        core.csrs.bind_hpm(n, hub.reader(own))
                    else:
                        core.csrs.bind_hpm(n, reader)
            for block in self.blocks.values():
                block.bind_counters(hub)
        except CounterError as e:
            raise NaiadBuildError(f"design '{self.design.name}': {e}") from None

    # -- lookup ------------------------------------------------------------

    def blocks_of_type(self, type_name):
        return [b for b in self.blocks.values() if b.spec.type == type_name]

    @property
    def htif(self):
        found = self.blocks_of_type("htif")
        return found[0] if found else None

    # -- program -----------------------------------------------------------

    def load(self, elf_path):
        image = load_elf(elf_path, self.memory)
        if image.xlen != self.design.cores.xlen:
            raise NaiadBuildError(
                f"'{elf_path}' is ELF{image.xlen}, but design '{self.design.name}' has xlen {self.design.cores.xlen}"
            )
        for block in self.blocks.values():
            block.on_load(image)
        self.image = image
        return image

    def reset(self):
        self.halted = False
        self.device.reset()
        self.counter_hub.reset()

    def halt(self):
        self.halted = True
        for core in self.cores:
            core.stop()

    def run(self, max_cycles):
        """Reset, then tick until the program exits through the HTIF or ``max_cycles`` pass."""
        self.reset()
        clock = self.clock
        cycle = -1
        try:
            for cycle in range(max_cycles):
                clock.clock_tick(cycle)
                if self.halted:
                    break
        except Exception as e:
            pcs = ", ".join(
                f"hart{c.core_id} pc={hex(c.pc_register.read_uint())}"
                for c in self.cores
            )
            raise NaiadRunError(f"at cycle {cycle} ({pcs}): {e}") from e
        clock.clock_tick_num = cycle + 1
        htif = self.htif
        return RunResult(
            exit_code=None if htif is None else htif.exit_code,
            cycles=cycle + 1,
            console=b"" if htif is None else bytes(htif.console),
            counters=self.counters(),
        )

    # -- counters ----------------------------------------------------------

    def counter(self, name):
        """The live value of counter-hub event ``name`` (see ``framework.ip.counters``)."""
        return self.counter_hub.read(name)

    def counters(self):
        """Every counter-hub event and its value, in report order."""
        return self.counter_hub.snapshot()


def build(design, count_all=False):
    """Build a :class:`NaiadSoC` from a validated :class:`~framework.naiad.design.Design`.

    ``count_all`` installs every on-demand counter family (instruction classes,
    per-block bus bytes) whether or not the design names one.
    """
    return NaiadSoC(design, count_all)
