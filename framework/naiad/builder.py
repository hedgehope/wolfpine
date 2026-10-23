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
from framework.matrix.ame_isa import attach as attach_matrix_unit
from framework.matrix.unit import COUNTER_NAMES as MATRIX_COUNTERS
from framework.matrix.unit import MatrixUnit
from framework.memory.memory import MemorySpace
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.naiad.registry import IP_REGISTRY, NAIAD_ISA_REGISTRY
from framework.pe.rv.isa.zicsr64_isa import CSRFile64
from framework.pe.rv.isa.zicsr_isa import CSRFile
from framework.pe.rv.rv32 import RV32I, RV32IM
from framework.pe.rv.rv64 import RV64I, rv64_extensions
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

    def clock_tick(self, cycle):
        self.current_cycle = cycle
        super().clock_tick(cycle)

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


def make_core(xlen, isa, reset_pc, memory, core_id):
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
        core = RV64I(
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
        return core
    if xlen != 32:
        raise NaiadBuildError(f"unsupported xlen {xlen}")
    extensions = [
        NAIAD_ISA_REGISTRY[name] for name in isa if NAIAD_ISA_REGISTRY[name] is not None
    ]
    core_class = RV32IM if "m" in isa else RV32I
    core = core_class(
        reset_pc,
        [memory],
        extensions=extensions,
        unknown_instr_is_error=True,
        core_id=core_id,
    )
    core.csrs = CSRFile(core.register_file, core_label=f"hart{core_id}")
    core.register_file.csrs = core.csrs
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
            k_chunk=spec.k_chunk,
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
    def __init__(self, design):
        self.design = design
        cores_spec = design.cores
        if cores_spec.count != 1:
            raise NaiadBuildError(
                f"cores.count is {cores_spec.count}, but only single-hart designs are supported so far "
                f"(the CSR file reports mhartid 0 for every hart, so crt0 cannot give them separate stacks)"
            )

        self.blocks = {
            spec.name: IP_REGISTRY[spec.type](spec) for spec in design.blocks
        }
        mem_map = MemoryMap()
        for block in self.blocks.values():
            if block.mapped:
                mem_map[AddressRange(block.base, block.size)] = block
        self.memory = NaiadMemory(mem_map)

        self.cores = [
            make_core(
                cores_spec.xlen, cores_spec.isa, cores_spec.reset_pc, self.memory, i
            )
            for i in range(cores_spec.count)
        ]
        # The matrix unit is attached after construction, so the core factory
        # stays about the core: AME goes last in each hart's ISA chain.
        self.matrix_units = []
        if design.matrix is not None:
            for core in self.cores:
                unit = make_matrix_unit(design.matrix, cores_spec.xlen)
                self.matrix_units.append(attach_matrix_unit(core, unit))
        self.clock = NaiadClock(self.cores)
        clints = [b for b in self.blocks.values() if b.spec.type == "clint"]
        for core in self.cores:
            core.csrs.clock_owner = self.clock
            if isinstance(core.csrs, CSRFile64) and clints:
                core.csrs.time_source = clints[0].mtime
        self.reset_line = Reset(
            [self.clock, *self.cores, *self.blocks.values(), *self.matrix_units]
        )
        self.device = Device(self.memory, [self.clock], [self.reset_line])
        self.halted = False
        self.image = None
        for block in self.blocks.values():
            block.attach(self)

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
        if name == "cycle":
            return self.clock.current_cycle
        if name == "instret":
            return sum(c.csrs.retired for c in self.cores)
        if name == "mtime":
            clints = self.blocks_of_type("clint")
            return clints[0].mtime() if clints else 0
        if name == "htif_chars":
            htif = self.htif
            return 0 if htif is None else len(htif.console)
        if name.startswith("matrix_") and name[7:] in MATRIX_COUNTERS:
            return sum(u.stats[name[7:]] for u in self.matrix_units)
        raise KeyError(f"unknown counter '{name}'")

    def counters(self):
        out = {"cycles": self.clock.clock_tick_num, "instret": self.counter("instret")}
        for core in self.cores:
            out[f"hart{core.core_id}.instret"] = core.csrs.retired
            unit = getattr(core, "matrix", None)
            if unit is not None:
                out.update(
                    {f"hart{core.core_id}.matrix.{k}": v for k, v in unit.stats.items()}
                )
        for block in self.blocks.values():
            out.update({f"{block.name}.{k}": v for k, v in block.counters().items()})
        return out


def build(design):
    """Build a :class:`NaiadSoC` from a validated :class:`~framework.naiad.design.Design`."""
    return NaiadSoC(design)
