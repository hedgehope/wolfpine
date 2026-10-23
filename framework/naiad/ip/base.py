"""The contract every Naiad IP block signs.

A block is a :class:`~framework.memory.mem_mapable.MemMapable` mapped at
``AddressRange(spec.base, spec.size)``. ``read``/``write`` receive offsets
relative to ``spec.base``: ``MemorySpace`` subtracts the range's low address
before dispatching, exactly as it does for a ``DRAM``.

Lifecycle, driven by :mod:`framework.naiad.builder`: construct from the spec,
:meth:`attach` once the SoC exists (clock, cores and memory are then
reachable), :meth:`on_load` after an ELF has been written, :meth:`reset` on
every device reset.
"""

from framework.memory.mem_mapable import MemMapable


class IPBlock(MemMapable):
    #: False for blocks with no fixed window (they take neither base nor window).
    mapped = True
    #: The ``params`` keys a design may give this block type.
    PARAMS = frozenset({"size"})

    def __init__(self, spec):
        self.spec = spec
        self.name = spec.name
        self.base = spec.base
        self.size = spec.size
        self.soc = None

    # -- schema ------------------------------------------------------------

    @classmethod
    def block_size(cls, params, window):
        """Bytes mapped for ``params`` in a ``window``-byte window."""
        return int(params.get("size", window))

    @classmethod
    def validate_params(cls, spec, error):
        """Raise ``error`` for a bad ``params`` value (keys are already checked)."""

    @classmethod
    def validate_in_design(cls, spec, design, error):
        """Raise ``error`` for a problem that needs the whole design to see."""

    # -- lifecycle ---------------------------------------------------------

    def attach(self, soc):
        self.soc = soc

    def on_load(self, image):
        pass

    def reset(self):
        pass

    def header_defines(self):
        """Extra ``(name, value)`` pairs for ``naiad.h``, beyond base/size."""
        return []

    def counters(self):
        """Counters this block contributes to the run report."""
        return {}

    # -- MemMapable --------------------------------------------------------

    def getSize(self):
        return self.size


class RegisterBlock(IPBlock):
    """An MMIO block of 64-bit registers at 8-byte strides.

    Accesses of 1-8 bytes are allowed as long as they stay inside one register,
    so an RV32 core reads a 64-bit register as two ``lw`` and an RV64 core as one
    ``ld``. Sub-register writes are read-modify-write of the register.
    """

    def read_reg(self, offset):
        raise NotImplementedError()

    def write_reg(self, offset, value):
        raise IndexError(f"{self.name}: register at offset {hex(offset)} is read-only")

    def _split(self, addr, size):
        reg, off = addr & ~7, addr & 7
        if size <= 0 or off + size > 8 or addr + size > self.size:
            raise IndexError(
                f"{self.name}: access of {size} bytes at offset {hex(addr)} crosses a register"
            )
        return reg, off

    def read(self, addr, size):
        reg, off = self._split(addr, size)
        return self.read_reg(reg).to_bytes(8, "little")[off : off + size]

    def write(self, addr, value, size=None):
        if size is None:
            size = len(value)
        reg, off = self._split(addr, size)
        if off == 0 and size == 8:
            self.write_reg(reg, int.from_bytes(value[:8], "little"))
            return
        current = bytearray(self.read_reg(reg).to_bytes(8, "little"))
        current[off : off + size] = value[:size]
        self.write_reg(reg, int.from_bytes(current, "little"))
