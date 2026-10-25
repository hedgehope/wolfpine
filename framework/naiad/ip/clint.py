"""CLINT: ``msip``, ``mtimecmp`` and ``mtime`` at the SiFive/Spike offsets.

Per hart ``h`` (SiFive layout, as Spike and QEMU's ``virt`` use):

* ``msip[h]`` at ``0x0000 + 4*h``: a 32-bit register whose bit 0 is the
  software-interrupt pending bit (WARL; the other bits read as zero);
* ``mtimecmp[h]`` at ``0x4000 + 8*h``: 64 bits;
* one shared ``mtime`` at ``0xBFF8``.

``mtime`` is the SoC clock's cycle number divided by ``cycles_per_tick``, plus
whatever offset a write to it established. Naiad cores take no interrupts, so
nothing ever traps; ``mtimecmp`` is plain storage, and ``msip`` matters in one
place only: **``wfi``**. On a multi-hart design the builder routes each hart's
``wfi`` to :meth:`CLINT.wfi`, which puts the hart to sleep (its core stops
ticking) unless a wake-up is already due, and a store that sets ``msip[h]``
wakes hart ``h``. A wake-up counts only when ``mie.MSIE`` (bit 3) is set, as on
Spike, where ``wfi`` resumes when ``mip & mie`` is nonzero; ``mstatus.MIE`` is
irrelevant to ``wfi``, so no trap is taken. A hart that has ``mie.MTIE`` set
does not sleep at all (a timer wake-up is not modelled, so ``wfi`` is then a
``nop`` and a software timer loop still progresses). Generated multi-hart
``crt0.S`` uses exactly this to park secondary harts until hart 0 releases them.

A sleeping hart's ``mcycle`` is advanced by the cycles it slept when it wakes.

Offsets outside these registers (inside the 64 KiB block) are plain storage.
"""

from framework.naiad.ip.base import RegisterBlock

MSIP = 0x0000
MTIMECMP = 0x4000
MTIME = 0xBFF8
MSIP_STRIDE = 4
MTIMECMP_STRIDE = 8
_MASK64 = (1 << 64) - 1
_MIE_MSIE = 1 << 3
_MIE_MTIE = 1 << 7
_CSR_MIE = 0x304


class CLINT(RegisterBlock):
    PARAMS = frozenset({"cycles_per_tick"})

    @classmethod
    def block_size(cls, params, window):
        return 0x10000

    @classmethod
    def validate_params(cls, spec, error):
        if int(spec.params.get("cycles_per_tick", 1)) < 1:
            raise error(f"block '{spec.name}': cycles_per_tick must be >= 1")

    def __init__(self, spec):
        super().__init__(spec)
        self.cycles_per_tick = int(spec.params.get("cycles_per_tick", 1))
        #: Harts with an ``msip`` register; the SoC's core count once attached.
        self.harts = 1
        #: hart id -> core, for the harts whose ``wfi`` sleeps here.
        self.cores = {}
        self.reset()

    def attach(self, soc):
        super().attach(soc)
        self.harts = len(soc.cores)
        self.msip = [0] * self.harts

    def reset(self):
        self.store = {}
        self.mtime_offset = 0
        self.msip = [0] * self.harts
        #: hart id -> the cycle it went to sleep at.
        self.sleeping = {}

    def _ticks(self):
        return self.soc.clock.current_cycle // self.cycles_per_tick

    def mtime(self):
        return (self._ticks() + self.mtime_offset) & _MASK64

    # -- wfi ------------------------------------------------------------------

    def bind_wfi(self, core):
        """Route ``core``'s ``wfi`` here (multi-hart designs; see the module docstring)."""
        self.cores[core.core_id] = core
        core.register_file.wfi = lambda c=core: self.wfi(c)

    def _mie(self, core):
        return core.csrs.read(_CSR_MIE) if core.csrs is not None else 0

    def wfi(self, core):
        """Hart ``core`` executed ``wfi``: sleep unless a wake-up is already due."""
        hart = core.core_id
        mie = self._mie(core)
        if mie & _MIE_MTIE:
            return  # a timer wake-up is not modelled: wfi is a nop
        if mie & _MIE_MSIE and hart < self.harts and self.msip[hart]:
            return
        self.sleeping[hart] = self.soc.clock.current_cycle
        core.active = False

    def _wake(self, hart):
        core = self.cores.get(hart)
        slept_at = self.sleeping.get(hart)
        if core is None or slept_at is None or not self._mie(core) & _MIE_MSIE:
            return
        del self.sleeping[hart]
        # The ticks it missed, so mcycle counts wall-clock cycles (to within one).
        if core.csrs is not None:
            core.csrs.cycles += max(0, self.soc.clock.current_cycle - slept_at - 1)
        core.active = True

    # -- registers --------------------------------------------------------------

    def read_reg(self, offset):
        if offset == MTIME:
            return self.mtime()
        if offset < MTIMECMP:
            # One 8-byte register holds msip[2k] (low word) and msip[2k + 1].
            hart = offset // MSIP_STRIDE
            lo = self.msip[hart] if hart < self.harts else 0
            hi = self.msip[hart + 1] if hart + 1 < self.harts else 0
            return lo | (hi << 32)
        return self.store.get(offset, 0)

    def write_reg(self, offset, value):
        if offset == MTIME:
            self.mtime_offset = (value - self._ticks()) & _MASK64
        elif offset < MTIMECMP:
            hart = offset // MSIP_STRIDE
            for h, word in ((hart, value & 0xFFFFFFFF), (hart + 1, value >> 32)):
                if h < self.harts:
                    bit = word & 1
                    self.msip[h] = bit
                    if bit and h in self.sleeping:
                        self._wake(h)
        else:
            self.store[offset] = value

    def header_defines(self):
        return [
            ("MSIP", MSIP),
            ("MTIMECMP", MTIMECMP),
            ("MTIME", MTIME),
            ("CYCLES_PER_TICK", self.cycles_per_tick),
        ]
