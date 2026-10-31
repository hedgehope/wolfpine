"""CLINT: ``msip``, ``mtimecmp`` and ``mtime`` at the SiFive/Spike offsets.

Per hart ``h`` (SiFive layout, as Spike and QEMU's ``virt`` use):

* ``msip[h]`` at ``0x0000 + 4*h``: a 32-bit register whose bit 0 is the
  software-interrupt pending bit (WARL; the other bits read as zero);
* ``mtimecmp[h]`` at ``0x4000 + 8*h``: 64 bits;
* one shared ``mtime`` at ``0xBFF8``.

``mtime`` is the SoC clock's cycle number divided by ``cycles_per_tick``, plus
whatever offset a write to it established. ``msip`` and ``mtimecmp`` matter in
two places.

**Interrupts.** :meth:`CLINT.pending` is a hart's ``mip`` (``MSIP`` = bit 3
from ``msip[h]``, ``MTIP`` = bit 7 when ``mtime >= mtimecmp[h]``), level
triggered. The builder wires it to each core's ``mip`` CSR, and a core whose
``mtvec`` is set takes the machine software / timer interrupt when
``mstatus.MIE`` and the matching ``mie`` bit are set
(:mod:`framework.pe.rv.trap`). A handler silences the timer by writing
``mtimecmp`` and the software interrupt by clearing ``msip``.

**``wfi``.** The builder routes each hart's ``wfi`` to :meth:`CLINT.wfi`, which
puts the hart to sleep (its core stops ticking) unless a wake-up is already
due. On a single-hart design a ``wfi`` with neither ``mie.MSIE`` nor
``mie.MTIE`` set is still a nop (nothing could ever wake the hart); on a
multi-hart design it sleeps regardless, as crt0's parked harts rely on. Hart
``h`` wakes when

* a store sets ``msip[h]`` while its ``mie.MSIE`` (bit 3) is set, or
* ``mtime >= mtimecmp[h]`` while its ``mie.MTIE`` (bit 7) is set (checked every
  cycle while a hart sleeps with MTIE set, so a later ``mtimecmp`` or ``mtime``
  write counts too; the hart resumes the cycle after the one it became due).

This is Spike's rule (``wfi`` resumes when ``mip & mie`` is nonzero). The hart
resumes at ``wfi + 4``; if ``mstatus.MIE`` is set and its ``mtvec`` is armed the
interrupt is then taken, with ``mepc`` = ``wfi + 4``. With ``mstatus.MIE``
clear (or ``mtvec`` unset) it just carries on, which is what generated
multi-hart ``crt0.S`` relies on to park secondary harts until hart 0 releases
them.

A sleeping hart's ``mcycle`` is advanced by the cycles it slept when it wakes.

Offsets outside these registers (inside the 64 KiB block) are plain storage.
"""

from framework.device.clock import Clockable
from framework.ip.base import RegisterBlock

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
        #: hart id -> True if its ``wfi`` sleeps even with no wake-up source enabled.
        self._always_sleeps = {}
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
        #: Sleeping harts that had ``mie.MTIE`` set (checked by :meth:`timer_tick`).
        self.timer_sleepers = set()

    def _ticks(self):
        return self.soc.clock.current_cycle // self.cycles_per_tick

    def mtime(self):
        return (self._ticks() + self.mtime_offset) & _MASK64

    # -- wfi ------------------------------------------------------------------

    def bind_wfi(self, core, always_sleeps=True):
        """Route ``core``'s ``wfi`` here (see the module docstring).

        ``always_sleeps`` False (single-hart designs): a ``wfi`` with no wake-up
        source enabled in ``mie`` is a nop rather than a sleep nobody ends.
        """
        self.cores[core.core_id] = core
        self._always_sleeps[core.core_id] = always_sleeps
        core.register_file.wfi = lambda c=core: self.wfi(c)

    def _mie(self, core):
        return core.csrs.read(_CSR_MIE) if core.csrs is not None else 0

    def mtimecmp(self, hart):
        return self.store.get(MTIMECMP + MTIMECMP_STRIDE * hart, 0)

    def pending(self, hart):
        """Hart ``hart``'s pending machine interrupts as ``mip`` bits (MSIP bit 3, MTIP bit 7)."""
        bits = 0
        if hart < self.harts and self.msip[hart]:
            bits |= _MIE_MSIE
        if self._timer_due(hart):
            bits |= _MIE_MTIE
        return bits

    def _timer_due(self, hart):
        # mtimecmp resets to 0 here (Spike's reset value too), so a hart that
        # enables MTIE without programming it falls straight through wfi.
        return self.mtime() >= self.mtimecmp(hart)

    def wfi(self, core):
        """Hart ``core`` executed ``wfi``: sleep unless a wake-up is already due."""
        hart = core.core_id
        mie = self._mie(core)
        if not mie & (_MIE_MSIE | _MIE_MTIE) and not self._always_sleeps.get(
            hart, True
        ):
            return
        if mie & _MIE_MTIE and self._timer_due(hart):
            return
        if mie & _MIE_MSIE and hart < self.harts and self.msip[hart]:
            return
        self.sleeping[hart] = self.soc.clock.current_cycle
        if mie & _MIE_MTIE:
            self.timer_sleepers.add(hart)
        core.active = False

    def timer_tick(self, cycle):
        """Wake the sleeping harts whose timer is due (the builder ticks this on multi-hart SoCs)."""
        if self.timer_sleepers:
            for hart in sorted(self.timer_sleepers):
                if self._timer_due(hart):
                    self._wake(hart, _MIE_MTIE)

    def _wake(self, hart, cause=_MIE_MSIE):
        core = self.cores.get(hart)
        slept_at = self.sleeping.get(hart)
        if core is None or slept_at is None or not self._mie(core) & cause:
            return
        del self.sleeping[hart]
        self.timer_sleepers.discard(hart)
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


class TimerWake(Clockable):
    """Ticks :meth:`CLINT.timer_tick` once a cycle, after the cores (multi-hart SoCs only)."""

    def __init__(self, clint):
        self.clint = clint

    def clock_tick(self, cycle_num):
        self.clint.timer_tick(cycle_num)
