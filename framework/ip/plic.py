"""PLIC-lite: a minimal machine external interrupt controller (``mip.MEIP``, cause 11).

Not the RISC-V PLIC register map: no priorities, thresholds or claim/complete.
Up to 64 level-triggered sources, one *pending* word, one *enable* word per
hart, and nothing else. A hart's machine external interrupt is pending while
``pending & enable[hart]`` is non-zero; the builder wires that to the hart's
``mip`` (bit 11) through the CLINT, and a core whose ``mtvec`` is armed takes it
(highest priority, ahead of software and timer) when ``mstatus.MIE`` and
``mie.MEIE`` are set (:mod:`framework.pe.rv.trap`). ``wfi`` wakes on it with
``mie.MEIE`` set (:mod:`framework.ip.clint`). Without a ``plic`` block nothing
changes anywhere: no design that omits it ever sees ``MEIP``.

Registers (64-bit, 8-byte stride; an RV32 hart uses two ``lw``). Source ``i``
is bit ``i`` (0-based, unlike the real PLIC's 1-based ids):

=========================== ===============================================
``0x0000`` ``PENDING``      read: pending sources; write: 1-bits are *set*
                            (software-raised sources, handy for testing)
``0x0008`` ``CLEAR``        write: 1-bits are cleared from ``PENDING``; reads 0
``0x1000 + 8*h`` ``ENABLE`` hart ``h``'s enable mask (read / write)
=========================== ===============================================

A handler finds its source with ``pending & enable`` and silences it by writing
``CLEAR`` (level-triggered: a source raised again re-pends it).

Device models and drivers raise a line with :meth:`PLICLite.raise_source` /
:meth:`PLICLite.lower_source`. A design can also schedule inputs with
``params: {schedule: [[cycle, source], ...]}``: source ``source`` goes pending
at the end of SoC cycle ``cycle`` (so an interrupt-driven program can be run
deterministically with no device model). ``params: {sources: N}`` (default 32,
at most 64) limits the valid sources; bits above it never pend.
"""

from framework.ip.base import RegisterBlock

PENDING = 0x0000
CLEAR = 0x0008
ENABLE = 0x1000
ENABLE_STRIDE = 8
MIP_MEIP = 1 << 11
MAX_SOURCES = 64


class PLICLite(RegisterBlock):
    PARAMS = frozenset({"sources", "schedule"})

    @classmethod
    def block_size(cls, params, window):
        return 0x4000

    @classmethod
    def validate_params(cls, spec, error):
        sources = spec.params.get("sources", 32)
        if not isinstance(sources, int) or not 1 <= sources <= MAX_SOURCES:
            raise error(
                f"block '{spec.name}': sources must be an integer in 1..{MAX_SOURCES}"
            )
        for entry in spec.params.get("schedule", []):
            ok = (
                isinstance(entry, (list, tuple))
                and len(entry) == 2
                and all(isinstance(v, int) for v in entry)
                and entry[0] >= 0
                and 0 <= entry[1] < sources
            )
            if not ok:
                raise error(
                    f"block '{spec.name}': schedule entries are [cycle, source] with "
                    f"cycle >= 0 and 0 <= source < sources ({sources}), got {entry!r}"
                )

    @classmethod
    def validate_in_design(cls, spec, design, error):
        if not design.blocks_of_type("clint"):
            raise error(
                f"block '{spec.name}': a plic needs a clint block (the interrupt and "
                f"wfi wake-up path run through it)"
            )
        if len(design.blocks_of_type("plic")) > 1:
            raise error("at most one plic block is supported")

    def __init__(self, spec):
        super().__init__(spec)
        self.sources = int(spec.params.get("sources", 32))
        self._mask = (1 << self.sources) - 1
        self._schedule = sorted(
            (int(c), int(s)) for c, s in spec.params.get("schedule", [])
        )
        self.harts = 1
        self.clint = None
        self.reset()

    def attach(self, soc):
        super().attach(soc)
        self.harts = len(soc.cores)
        self.enable = [0] * self.harts
        self.clint = next(
            (b for b in soc.blocks.values() if b.spec.type == "clint"), None
        )
        if self.clint is not None:
            self.clint.external_pending = self.meip

    def reset(self):
        self.pending = 0
        self.enable = [0] * self.harts
        self._next = 0

    # -- the interrupt line ----------------------------------------------------

    def meip(self, hart):
        """``mip`` bits for ``hart``: ``MEIP`` while an enabled source is pending."""
        if hart < self.harts and self.pending & self.enable[hart]:
            return MIP_MEIP
        return 0

    def raise_source(self, source):
        """Make ``source`` pending (and wake a hart sleeping in ``wfi`` that is waiting for it)."""
        if not 0 <= source < self.sources:
            raise ValueError(
                f"{self.name}: source {source} outside 0..{self.sources - 1}"
            )
        self.pending |= 1 << source
        self._notify()

    def lower_source(self, source):
        self.pending &= ~(1 << source)

    def _notify(self):
        if self.clint is not None and self.clint.sleeping:
            self.clint.wake_external()

    def clock_tick(self, cycle_num):
        schedule = self._schedule
        while self._next < len(schedule) and schedule[self._next][0] <= cycle_num:
            self.raise_source(schedule[self._next][1])
            self._next += 1

    # -- registers ---------------------------------------------------------------

    def read_reg(self, offset):
        if offset == PENDING:
            return self.pending
        if offset >= ENABLE:
            hart = (offset - ENABLE) // ENABLE_STRIDE
            return self.enable[hart] if hart < self.harts else 0
        return 0

    def write_reg(self, offset, value):
        if offset == PENDING:
            self.pending |= value & self._mask
            self._notify()
        elif offset == CLEAR:
            self.pending &= ~value
        elif offset >= ENABLE:
            hart = (offset - ENABLE) // ENABLE_STRIDE
            if hart < self.harts:
                self.enable[hart] = value & self._mask
                self._notify()

    def header_defines(self):
        return [
            ("PENDING", PENDING),
            ("CLEAR", CLEAR),
            ("ENABLE", ENABLE),
            ("SOURCES", self.sources),
        ]
