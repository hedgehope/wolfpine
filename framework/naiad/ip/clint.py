"""CLINT: ``msip``, ``mtimecmp`` and ``mtime`` at the SiFive/Spike offsets.

``mtime`` is the SoC clock's cycle number divided by ``cycles_per_tick``, plus
whatever offset a write to it established. Naiad cores take no interrupts yet,
so ``msip`` and ``mtimecmp`` are plain storage: software can read back what it
wrote, and nothing fires.
"""

from framework.naiad.ip.base import RegisterBlock

MSIP = 0x0000
MTIMECMP = 0x4000
MTIME = 0xBFF8
_MASK64 = (1 << 64) - 1


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
        self.reset()

    def reset(self):
        self.store = {}
        self.mtime_offset = 0

    def _ticks(self):
        return self.soc.clock.current_cycle // self.cycles_per_tick

    def mtime(self):
        return (self._ticks() + self.mtime_offset) & _MASK64

    def read_reg(self, offset):
        if offset == MTIME:
            return self.mtime()
        return self.store.get(offset, 0)

    def write_reg(self, offset, value):
        if offset == MTIME:
            self.mtime_offset = (value - self._ticks()) & _MASK64
        else:
            self.store[offset] = value

    def header_defines(self):
        return [
            ("MSIP", MSIP),
            ("MTIMECMP", MTIMECMP),
            ("MTIME", MTIME),
            ("CYCLES_PER_TICK", self.cycles_per_tick),
        ]
