"""Performance counters as MMIO: counter ``i`` reads at offset ``8 * i``.

The block exposes, read-only, a chosen subset of the SoC's counter dict (see
``NaiadSoC.counter``), so a program without Zicsr can still time itself and a
program with it can see counters no CSR names.
"""

from framework.matrix.unit import COUNTER_NAMES as _MATRIX_COUNTERS
from framework.naiad.ip.base import RegisterBlock

#: Names a design may list; ``NaiadSoC.counter`` answers each. The ``matrix_*``
#: ones are summed over the harts' matrix units (zero without one).
KNOWN_COUNTERS = ("cycle", "instret", "mtime", "htif_chars") + tuple(
    f"matrix_{name}" for name in _MATRIX_COUNTERS
)


class PerfCounters(RegisterBlock):
    PARAMS = frozenset({"counters"})
    DEFAULT_COUNTERS = ("cycle", "instret")

    @classmethod
    def _names(cls, params):
        return tuple(params.get("counters", cls.DEFAULT_COUNTERS))

    @classmethod
    def block_size(cls, params, window):
        return 8 * max(1, len(cls._names(params)))

    @classmethod
    def validate_params(cls, spec, error):
        names = cls._names(spec.params)
        unknown = [n for n in names if n not in KNOWN_COUNTERS]
        if unknown:
            raise error(
                f"block '{spec.name}': unknown counter(s) {unknown}; known: {list(KNOWN_COUNTERS)}"
            )

    def __init__(self, spec):
        super().__init__(spec)
        self.names = self._names(spec.params)

    def read_reg(self, offset):
        index = offset // 8
        if index >= len(self.names):
            return 0
        return self.soc.counter(self.names[index]) & ((1 << 64) - 1)

    def header_defines(self):
        return [(name.upper(), 8 * i) for i, name in enumerate(self.names)]
