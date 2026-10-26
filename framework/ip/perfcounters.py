"""Performance counters as MMIO: counter ``i`` reads at offset ``8 * i``.

The block exposes, read-only, the counter-hub events its ``counters`` param
lists (``framework.ip.counters``), so a program without Zicsr can still time
itself and a program with it can read events no Zihpm counter is bound to. Any
hub event may be listed; the names are checked against the built SoC's hub, so
an event a new IP block contributes needs no change here.
"""

import re

from framework.ip.base import RegisterBlock

_M64 = (1 << 64) - 1


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
        bad = [n for n in names if not isinstance(n, str) or not n]
        if bad:
            raise error(f"block '{spec.name}': counters must be event names, got {bad}")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise error(f"block '{spec.name}': counter(s) {dupes} listed twice")

    def __init__(self, spec):
        super().__init__(spec)
        self.names = self._names(spec.params)
        self._readers = ()

    def required_events(self):
        return self.names

    def bind_counters(self, hub):
        self._readers = tuple(hub.reader(name) for name in self.names)

    def read_reg(self, offset):
        index = offset // 8
        if index >= len(self._readers):
            return 0
        return self._readers[index]() & _M64

    def header_defines(self):
        return [
            (re.sub(r"[^A-Za-z0-9]", "_", name).upper(), 8 * i)
            for i, name in enumerate(self.names)
        ]
