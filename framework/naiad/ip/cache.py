"""A timing-only set-associative cache in front of one or more memory blocks.

The cache never holds or changes data: the cores and the DMA still read and
write the fronted blocks directly, and this model only answers how long an
access would have taken. It tracks tags, LRU order and dirty bits, nothing else,
so a program's results are the same with or without it. Like the HTIF it has no
window; a design names the blocks it fronts::

    - name: l1d
      type: cache
      params: {fronts: [dram], size: 32KiB, assoc: 4, line: 64,
               hit_latency: 1, miss_latency: 40, write_policy: write_back}

The timing model plugs it in per region through :meth:`Cache.access`, which
returns the cycles an access spends in the hierarchy: every line it touches
costs ``hit_latency``, a line fill adds ``miss_latency``, and evicting a dirty
line adds ``writeback_latency`` (default: ``miss_latency``). An access spanning
lines costs the sum over its lines, as if they were looked up one after the
other. ``miss_latency: backing`` takes the fill cost from the fronted region
instead: whoever integrates the cache sets :attr:`Cache.backing_latency` to an
int or a callable ``addr -> cycles``.

Write policies:

* ``write_back`` (default): write-allocate; a write hit or fill marks the line
  dirty, and the dirty line is written back when evicted.
* ``write_through``: no-write-allocate; every write goes to memory as a posted
  store (``write_throughs`` counts them) and costs ``hit_latency``, a write miss
  leaves the cache unchanged, and lines are never dirty.

Replacement is LRU within a set. Single-hart only: there is no coherence, so a
DMA write into a fronted region does not invalidate a line here. Snooping MSI
for multicore is Tier 3 work, out of scope.
"""

from framework.naiad.design import parse_int
from framework.naiad.ip.base import IPBlock

WRITE_POLICIES = ("write_back", "write_through")
_DEFAULTS = {
    "size": 32 * 1024,
    "assoc": 4,
    "line": 64,
    "hit_latency": 1,
    "miss_latency": 20,
}


def _is_pow2(n):
    return n > 0 and not n & (n - 1)


class Cache(IPBlock):
    mapped = False
    PARAMS = frozenset({*_DEFAULTS, "writeback_latency", "write_policy", "fronts"})

    @classmethod
    def _geometry(cls, spec, error):
        params = spec.params
        what = f"block '{spec.name}'"
        values = {}
        for key in ("size", "assoc", "line", "hit_latency"):
            values[key] = parse_int(
                params.get(key, _DEFAULTS[key]), f"{what}.params.{key}"
            )
        miss = params.get("miss_latency", _DEFAULTS["miss_latency"])
        values["miss_latency"] = (
            None
            if miss == "backing"
            else parse_int(miss, f"{what}.params.miss_latency")
        )
        wb = params.get("writeback_latency")
        values["writeback_latency"] = (
            None if wb is None else parse_int(wb, f"{what}.params.writeback_latency")
        )
        for key in ("size", "line", "assoc"):
            if values[key] < 1:
                raise error(f"{what}: {key} must be >= 1")
        for key in ("hit_latency", "miss_latency", "writeback_latency"):
            if values[key] is not None and values[key] < 0:
                raise error(f"{what}: {key} must be >= 0")
        if not _is_pow2(values["line"]):
            raise error(f"{what}: line must be a power of two, got {values['line']}")
        way_bytes = values["assoc"] * values["line"]
        if values["size"] % way_bytes or not _is_pow2(values["size"] // way_bytes):
            raise error(
                f"{what}: size {values['size']} must be assoc x line ({way_bytes}) times a power-of-two set count"
            )
        values["write_policy"] = str(params.get("write_policy", "write_back"))
        if values["write_policy"] not in WRITE_POLICIES:
            raise error(
                f"{what}: write_policy must be one of {list(WRITE_POLICIES)}, got {values['write_policy']!r}"
            )
        fronts = params.get("fronts")
        if not isinstance(fronts, list) or not fronts:
            raise error(f"{what}: fronts must be a non-empty list of block names")
        values["fronts"] = tuple(str(f) for f in fronts)
        return values

    @classmethod
    def validate_params(cls, spec, error):
        cls._geometry(spec, error)

    @classmethod
    def validate_in_design(cls, spec, design, error):
        names = {b.name: b for b in design.blocks}
        for target in spec.params["fronts"]:
            block = names.get(str(target))
            if block is None:
                raise error(
                    f"block '{spec.name}': fronts '{target}', which the design does not have"
                )
            if block.base is None or block.type == "cache":
                raise error(
                    f"block '{spec.name}': fronts '{target}', which is not a memory-mapped block"
                )
        for other in design.blocks_of_type("cache"):
            if other.name != spec.name:
                shared = set(map(str, spec.params["fronts"])) & set(
                    map(str, other.params["fronts"])
                )
                if shared:
                    raise error(
                        f"blocks '{spec.name}' and '{other.name}' both front {sorted(shared)}"
                    )

    def __init__(self, spec):
        super().__init__(spec)
        g = self._geometry(spec, ValueError)
        self.capacity = g["size"]
        self.assoc = g["assoc"]
        self.line = g["line"]
        self.hit_latency = g["hit_latency"]
        self.miss_latency = g["miss_latency"]
        self.writeback_latency = g["writeback_latency"]
        self.write_policy = g["write_policy"]
        self.fronts = g["fronts"]
        self.sets = self.capacity // (self.assoc * self.line)
        self._line_shift = self.line.bit_length() - 1
        #: Fill cost for ``miss_latency: backing``: an int or ``addr -> cycles``.
        self.backing_latency = None
        #: ``(low, high)`` inclusive address ranges of the fronted blocks, once attached.
        self.regions = ()
        self.reset()

    def attach(self, soc):
        super().attach(soc)
        self.regions = tuple(
            (soc.blocks[name].base, soc.blocks[name].base + soc.blocks[name].size - 1)
            for name in self.fronts
        )

    def reset(self):
        # Per set: line numbers in LRU order (least recent first), and the dirty ones.
        self._tags = [[] for _ in range(self.sets)]
        self._dirty = set()
        self.stats = dict.fromkeys(
            ("accesses", "hits", "misses", "evictions", "writebacks", "write_throughs"),
            0,
        )

    # -- the timing API ----------------------------------------------------

    def covers(self, addr):
        return any(low <= addr <= high for low, high in self.regions)

    def contains(self, addr):
        """Whether the line holding ``addr`` is resident (no LRU update, no counting)."""
        line_no = addr >> self._line_shift
        return line_no in self._tags[line_no % self.sets]

    def _fill_cost(self, addr):
        if self.miss_latency is not None:
            return self.miss_latency
        backing = self.backing_latency
        if backing is None:
            raise RuntimeError(
                f"{self.name}: miss_latency is 'backing' but no backing_latency was set"
            )
        return backing(addr) if callable(backing) else backing

    def _writeback_cost(self, addr):
        return (
            self._fill_cost(addr)
            if self.writeback_latency is None
            else self.writeback_latency
        )

    def access(self, addr, size, is_write):
        """Cycles for a ``size``-byte access at ``addr``, updating tags, LRU and counters."""
        first = addr >> self._line_shift
        last = (addr + max(size, 1) - 1) >> self._line_shift
        stats = self.stats
        cycles = 0
        for line_no in range(first, last + 1):
            ways = self._tags[line_no % self.sets]
            stats["accesses"] += 1
            cycles += self.hit_latency
            if is_write and self.write_policy == "write_through":
                stats["write_throughs"] += 1
                if line_no in ways:
                    stats["hits"] += 1
                    ways.remove(line_no)
                    ways.append(line_no)
                else:
                    stats["misses"] += 1
                continue
            if line_no in ways:
                stats["hits"] += 1
                ways.remove(line_no)
                ways.append(line_no)
            else:
                stats["misses"] += 1
                line_addr = line_no << self._line_shift
                cycles += self._fill_cost(line_addr)
                if len(ways) == self.assoc:
                    victim = ways.pop(0)
                    stats["evictions"] += 1
                    if victim in self._dirty:
                        self._dirty.discard(victim)
                        stats["writebacks"] += 1
                        cycles += self._writeback_cost(victim << self._line_shift)
                ways.append(line_no)
            if is_write:
                self._dirty.add(line_no)
        return cycles

    def flush(self):
        """Write back every dirty line (they stay resident, now clean); returns the cycles."""
        cycles = 0
        for line_no in sorted(self._dirty):
            self.stats["writebacks"] += 1
            cycles += self._writeback_cost(line_no << self._line_shift)
        self._dirty.clear()
        return cycles

    def counters(self):
        return dict(self.stats)

    def header_defines(self):
        return [("SIZE", self.capacity), ("ASSOC", self.assoc), ("LINE", self.line)]

    # Never mapped, so never dispatched to: the cache holds no data.
    def read(self, addr, size):
        raise IndexError(f"{self.name}: the cache model has no window of its own")

    def write(self, addr, value, size=None):
        raise IndexError(f"{self.name}: the cache model has no window of its own")
