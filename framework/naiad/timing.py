"""A Naiad design's ``timing:`` section, and installing its cycle model on each hart.

The model itself is generic RV64 core timing (:mod:`framework.pe.rv.timing.core`,
with the vector unit in :mod:`framework.pe.rv.timing.vector` and memory RAW in
:mod:`framework.pe.rv.timing.store_hazards`); this module is the SoC glue:
:func:`parse_timing` checks the section against the design (its blocks and
block types), and :func:`install` gives every hart of a built SoC its
:class:`~framework.pe.rv.timing.core.NaiadCostState`. The model's names are
re-exported here for older imports.
"""

from framework.pe.rv.timing.core import (  # noqa: F401  (re-exported)
    DEFAULT_REGION,
    REGION_DEFAULTS,
    STALL_REASONS,
    DualIssueCostState,
    MemoryTiming,
    NaiadCostState,
    Region,
    RegionTimingSpec,
    ScalarTimingSpec,
    SharedPort,
    SharedPortRegion,
    TimingSpec,
    _bandwidth,
    _flag,
    _number,
    _parse_fields,
    _parse_matrix,
    _parse_scalar,
)
from framework.pe.rv.timing.store_hazards import StoresInFlight
from framework.pe.rv.timing.vector import parse_vector
from framework.util.schema import DesignError
from framework.util.schema import check_keys as _check_keys


def _parse_memory(raw, design):
    from framework.naiad.registry import IP_REGISTRY

    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise DesignError(
            "timing.memory: expected a mapping of block name or type -> {latency, bandwidth}"
        )
    names = {b.name for b in design.blocks if b.base is not None}
    out = {}
    for key, entry in raw.items():
        key = str(key)
        if key != "default" and key not in names and key not in IP_REGISTRY:
            raise DesignError(
                f"timing.memory: '{key}' is neither a mapped block of this design ({sorted(names)}), "
                f"a block type, nor 'default'"
            )
        out[key] = _parse_fields(
            entry,
            RegionTimingSpec,
            f"timing.memory.{key}",
            {"latency": lambda v, w: _number(v, w, minimum=0), "bandwidth": _bandwidth},
        )
    return out


def parse_timing(raw, design):
    """The ``timing:`` section of ``design`` (blocks and ``matrix:`` already parsed), or None."""
    if raw is None:
        return None
    _check_keys(raw, {"scalar", "memory", "matrix", "vector", "shared_ports"}, "timing")
    return TimingSpec(
        scalar=_parse_scalar(raw.get("scalar")),
        memory=_parse_memory(raw.get("memory"), design),
        matrix=_parse_matrix(raw.get("matrix"), design),
        vector=parse_vector(raw.get("vector"), design, _parse_fields, _number, _flag),
        shared_ports=_flag(raw.get("shared_ports", True), "timing.shared_ports"),
    )


def install(soc):
    """Give every hart of ``soc`` a :class:`NaiadCostState` from its design's ``timing:`` section.

    Called by the builder after every block is attached. A block that declares
    ``fronts`` and has an ``access`` method (the ``cache`` IP,
    ``framework.ip.cache``) becomes the latency provider of each region
    it fronts. ``Cache.access`` returns the whole time spent in the cache, hit
    latency included, so it **replaces** the region's latency: a cached load
    costs ``scalar.load + access(...)``, and a tile transfer
    ``tile_load_latency + access(...)``. A cache whose ``backing_latency`` is
    unset (``miss_latency: backing``) gets the fronted region's own latency as
    its fill cost. Stores update the cache's state but are posted, so their
    answer is not charged; DMA traffic does not go through the cache or the
    regions' ports.

    On a multi-hart SoC each hart has its own cost state (scoreboard, units,
    matrix and vector models). With ``timing.shared_ports`` (the default) the
    harts share each region's port, so a region's ``bandwidth`` is the harts'
    combined cap (fixed priority, hart 0 first); a region with unlimited
    bandwidth (``null``, as the dwarfs' L1) has no contention. A cache block
    is one shared cache (the harts' accesses all move its state) unless it is
    ``private: true``: then each hart gets its own L1 view
    (``block.for_hart(h)``), kept coherent by snooping MSI, whose bus cycles
    loads pay as latency and scalar stores as LSU occupancy; AMOs (not lr) are
    timed as writes there (``framework.ip.cache``).
    """
    timing = soc.design.timing
    # Multi-hart: one port per region for all harts (timing.shared_ports).
    ports = {} if len(soc.cores) > 1 and timing.shared_ports else None
    # One memory, so one table of stores in flight that every hart's loads check.
    inflight = StoresInFlight()
    width = timing.scalar.issue_width
    cost_state = DualIssueCostState if width == 2 else NaiadCostState
    for core in soc.cores:
        memory = MemoryTiming(timing, soc.blocks, ports, inflight)
        for block in soc.blocks.values():
            fronts = getattr(block, "fronts", None)
            if fronts and callable(getattr(block, "access", None)):
                for_hart = getattr(block, "for_hart", None)
                provider = block if for_hart is None else for_hart(core.core_id)
                for name in fronts:
                    memory.attach_provider(name, provider)
                if hasattr(block, "backing_latency") and block.backing_latency is None:
                    block.backing_latency = lambda addr, m=memory: (
                        m.lookup(addr).latency
                    )
        core.rv_cost = cost_state(
            timing,
            memory,
            soc.design.cores.xlen,
            getattr(core, "matrix", None),
            getattr(core, "vector", None),
        )
    if width > 1:
        soc.clock.set_issue_width(width, soc.cores)
