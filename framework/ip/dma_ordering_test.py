"""DMA memory ordering and cache coherence (``timing.memory.{dma_coherence, war, waw}``).

The DMA is clocked directly (as in :mod:`framework.ip.dma_test`); the cores' stores
in flight are registered by hand in the SoC's table, which is what the
core-side tests of :mod:`framework.pe.rv.timing.memory_ordering_test` show
the cores doing.
"""

import pytest

from framework.ip import dma_test as dt
from framework.naiad.builder import build
from framework.naiad.design import parse_design

N = 64  # bytes moved: 8 cycles at 8 B/cycle
SETUP = 16
BASE = SETUP + N // 8  # undisturbed length of the transfer
SRC = dt.DRAM + 0x1000
DST = dt.SP + 0x100


def _soc(memory=None, cache=None):
    raw = dt._raw()
    raw["timing"] = {
        "scalar": {"store": 1},
        "memory": {
            "scratchpad": {"latency": 0},
            "dram": {"latency": 0},
            **(memory or {}),
        },
    }
    if cache is not None:
        raw["blocks"].append(
            {
                "name": "l1d",
                "type": "cache",
                "params": {
                    "fronts": ["dram", "scratchpad"],
                    "size": 4096,
                    "assoc": 2,
                    "line": 64,
                    "hit_latency": 1,
                    "miss_latency": 10,
                    "writeback_latency": 7,
                    **cache,
                },
            }
        )
    soc = build(parse_design(raw))
    soc.reset()
    return soc


def _finish(soc):
    start = soc.clock.current_cycle
    return dt._tick_until_idle(soc) - start


def _go(soc, src=SRC, dst=DST, length=N):
    soc.memory.write(src, dt._pattern(length))
    dt._start(soc, src, dst, length)
    return _finish(soc)


# -- DMA reads vs stores in flight ----------------------------------------------


def test_default_dma_ignores_stores_in_flight():
    soc = _soc()
    soc.cores[0].rv_cost.inflight.add_span(SRC, N, 100, 0)
    assert _go(soc) == BASE


def test_dma_read_waits_for_an_in_flight_store_to_its_source():
    soc = _soc(memory={"dma_coherence": True})
    soc.cores[0].rv_cost.inflight.add_span(SRC + 16, 8, 100, 0)
    # The 8-byte chunk holding the store (chunk 2) cannot move before cycle 100.
    # The chunk holding the store (the third, bytes 16..24) cannot move before cycle 100.
    assert _go(soc) > BASE
    assert soc.clock.current_cycle == 100 + N // 8 - 2 - 1
    assert soc.blocks["dma"].counters()["stall_cycles"] > 0
    assert bytes(soc.memory.read(DST, N)) == dt._pattern(
        N
    )  # bytes unchanged by waiting


def test_dma_read_of_other_bytes_does_not_wait():
    soc = _soc(memory={"dma_coherence": True})
    soc.cores[0].rv_cost.inflight.add_span(SRC + 4096, 64, 100, 0)
    assert _go(soc) == BASE
    assert soc.blocks["dma"].counters()["stall_cycles"] == 0


def test_dma_read_resumes_the_cycle_the_store_completes():
    soc = _soc(memory={"dma_coherence": True})
    soc.cores[0].rv_cost.inflight.add_span(SRC, N, 50, 0)
    _go(soc)
    # Cycle 50 is the first the store's bytes are visible: 8 chunks from there.
    assert soc.clock.current_cycle == 50 + N // 8 - 1


# -- DMA writes vs WAW / WAR -----------------------------------------------------


def test_dma_write_waits_for_a_store_still_writing_its_destination_waw():
    soc = _soc(memory={"waw": True})
    soc.cores[0].rv_cost.inflight.add_span(DST, 8, 60, 0)
    _go(soc)
    assert soc.clock.current_cycle == 60 + N // 8 - 1
    assert soc.blocks["dma"].counters()["stall_cycles"] > 0


def test_dma_write_ignores_that_store_without_waw():
    soc = _soc(memory={"war": True})
    soc.cores[0].rv_cost.inflight.add_span(DST, 8, 60, 0)
    assert _go(soc) == BASE


def test_dma_write_waits_for_a_load_still_reading_its_destination_war():
    soc = _soc(memory={"war": True})
    soc.cores[0].rv_cost.inflight.reads.add_span(DST, 8, 60, 0)
    _go(soc)
    assert soc.clock.current_cycle == 60 + N // 8 - 1


# -- core stores vs the DMA's own ranges ------------------------------------------


def test_a_core_store_into_the_dma_destination_waits_waw():
    soc = _soc(memory={"waw": True})
    soc.memory.write(SRC, dt._pattern(N))
    dt._start(soc, SRC, DST, N)
    inflight = soc.cores[0].rv_cost.inflight
    at, why = inflight.store_wait(
        (DST + 8, DST + 16, DST + 8, 0, 1, 8), soc.clock.current_cycle
    )
    assert why == "mem_waw"
    assert at == soc.clock.current_cycle + SETUP + N // 8
    assert inflight.store_wait(
        (DST + N, DST + N + 8, DST + N, 0, 1, 8), soc.clock.current_cycle
    ) == (0, None)


def test_a_core_store_over_the_dma_source_waits_war():
    soc = _soc(memory={"war": True})
    soc.memory.write(SRC, dt._pattern(N))
    dt._start(soc, SRC, DST, N)
    inflight = soc.cores[0].rv_cost.inflight
    at, why = inflight.store_wait((SRC, SRC + 8, SRC, 0, 1, 8), soc.clock.current_cycle)
    assert why == "mem_war"
    assert at > soc.clock.current_cycle
    # Nothing registered without the options.
    soc = _soc()
    soc.memory.write(SRC, dt._pattern(N))
    dt._start(soc, SRC, DST, N)
    assert not soc.cores[0].rv_cost.inflight.ordered


def test_a_load_of_the_dma_destination_never_waits():
    soc = _soc(memory={"hazards": True})
    soc.memory.write(SRC, dt._pattern(N))
    dt._start(soc, SRC, DST, N)
    inflight = soc.cores[0].rv_cost.inflight
    assert inflight.wait_span(DST, N, soc.clock.current_cycle) == 0


# -- cache coherence --------------------------------------------------------------


def _cache(soc):
    return soc.blocks["l1d"]


def test_default_dma_bypasses_a_dirty_line_and_leaves_stale_ones():
    soc = _soc(cache={})
    cache = _cache(soc)
    cache.access(SRC, 8, True)  # dirty source line
    cache.access(DST, 8, False)  # resident destination line
    assert _go(soc) == BASE
    assert cache.contains(DST)
    assert SRC >> 6 in cache._dirty
    assert "dma_writebacks" not in cache.counters()


def test_a_dma_read_writes_back_a_dirty_source_line_and_pays_for_it():
    soc = _soc(memory={"dma_coherence": True}, cache={})
    cache = _cache(soc)
    cache.access(SRC, 8, True)
    cycles = _go(soc)
    assert cycles == BASE + 7  # one writeback, writeback_latency 7
    counters = cache.counters()
    assert counters["dma_writebacks"] == 1
    assert cache.contains(SRC)
    assert SRC >> 6 not in cache._dirty
    assert soc.blocks["dma"].counters()["coherence_cycles"] == 7


def test_a_dma_write_evicts_the_destination_line_so_the_next_core_access_misses():
    soc = _soc(memory={"dma_coherence": True}, cache={})
    cache = _cache(soc)
    cache.access(DST, 8, False)  # clean and resident
    assert _go(soc) == BASE  # clean line: no write-back cost
    assert not cache.contains(DST)
    assert cache.counters()["dma_invalidations"] == 1
    assert (
        cache.access(DST, 8, False) == 1 + 10
    )  # miss again: hit_latency + miss_latency


def test_a_dma_write_over_a_dirty_line_writes_it_back_first():
    soc = _soc(memory={"dma_coherence": True}, cache={})
    cache = _cache(soc)
    cache.access(DST, 8, True)
    assert _go(soc) == BASE + 7
    counters = cache.counters()
    assert (counters["dma_writebacks"], counters["dma_invalidations"]) == (1, 1)
    assert not cache.contains(DST)


def test_a_line_the_dma_does_not_touch_is_left_alone():
    soc = _soc(memory={"dma_coherence": True}, cache={})
    cache = _cache(soc)
    cache.access(SRC + 1024, 8, True)
    assert _go(soc) == BASE
    assert cache.contains(SRC + 1024)
    assert (SRC + 1024) >> 6 in cache._dirty


@pytest.mark.parametrize("private", [False, True])
def test_every_private_l1_is_snooped(private):
    raw_cache = {"private": True} if private else {}
    soc = _soc(memory={"dma_coherence": True}, cache=raw_cache)
    cache = _cache(soc)
    cache.for_hart(0).access(SRC, 8, True)
    assert _go(soc) == BASE + 7
    assert cache.counters()["dma_writebacks"] == 1


def test_results_are_identical_with_and_without_coherence():
    out = []
    for memory in ({}, {"hazards": True}):
        soc = _soc(memory=memory, cache={})
        _cache(soc).access(SRC, 8, True)
        _go(soc)
        out.append(bytes(soc.memory.read(DST, N)))
    assert out[0] == out[1] == dt._pattern(N)
