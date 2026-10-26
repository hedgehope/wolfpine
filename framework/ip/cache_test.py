import copy

import pytest

from framework.ip.cache import Cache
from framework.naiad.builder import build
from framework.naiad.design import BlockSpec, DesignError, parse_design

HIT, MISS, WB = 1, 10, 7

BASE = {
    "name": "t",
    "cores": {"xlen": 64, "isa": ["i", "m"], "reset_pc": 0x1000},
    "blocks": [
        {"name": "bootrom", "type": "bootrom", "base": 0x1000, "window": 0x1000},
        {"name": "sp", "type": "scratchpad", "base": 0x20000000, "window": 0x1000},
        {"name": "dram", "type": "dram", "base": 0x80000000, "window": "1MiB"},
        {"name": "htif", "type": "htif"},
    ],
}


def _cache(**params):
    params = {"fronts": ["dram"], "hit_latency": HIT, "miss_latency": MISS, **params}
    return Cache(BlockSpec(name="c", type="cache", params=params))


def _design_with(*caches):
    raw = copy.deepcopy(BASE)
    raw["blocks"] += [
        {"name": f"c{i}", "type": "cache", "params": p} for i, p in enumerate(caches)
    ]
    return raw


# -- direct-mapped -----------------------------------------------------------


def test_direct_mapped_hit_miss_sequence():
    c = _cache(size=256, assoc=1, line=64)  # 4 sets
    assert c.sets == 4
    # 0x000 and 0x100 map to set 0 and evict each other; 0x040 is set 1.
    seq = [0x000, 0x004, 0x040, 0x100, 0x000, 0x044, 0x13C]
    costs = [c.access(a, 4, False) for a in seq]
    assert costs == [
        HIT + MISS,
        HIT,
        HIT + MISS,
        HIT + MISS,
        HIT + MISS,
        HIT,
        HIT + MISS,
    ]
    assert c.counters() == {
        "accesses": 7,
        "hits": 2,
        "misses": 5,
        "evictions": 3,
        "writebacks": 0,
        "write_throughs": 0,
    }


# -- 2-way and LRU -------------------------------------------------------------


def test_two_way_keeps_two_conflicting_lines():
    c = _cache(size=256, assoc=2, line=64)  # 2 sets
    a, b = 0x000, 0x080  # both set 0
    assert c.access(a, 8, False) == HIT + MISS
    assert c.access(b, 8, False) == HIT + MISS
    assert c.access(a, 8, False) == HIT
    assert c.access(b, 8, False) == HIT
    assert c.counters()["evictions"] == 0


def test_lru_evicts_the_least_recently_used_way():
    c = _cache(size=256, assoc=2, line=64)
    a, b, d = 0x000, 0x080, 0x100  # all set 0
    c.access(a, 4, False)
    c.access(b, 4, False)
    c.access(a, 4, False)  # a is now most recent, b least
    c.access(d, 4, False)  # evicts b
    assert [c.contains(x) for x in (a, b, d)] == [True, False, True]
    c.access(b, 4, False)  # evicts a (d was used more recently)
    assert [c.contains(x) for x in (a, b, d)] == [False, True, True]
    assert c.counters()["evictions"] == 2


def test_four_way_lru_order_over_a_cyclic_sweep():
    c = _cache(size=4 * 64, assoc=4, line=64)  # one set
    lines = [i * 64 for i in range(5)]
    for a in lines[:4]:
        c.access(a, 1, False)
    for a in lines[:4]:
        assert c.access(a, 1, False) == HIT
    c.access(lines[4], 1, False)  # evicts lines[0], the oldest
    assert [c.contains(a) for a in lines] == [False, True, True, True, True]


# -- spanning accesses ---------------------------------------------------------


def test_an_access_spanning_lines_costs_each_line():
    c = _cache(size=1024, assoc=2, line=64)
    assert c.access(0x3C, 8, False) == 2 * (
        HIT + MISS
    )  # 0x3C..0x43 straddles two lines
    assert c.access(0x00, 128, False) == 2 * HIT
    assert c.access(0x7F, 2, False) == HIT + (HIT + MISS)  # 0x7F hits, 0x80 misses
    assert c.counters()["accesses"] == 6


# -- write policies --------------------------------------------------------------


def test_write_back_allocates_and_writes_back_dirty_victims():
    c = _cache(size=128, assoc=1, line=64, writeback_latency=WB)  # 2 sets
    assert c.access(0x000, 4, True) == HIT + MISS  # write-allocate, now dirty
    assert c.contains(0x000)
    assert c.access(0x004, 4, False) == HIT
    assert c.access(0x080, 4, False) == HIT + MISS + WB  # evicts the dirty line
    assert c.access(0x000, 4, False) == HIT + MISS  # evicts a clean line: no writeback
    stats = c.counters()
    assert (stats["evictions"], stats["writebacks"], stats["write_throughs"]) == (
        2,
        1,
        0,
    )


def test_writeback_latency_defaults_to_miss_latency():
    c = _cache(size=64, assoc=1, line=64)
    c.access(0x000, 4, True)
    assert c.access(0x040, 4, False) == HIT + MISS + MISS


def test_write_through_does_not_allocate_or_dirty():
    c = _cache(size=128, assoc=1, line=64, write_policy="write_through")
    assert c.access(0x000, 4, True) == HIT  # posted, no allocate
    assert not c.contains(0x000)
    assert c.access(0x000, 4, False) == HIT + MISS
    assert c.access(0x000, 4, True) == HIT  # write hit: still clean
    assert c.access(0x080, 4, False) == HIT + MISS  # evicts it with no writeback
    stats = c.counters()
    assert stats["write_throughs"] == 2
    assert stats["writebacks"] == 0
    assert (stats["hits"], stats["misses"]) == (1, 3)


def test_flush_writes_back_dirty_lines_once():
    c = _cache(size=256, assoc=2, line=64, writeback_latency=WB)
    c.access(0x000, 4, True)
    c.access(0x040, 4, True)
    c.access(0x080, 4, False)
    assert c.flush() == 2 * WB
    assert c.flush() == 0
    assert c.contains(0x000)
    assert c.counters()["writebacks"] == 2


def test_backing_miss_latency_comes_from_the_integrator():
    c = _cache(size=128, assoc=1, line=64, miss_latency="backing")
    with pytest.raises(RuntimeError, match="backing_latency"):
        c.access(0, 4, False)
    c.backing_latency = lambda addr: 100 + addr // 64
    assert c.access(0x40, 4, False) == HIT + 101
    c.backing_latency = 5
    assert c.access(0x80, 4, False) == HIT + 5


def test_reset_empties_the_cache_and_counters():
    c = _cache(size=128, assoc=1, line=64)
    c.access(0, 4, True)
    c.reset()
    assert not c.contains(0)
    assert set(c.counters().values()) == {0}


# -- the design file ---------------------------------------------------------------


def test_cache_block_builds_unmapped_and_leaves_data_alone():
    soc = build(
        parse_design(
            _design_with({"fronts": ["dram", "sp"], "size": "4KiB", "assoc": 2})
        )
    )
    cache = soc.blocks["c0"]
    assert not cache.mapped
    assert cache.regions == ((0x80000000, 0x800FFFFF), (0x20000000, 0x20000FFF))
    assert cache.covers(0x80000010)
    assert not cache.covers(0x1000)
    soc.memory.write(0x80000000, b"\x11" * 64)
    cache.access(0x80000000, 64, True)
    cache.access(0x80001000, 64, False)
    assert soc.memory.read(0x80000000, 64) == b"\x11" * 64
    assert soc.counters()["c0.misses"] == 2
    assert soc.counters()["c0.write_throughs"] == 0
    with pytest.raises(IndexError):
        cache.read(0, 4)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"fronts": ["nope"]}, "does not have"),
        ({"fronts": ["htif"]}, "not a memory-mapped block"),
        ({"fronts": []}, "non-empty list"),
        ({}, "non-empty list"),
        ({"fronts": ["dram"], "line": 48}, "power of two"),
        ({"fronts": ["dram"], "size": 3 * 64 * 4}, "power-of-two set count"),
        ({"fronts": ["dram"], "assoc": 0}, "assoc must be >= 1"),
        ({"fronts": ["dram"], "hit_latency": -1}, "hit_latency must be >= 0"),
        ({"fronts": ["dram"], "write_policy": "write_around"}, "write_policy"),
        ({"fronts": ["dram"], "miss_latency": "slow"}, "expected an integer"),
        ({"fronts": ["dram"], "ways": 2}, "unknown param"),
    ],
)
def test_bad_cache_params_are_rejected(params, match):
    with pytest.raises(DesignError, match=match):
        parse_design(_design_with(params))


def test_a_block_has_at_most_one_cache():
    with pytest.raises(DesignError, match="both front"):
        parse_design(_design_with({"fronts": ["dram"]}, {"fronts": ["sp", "dram"]}))


def test_cache_takes_no_window():
    raw = copy.deepcopy(BASE)
    raw["blocks"].append(
        {
            "name": "c",
            "type": "cache",
            "base": 0x3000,
            "window": 0x1000,
            "params": {"fronts": ["dram"]},
        }
    )
    with pytest.raises(DesignError, match="takes no base/window"):
        parse_design(raw)
