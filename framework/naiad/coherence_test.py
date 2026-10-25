"""Private per-hart caches with MSI snooping (``cache.private: true``), and wfi's timer wake-up.

Unit checks drive :class:`~framework.naiad.ip.cache.CoherentL1` views directly;
the end-to-end ones build ``kernels/naiad/coherence`` (false sharing, padded
counters, a shared AMO counter) and ``kernels/naiad/wfi_timer`` (every hart
sleeps in ``wfi`` until its own ``mtimecmp``).
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.difftest import difftest, find_spike
from framework.naiad.dwarfs import NAIAD_CC, multicore_design
from framework.naiad.ip.cache import Cache, CoherentL1

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "kernels" / "naiad"
DWARFS_4C = multicore_design(
    4
)  # naiad-dwarfs with 4 harts (generated, not a hand copy)
RV64GC = DESIGNS_DIR / "naiad-rv64gc.yaml"
L1 = 0x10000000
LINE = 64
HIT, FILL = (
    1,
    6,
)  # hit_latency, and the fill (= writeback = snoop) cost: l1's region latency
INVALIDATE = 8  # the invalidate_latency default

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
needs_spike = pytest.mark.skipif(
    find_spike() is None, reason="needs spike (tools/naiad/build_spike.sh)"
)


def _raw(harts=2, cache=True, path=DWARFS_4C, **params):
    """naiad-dwarfs at ``harts`` harts; ``cache``: None (none), False (shared) or True (private)."""
    raw = yaml.safe_load(Path(path).read_text())
    raw["cores"]["count"] = harts
    raw["name"] = (
        f"coh-{harts}-{ {None: 'none', False: 'shared', True: 'private'}[cache] }"
    )
    if cache is not None:
        p = {
            "fronts": ["l1"],
            "size": "32KiB",
            "assoc": 4,
            "line": LINE,
            "hit_latency": HIT,
            "miss_latency": "backing",
            **params,
        }
        if cache:
            p["private"] = True
        raw["blocks"].append({"name": "l1d", "type": "cache", "params": p})
    return raw


def _soc(harts=2, cache=True, **params):
    soc = build(parse_design(_raw(harts, cache, **params)))
    soc.reset()
    return soc


# -- the MSI model ---------------------------------------------------------------


def test_private_cache_gives_each_hart_its_own_view():
    soc = _soc(4)
    cache = soc.blocks["l1d"]
    assert len(cache.l1s) == 4
    assert len({id(c) for c in cache.l1s}) == 4
    views = [c.rv_cost.memory.by_name["l1"].provider for c in soc.cores]
    assert all(isinstance(v, CoherentL1) for v in views)
    assert [v.l1 for v in views] == list(cache.l1s)
    assert all(c.rv_cost.memory.by_name["l1"].coherent for c in soc.cores)
    shared = _soc(4, cache=False)
    regions = [c.rv_cost.memory.by_name["l1"] for c in shared.cores]
    assert all(r.provider is shared.blocks["l1d"] and not r.coherent for r in regions)


def test_msi_read_snoops_an_m_line_and_a_write_invalidates():
    a, b = _soc(2).blocks["l1d"].views
    line = L1 + 0x1000
    assert a.access(line, 8, True) == HIT + FILL  # I -> M, nobody else has it
    assert a.coherence_cycles == 0
    assert b.access(line, 8, False) == HIT + FILL + FILL  # fill + A's writeback
    assert b.coherence_cycles == FILL
    assert line >> 6 not in a.l1._dirty
    assert a.l1.contains(line)
    assert a.l1.stats["snoop_writebacks"] == 1
    assert b.access(line, 8, True) == HIT + INVALIDATE  # S -> M, invalidates A's S copy
    assert not a.l1.contains(line)
    assert a.l1.stats["invalidations"] == 1
    assert b.l1.stats["upgrades"] == 1
    assert b.access(line, 8, True) == HIT  # M: a plain hit
    # A writes the line B holds in M: a miss, plus B's writeback.
    assert a.access(line + 8, 8, True) == HIT + FILL + FILL
    assert not b.l1.contains(line)
    assert b.l1.stats["snoop_writebacks"] == 1


def test_lone_s_line_upgrades_silently_and_reads_share():
    a, b = _soc(2).blocks["l1d"].views
    line = L1 + 0x2000
    a.access(line, 8, False)
    assert a.access(line, 8, True) == HIT  # no other copy: no bus cost
    b2 = L1 + 0x3000
    a.access(b2, 8, False)
    assert b.access(b2, 8, False) == HIT + FILL  # S in both: no snoop
    assert a.l1.contains(b2)
    assert b.l1.contains(b2)


def test_coherence_costs_are_configurable_and_counted_per_hart():
    soc = _soc(2, invalidate_latency=3, snoop_latency=11)
    a, b = soc.blocks["l1d"].views
    a.access(L1, 8, True)
    assert b.access(L1, 8, False) == HIT + FILL + 11
    assert a.access(L1, 8, True) == HIT + 3
    c = soc.counters()
    assert c["l1d.hart0.coherence_cycles"] == 3
    assert c["l1d.hart1.coherence_cycles"] == 11
    assert c["l1d.coherence_cycles"] == 14
    assert c["l1d.hart1.invalidations"] == 1
    assert c["l1d.hart0.snoop_writebacks"] == 1
    assert c["l1d.invalidations"] == 1
    assert c["l1d.misses"] == 2
    soc.reset()
    assert soc.counters()["l1d.coherence_cycles"] == 0
    assert not soc.blocks["l1d"].l1s[0].contains(L1)


def test_one_hart_private_cache_times_like_a_shared_one():
    private = _soc(1).blocks["l1d"].views[0]
    shared = _soc(1, cache=False).blocks["l1d"]
    trace = [(L1 + (i * 136) % 0x9000, 8 + i % 3 * 60, i % 4 == 0) for i in range(3000)]
    assert [private.access(*t) for t in trace] == [shared.access(*t) for t in trace]


@pytest.mark.parametrize(
    "params, match",
    [
        ({"private": "yes"}, "private must be true or false"),
        (
            {"private": True, "write_policy": "write_through"},
            "need write_policy write_back",
        ),
        ({"invalidate_latency": -1}, "invalidate_latency must be >= 0"),
        ({"snoop_latency": -2}, "snoop_latency must be >= 0"),
    ],
)
def test_bad_private_params_are_rejected(params, match):
    with pytest.raises(DesignError, match=match):
        parse_design(_raw(2, cache=False, **params))


def test_shared_cache_counter_keys_are_unchanged():
    keys = set(_soc(2, cache=False).blocks["l1d"].counters())
    assert keys == {
        "accesses",
        "hits",
        "misses",
        "evictions",
        "writebacks",
        "write_throughs",
    }
    assert isinstance(_soc(2, cache=False).blocks["l1d"].for_hart(1), Cache)


# -- wfi timer wake-up (unit) ----------------------------------------------------------


def test_wfi_wakes_when_mtime_reaches_mtimecmp():
    soc = build(parse_design(_raw(2, cache=None, path=RV64GC)))
    soc.reset()
    clint = soc.blocks["clint"]
    hart1 = soc.cores[1]
    hart1.csrs.write(0x304, 1 << 7)  # mie.MTIE
    clint.write_reg(0x4000 + 8, 100)  # mtimecmp[1]
    soc.clock.current_cycle = 40
    hart1.register_file.wfi()
    assert not hart1.active
    assert clint.timer_sleepers == {1}
    clint.timer_tick(40)
    assert not hart1.active
    soc.clock.current_cycle = 100  # mtime == mtimecmp: due
    clint.timer_tick(100)
    assert hart1.active
    assert not clint.sleeping
    assert not clint.timer_sleepers
    assert hart1.csrs.cycles >= 59  # the slept cycles are credited to mcycle
    hart1.register_file.wfi()  # still due: falls straight through
    assert hart1.active
    # Without MTIE the timer does not wake it (msip with MSIE still would).
    hart1.csrs.write(0x304, 1 << 3)
    clint.write_reg(0x4000 + 8, 1000)
    hart1.register_file.wfi()
    soc.clock.current_cycle = 2000
    clint.timer_tick(2000)
    assert not hart1.active
    clint.write_reg(4, 1)
    assert hart1.active


# -- end to end -----------------------------------------------------------------


def _build(tmp_path, raw, prog):
    design = tmp_path / f"{raw['name']}.yaml"
    design.write_text(yaml.safe_dump(raw))
    build_dir = tmp_path / f"build-{raw['name']}-{prog}"
    subprocess.run(
        [
            "make",
            "-s",
            "-C",
            str(KERNELS),
            f"DESIGN={design}",
            f"PROG={prog}",
            f"BUILD={build_dir}",
            f"NAIAD_CC={NAIAD_CC}",
            f"PYTHON={sys.executable}",
        ],
        check=True,
        capture_output=True,
    )
    return design, build_dir / f"{prog}.elf"


def _run(design, elf):
    soc = build(load_design(design))
    soc.htif.console_stream = None
    soc.load(elf)
    r = soc.run(5_000_000)
    assert r.exit_code == 0, r.console.decode()
    return r


def _phases(console):
    return {
        m[1]: (int(m[2]), int(m[3]))
        for m in re.finditer(r"(\w+) cycles=(\d+) invalidations=(\d+)", console)
    }


@needs_clang
@pytest.mark.parametrize("harts", [2, 4])
def test_false_sharing_ping_pongs_only_with_private_caches(tmp_path, harts):
    raw = _raw(harts, cache=True)
    raw["cores"]["isa"].append("zihpm")
    raw["counters"] = {"hpm": {3: "l1d.invalidations"}}
    private = _run(*_build(tmp_path, raw, "coherence"))
    shared = _run(*_build(tmp_path, _raw(harts, cache=False), "coherence"))
    p, s = _phases(private.console.decode()), _phases(shared.console.decode())
    assert b"PASS" in private.console
    assert b"PASS" in shared.console
    iters = 200
    # Private MSI: every false-sharing write misses the line another hart took.
    assert p["false"][1] >= iters * (harts - 1)
    assert p["padded"][1] < iters // 4  # only the barrier's lines
    assert p["true"][1] >= iters * (harts - 1)
    assert p["false"][0] > 2 * p["padded"][0]
    # One shared cache: false and padded sharing cost the same.
    assert abs(s["false"][0] - s["padded"][0]) < 0.05 * s["padded"][0]
    assert private.counters["l1d.coherence_cycles"] > 0
    assert private.counters["l1d.snoop_writebacks"] > 0


@needs_clang
@pytest.mark.parametrize("harts", [2, 4])
def test_wfi_timer_wakes_each_hart(tmp_path, harts):
    r = _run(*_build(tmp_path, _raw(harts, cache=None), "wfi_timer"))
    lines = r.console.decode().splitlines()
    assert lines[-1] == "PASS"
    for h in range(harts):
        m = re.fullmatch(
            rf"hart {h}: target (\d+) woke (\d+) after (\d+) wfi", lines[h]
        )
        target, woke, wfis = map(int, m.groups())
        assert wfis == 1  # slept once: a nop wfi would spin thousands of times
        assert target <= woke < target + 50
    assert r.cycles > 5000 * harts
    # The harts slept: far fewer instructions than a spin loop over the waits.
    assert r.counters["instret"] < r.cycles


@needs_clang
@needs_spike
def test_wfi_timer_agrees_with_spike(tmp_path):
    design, elf = _build(tmp_path, _raw(4, cache=None, path=RV64GC), "wfi_timer")
    result = difftest(design, elf, ignore=["hart"])
    assert result.ok, result.report()
    assert b"PASS" in result.naiad.console
