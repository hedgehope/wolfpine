"""The multicore-scaling harness pieces: the hart and cache override axes, ``scaling.csv`` and its figure."""

import json
from pathlib import Path

import pytest

from experiments.naiad import metrics, scaling
from experiments.naiad.sweep import (
    AddressMoved,
    DesignFactory,
    SweepError,
    apply_override,
    load_sweep,
)
from framework.naiad.design import load_design

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2] / "framework" / "naiad" / "designs" / "naiad-dwarfs.yaml"
SWEEP = HERE.parent / "sweeps" / "multicore_scaling.yaml"
CACHE = {
    "type": "cache",
    "params": {
        "fronts": ["l1"],
        "size": "32KiB",
        "hit_latency": 1,
        "miss_latency": "backing",
    },
}


def test_override_adds_a_named_list_element_and_merges_an_existing_one():
    raw = {"blocks": [{"name": "l1", "type": "scratchpad", "params": {"size": 4}}]}
    apply_override(raw, "blocks.l1d", CACHE)
    assert raw["blocks"][1] == {"name": "l1d", **CACHE}
    apply_override(raw, "blocks.l1", {"params": {"latency": 3}})
    assert raw["blocks"][0]["params"] == {"size": 4, "latency": 3}
    with pytest.raises(SweepError):
        apply_override(raw, "blocks.l1", 5)


def test_harts_and_an_added_cache_derive_without_moving_addresses(tmp_path):
    factory = DesignFactory(tmp_path)
    _, path, _ = factory.derive(
        BASE,
        {
            "cores.count": 4,
            "blocks.l1d": {**CACHE, "params": {**CACHE["params"], "private": True}},
        },
    )
    design = load_design(path)
    assert design.cores.count == 4
    assert [b.name for b in design.blocks_of_type("cache")] == ["l1d"]
    # A mapped block may not be added.
    with pytest.raises(AddressMoved):
        factory.derive(
            BASE,
            {
                "blocks.extra": {
                    "type": "scratchpad",
                    "base": 0x30000000,
                    "window": 0x1000,
                    "params": {"size": 0x1000},
                }
            },
        )


def test_multicore_sweep_expands(tmp_path, monkeypatch):
    from experiments.naiad import sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "OUT", tmp_path)
    sw = load_sweep(SWEEP, quick=True)
    points = sweep_mod.expand(sw, log=lambda *a: None)
    by_study = {}
    for p in points:
        by_study.setdefault(p.study, []).append(p)
    assert set(by_study) == {"scaling", "scaling_gemm", "bandwidth", "bandwidth_gemm"}
    # The plain variants run on one hart only; the _mc ones on 1, 2, 4, 8.
    for p in by_study["scaling"]:
        count = p.overrides["cores.count"]
        assert (count is None) != p.variant.endswith("_mc")
    assert {
        p.overrides["cores.count"]
        for p in by_study["scaling"]
        if p.variant.endswith("_mc")
    } == {1, 2, 4, 8}
    caches = {
        json.dumps(p.overrides["blocks.l1d"], sort_keys=True)
        for p in by_study["scaling"]
    }
    assert len(caches) == 3  # none (null), shared, private


def _row(
    study, dwarf, variant, harts, cycles, cache=None, bw=None, pid=None, fi="fp32"
):
    return {
        "study": study,
        "dwarf": dwarf,
        "variant": variant,
        "size": "64",
        "in_fmt": fi,
        "acc_fmt": "fp32",
        "status": "ok",
        "passed": True,
        "region_cycles": cycles,
        "normwise": 1e-8,
        "point_id": pid or f"{dwarf}-{variant}-{harts}-{cache}-{bw}",
        scaling.HARTS_KEY: "" if harts is None else harts,
        scaling.CACHE_KEY: "" if cache is None else json.dumps(cache),
        scaling.BW_KEY: "" if bw is None else bw,
    }


def test_scaling_records_csv_and_figure(tmp_path):
    private = {**CACHE, "params": {**CACHE["params"], "private": True}}
    rows = [
        _row("s", "gemm", "ame", None, 1000),
        _row("s", "gemm", "ame_mc", 1, 1010),
        _row("s", "gemm", "ame_mc", 2, 505),
        _row("s", "gemm", "ame_mc", 4, 300),
        _row("s", "gemm", "ame", None, 2000, cache=private),
        _row("s", "gemm", "ame_mc", 1, 2000, cache=private),
        _row("s", "gemm", "ame_mc", 2, 1600, cache=private),
        _row("s", "jacobi", "rvv", None, 900, bw=64),
        _row("s", "jacobi", "rvv_mc", 1, 900, bw=64),
        _row("s", "jacobi", "rvv_mc", 2, 600, bw=64),
        _row("other", "gemm", "ame_mc", 2, 1),  # a study that does not feed the figure
    ]
    points = tmp_path / "points"
    points.mkdir()
    (points / "pid-c.json").write_text(
        json.dumps(
            {
                "counters": {
                    "instret": 200,
                    "timing.stall_mem": 50,
                    "l1d.invalidations": 7,
                    "l1d.coherence_cycles": 30,
                }
            }
        )
    )
    rows[6]["point_id"] = "pid-c"
    recs = scaling.scaling_records(rows, {"s"}, points)
    assert len(recs) == 7
    by = {(r["dwarf"], r["cache"], r["l1_bandwidth"], r["harts"]): r for r in recs}
    r = by[("gemm", "none", "", 2)]
    assert r["speedup"] == pytest.approx(1010 / 505)
    assert r["efficiency"] == pytest.approx(1010 / 505 / 2)
    assert r["speedup_vs_single"] == pytest.approx(1000 / 505)
    assert r["single_variant"] == "ame"
    r = by[("gemm", "private", "", 2)]
    assert r["speedup"] == pytest.approx(2000 / 1600)
    assert r["instret_per_hart"] == 100
    assert r["stall_mem_per_hart"] == 25
    assert r["l1d.invalidations"] == 7
    assert r["l1d.coherence_cycles"] == 30
    r = by[("jacobi", "none", 64, 2)]
    assert r["speedup"] == pytest.approx(1.5)
    assert r["single_variant"] == "rvv"

    path = metrics.write_csv(tmp_path / "scaling.csv", recs, scaling.COLUMNS)
    back = metrics.read_csv(path)
    assert len(back) == 7
    assert set(scaling.COLUMNS) <= set(back[0])
    pytest.importorskip("matplotlib")
    made = scaling.plot_scaling(recs, tmp_path / "figures", "t")
    assert {p.suffix for p in made} == {".png", ".svg"}
    assert all(p.exists() for p in made)
