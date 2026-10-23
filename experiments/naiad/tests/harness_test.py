"""Tests for the Naiad experiment harness (``pytest experiments/naiad/tests``).

Everything but the last test runs without a toolchain: the end-to-end test
builds the toy dwarf (``toy_dwarf/toymm``) with clang and is skipped without it.
"""

import json
import math
import shutil
from pathlib import Path

import pytest
import yaml

from experiments.naiad import metrics, run
from experiments.naiad.sweep import (
    AddressMoved,
    DesignFactory,
    Study,
    SweepError,
    apply_override,
    expand,
    load_sweep,
)
from framework.naiad.design import load_design
from framework.naiad.dwarfs import NAIAD_CC

HERE = Path(__file__).resolve().parent
TOY_ROOT = "experiments/naiad/tests/toy_dwarf"
TOY_SWEEP = HERE / "toy_sweep.yaml"
BASE = HERE.parents[2] / "framework" / "naiad" / "designs" / "naiad-dwarfs.yaml"
HAVE_CC = shutil.which(NAIAD_CC) is not None and shutil.which("make") is not None


def write_sweep(tmp_path, body):
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


# -- config expansion -----------------------------------------------------------------------


def test_expansion_products_only_and_dedupe(tmp_path):
    path = write_sweep(
        tmp_path,
        {
            "name": "x",
            "dwarfs": ["toymm", "missing", "broken"],
            "dwarfs_root": TOY_ROOT,
            "variants": ["ame", "scalar", "nosuch"],
            "sizes": "all",
            "formats": [["fp32", "fp32"], ["bf16", "fp32"]],
            "overrides": {
                "timing.matrix.macs_per_cycle": [64, 128, 256],
                "timing.scalar.fp_fma": [2, 4],
            },
            "only": {"timing.matrix.macs_per_cycle": ["ame"]},
        },
    )
    logs = []
    sweep = load_sweep(path)
    points = expand(sweep, DesignFactory(tmp_path / "out"), log=logs.append)
    ame = [p for p in points if p.variant == "ame"]
    scalar = [p for p in points if p.variant == "scalar"]
    # ame: 2 sizes x 2 formats x 3 macs x 2 fma; scalar: 2 sizes x fp32 only (bf16 is not an F/D pair) x 2 fma.
    assert len(ame) == 2 * 2 * 3 * 2
    assert len(scalar) == 2 * 1 * 2
    assert all(p.overrides["timing.matrix.macs_per_cycle"] is None for p in scalar)
    assert len({p.point_id for p in points}) == len(points)
    text = "\n".join(logs)
    assert "skip dwarf 'missing'" in text
    assert "skip dwarf 'broken'" in text
    assert "deliberately broken" in text
    assert "no such variant" in text
    # Timing-only overrides share one design per value set and leave addresses alone.
    designs = {p.design_hash for p in points}
    assert len(designs) == 3 * 2 + 2  # ame combos + scalar (macs at base) combos
    for p in points:
        assert Path(p.design_path).exists()


def test_size_selectors_quick_and_study_defaults(tmp_path):
    path = write_sweep(
        tmp_path,
        {
            "name": "y",
            "dwarfs": "all",
            "dwarfs_root": TOY_ROOT,
            "variants": ["ame"],
            "sizes": "largest",
            "studies": [
                {
                    "name": "a",
                    "quick": {"overrides": {"timing.matrix.macs_per_cycle": [8]}},
                },
                {"name": "b", "sizes": [16], "formats": "all"},
            ],
            "overrides": {"timing.matrix.macs_per_cycle": [8, 16]},
        },
    )
    full = expand(load_sweep(path), DesignFactory(tmp_path / "o1"), log=lambda m: None)
    a = [p for p in full if p.study == "a"]
    assert {p.size for p in a} == {32}
    assert len(a) == 2
    b = [p for p in full if p.study == "b"]
    assert {(p.in_fmt, p.acc_fmt) for p in b} == {
        ("fp32", "fp32"),
        ("fp64", "fp64"),
        ("bf16", "fp32"),
        ("fp16", "fp32"),
    }
    quick = expand(
        load_sweep(path, quick=True), DesignFactory(tmp_path / "o2"), log=lambda m: None
    )
    qa = [p for p in quick if p.study == "a"]
    assert {p.size for p in qa} == {16}
    assert len(qa) == 1


def test_unknown_study_key_is_an_error(tmp_path):
    with pytest.raises(SweepError, match="unknown keys"):
        load_sweep(write_sweep(tmp_path, {"name": "z", "overides": {}}))


def test_override_paths_and_deep_merge():
    raw = yaml.safe_load(BASE.read_text())
    apply_override(raw, "timing.matrix.format_passes", {"fp32": 1})
    assert raw["timing"]["matrix"]["format_passes"] == {
        "fp64": 16,
        "fp32": 1,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
    }
    apply_override(raw, "blocks.dram.params.foo", 3)
    assert next(b for b in raw["blocks"] if b["name"] == "dram")["params"] == {"foo": 3}
    apply_override(raw, "timing.memory.l1.latency", 1)
    assert raw["timing"]["memory"]["l1"] == {"latency": 1, "bandwidth": None}
    with pytest.raises(SweepError, match="no list element named"):
        apply_override(raw, "blocks.nosuch.base", 0)


# -- addresses never move --------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, value",
    [
        ("blocks.l1.base", 0x30000000),
        ("blocks.l1.params.size", 0x100000),
        ("blocks.dram.window", "512MiB"),
    ],
)
def test_overrides_that_move_addresses_are_refused(tmp_path, key, value):
    factory = DesignFactory(tmp_path)
    with pytest.raises(AddressMoved, match="moves addresses"):
        factory.derive(BASE, {key: value})


def test_address_move_aborts_expansion(tmp_path):
    path = write_sweep(
        tmp_path,
        {
            "dwarfs_root": TOY_ROOT,
            "dwarfs": ["toymm"],
            "variants": ["ame"],
            "overrides": {"blocks.l1.base": 0x30000000},
        },
    )
    with pytest.raises(AddressMoved):
        expand(load_sweep(path), DesignFactory(tmp_path / "o"), log=lambda m: None)


def test_derived_design_keeps_layout_and_takes_overrides(tmp_path):
    factory = DesignFactory(tmp_path)
    h, path, _ = factory.derive(
        BASE, {"timing.matrix.macs_per_cycle": 128, "matrix.k": 8}
    )
    d, base = load_design(path), load_design(BASE)
    assert d.timing.matrix.macs_per_cycle == 128
    assert d.matrix.k == 8
    assert [(b.name, b.base, b.size, b.window) for b in d.blocks] == [
        (b.name, b.base, b.size, b.window) for b in base.blocks
    ]
    assert (
        factory.derive(BASE, {"matrix.k": 8, "timing.matrix.macs_per_cycle": 128})[0]
        == h
    )
    assert path.read_text().startswith("# Derived by experiments.naiad")


def test_normalise_points(tmp_path):
    factory = DesignFactory(tmp_path)
    params = {
        "peak_flops_per_cycle": 512,
        "operand_bytes": 8 * 1024 * 3,
        "mem_bytes_per_cycle": 32,
    }
    _, path, derived = factory.derive(BASE, {}, params, "fp32", "fp32")
    d = load_design(path)
    # 24 KiB / (8 tiles * 4 B + 4 accs * 4 B) = 512 elements per d^2 -> d = 22 -> 16 (power of two).
    assert derived["d"] == 16
    assert d.matrix.m == d.matrix.n == d.matrix.k == 16
    assert d.timing.matrix.macs_per_cycle == 256
    assert d.timing.matrix.tile_load_bandwidth == 32
    assert (
        d.matrix.in_formats == load_design(BASE).matrix.in_formats
    )  # the base's format list is kept


# -- metrics on synthetic rows ---------------------------------------------------------------


def test_interpolate_crossover():
    value, status = metrics.interpolate_crossover([1, 4, 16], [0.5, 2, 8])
    assert status == "bracketed"
    assert value == pytest.approx(4 ** (1 / 3))
    assert metrics.interpolate_crossover([4, 1], [2, 0.5])[0] == pytest.approx(
        4 ** (1 / 3)
    )  # order-free
    assert metrics.interpolate_crossover([1, 2], [1.5, 3]) == (None, "below_range")
    assert metrics.interpolate_crossover([1, 2], [0.1, 0.2]) == (None, "above_range")
    assert metrics.interpolate_crossover([1, 2, 4], [0.5, 1.0, 2]) == (2, "exact")
    assert metrics.interpolate_crossover([], []) == (None, "empty")


def test_peaks_and_ratio(tmp_path):
    base = load_design(BASE)
    # 16^3 MACs at 2048 MACs/cycle and 4 fp32 passes: ii = 8, peak 512 MACs/cycle.
    assert metrics.matrix_peak(base, "fp32") == 512
    assert metrics.matrix_peak(base, "int8") == 2048
    assert metrics.vector_peak(base, "fp32") == 1
    assert metrics.vector_peak(base, "fp32", 8) == 8
    with pytest.raises(ValueError, match="Phase C"):
        metrics.vector_peak(base, "fp32", "rvv")
    _, path, _ = DesignFactory(tmp_path).derive(BASE, {"timing.matrix.mma_ii": 64})
    assert metrics.matrix_peak(load_design(path), "fp32") == 64


def _row(variant, region, macs=None, fmts=("fp32", "fp32"), size="16", **ovr):
    row = {
        "study": "s",
        "dwarf": "d",
        "variant": variant,
        "size": size,
        "in_fmt": fmts[0],
        "acc_fmt": fmts[1],
        "seed": 0,
        "status": "ok",
        "region_cycles": region,
        "point_id": f"{variant}-{region}",
    }
    row.update({f"ovr:{k}": v for k, v in ovr.items()})
    if macs is not None:
        row["mv_ratio"] = macs / 4
    return row


def test_speedup_matching_and_crossover_on_synthetic_rows():
    study = Study(
        name="s",
        overrides={
            "timing.matrix.macs_per_cycle": [4, 16, 64],
            "timing.scalar.fp_fma": [2],
        },
        only={"timing.matrix.macs_per_cycle": ["ame"]},
    )
    rows = [
        _row(
            "scalar",
            1000,
            **{"timing.matrix.macs_per_cycle": "", "timing.scalar.fp_fma": 2},
        ),
        _row(
            "scalar",
            9999,
            fmts=("fp64", "fp64"),
            **{"timing.matrix.macs_per_cycle": "", "timing.scalar.fp_fma": 2},
        ),
        _row(
            "scalar",
            7777,
            size="32",
            **{"timing.matrix.macs_per_cycle": "", "timing.scalar.fp_fma": 2},
        ),
        _row(
            "ame",
            2000,
            4,
            **{"timing.matrix.macs_per_cycle": 4, "timing.scalar.fp_fma": 2},
        ),
        _row(
            "ame",
            500,
            16,
            **{"timing.matrix.macs_per_cycle": 16, "timing.scalar.fp_fma": 2},
        ),
        _row(
            "ame",
            100,
            64,
            **{"timing.matrix.macs_per_cycle": 64, "timing.scalar.fp_fma": 2},
        ),
        _row(
            "ame",
            100,
            64,
            fmts=("bf16", "fp32"),
            **{"timing.matrix.macs_per_cycle": 64, "timing.scalar.fp_fma": 2},
        ),
    ]
    metrics.add_speedups(rows, study)
    assert [r.get("speedup") for r in rows[3:]] == [
        0.5,
        2.0,
        10.0,
        10.0,
    ]  # bf16 compares with the fp32 baseline
    assert rows[3]["baseline_point"] == "scalar-1000"
    cross = metrics.crossovers(rows, study)
    fp32 = next(c for c in cross if c["in_fmt"] == "fp32")
    assert fp32["status"] == "bracketed"
    assert fp32["points"] == 3
    # Between ratio 1 (0.5x) and 4 (2x): 4^(1/3), as in test_interpolate_crossover.
    assert fp32["crossover_ratio"] == pytest.approx(4 ** (1 / 3))
    assert "timing.scalar.fp_fma=2" in fp32["group"]
    # baseline.formats: same -> bf16 finds no bf16 scalar row.
    same = Study(
        name="s",
        overrides=study.overrides,
        only=study.only,
        baseline={"variant": "scalar", "formats": "same"},
    )
    rows2 = [dict(r) for r in rows]
    for r in rows2:
        r.pop("speedup", None)
    metrics.add_speedups(rows2, same)
    assert rows2[-1].get("speedup") is None
    assert rows2[3]["speedup"] == 0.5


def _result(point_id="p1", variant="ame", status="ok"):
    return {
        "point_id": point_id,
        "dwarf": "toymm",
        "variant": variant,
        "size": 16,
        "size_str": "16",
        "in_fmt": "fp32",
        "acc_fmt": "fp32",
        "seed": 0,
        "status": status,
        "passed": True,
        "exit_code": 0,
        "cycles": 120,
        "region_cycles": 100,
        "useful_flops": 8192,
        "tolerance": 1e-5,
        "design_hash": "abc",
        "design_file": "designs/abc.yaml",
        "design_path": str(BASE),
        "git_commit": "deadbeef",
        "overrides": {"timing.matrix.macs_per_cycle": 64},
        "error": {
            "C": {"normwise": 1e-7, "max_rel": 1e-5},
            "D": {"normwise": 2e-7, "max_rel": 1e-6},
        },
        "counters": {
            "matrix.macs": 8192,
            "matrix.mma": 2,
            "matrix.mem_to_tile_bytes": 2048,
            "matrix.acc_to_mem_bytes": 1024,
            "bus.l1.read_bytes": 64,
            "bus.l1.write_bytes": 32,
            "dma.bytes": 4096,
            "dma.dram_to_scratchpad_bytes": 4096,
            "timing.mem.l1.bytes": 96,
            "timing.matrix_busy_cycles": 40,
            "instret": 50,
            "instret.scalar": 40,
            "instret.fp": 2,
            "instret.vector": 0,
            "instret.matrix": 8,
        },
    }


def test_csv_schema_and_round_trip(tmp_path):
    study = Study(name="s", overrides={"timing.matrix.macs_per_cycle": [64]})
    row = metrics.flatten(_result(), study)
    assert row["redundancy"] == 2.0
    assert row["matrix_util"] == 0.4
    assert row["normwise"] == 2e-7
    assert row["max_rel"] == 1e-5
    assert row["mv_ratio"] == 512
    assert row["vector_baseline"] == "scalar"
    assert row["macs_util"] == pytest.approx(8192 / (512 * 100))
    for col in (
        "bytes:matrix.mem_to_tile",
        "bytes:matrix.acc_to_mem",
        "bytes:bus.l1.read",
        "bytes:bus.l1.write",
        "bytes:dma.bytes",
        "bytes:dma.dram_to_scratchpad_bytes",
        "err:C.normwise",
        "err:D.max_rel",
        "ovr:timing.matrix.macs_per_cycle",
        "instret.matrix",
    ):
        assert col in row, col
    assert not any(c.startswith("bytes:timing") for c in row)
    failed = metrics.flatten(
        {
            **_result("p2", status="error"),
            "counters": {},
            "error": {},
            "message": "boom",
        },
        study,
    )
    path = metrics.write_csv(tmp_path / "r.csv", [row, failed])
    header = path.read_text().splitlines()[0].split(",")
    assert header[: len(metrics.FIXED_COLUMNS)] == metrics.FIXED_COLUMNS
    ovr_at = header.index("ovr:timing.matrix.macs_per_cycle")
    assert ovr_at < header.index("bytes:bus.l1.read") < header.index("err:C.max_rel")
    back = metrics.read_csv(path)
    assert back[0]["region_cycles"] == 100
    assert back[0]["passed"] is True
    assert back[1]["message"] == "boom"
    assert back[1]["redundancy"] is None


# -- resume ------------------------------------------------------------------------------------


def test_resume_skips_completed_points(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(point, result_path, build_root, commit):
        calls.append(point["point_id"])
        failed = point["variant"] == "scalar"
        result = {
            **point,
            "size_str": str(point["size"]),
            "design_file": "designs/x.yaml",
            "git_commit": commit,
            "status": "error" if failed else "ok",
            "message": "build failed" if failed else "",
            "region_cycles": 10,
            "cycles": 20,
            "passed": not failed,
            "wall_s": 0,
        }
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(json.dumps(result))
        return result

    monkeypatch.setattr(run, "run_one", fake_run_one)
    kw = dict(out_root=tmp_path, jobs=1, log=lambda m: None)
    _, points, rows, _ = run.run_sweep(TOY_SWEEP, **kw)
    unique = {p.point_id for p in points}
    assert sorted(calls) == sorted(unique)
    assert len(rows) == len(points)
    calls.clear()
    run.run_sweep(TOY_SWEEP, **kw)
    assert calls == []
    run.run_sweep(TOY_SWEEP, retry_failed=True, **kw)
    assert calls
    assert all(
        pid in {p.point_id for p in points if p.variant == "scalar"} for pid in calls
    )
    calls.clear()
    run.run_sweep(TOY_SWEEP, filters=["precision:*fp16*"], force=True, **kw)
    assert len(calls) == 1
    assert (tmp_path / "toy" / "results.csv").exists()


# -- end to end --------------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_CC, reason=f"needs make and {NAIAD_CC}")
def test_toy_sweep_end_to_end(tmp_path):
    from experiments.naiad.plot import plot_sweep

    _, points, rows, cross = run.run_sweep(
        TOY_SWEEP, out_root=tmp_path, jobs=2, log=lambda m: None
    )
    assert rows
    failed = [r["message"] for r in rows if r["status"] != "ok" or not r["passed"]]
    assert not failed
    assert all(r["git_commit"] and r["design_hash"] and r["seed"] == 0 for r in rows)
    ame = [r for r in rows if r["study"] == "crossover" and r["variant"] == "ame"]
    assert len(ame) == 5
    assert all(r.get("speedup") for r in ame)
    # More matrix throughput never makes the toy slower.
    ame.sort(key=lambda r: r["mv_ratio"])
    assert [r["speedup"] for r in ame] == sorted(r["speedup"] for r in ame)
    c = next(c for c in cross if c["study"] == "crossover")
    assert c["status"] == "bracketed"
    assert c["min_ratio"] < c["crossover_ratio"] < c["max_ratio"]
    assert all(
        r["redundancy"] == 1.0 for r in ame
    )  # a dense GEMM does no redundant work
    precision = {
        (r["in_fmt"], r["acc_fmt"]): r["normwise"]
        for r in rows
        if r["study"] == "precision"
    }
    assert (
        precision[("fp64", "fp64")]
        < precision[("fp32", "fp32")]
        < precision[("bf16", "fp32")]
    )
    paths = plot_sweep(tmp_path / "toy", log=lambda m: None)
    names = {p.name for p in paths}
    for stem in (
        "crossover-crossover",
        "accuracy-precision",
        "bytes-bytes",
        "redundancy-bytes",
    ):
        assert f"{stem}.png" in names
        assert f"{stem}.svg" in names
    assert all(p.stat().st_size > 1000 for p in paths)
    # Resumed: nothing reruns, and the CSV is rewritten identically.
    before = (tmp_path / "toy" / "results.csv").read_text()
    _, _, rows2, _ = run.run_sweep(
        TOY_SWEEP, out_root=tmp_path, jobs=2, log=lambda m: None
    )
    assert (tmp_path / "toy" / "results.csv").read_text() == before
    assert not math.isnan(rows2[0]["cycles"])


def test_only_takes_variant_globs():
    study = Study(name="s", only={"timing.matrix.macs_per_cycle": ["ame*"]})
    assert study.applies("timing.matrix.macs_per_cycle", "ame_gauss")
    assert not study.applies("timing.matrix.macs_per_cycle", "scalar")
    assert study.applies("timing.scalar.fp_fma", "scalar")


def test_crossover_needs_a_swept_ratio():
    study = Study(name="s")
    rows = [{**_row("ame", 10, 64), "speedup": 2.0}]
    assert metrics.crossovers(rows, study) == []
