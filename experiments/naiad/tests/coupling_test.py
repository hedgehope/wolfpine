"""The coupling study's harness pieces: tensix normalisation, labels by normalise key, coupling figures."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from experiments.naiad import plot
from experiments.naiad.sweep import DesignFactory, SweepError, expand, load_sweep

REPO = Path(__file__).resolve().parents[3]
DESIGNS = REPO / "framework" / "naiad" / "designs"
SWEEP = REPO / "experiments" / "naiad" / "sweeps" / "coupling_all.yaml"
BUDGET = {"operand_bytes": 12288, "mem_bytes_per_cycle": 64, "mode": "tensix"}
needs_mpl = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="needs matplotlib"
)


def _blocks(path):
    raw = yaml.safe_load(Path(path).read_text())
    return {b["name"]: b.get("params") or {} for b in raw["blocks"]}


@pytest.mark.parametrize("peak, ratio", [(4096, 1), (1024, "1/4")])
def test_tensix_normalise_sets_clock_and_dma(tmp_path, peak, ratio):
    factory = DesignFactory(tmp_path)
    _, path, derived = factory.derive(
        DESIGNS / "naiad-tensix.yaml",
        {},
        {"peak_flops_per_cycle": peak, **BUDGET},
        "bf16",
        "fp32",
    )
    blocks = _blocks(path)
    assert blocks["tensix"]["clock_ratio"] == ratio
    assert blocks["dma"]["bytes_per_cycle"] == 16
    assert blocks["dma"]["channels"] == 4  # the window depends on it: never normalised
    assert derived["clock_ratio"] == ratio
    raw = yaml.safe_load(path.read_text())
    assert raw["timing"]["matrix"]["macs_per_cycle"] == peak // 2


def test_tensix_normalise_needs_the_blocks(tmp_path):
    with pytest.raises(SweepError, match="'tensix' block"):
        DesignFactory(tmp_path).derive(
            DESIGNS / "naiad-dwarfs.yaml",
            {},
            {"peak_flops_per_cycle": 4096, **BUDGET},
            "bf16",
            "fp32",
        )


def test_coupling_all_expands_every_design_point(tmp_path):
    sweep = load_sweep(SWEEP)
    points = expand(sweep, DesignFactory(tmp_path), log=lambda *_: None)
    gemm = {(p.variant, p.in_fmt) for p in points if p.dwarf == "gemm"}
    assert gemm == {
        (v, f) for v in ("ame", "ime", "vme", "rvv") for f in ("fp32", "bf16")
    } | {("tensix", "bf16")}
    reduce_variants = {p.variant for p in points if p.dwarf == "reduce"}
    assert reduce_variants == {"ame", "ime", "vme", "rvv"}
    peaks = {p.normalise["peak_flops_per_cycle"] for p in points if p.normalise}
    assert peaks == {4096, 1024}
    ime = {p.normalised["vlen"] for p in points if p.variant == "ime"}
    vme = {p.normalised["te"] for p in points if p.variant == "vme"}
    assert ime == {2048, 256}
    assert vme == {16, 32}


def _row(variant, cycles, size=128, peak=4096, study="s", fmt="fp32", **norm):
    r = {
        "study": study,
        "dwarf": "gemm",
        "variant": variant,
        "size": size,
        "in_fmt": fmt,
        "acc_fmt": "fp32",
        "status": "ok",
        "region_cycles": cycles,
        "useful_flops": 2 * size**3,
        "point_id": f"{variant}{size}{peak}{fmt}{sorted(norm.items())}",
        "bytes:matrix.mem_to_tile": 1000,
        "bytes:matrix.vrf_a_read": 500,
        "bytes:vector.mem_to_vec_bytes": 300,
    }
    if variant != "rvv":
        r["norm:peak_flops_per_cycle"] = peak
    r.update({f"norm:{k}": v for k, v in norm.items()})
    return r


def test_norm_tags_tell_two_points_of_a_variant_apart():
    rows = [_row("ime", 10, vlen=2048), _row("ime", 20, vlen=256), _row("ame", 5, d=16)]
    tags = plot.norm_tags(rows)
    assert tags[id(rows[0])] == "vlen=2048"
    assert tags[id(rows[1])] == "vlen=256"
    assert tags[id(rows[2])] == ""  # nothing varies for ame


def test_coupling_points_label_by_key_not_peak():
    rows = [
        _row("ime", 10, vlen=2048),
        _row("ime", 12, peak=1024, vlen=2048),
        _row("ime", 20, vlen=256),
        _row("rvv", 99),
        _row("ame", 5),
    ]
    labels = {
        id(r): name for r in rows for name in [plot.coupling_points(rows)[id(r)][1]]
    }
    assert labels[id(rows[0])] == labels[id(rows[1])] == "IME vlen=2048"
    assert labels[id(rows[2])] == "IME vlen=256"
    order = sorted(set(plot.coupling_points(rows).values()))
    assert [n for _, n in order] == ["AME", "IME vlen=256", "IME vlen=2048", "RVV"]


def test_bytes_bars_keep_two_normalised_points_apart():
    rows = [_row("ime", 10, vlen=2048), _row("ime", 20, vlen=256), _row("rvv", 99)]
    keys = sorted(k[1] for k in plot.bytes_bars(rows))
    assert keys == ["ime\nvlen=2048", "ime\nvlen=256", "rvv"]


def test_segments_include_vrf_and_vector_paths():
    segs = plot._segments(_row("vme", 1))
    assert segs["VRF → matrix (A, B)"] == 500
    assert segs["memory → VRF"] == 300


@needs_mpl
def test_coupling_and_bytes_figures(tmp_path):
    rows = []
    for size in (64, 128):
        rows.append(_row("rvv", 1000 * size, size=size, study="a"))
        rows.append(_row("rvv", 1000 * size, size=size, study="b"))
        for peak in (4096, 1024):
            rows.append(_row("ame", 10 * size, size=size, peak=peak, study="a"))
            for vlen in (2048, 256):
                rows.append(
                    _row(
                        "ime",
                        vlen * size // 64,
                        size=size,
                        peak=peak,
                        study="b",
                        vlen=vlen,
                    )
                )
    paths = plot.plot_coupling(rows, ["a", "b"], tmp_path, "t")
    names = sorted(p.name for p in paths)
    assert names == sorted(
        f"coupling_{m}-t-gemm.{ext}"
        for m in ("cycles", "speedup", "bytes")
        for ext in ("png", "svg")
    )
    made = plot.plot_bytes(rows, "b", tmp_path)
    assert made
    assert all(p.exists() for p in made)
