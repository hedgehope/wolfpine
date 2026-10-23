"""``python -m experiments.naiad.plot <sweep> [--out DIR]``: the figures for one sweep's ``results.csv``.

Each study lists the figures it feeds (``figures:`` in the sweep file); every
figure is written as PNG and SVG to ``out/<sweep>/figures/<figure>-<study>``:

``crossover``
    Speedup over the vector baseline against the matrix-to-vector peak
    throughput ratio, log-log, one line per dwarf (and size / formats when a
    study has several), the interpolated crossover marked.
``bytes``
    Bytes moved per path, stacked bars per dwarf and variant: memory to and
    from the matrix unit's tiles and accumulators, scalar loads and stores per
    memory block, and DMA.
``accuracy``
    Normwise error against the fp64 golden per input/accumulator format pair,
    one line per dwarf, log axis; the fp64 unit roundoff is the floor (exact
    results sit on it).
``redundancy``
    Matrix-unit FLOPs over the algorithm's useful FLOPs, per dwarf.

Colours follow the dwarf, never its position: :data:`DWARF_ORDER` fixes the
slot of every known dwarf, so a dwarf keeps its colour across figures and sweeps.
"""

import argparse
import json
import sys
from pathlib import Path

from experiments.naiad import metrics
from experiments.naiad.run import resolve_sweep
from experiments.naiad.sweep import OUT, load_sweep

#: The reference categorical palette (fixed order).
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
OTHER = "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
#: Dwarf -> palette slot: Tier-1 dwarfs first, then the harness's toy.
DWARF_ORDER = [
    "gemm",
    "jacobi",
    "fft",
    "nbody",
    "reduce",
    "scan",
    "tensorprod",
    "toymm",
]


def dwarf_colour(name, extra=()):
    if name in DWARF_ORDER:
        return PALETTE[DWARF_ORDER.index(name)]
    extra = sorted(set(extra) - set(DWARF_ORDER))
    return (
        OTHER
        if name not in extra
        else PALETTE[(len(DWARF_ORDER) + extra.index(name)) % len(PALETTE)]
    )


def variant_style(variant):
    """Line style per variant: the dwarf's colour is shared, so the variant shows in the dash."""
    if variant == "ame":
        return "-"
    if variant == "scalar":
        return ":"
    return "--"


def _variant_suffix(variant):
    return "" if variant == "ame" else f" ({variant})"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )
    return plt


def _save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
        paths.append(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def _note(fig, text):
    fig.text(0.01, -0.02, text, fontsize=8, color=INK_2, ha="left", va="top", wrap=True)


def _ok(rows, study):
    return [r for r in rows if r["study"] == study and r["status"] == "ok"]


def _series_label(dwarf, size, fmts, multi_size, multi_fmt):
    parts = [dwarf]
    if multi_size:
        parts.append(f"n={size}")
    if multi_fmt:
        parts.append(fmts)
    return " ".join(parts)


# -- figures ------------------------------------------------------------------------------


def plot_crossover(rows, cross, study, out_dir, vector_baseline="scalar"):
    plt = _plt()
    pts = [
        r
        for r in _ok(rows, study)
        if r.get("speedup") is not None and r.get("mv_ratio") is not None
    ]
    if not pts:
        return []
    groups = {}
    for r in pts:
        groups.setdefault(
            (r["dwarf"], r["variant"], r["size"], f"{r['in_fmt']}/{r['acc_fmt']}"), []
        ).append(r)
    dwarfs = sorted({g[0] for g in groups})
    multi_size = len({(g[0], g[2]) for g in groups}) > len(dwarfs)
    multi_fmt = len({g[3] for g in groups}) > 1
    xo = {
        (c["dwarf"], c["variant"], c["size"], f"{c['in_fmt']}/{c['acc_fmt']}"): c
        for c in cross
        if c["study"] == study
    }
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for key in sorted(groups):
        dwarf, variant, size, fmts = key
        members = sorted(groups[key], key=lambda r: r["mv_ratio"])
        colour = dwarf_colour(dwarf, dwarfs)
        xs = [r["mv_ratio"] for r in members]
        ys = [r["speedup"] for r in members]
        name = _series_label(
            dwarf, size, fmts, multi_size, multi_fmt
        ) + _variant_suffix(variant)
        ax.plot(
            xs,
            ys,
            variant_style(variant),
            marker="o",
            color=colour,
            label=name,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
        c = xo.get(key)
        if c and c["crossover_ratio"] not in (None, ""):
            x = float(c["crossover_ratio"])
            ax.plot(
                [x],
                [1],
                marker="D",
                markersize=9,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=1.5,
                zorder=4,
            )
            ax.annotate(
                f"{x:.3g}",
                (x, 1),
                xytext=(4, -14),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
            )
    ax.axhline(1, color=INK_2, linewidth=1, linestyle="--", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(
        "Matrix-to-vector peak throughput ratio (matrix MACs/cycle ÷ vector FMAs/cycle)"
    )
    ax.set_ylabel(f"Speedup over {vector_baseline} baseline (×)")
    ax.set_title("Crossover: where the matrix reformulation pays off")
    ax.legend(loc="upper left", fontsize=9)
    _note(
        fig,
        "Diamonds mark the crossover (speedup 1, log-linear interpolation). "
        f"Vector baseline: {vector_baseline}"
        + (
            " F/D, standing in for RVV until Phase C."
            if vector_baseline == "scalar"
            else "."
        ),
    )
    return _save(fig, out_dir, f"crossover-{study}")


def _segments(row):
    """``[(name, bytes)]`` for one row, grouped into display paths."""
    segs = {}
    names = {
        "matrix.mem_to_tile": "memory → tile",
        "matrix.tile_to_mem": "tile → memory",
        "matrix.mem_to_acc": "memory → acc",
        "matrix.acc_to_mem": "acc → memory",
    }
    for col, value in row.items():
        if not col.startswith("bytes:") or not value:
            continue
        path = col[len("bytes:") :]
        if path in names:
            segs[names[path]] = segs.get(names[path], 0) + value
        elif path.startswith("bus."):
            _, block, kind = path.split(".", 2)
            if block in ("bootrom", "clint", "perf"):
                continue  # boot and counter reads, not data movement
            label = f"scalar {'load' if kind == 'read' else 'store'}: {block}"
            segs[label] = segs.get(label, 0) + value
        else:
            block, _, key = path.partition(".")
            if key == "bytes":
                segs[f"DMA: {block}"] = segs.get(f"DMA: {block}", 0) + value
    return segs


def plot_bytes(rows, study, out_dir):
    plt = _plt()
    pts = _ok(rows, study)
    if not pts:
        return []
    bars = {}
    for r in pts:
        key = (r["dwarf"], r["variant"], r["size"], f"{r['in_fmt']}/{r['acc_fmt']}")
        bars.setdefault(key, _segments(r))
    order = ["memory → tile", "tile → memory", "memory → acc", "acc → memory"]
    totals = {}
    for segs in bars.values():
        for name, v in segs.items():
            totals[name] = totals.get(name, 0) + v
    names = [n for n in order if totals.get(n)] + sorted(
        (n for n in totals if n not in order and totals[n]), key=lambda n: -totals[n]
    )
    if len(names) > len(PALETTE):
        keep, fold = names[: len(PALETTE) - 1], set(names[len(PALETTE) - 1 :])
        for segs in bars.values():
            segs["other"] = sum(segs.pop(n, 0) for n in fold)
        names = [*keep, "other"]
    keys = sorted(bars)
    multi_size = len({k[2] for k in keys}) > 1
    multi_fmt = len({k[3] for k in keys}) > 1
    # One fact per line (dwarf, size, formats, variant) so neighbouring labels never collide.
    labels = [
        "\n".join(
            [d, *([f"n={s}"] if multi_size else []), *([f] if multi_fmt else []), v]
        )
        for d, v, s, f in keys
    ]
    fig, ax = plt.subplots(figsize=(max(5, 1.25 * len(keys) + 2.5), 4.6))
    bottom = [0.0] * len(keys)
    for i, name in enumerate(names):
        vals = [bars[k].get(name, 0) / 1024 for k in keys]
        colour = OTHER if name == "other" else PALETTE[i % len(PALETTE)]
        ax.bar(
            range(len(keys)),
            vals,
            0.6,
            bottom=bottom,
            color=colour,
            label=name,
            edgecolor="white",
            linewidth=1.5,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(keys)), labels, fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Bytes moved (KiB)")
    ax.set_title("Bytes moved per path")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    _note(
        fig,
        "Whole run, from the counters: matrix-unit transfers, scalar loads/stores per memory block, and DMA.",
    )
    return _save(fig, out_dir, f"bytes-{study}")


def plot_accuracy(rows, study, out_dir):
    plt = _plt()
    pts = [r for r in _ok(rows, study) if r.get("normwise") is not None]
    if not pts:
        return []
    pairs = []
    for r in pts:
        p = f"{r['in_fmt']}/{r['acc_fmt']}"
        if p not in pairs:
            pairs.append(p)
    pairs.sort(key=lambda p: (-_bits(p.split("/")[0]), -_bits(p.split("/")[1]), p))
    floor = metrics.FP64_UNIT_ROUNDOFF
    series = {}
    for r in pts:
        series.setdefault((r["dwarf"], r["variant"], r["size"]), {})[
            f"{r['in_fmt']}/{r['acc_fmt']}"
        ] = max(float(r["normwise"]), floor)
    dwarfs = sorted({k[0] for k in series})
    multi_size = len({(k[0], k[2]) for k in series}) > len(dwarfs)
    multi_var = len({k[1] for k in series}) > 1
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(pairs) + 3), 4.4))
    for key in sorted(series):
        dwarf, variant, size = key
        xs = [i for i, p in enumerate(pairs) if p in series[key]]
        ys = [series[key][pairs[i]] for i in xs]
        name = _series_label(dwarf, size, "", multi_size, False) + (
            _variant_suffix(variant) if multi_var else ""
        )
        ax.plot(
            xs,
            ys,
            variant_style(variant),
            marker="o",
            color=dwarf_colour(dwarf, dwarfs),
            label=name,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
    ax.axhline(floor, color=INK_2, linestyle="--", linewidth=1)
    ax.annotate(
        "fp64 unit roundoff (floor)",
        (len(pairs) - 1, floor),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_2,
    )
    ax.set_yscale("log")
    ax.set_xticks(range(len(pairs)), pairs)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Input / accumulator format")
    ax.set_ylabel("Normwise error vs fp64 golden (‖Δ‖₂ / ‖ref‖₂)")
    ax.set_title("Accuracy per precision mode")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    _note(
        fig,
        "Golden: fp64 on the unrounded inputs, so input rounding counts. Exact results are drawn on the floor.",
    )
    return _save(fig, out_dir, f"accuracy-{study}")


def _bits(fmt):
    widths = {
        "fp64": 64,
        "int32": 32,
        "fp32": 32,
        "tf32": 19,
        "fp16": 16,
        "bf16": 16,
        "fp8_e4m3": 8,
        "fp8_e5m2": 8,
        "int8": 8,
    }
    return widths.get(fmt, 0)


def plot_redundancy(rows, study, out_dir):
    plt = _plt()
    pts = [r for r in _ok(rows, study) if r.get("redundancy") is not None]
    if not pts:
        return []
    bars = {}
    for r in pts:
        bars.setdefault(
            (r["dwarf"], r["size"], f"{r['in_fmt']}/{r['acc_fmt']}", r["variant"]),
            r["redundancy"],
        )
    keys = sorted(bars)
    dwarfs = sorted({k[0] for k in keys})
    multi_size = len({(k[0], k[1]) for k in keys}) > len(dwarfs)
    multi_fmt = len({k[2] for k in keys}) > 1
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(keys) + 2), 4))
    vals = [bars[k] for k in keys]
    ax.bar(
        range(len(keys)),
        vals,
        0.6,
        color=[dwarf_colour(k[0], dwarfs) for k in keys],
        edgecolor="white",
        linewidth=1.5,
    )
    for i, v in enumerate(vals):
        ax.annotate(
            f"{v:.3g}×",
            (i, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax.axhline(1, color=INK_2, linestyle="--", linewidth=1)
    ax.set_xticks(
        range(len(keys)),
        [
            _series_label(d, s, f, multi_size, multi_fmt)
            + _variant_suffix(v).replace(" (", "\n(")
            for d, s, f, v in keys
        ],
        fontsize=9,
    )
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Matrix FLOPs ÷ useful FLOPs (×)")
    ax.set_title("Redundant arithmetic of each reformulation")
    _note(
        fig,
        "1× (dashed) is no redundant work; matrix FLOPs = 2 × MACs issued, useful FLOPs from each dwarf's spec.",
    )
    return _save(fig, out_dir, f"redundancy-{study}")


# -- driver ----------------------------------------------------------------------------------


def plot_sweep(sweep_dir, studies=None, log=print):
    """Every figure the studies ask for, from ``sweep_dir/results.csv``; returns the paths written."""
    sweep_dir = Path(sweep_dir)
    rows = metrics.read_csv(sweep_dir / "results.csv")
    cross_path = sweep_dir / "crossover.csv"
    cross = metrics.read_csv(cross_path) if cross_path.exists() else []
    meta = json.loads((sweep_dir / "meta.json").read_text())
    out_dir = sweep_dir / "figures"
    paths = []
    for name, info in meta["studies"].items():
        if studies and name not in studies:
            continue
        for fig in info.get("figures") or []:
            if fig == "crossover":
                made = plot_crossover(
                    rows, cross, name, out_dir, info.get("vector_baseline", "scalar")
                )
            elif fig == "bytes":
                made = plot_bytes(rows, name, out_dir)
            elif fig == "accuracy":
                made = plot_accuracy(rows, name, out_dir)
            else:
                made = plot_redundancy(rows, name, out_dir)
            if not made:
                log(f"{name}: no data for the {fig} figure")
            paths += made
    for p in paths:
        log(f"wrote {p}")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m experiments.naiad.plot", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument(
        "sweep", help="sweep name (out/<name>), sweep YAML, or a sweep output directory"
    )
    ap.add_argument("--out", help=f"output root (default {OUT})")
    ap.add_argument(
        "--quick", action="store_true", help="plot the --quick run of a sweep YAML"
    )
    args = ap.parse_args(argv)
    target = Path(args.sweep)
    if target.is_dir():
        sweep_dir = target
    else:
        root = Path(args.out or OUT)
        try:
            name = load_sweep(resolve_sweep(args.sweep)).name
        except SystemExit:
            name = args.sweep
        sweep_dir = root / (f"{name}-quick" if args.quick else name)
    if not (sweep_dir / "results.csv").exists():
        raise SystemExit(
            f"no results in {sweep_dir}; run python -m experiments.naiad.run first"
        )
    plot_sweep(sweep_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
