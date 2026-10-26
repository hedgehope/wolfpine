"""``python -m experiments.naiad.plot <sweep> [--out DIR]``: the figures for one sweep's ``results.csv``.

Each study lists the figures it feeds (``figures:`` in the sweep file); every
figure is written as PNG and SVG to ``out/<sweep>/figures/<figure>-<study>``:

``crossover``
    Speedup over the vector baseline against the matrix-to-vector peak
    throughput ratio, log-log, one line per dwarf (and size / formats when a
    study has several), the interpolated crossover marked, and its value in a
    table beside the axes; variants that run once (the scalar reference) are
    flat dotted lines.
``bytes``
    Bytes moved per path, stacked bars per dwarf and variant: memory to and
    from the matrix unit's tiles and accumulators, scalar loads and stores per
    memory block, and DMA.
``accuracy``
    Normwise error against the fp64 golden per input/accumulator format pair,
    one line per dwarf, log axis; the fp64 unit roundoff is the floor (exact
    results sit on it). With several seeds a point is the median, with a bar
    up to the worst seed.
``redundancy``
    Matrix-unit FLOPs over the algorithm's useful FLOPs, per dwarf.
``coupling``
    One set over every study that lists it (a study per design point, each
    normalised its own way): ``coupling_cycles`` / ``coupling_speedup`` (region
    cycles and speedup over the baseline against size, a panel per format pair
    and normalised peak, a line per design point labelled by its normalise
    keys) and ``coupling_bytes`` (bytes per useful FLOP and path).
``sparsity``
    Speedup over the baseline against block density, for sizes ``(n, density[, nv])``
    (spmv), one line per variant, nv and swept override.
``scaling``
    One figure (and ``scaling.csv``) over every study that lists it: speedup of
    the ``*_mc`` variants against ``cores.count``, a panel per cache / L1-port
    configuration (:mod:`experiments.naiad.scaling`).

Colours follow the dwarf, never its position: :data:`DWARF_ORDER` fixes the
slot of every known dwarf, so a dwarf keeps its colour across figures and sweeps.
"""

import argparse
import json
import statistics
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
    "#8f5b2e",
    "#00838f",
]
OTHER = "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
#: Dwarf -> palette slot: Tier-1 dwarfs first, then the Phase D dwarfs, then the harness's toy.
DWARF_ORDER = [
    "gemm",
    "jacobi",
    "fft",
    "nbody",
    "reduce",
    "scan",
    "tensorprod",
    "spmv",
    "apsp",
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
    if variant.endswith("_fused"):
        return "-."
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


def _key_order(key):
    """A sort key for a tuple of labels whose size cell may be a number (66) or a string (68x2)."""
    return tuple((_size_parts(x), str(x)) for x in key)


def _series_label(dwarf, size, fmts, multi_size, multi_fmt):
    parts = [dwarf]
    if multi_size:
        parts.append(f"n={size}")
    if multi_fmt:
        parts.append(fmts)
    return " ".join(parts)


# -- figures ------------------------------------------------------------------------------


def _crossover_text(c):
    """The crossover cell of the table inset: the ratio, or which side of the swept range it lies."""
    if c is None:
        return "-"
    if c.get("crossover_ratio") not in (None, ""):
        return f"{float(c['crossover_ratio']):.3g}"
    if c.get("status") == "below_range":
        return f"< {float(c['min_ratio']):.3g}"
    if c.get("status") == "above_range":
        return f"> {float(c['max_ratio']):.3g}"
    return "-"


def plot_crossover(rows, cross, study, out_dir, vector_baseline="scalar"):
    """Speedup vs ratio, a line per swept series; unswept variants (a reference such as
    the scalar baseline, which runs once) are flat dotted lines across the axis. The
    crossover values sit in a table beside the axes, so they never overlap."""
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
    swept = {k: v for k, v in groups.items() if len({r["mv_ratio"] for r in v}) > 1}
    flat = {k: v for k, v in groups.items() if k not in swept}
    dwarfs = sorted({g[0] for g in groups})
    multi_size = len({(g[0], g[2]) for g in swept}) > len({g[0] for g in swept})
    multi_fmt = len({g[3] for g in swept}) > 1
    xo = {
        (c["dwarf"], c["variant"], c["size"], f"{c['in_fmt']}/{c['acc_fmt']}"): c
        for c in cross
        if c["study"] == study
    }

    # One dwarf at several sizes: the size, not the dwarf, picks the colour (each
    # size's unswept baselines take its colour too), or every line would share one.
    sizes = sorted({k[2] for k in groups}, key=lambda s: _key_order((s,)))
    by_size = len(dwarfs) == 1 and multi_size

    def colour_of(dwarf, size):
        if by_size:
            return PALETTE[sizes.index(size) % len(PALETTE)]
        return dwarf_colour(dwarf, dwarfs)

    fig = plt.figure(figsize=(10.5, 5.0))
    ax = fig.add_axes([0.07, 0.12, 0.6, 0.8])
    table = []
    for key in sorted(swept, key=_key_order):
        dwarf, variant, size, fmts = key
        members = sorted(swept[key], key=lambda r: r["mv_ratio"])
        colour = colour_of(dwarf, size)
        name = _series_label(
            dwarf, size, fmts, multi_size, multi_fmt
        ) + _variant_suffix(variant)
        ax.plot(
            [r["mv_ratio"] for r in members],
            [r["speedup"] for r in members],
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
            ax.plot(
                [float(c["crossover_ratio"])],
                [1],
                marker="D",
                markersize=9,
                color=colour,
                markeredgecolor="white",
                markeredgewidth=1.5,
                zorder=4,
            )
        table.append((name, _crossover_text(c), colour))
    ref_variants = sorted({k[1] for k in flat})
    for key in sorted(flat, key=_key_order):
        dwarf, variant, size, _ = key
        speedup = statistics.median(r["speedup"] for r in flat[key])
        ax.axhline(
            speedup,
            color=colour_of(dwarf, size),
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )
    if ref_variants:
        ax.plot(
            [],
            [],
            ":",
            color=INK_2,
            linewidth=1.2,
            label=f"{', '.join(ref_variants)} (runs once; {'size' if by_size else 'dwarf'} colour)",
        )
    ax.axhline(1, color=INK_2, linewidth=1, linestyle="--", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(
        "Matrix-to-vector peak throughput ratio (matrix MACs/cycle ÷ vector FMAs/cycle)"
    )
    ax.set_ylabel(f"Speedup over {vector_baseline} baseline (×)")
    ax.set_title("Crossover: where the matrix reformulation pays off")
    ax.legend(loc="upper left", fontsize=8)
    if table:
        table.sort(key=lambda t: t[0])
        tax = fig.add_axes([0.70, 0.12, 0.29, 0.8])
        tax.axis("off")
        tab = tax.table(
            cellText=[[name, value] for name, value, _ in table],
            colLabels=["series", "crossover ratio"],
            loc="upper left",
            cellLoc="left",
            colLoc="left",
            edges="horizontal",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        tab.scale(1.0, 1.35)
        for i, (_, _, colour) in enumerate(table, start=1):
            tab[i, 0].get_text().set_color(colour)
        for cell in tab.get_celld().values():
            cell.set_edgecolor(GRID)
    _note(
        fig,
        "Diamonds mark the crossover (speedup 1, log-linear interpolation); the table gives each "
        "value (< / >: outside the swept range). "
        f"Vector baseline: {vector_baseline}"
        + (" F/D." if vector_baseline == "scalar" else "."),
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
        "matrix.vec_to_tile": "vector → tile",  # VME (Zvt) moves
        "matrix.tile_to_vec": "tile → vector",
        "matrix.vrf_a_read": "VRF → matrix (A, B)",  # IME / VME operand reads
        "matrix.vrf_b_read": "VRF → matrix (A, B)",
        "matrix.vrf_c_read": "VRF ↔ matrix (C)",  # IME: C lives in the VRF
        "matrix.vrf_c_write": "VRF ↔ matrix (C)",
        "vector.mem_to_vec_bytes": "memory → VRF",  # vector loads (byte_paths keeps the suffix)
        "vector.vec_to_mem_bytes": "VRF → memory",
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


#: Normaliser parameters (resolved, as the CSV's ``norm:`` columns hold them) that
#: tell two points of one variant apart: a bar or series is labelled with each that varies.
NORM_LABEL_KEYS = ("peak_flops_per_cycle", "vlen", "te", "d", "clock_ratio")
_NORM_SHORT = {"peak_flops_per_cycle": "peak", "clock_ratio": "clock"}


def norm_tags(rows, group=lambda r: (r["dwarf"], r["variant"])):
    """``{id(row): "vlen=2048, ..."}``: the :data:`NORM_LABEL_KEYS` that vary within each row's ``group``."""
    groups = {}
    for r in rows:
        groups.setdefault(group(r), []).append(r)
    tags = {}
    for members in groups.values():
        keys = [
            k
            for k in NORM_LABEL_KEYS
            if len({str(r.get(f"norm:{k}")) for r in members}) > 1
        ]
        for r in members:
            tags[id(r)] = ", ".join(
                f"{_NORM_SHORT.get(k, k)}={r.get(f'norm:{k}')}"
                for k in keys
                if r.get(f"norm:{k}") not in (None, "")
            )
    return tags


def bytes_bars(pts):
    """``{(dwarf, variant label, size, formats): segments}``, one bar per point of the bytes figure.

    Two normalised points of one variant (IME at two VLENs, VME at two TEs, two
    peaks) are two bars: the variant label carries every normalise key that varies.
    """
    tags = norm_tags(
        pts, lambda r: (r["dwarf"], r["variant"], r["size"], r["in_fmt"], r["acc_fmt"])
    )
    bars = {}
    for r in pts:
        tag = tags[id(r)].replace(
            ", ", "\n"
        )  # one fact per line, as the rest of the label
        variant = r["variant"] + (f"\n{tag}" if tag else "")
        key = (r["dwarf"], variant, r["size"], f"{r['in_fmt']}/{r['acc_fmt']}")
        bars.setdefault(key, _segments(r))
    return bars


def plot_bytes(rows, study, out_dir):
    plt = _plt()
    pts = _ok(rows, study)
    if not pts:
        return []
    bars = bytes_bars(pts)
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
    keys = sorted(bars, key=_key_order)
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


#: Coupling-figure design-point order (and colour slot): the variant, then its label keys.
COUPLING_ORDER = ("ame", "ime", "vme", "tensix")
_COUPLING_NAMES = {
    "ame": "AME",
    "ime": "IME",
    "vme": "VME",
    "tensix": "Tensix",
    "rvv": "RVV",
}
#: Keys that name a design point; the peak is a panel, and the clock ratio follows it.
_POINT_KEYS = tuple(
    k for k in NORM_LABEL_KEYS if k not in ("peak_flops_per_cycle", "clock_ratio")
)


def _peak(r):
    v = r.get("norm:peak_flops_per_cycle")
    return None if v in (None, "") else int(float(v))


def coupling_points(rows, baseline="rvv"):
    """``{id(row): (order, label)}`` for the coupling figures: one design point per
    variant and :data:`_POINT_KEYS` value set (``IME vlen=2048``), the baseline last."""
    tags = {}
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    for variant, members in by_variant.items():
        keys = [
            k
            for k in _POINT_KEYS
            if len({str(r.get(f"norm:{k}")) for r in members}) > 1
        ]
        for r in members:
            tag = " ".join(
                f"{k}={r.get(f'norm:{k}')}"
                for k in keys
                if r.get(f"norm:{k}") not in (None, "")
            )
            name = _COUPLING_NAMES.get(variant, variant)
            rank = (
                COUPLING_ORDER.index(variant)
                if variant in COUPLING_ORDER
                else len(COUPLING_ORDER)
            )
            if variant == baseline:
                rank = len(COUPLING_ORDER) + 1
            sort_tag = tuple(float(r.get(f"norm:{k}") or 0) for k in keys)
            tags[id(r)] = ((rank, variant, sort_tag), f"{name} {tag}".strip())
    return tags


def _size_num(size):
    parts = _size_parts(size)
    return parts[0] if parts else 0


def plot_coupling(rows, studies, out_dir, stem, baseline="rvv"):
    """The coupling comparison over several studies (one per design point, each with its own
    normalisation): region cycles and speedup over ``baseline`` against problem size, one
    panel per format pair (rows) and normalised peak (columns), one line per design point
    (labelled by its normalise keys, so IME at two VLENs is two lines); plus bytes moved per
    useful FLOP and path, at the highest peak. The baseline (peak-independent) is drawn in every
    peak panel."""
    pts = [
        r
        for r in rows
        if r["study"] in studies and r["status"] == "ok" and r.get("region_cycles")
    ]
    paths = []
    for dwarf in sorted({r["dwarf"] for r in pts}, key=lambda d: (d != "gemm", d)):
        paths += _plot_coupling_dwarf(
            [r for r in pts if r["dwarf"] == dwarf], dwarf, out_dir, stem, baseline
        )
    return paths


def _plot_coupling_dwarf(pts, dwarf, out_dir, stem, baseline):
    plt = _plt()
    stem = f"{stem}-{dwarf}"
    cube = "³" if dwarf == "gemm" else ""
    # One row per (design point, size, formats, peak): the same baseline point recurs per study.
    uniq = {}
    for r in pts:
        uniq.setdefault((r["point_id"], _peak(r)), r)
    pts = list(uniq.values())
    labels = coupling_points(pts, baseline)
    order = sorted({labels[id(r)] for r in pts})
    colour = {}
    slot = 0
    for _, name in order:
        if name.startswith(_COUPLING_NAMES.get(baseline, baseline)):
            colour[name] = INK_2
        else:
            colour[name] = PALETTE[slot % len(PALETTE)]
            slot += 1
    fmts = sorted(
        {f"{r['in_fmt']}/{r['acc_fmt']}" for r in pts},
        key=lambda f: (-_bits(f.split("/")[0]), f),
    )
    peaks = sorted({_peak(r) for r in pts if _peak(r) is not None}, reverse=True) or [
        None
    ]
    base_rows = {
        (r["dwarf"], r["size"], r["in_fmt"], r["acc_fmt"]): r["region_cycles"]
        for r in pts
        if r["variant"] == baseline
    }
    paths = []
    for metric in ("cycles", "speedup"):
        fig, axes = plt.subplots(
            len(fmts),
            len(peaks),
            figsize=(5.2 * len(peaks) + 2.4, 3.6 * len(fmts)),
            squeeze=False,
            sharex=True,
        )
        for i, fmt in enumerate(fmts):
            for j, peak in enumerate(peaks):
                ax = axes[i][j]
                series = {}
                for r in pts:
                    if f"{r['in_fmt']}/{r['acc_fmt']}" != fmt:
                        continue
                    if r["variant"] != baseline and _peak(r) != peak:
                        continue
                    if metric == "speedup":
                        b = base_rows.get(
                            (r["dwarf"], r["size"], r["in_fmt"], r["acc_fmt"])
                        )
                        if r["variant"] == baseline or not b:
                            continue
                        y = b / r["region_cycles"]
                    else:
                        y = r["region_cycles"]
                    series.setdefault(labels[id(r)], []).append(
                        (_size_num(r["size"]), y)
                    )
                for key in sorted(series):
                    name = key[1]
                    line = sorted(series[key])
                    is_base = colour[name] == INK_2
                    ax.plot(
                        [p[0] for p in line],
                        [p[1] for p in line],
                        "--" if is_base else "-",
                        marker="o",
                        color=colour[name],
                        label=name,
                        markeredgecolor="white",
                        markeredgewidth=1.5,
                        zorder=3,
                    )
                if metric == "speedup":
                    ax.axhline(1, color=INK_2, linewidth=1, linestyle="--", zorder=1)
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                sizes = sorted({_size_num(r["size"]) for r in pts})
                ax.set_xticks(sizes, [f"{int(s)}{cube}" for s in sizes])
                ax.minorticks_off()
                ax.set_title(f"{fmt}, peak {peak} FLOPs/cycle" if peak else fmt)
                if i == len(fmts) - 1:
                    ax.set_xlabel(f"{dwarf} size" + (" (n³)" if cube else " (n)"))
                if j == 0:
                    ax.set_ylabel(
                        "Region cycles"
                        if metric == "cycles"
                        else f"Speedup over {baseline} (×)"
                    )
        handles = {}
        for row_axes in axes:
            for ax in row_axes:
                for h, lbl in zip(*ax.get_legend_handles_labels()):
                    handles.setdefault(lbl, h)
        names = [n for _, n in order if n in handles]
        fig.legend(
            [handles[n] for n in names],
            names,
            loc="upper left",
            bbox_to_anchor=(1.0, 0.95),
            fontsize=9,
        )
        fig.suptitle(
            f"Coupling at equal resources, {dwarf}: "
            + ("region cycles" if metric == "cycles" else f"speedup over {baseline}"),
            fontweight="bold",
        )
        fig.tight_layout()
        _note(
            fig,
            "Every design point normalised to the panel's peak matrix FLOPs/cycle (per pass; format_passes / "
            "fidelity apply on top), 12 KiB operand storage where the mode allows it, 64 B/cycle memory bandwidth. "
            f"{baseline} (dashed) does not depend on the peak. Tensix runs bf16 inputs only.",
        )
        paths += _save(fig, out_dir, f"coupling_{metric}-{stem}")
    paths += _plot_coupling_bytes(
        plt, pts, labels, order, colour, fmts, peaks[0], baseline, out_dir, stem, cube
    )
    return paths


def _plot_coupling_bytes(
    plt, pts, labels, order, colour, fmts, peak, baseline, out_dir, stem, cube
):
    """Stacked bytes per useful FLOP and path, a bar per design point and size, a panel per format pair."""
    chosen = [r for r in pts if r["variant"] == baseline or _peak(r) == peak]
    segs = {id(r): _segments(r) for r in chosen}
    totals = {}
    for sg in segs.values():
        for n, v in sg.items():
            totals[n] = totals.get(n, 0) + v
    names = sorted((n for n in totals if totals[n]), key=lambda n: -totals[n])
    if len(names) > len(PALETTE):
        keep, fold = names[: len(PALETTE) - 1], set(names[len(PALETTE) - 1 :])
        for sg in segs.values():
            sg["other"] = sum(sg.pop(n, 0) for n in fold)
        names = [*keep, "other"]
    sizes = sorted({_size_num(r["size"]) for r in chosen})
    fig, axes = plt.subplots(len(fmts), 1, figsize=(13, 3.8 * len(fmts)), squeeze=False)
    for i, fmt in enumerate(fmts):
        ax = axes[i][0]
        here = [r for r in chosen if f"{r['in_fmt']}/{r['acc_fmt']}" == fmt]
        points = [n for _, n in order if any(labels[id(r)][1] == n for r in here)]
        ticks, xs = [], []
        x = 0
        for s in sizes:
            for n in points:
                r = next(
                    (
                        r
                        for r in here
                        if labels[id(r)][1] == n and _size_num(r["size"]) == s
                    ),
                    None,
                )
                if r is not None:
                    flops = float(r.get("useful_flops") or 0) or 1.0
                    bottom = 0.0
                    for k, seg in enumerate(names):
                        v = segs[id(r)].get(seg, 0) / flops
                        ax.bar(
                            x,
                            v,
                            0.8,
                            bottom=bottom,
                            color=OTHER
                            if seg == "other"
                            else PALETTE[k % len(PALETTE)],
                            label=seg,
                            edgecolor="white",
                            linewidth=1.0,
                        )
                        bottom += v
                    ticks.append(x)
                    xs.append(f"{n}\n{int(s)}{cube}")
                x += 1
            x += 1
        ax.set_xticks(ticks, xs, fontsize=7, rotation=90)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Bytes per useful FLOP")
        ax.set_title(f"{fmt}, peak {peak} FLOPs/cycle" if peak else fmt)
    handles = {}
    for row_axes in axes:
        for h, lbl in zip(*row_axes[0].get_legend_handles_labels()):
            handles.setdefault(lbl, h)
    fig.legend(
        [handles[n] for n in names if n in handles],
        [n for n in names if n in handles],
        loc="upper left",
        bbox_to_anchor=(1.0, 0.95),
        fontsize=9,
    )
    fig.suptitle(
        f"Where each coupling spends its traffic, {stem.rsplit('-', 1)[-1]} (bytes per useful FLOP)",
        fontweight="bold",
    )
    fig.tight_layout()
    _note(
        fig,
        "Whole-run counters over the dwarf's useful FLOPs. memory → tile is AME's tile loads, IME's tile loads "
        "into vector registers and VME's accumulator tile loads; VRF → matrix is the operand traffic IME / VME read "
        "through the vector register file ports; DMA is the Tensix offload's traffic over the bus.",
    )
    return _save(fig, out_dir, f"coupling_bytes-{stem}")


def plot_accuracy(rows, study, out_dir):
    """Normwise error per format pair, a line per dwarf/variant: the median over the
    study's seeds, with a bar up to the worst seed when there are several."""
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
        series.setdefault((r["dwarf"], r["variant"], r["size"]), {}).setdefault(
            f"{r['in_fmt']}/{r['acc_fmt']}", []
        ).append(max(float(r["normwise"]), floor))
    n_seeds = max(len(v) for s_ in series.values() for v in s_.values())
    dwarfs = sorted({k[0] for k in series})
    multi_size = len({(k[0], k[2]) for k in series}) > len(dwarfs)
    multi_var = len({k[1] for k in series}) > 1
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(pairs) + 3), 4.4))
    for key in sorted(series):
        dwarf, variant, size = key
        xs = [i for i, p in enumerate(pairs) if p in series[key]]
        med = [statistics.median(series[key][pairs[i]]) for i in xs]
        worst = [max(series[key][pairs[i]]) for i in xs]
        name = _series_label(dwarf, size, "", multi_size, False) + (
            _variant_suffix(variant) if multi_var else ""
        )
        colour = dwarf_colour(dwarf, dwarfs)
        ax.plot(
            xs,
            med,
            variant_style(variant),
            marker="o",
            color=colour,
            label=name,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
        if n_seeds > 1:
            ax.errorbar(
                xs,
                med,
                yerr=[[0] * len(xs), [w - m for w, m in zip(worst, med)]],
                fmt="none",
                ecolor=colour,
                elinewidth=1,
                capsize=3,
                zorder=2,
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
        "Golden: fp64 on the unrounded inputs, so input rounding counts. Exact results are drawn on the floor."
        + (
            f" Points are the median over {n_seeds} input seeds; bars reach the worst seed."
            if n_seeds > 1
            else " One input seed."
        ),
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
    keys = sorted(bars, key=_key_order)
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


def feature_deltas(rows, study):
    """``[(label, dwarf, delta)]`` for a feature study, sorted best first (most negative delta).

    ``delta`` is ``region_cycles / reference - 1``. In a study with a
    ``kernel.cflags`` axis the reference is the same point with no cflags (the
    default kernel), and each row is one switch. A study without one (a design
    change, e.g. ``ztt_preset``) is compared with the same point in the study
    named ``<prefix>_default``, ``<prefix>`` being the study name up to its
    first ``_``; a study with neither has no deltas.
    """
    key = lambda r: (r["dwarf"], r["variant"], r["size"], r["in_fmt"], r["acc_fmt"])  # noqa: E731
    own = [r for r in _ok(rows, study) if r.get("region_cycles")]
    flags = lambda r: str(r.get("ovr:kernel.cflags") or "").strip("[]()' ")  # noqa: E731
    if any(flags(r) for r in own):
        ref = {key(r): r["region_cycles"] for r in own if not flags(r)}
        pts = [(r, flags(r).replace("'", "")) for r in own if flags(r)]
    else:
        prefix = study.split("_", 1)[0]
        ref_study = f"{prefix}_default"
        if ref_study == study:
            return []
        ref = {
            key(r): r["region_cycles"]
            for r in _ok(rows, ref_study)
            if r.get("region_cycles")
        }
        pts = [(r, study) for r in own]
    out = []
    for r, what in pts:
        base = ref.get(key(r))
        if base:
            label = f"{what}  [{r['dwarf']} n={r['size']} {r['in_fmt']}/{r['acc_fmt']}]"
            label = label.replace('"', "").replace(",", " ")
            out.append((label, r["dwarf"], r["region_cycles"] / base - 1))
    return sorted(out, key=lambda t: t[2])


def plot_feature_ranking(rows, study, out_dir):
    """Ranked horizontal bars: each switch's (or design's) change in region cycles."""
    deltas = feature_deltas(rows, study)
    if not deltas:
        return []
    plt = _plt()
    dwarfs = sorted({d for _, d, _ in deltas})
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(deltas) + 1.6))
    ys = range(len(deltas))
    vals = [100 * v for _, _, v in deltas]
    ax.barh(
        ys,
        vals,
        0.6,
        color=[dwarf_colour(d, dwarfs) for _, d, _ in deltas],
        edgecolor="white",
        linewidth=1.5,
    )
    for y, v in zip(ys, vals):
        ax.annotate(
            f"{v:+.1f}%",
            (v, y),
            xytext=(4 if v >= 0 else -4, 0),
            textcoords="offset points",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.axvline(0, color=INK_2, linewidth=1)
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    pad = 0.15 * max(hi - lo, 1.0)  # room for the value labels at both ends
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
    ax.set_yticks(list(ys), [lbl for lbl, _, _ in deltas], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in region cycles vs the reference (%)")
    ax.set_title(f"Feature ranking: {study}")
    _note(
        fig,
        "Negative is faster. Reference: the default kernel (no cflags) on the same design, "
        "or the <prefix>_default study's point for a design change.",
    )
    return _save(fig, out_dir, f"feature_ranking-{study}")


def _size_parts(size):
    """A size cell (``256x0.25x16``, or a bare number) as a list of numbers."""
    out = []
    for part in str(size).split("x"):
        try:
            out.append(float(part))
        except ValueError:
            return []
    return out


def plot_sparsity(rows, study, out_dir):
    """Speedup over the baseline against block density (a size's second component, log axis).

    For sizes ``(n, density[, nv])`` (spmv): one line per variant, ``nv`` and
    swept override (the matrix:vector ratio); colour follows the ratio, the dash
    follows nv. The dashed line at 1 is parity with the baseline.
    """
    pts = [
        r
        for r in _ok(rows, study)
        if r.get("speedup") is not None and len(_size_parts(r["size"])) >= 2
    ]
    if not pts:
        return []
    plt = _plt()
    ovr_cols = sorted(
        {
            k
            for r in pts
            for k in r
            if k.startswith("ovr:") and r.get(k) not in (None, "")
        }
    )
    ovr_cols = [c for c in ovr_cols if len({r.get(c) for r in pts}) > 1]

    def ovr_key(r):
        return tuple(r.get(c) for c in ovr_cols)

    levels = sorted(
        {ovr_key(r) for r in pts},
        key=lambda t: [float(v) if isinstance(v, (int, float)) else 0 for v in t],
    )
    groups = {}
    for r in pts:
        parts = _size_parts(r["size"])
        nv = int(parts[2]) if len(parts) > 2 else None
        groups.setdefault((r["dwarf"], r["variant"], nv, ovr_key(r)), []).append(
            (parts[1], r["speedup"], r.get("mv_ratio"))
        )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    nvs = sorted({k[2] for k in groups if k[2] is not None}, reverse=True)
    styles = ["-", "--", ":", "-."]
    for key in sorted(
        groups, key=lambda k: (k[0], k[1], -(k[2] or 0), levels.index(k[3]))
    ):
        dwarf, variant, nv, level = key
        line = sorted(groups[key], key=lambda t: t[0])
        colour = PALETTE[levels.index(level) % len(PALETTE)]
        ratios = {t[2] for t in line}
        ratio = (
            [f"ratio {float(next(iter(ratios))):.3g}"]
            if len(ratios) == 1 and None not in ratios and "" not in ratios
            else [f"{c[4:].rsplit('.', 1)[-1]}={v}" for c, v in zip(ovr_cols, level)]
        )
        label = " ".join(
            [dwarf + _variant_suffix(variant)]
            + ([f"nv={nv}"] if nv is not None else [])
            + ratio
        )
        ax.plot(
            [t[0] for t in line],
            [t[1] for t in line],
            styles[nvs.index(nv) % len(styles)] if nv in nvs else "-",
            marker="o",
            color=colour,
            label=label,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
    ax.axhline(1, color=INK_2, linewidth=1, linestyle="--", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Block density (nonzero blocks ÷ block positions)")
    ax.set_ylabel("Speedup over the baseline (×)")
    ax.set_title(f"Speedup vs block density: {study}")
    ax.legend(loc="best", fontsize=8, handlelength=4)
    _note(
        fig,
        "Colour: the swept override (ratio = matrix peak MACs/cycle ÷ vector peak FMAs/cycle); "
        "line style: vectors per product (nv). Grey dashed line: parity.",
    )
    return _save(fig, out_dir, f"sparsity-{study}")


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
    coupling = [
        name
        for name, info in meta["studies"].items()
        if "coupling" in (info.get("figures") or [])
        and (not studies or name in studies)
    ]
    if coupling:  # one set of coupling figures over every study that feeds it
        made = plot_coupling(rows, coupling, out_dir, sweep_dir.name)
        if not made:
            log(f"{', '.join(coupling)}: no data for the coupling figures")
        paths += made
    scaling_studies = [
        name
        for name, info in meta["studies"].items()
        if "scaling" in (info.get("figures") or []) and (not studies or name in studies)
    ]
    if scaling_studies:  # one figure + scaling.csv over every study that feeds it
        from experiments.naiad import scaling

        paths += scaling.make(sweep_dir, scaling_studies, rows, log)[1:]
    for name, info in meta["studies"].items():
        if studies and name not in studies:
            continue
        for fig in info.get("figures") or []:
            if fig in ("coupling", "scaling"):
                continue
            if fig == "crossover":
                made = plot_crossover(
                    rows, cross, name, out_dir, info.get("vector_baseline", "scalar")
                )
            elif fig == "bytes":
                made = plot_bytes(rows, name, out_dir)
            elif fig == "accuracy":
                made = plot_accuracy(rows, name, out_dir)
            elif fig == "feature_ranking":
                made = plot_feature_ranking(rows, name, out_dir)
            elif fig == "sparsity":
                made = plot_sparsity(rows, name, out_dir)
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
