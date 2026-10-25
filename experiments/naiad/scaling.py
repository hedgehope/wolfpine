"""``python -m experiments.naiad.scaling <sweep>``: the multicore scaling table and figure for one sweep.

For every study that lists the ``scaling`` figure (``sweeps/multicore_scaling.yaml``)
this reads ``results.csv`` and the stored point results, and writes

``scaling.csv``
    one row per ``*_mc`` point: ``harts`` (the ``cores.count`` override, 1 when
    blank), ``cache`` (``none`` / ``shared`` / ``private``, from a ``blocks.l1d``
    override), ``l1_bandwidth`` (``timing.memory.l1.bandwidth``, blank =
    unlimited), ``speedup`` over the same variant on 1 hart with the same
    cache, L1 port, size and formats, ``efficiency`` = speedup / harts,
    ``speedup_vs_single`` over the single-hart variant the ``_mc`` one splits
    (``ame`` / ``rvv``) at the same cache and port, the normwise error of both,
    the per-hart mean of every ``timing.stall_*`` counter (``stall_<reason>_per_hart``)
    and of the instructions retired, and the cache's coherence counters summed
    over the harts (``l1d.{misses,invalidations,snoop_writebacks,upgrades,coherence_cycles}``).
    Counters cover the whole run (hart 0's setup included), cycles only the
    timed region; the per-hart stall means are what each hart waited, of which
    the barrier wait shows up as ``mem`` (spinning loads) and ``mem_raw``.
``figures/scaling-<sweep>.{png,svg}``
    speedup against harts, log-log, one panel per (cache, L1 port)
    configuration, one line per dwarf / variant / formats: colour = dwarf
    (:func:`experiments.naiad.plot.dwarf_colour`), solid = ``ame_mc``, dashed =
    ``rvv_mc``, dash-dot = a 16-bit input format; the grey dotted line is ideal
    (speedup = harts).
"""

import argparse
import json
import sys
from pathlib import Path

from experiments.naiad import metrics

HARTS_KEY = "ovr:cores.count"
CACHE_KEY = "ovr:blocks.l1d"
BW_KEY = "ovr:timing.memory.l1.bandwidth"
COHERENCE = (
    "misses",
    "invalidations",
    "snoop_writebacks",
    "upgrades",
    "coherence_cycles",
)
STALLS = (
    "cycles",
    "mem",
    "mem_raw",
    "matrix_busy",
    "matrix_raw",
    "vector_busy",
    "vector_raw",
    "raw",
    "structural",
    "branch",
)
COLUMNS = [
    "study",
    "dwarf",
    "variant",
    "size",
    "in_fmt",
    "acc_fmt",
    "cache",
    "l1_bandwidth",
    "harts",
    "region_cycles",
    "base_region_cycles",
    "speedup",
    "efficiency",
    "single_variant",
    "single_region_cycles",
    "speedup_vs_single",
    "normwise",
    "single_normwise",
    "passed",
    "instret_per_hart",
    *[f"stall_{s}_per_hart" for s in STALLS],
    *[f"l1d.{c}" for c in COHERENCE],
    "point_id",
]


def harts_of(row):
    v = row.get(HARTS_KEY)
    return 1 if v in (None, "") else int(v)


def cache_of(row):
    v = row.get(CACHE_KEY)
    if v in (None, ""):
        return "none"
    params = (json.loads(v) if isinstance(v, str) else v).get("params") or {}
    return "private" if params.get("private") else "shared"


def bandwidth_of(row):
    v = row.get(BW_KEY)
    return "" if v in (None, "") else v


def _config(row):
    return (
        row["study"],
        row["dwarf"],
        row["size"],
        row["in_fmt"],
        row["acc_fmt"],
        cache_of(row),
        bandwidth_of(row),
    )


def _counters(points_dir, point_id):
    path = Path(points_dir) / f"{point_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("counters") or {}


def scaling_records(rows, studies, points_dir):
    """One record per ok ``*_mc`` row of ``studies`` (see the module docstring)."""
    ok = [
        r
        for r in rows
        if r["study"] in studies and r["status"] == "ok" and r.get("region_cycles")
    ]
    by_key = {}
    for r in ok:
        by_key.setdefault((*_config(r), r["variant"]), {})[harts_of(r)] = r
    out = []
    for r in ok:
        if not r["variant"].endswith("_mc"):
            continue
        h = harts_of(r)
        series = by_key[(*_config(r), r["variant"])]
        base = series.get(1)
        single_variant = r["variant"][: -len("_mc")]
        single = by_key.get((*_config(r), single_variant), {}).get(1)
        c = _counters(points_dir, r["point_id"])
        rec = {
            "study": r["study"],
            "dwarf": r["dwarf"],
            "variant": r["variant"],
            "size": r["size"],
            "in_fmt": r["in_fmt"],
            "acc_fmt": r["acc_fmt"],
            "cache": cache_of(r),
            "l1_bandwidth": bandwidth_of(r),
            "harts": h,
            "region_cycles": r["region_cycles"],
            "normwise": r.get("normwise"),
            "passed": r.get("passed"),
            "single_variant": single_variant,
            "point_id": r["point_id"],
        }
        if base:
            rec["base_region_cycles"] = base["region_cycles"]
            rec["speedup"] = base["region_cycles"] / r["region_cycles"]
            rec["efficiency"] = rec["speedup"] / h
        if single:
            rec["single_region_cycles"] = single["region_cycles"]
            rec["speedup_vs_single"] = single["region_cycles"] / r["region_cycles"]
            rec["single_normwise"] = single.get("normwise")
        if c:
            rec["instret_per_hart"] = c.get("instret", 0) / h
            for s in STALLS:
                rec[f"stall_{s}_per_hart"] = c.get(f"timing.stall_{s}", 0) / h
            for k in COHERENCE:
                if f"l1d.{k}" in c:
                    rec[f"l1d.{k}"] = c[f"l1d.{k}"]
        out.append(rec)
    out.sort(
        key=lambda x: (
            x["study"],
            x["dwarf"],
            x["variant"],
            x["in_fmt"],
            x["cache"],
            str(x["l1_bandwidth"]),
            x["harts"],
        )
    )
    return out


def write_scaling_csv(sweep_dir, studies, rows=None):
    """``scaling.csv`` in ``sweep_dir`` for ``studies``; returns ``(path, records)``."""
    sweep_dir = Path(sweep_dir)
    rows = rows if rows is not None else metrics.read_csv(sweep_dir / "results.csv")
    records = scaling_records(rows, set(studies), sweep_dir / "points")
    path = metrics.write_csv(sweep_dir / "scaling.csv", records, COLUMNS)
    return path, records


def _panel_label(cache, bw):
    port = "L1 unlimited" if bw in ("", None) else f"L1 {bw:g} B/cycle shared"
    cache_txt = {
        "none": "no cache",
        "shared": "shared L1 cache",
        "private": "private MSI L1s",
    }[cache]
    return f"{cache_txt}, {port}"


def plot_scaling(records, out_dir, stem):
    """Speedup vs harts, one panel per (cache, L1 port) configuration; returns the paths written."""
    from experiments.naiad import plot as P

    pts = [r for r in records if r.get("speedup")]
    if not pts:
        return []
    plt = P._plt()
    order = ["none", "shared", "private"]
    configs = sorted(
        {(r["cache"], r["l1_bandwidth"]) for r in pts},
        key=lambda c: (c[1] not in ("", None), -(c[1] or 0), order.index(c[0])),
    )
    dwarfs = sorted({r["dwarf"] for r in pts})
    harts = sorted({r["harts"] for r in pts})
    ncols = min(3, len(configs))
    nrows = -(-len(configs) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False, sharey=True
    )
    handles = {}
    for ax, (cache, bw) in zip(axes.flat, configs):
        ax.plot(harts, harts, color=P.INK_2, linestyle=":", linewidth=1, zorder=1)
        series = {}
        for r in pts:
            if (r["cache"], r["l1_bandwidth"]) == (cache, bw):
                series.setdefault(
                    (r["dwarf"], r["variant"], r["in_fmt"], r["acc_fmt"]), []
                ).append(r)
        for (dwarf, variant, fi, fa), members in sorted(series.items()):
            members.sort(key=lambda r: r["harts"])
            style = (
                "-."
                if fi in ("bf16", "fp16")
                else ("--" if variant.startswith("rvv") else "-")
            )
            label = (
                dwarf
                + ("" if variant == "ame_mc" else f" ({variant})")
                + ("" if fi == "fp32" else f" {fi}")
            )
            (line,) = ax.plot(
                [r["harts"] for r in members],
                [r["speedup"] for r in members],
                color=P.dwarf_colour(dwarf, dwarfs),
                linestyle=style,
                marker="o",
                markersize=4,
                zorder=3,
            )
            handles.setdefault(label, line)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(harts, [str(h) for h in harts])
        top = max(max(harts), max(r["speedup"] for r in pts))
        ticks = [
            t for t in (0.25, 0.5, 1, 2, 4, 8, 16) if t <= top * 1.01 or t <= max(harts)
        ]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_title(_panel_label(cache, bw), fontsize=10)
        ax.set_xlabel("Harts")
    for ax in axes[:, 0]:
        ax.set_ylabel("Speedup over 1 hart (×)")
    for ax in list(axes.flat)[len(configs) :]:
        ax.axis("off")
    fig.legend(
        list(handles.values()),
        list(handles.keys()),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        handlelength=3,
    )
    fig.suptitle(
        "Multicore scaling of the dwarfs (each hart has its own matrix and vector unit)",
        fontweight="bold",
    )
    fig.tight_layout()
    P._note(
        fig,
        "Speedup = region cycles of the same *_mc kernel on 1 hart (same cache and L1 port) over N harts. "
        "Grey dotted: ideal. Colour: dwarf; solid: ame_mc, dashed: rvv_mc, dash-dot: bf16 inputs.",
    )
    return P._save(fig, Path(out_dir), f"scaling-{stem}")


def make(sweep_dir, studies, rows=None, log=print):
    """``scaling.csv`` and the figure for ``studies`` of the sweep in ``sweep_dir``; returns the paths written."""
    sweep_dir = Path(sweep_dir)
    path, records = write_scaling_csv(sweep_dir, studies, rows)
    log(f"wrote {path} ({len(records)} rows)")
    return [path, *plot_scaling(records, sweep_dir / "figures", sweep_dir.name)]


def main(argv=None):
    from experiments.naiad.run import resolve_sweep
    from experiments.naiad.sweep import OUT, load_sweep

    ap = argparse.ArgumentParser(
        prog="python -m experiments.naiad.scaling", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument(
        "sweep", help="sweep name (out/<name>), sweep YAML, or a sweep output directory"
    )
    ap.add_argument(
        "--quick", action="store_true", help="the --quick run of a sweep YAML"
    )
    args = ap.parse_args(argv)
    target = Path(args.sweep)
    if target.is_dir():
        sweep_dir = target
    else:
        name = load_sweep(resolve_sweep(args.sweep)).name
        sweep_dir = OUT / (f"{name}-quick" if args.quick else name)
    meta = json.loads((sweep_dir / "meta.json").read_text())
    studies = [
        n
        for n, info in meta["studies"].items()
        if "scaling" in (info.get("figures") or [])
    ]
    for p in make(sweep_dir, studies)[1:]:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
