"""``python -m experiments.naiad.costmodel_validate [--sweeps ...] [--extra DIR ...] [--out DIR]``.

Predicted vs simulated region cycles for every point of the experiment
harness's result sets (``experiments/naiad/out/<sweep>/results.csv`` plus the
stored point JSONs for sizes and feature switches), and the three numbers a
compiler cares about:

* **error**: per dwarf and variant, the median and max relative error
  ``pred / sim - 1``;
* **decision accuracy**: for every *decision* -- one problem (dwarf, size,
  formats, switches) on one design, with every variant the result set ran
  there -- whether :func:`~framework.naiad.costmodel.choose` picks the variant
  the simulator measured fastest, and the cycles lost when it does not
  (``sim[chosen] / sim[best] - 1``); also the coarser *reformulate or not*
  decision (best matrix variant vs best baseline);
* **crossovers**: per crossover group, the matrix:vector ratio where the
  predicted and the simulated speedup cross 1 (the harness's own
  log-interpolation, :func:`experiments.naiad.metrics.interpolate_crossover`).

Writes ``predictions.csv`` (one row per point), ``summary.csv`` (per dwarf and
variant), ``decisions.csv``, ``crossovers.csv`` and the figure
``costmodel_pred_vs_sim`` (PNG + SVG, the harness's palette) to ``--out``.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from framework.naiad import costmodel
from framework.naiad.costmodel.dwarfs.common import Unsupported

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SWEEPS = ("talk_tier1", "talk_tier2", "feature_ranking")


def _size(value):
    if isinstance(value, list):
        return tuple(_size(v) for v in value)
    return value


def load_points(out_dir, source=None):
    """Rows of ``out_dir/results.csv`` joined with their point JSON (size, cflags, design path)."""
    out_dir = Path(out_dir)
    rows = []
    with open(out_dir / "results.csv") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok" or not r.get("region_cycles"):
                continue
            pj = out_dir / "points" / f"{r['point_id']}.json"
            if not pj.exists():
                continue
            p = json.loads(pj.read_text())
            rows.append(
                {
                    "source": source or out_dir.name,
                    "study": r["study"],
                    "dwarf": r["dwarf"],
                    "variant": r["variant"],
                    "size": _size(p["size"]),
                    "size_str": r["size"],
                    "in_fmt": r["in_fmt"],
                    "acc_fmt": r["acc_fmt"],
                    "seed": int(r["seed"] or 0),
                    "cflags": tuple(p.get("extra_cflags") or ()),
                    "design": str(out_dir / r["design_file"]),
                    "design_hash": r["design_hash"],
                    "mv_ratio": float(r["mv_ratio"]) if r.get("mv_ratio") else None,
                    "sim": int(r["region_cycles"]),
                    "commit": p.get("git_commit", ""),
                    "point_id": r["point_id"],
                }
            )
    return rows


def predict_row(row, model="schedule"):
    """Predicted cycles for one row, or ``(None, reason)``."""
    try:
        mod = costmodel.dwarf_module(row["dwarf"])
    except (KeyError, ModuleNotFoundError) as e:
        return None, str(e)
    switches = {}
    if row["cflags"]:
        parse = getattr(mod, "switches_from_cflags", None)
        switches = parse(row["cflags"]) if parse else None
        if switches is None:
            return None, f"switches {row['cflags']} not modelled"
    if row["variant"] not in mod.VARIANTS:
        return None, f"variant {row['variant']} not modelled"
    try:
        pred = costmodel.predict(
            row["dwarf"],
            row["size"],
            row["in_fmt"],
            row["acc_fmt"],
            row["design"],
            [row["variant"]],
            switches,
            model,
        )
    except Unsupported as e:
        return None, f"unsupported: {e}"
    if row["variant"] not in pred:
        return None, "unsupported"
    return pred[row["variant"]], ""


def evaluate(rows, model="schedule"):
    out, skipped = [], defaultdict(int)
    for r in rows:
        pred, why = predict_row(r, model)
        if pred is None:
            skipped[(r["dwarf"], r["variant"], why)] += 1
            continue
        out.append({**r, "pred": pred, "rel_err": pred / r["sim"] - 1})
    return out, skipped


def summarise(preds, key=("dwarf", "variant")):
    groups = defaultdict(list)
    for p in preds:
        groups[tuple(p[k] for k in key)].append(p)
    table = []
    for k, ps in sorted(groups.items()):
        errs = [abs(p["rel_err"]) for p in ps]
        signed = [p["rel_err"] for p in ps]
        table.append(
            {
                **dict(zip(key, k)),
                "n": len(ps),
                "median_abs_err": statistics.median(errs),
                "max_abs_err": max(errs),
                "mean_signed_err": statistics.fmean(signed),
            }
        )
    return table


def _unique_timing(preds):
    """One row per (source, point): seeds and studies sharing a point time identically."""
    seen, out = set(), []
    for p in preds:
        k = (p["source"], p["point_id"])
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def decisions(preds):
    """Decision instances: a problem on one design with every variant that ran there.

    Matrix variants are tied to their design (hash); baselines (scalar / rvv) are
    independent of the matrix unit, so within a study they join every matrix
    design point of the same problem.
    """
    by_problem = defaultdict(lambda: {"matrix": defaultdict(dict), "base": {}})
    for p in preds:
        key = (
            p["source"],
            p["study"],
            p["dwarf"],
            str(p["size"]),
            p["in_fmt"],
            p["acc_fmt"],
            p["cflags"],
        )
        mod = costmodel.dwarf_module(p["dwarf"])
        if p["variant"] in getattr(mod, "redundant", lambda _m: set())(
            costmodel.machine(p["design"])
        ):
            continue  # the same code as another variant on this design (one hart's ame_mc)
        if costmodel.is_matrix_variant(p["variant"]):
            by_problem[key]["matrix"][p["design_hash"]][p["variant"]] = p
        else:
            by_problem[key]["base"][p["variant"]] = p
    out = []
    for key, g in by_problem.items():
        designs = g["matrix"] or {None: {}}
        for dh, mats in designs.items():
            cands = {**g["base"], **mats}
            if len(cands) < 2:
                continue
            sim = {v: c["sim"] for v, c in cands.items()}
            pred = {v: c["pred"] for v, c in cands.items()}
            best_sim = min(sim, key=sim.get)
            best_pred = min(pred, key=pred.get)
            lost = sim[best_pred] / sim[best_sim] - 1
            mat_s = {v: s for v, s in sim.items() if costmodel.is_matrix_variant(v)}
            base_s = {
                v: s for v, s in sim.items() if not costmodel.is_matrix_variant(v)
            }
            reform = None
            if mat_s and base_s:
                mat_p = {v: pred[v] for v in mat_s}
                base_p = {v: pred[v] for v in base_s}
                reform = (min(mat_s.values()) < min(base_s.values())) == (
                    min(mat_p.values()) < min(base_p.values())
                )
            out.append(
                {
                    "source": key[0],
                    "study": key[1],
                    "dwarf": key[2],
                    "size": key[3],
                    "in_fmt": key[4],
                    "acc_fmt": key[5],
                    "cflags": " ".join(key[6]),
                    "design_hash": dh or "",
                    "mv_ratio": next(
                        (c["mv_ratio"] for c in mats.values() if c["mv_ratio"]), None
                    ),
                    "variants": " ".join(sorted(cands)),
                    "best_sim": best_sim,
                    "best_pred": best_pred,
                    "correct": best_sim == best_pred,
                    "cycles_lost": lost,
                    "sim_margin": sorted(sim.values())[1] / sorted(sim.values())[0],
                    "reformulate_correct": reform,
                }
            )
    return out


def crossovers(preds, baseline="rvv"):
    """Predicted vs simulated crossover ratio per group swept over the matrix peak."""
    from experiments.naiad.metrics import interpolate_crossover

    groups = defaultdict(list)
    bases = {}
    for p in preds:
        key = (
            p["source"],
            p["study"],
            p["dwarf"],
            str(p["size"]),
            p["in_fmt"],
            p["acc_fmt"],
            p["cflags"],
        )
        if p["variant"] == baseline:
            bases[key] = p
        elif costmodel.is_matrix_variant(p["variant"]) and p["mv_ratio"]:
            groups[key + (p["variant"],)].append(p)
    out = []
    for gk, ps in groups.items():
        base = bases.get(gk[:-1])
        if base is None or len({p["mv_ratio"] for p in ps}) < 3:
            continue
        ps.sort(key=lambda p: p["mv_ratio"])
        ratios = [p["mv_ratio"] for p in ps]
        s_sim = [base["sim"] / p["sim"] for p in ps]
        s_pred = [base["pred"] / p["pred"] for p in ps]
        x_sim, st_sim = interpolate_crossover(ratios, s_sim)
        x_pred, st_pred = interpolate_crossover(ratios, s_pred)
        out.append(
            {
                "source": gk[0],
                "study": gk[1],
                "dwarf": gk[2],
                "size": gk[3],
                "in_fmt": gk[4],
                "acc_fmt": gk[5],
                "variant": gk[7],
                "baseline": baseline,
                "points": len(ps),
                "sim_crossover": x_sim,
                "sim_status": st_sim,
                "pred_crossover": x_pred,
                "pred_status": st_pred,
                "ratio_err": (x_pred / x_sim - 1) if x_sim and x_pred else None,
                "sim_speedup_at_max": s_sim[-1],
                "pred_speedup_at_max": s_pred[-1],
            }
        )
    return out


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()}
            )


def _marker(variant):
    if variant == "scalar":
        return "s"
    if variant == "rvv":
        return "^"
    return "o" if variant.startswith("ame") else "D"


def figure(panels, path):
    """Predicted vs simulated, log-log, one panel per model; colour = dwarf (the harness palette)."""
    from experiments.naiad import plot

    plt = plot._plt()
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.4 * len(panels), 5.2), sharex=True, sharey=True
    )
    axes = list(axes) if len(panels) > 1 else [axes]
    allp = [p for _, ps in panels for p in ps]
    lo = min(min(p["sim"], p["pred"]) for p in allp) / 1.5
    hi = max(max(p["sim"], p["pred"]) for p in allp) * 1.5
    dwarfs = sorted(
        {p["dwarf"] for p in allp},
        key=lambda d: plot.DWARF_ORDER.index(d) if d in plot.DWARF_ORDER else 99,
    )
    for ax, (title, preds) in zip(axes, panels):
        ax.plot([lo, hi], [lo, hi], color=plot.INK_2, lw=0.8, zorder=1)
        for band in (1.1, 1 / 1.1):
            ax.plot(
                [lo, hi],
                [lo * band, hi * band],
                color=plot.INK_2,
                lw=0.6,
                ls=":",
                zorder=1,
            )
        for d in dwarfs:
            ps = [p for p in preds if p["dwarf"] == d]
            for mk in ("o", "^", "s", "D"):
                q = [p for p in ps if _marker(p["variant"]) == mk]
                if q:
                    ax.scatter(
                        [p["sim"] for p in q],
                        [p["pred"] for p in q],
                        s=16,
                        marker=mk,
                        color=plot.dwarf_colour(d),
                        edgecolors="white",
                        linewidths=0.4,
                        alpha=0.9,
                        zorder=2,
                    )
        errs = sorted(abs(p["rel_err"]) for p in preds)
        med = errs[len(errs) // 2] if errs else 0
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("simulated region cycles")
        ax.set_title(f"{title}: median |error| {med:.1%}")
    axes[0].set_ylabel("predicted region cycles")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], ls="", marker="o", color=plot.dwarf_colour(d), label=d)
        for d in dwarfs
    ]
    handles += [
        Line2D([], [], ls="", marker=mk, color=plot.INK_2, label=lab)
        for mk, lab in (("o", "AME"), ("^", "RVV"), ("s", "scalar"), ("D", "IME / VME"))
    ]
    axes[-1].legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8
    )
    fig.suptitle(
        "Dwarf cost model vs the Naiad simulator (dotted: +/-10%)", fontweight="bold"
    )
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def report(name, timing, skipped=None):
    summary = summarise(timing)
    dec = decisions(timing)
    xo = crossovers(timing) + crossovers(timing, "scalar")
    print(f"\n=== {name} model: {len(timing)} unique points")
    for (d, v, why), n in sorted((skipped or {}).items()):
        print(f"  skipped {d}/{v}: {n} ({why})")
    print(f"{'dwarf':12s}{'variant':11s}{'n':>5s}{'median':>9s}{'max':>9s}{'bias':>9s}")
    for s in summary:
        print(
            f"{s['dwarf']:12s}{s['variant']:11s}{s['n']:5d}{s['median_abs_err']:9.1%}{s['max_abs_err']:9.1%}{s['mean_signed_err']:+9.1%}"
        )
    if dec:
        ok = sum(d["correct"] for d in dec)
        lost = [d["cycles_lost"] for d in dec if not d["correct"]]
        ref = [
            d["reformulate_correct"]
            for d in dec
            if d["reformulate_correct"] is not None
        ]
        print(
            f"decisions: {ok}/{len(dec)} correct ({ok / len(dec):.1%}); when wrong, cycles lost median "
            f"{statistics.median(lost) if lost else 0:.1%}, max {max(lost) if lost else 0:.1%}; "
            f"reformulate-or-not {sum(ref)}/{len(ref)}"
        )
    for x in xo:
        if "bracketed" in (x["sim_status"], x["pred_status"]):
            print(
                f"  crossover {x['source']}:{x['study']} {x['dwarf']}/{x['variant']}@{x['size']} vs {x['baseline']}: "
                f"sim {_fmt(x['sim_crossover'], x['sim_status'])} pred {_fmt(x['pred_crossover'], x['pred_status'])}"
            )
    return summary, dec, xo


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.naiad.costmodel_validate")
    ap.add_argument(
        "--sweeps",
        nargs="*",
        default=list(DEFAULT_SWEEPS),
        help="sweep names under experiments/naiad/out",
    )
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="further sweep output dirs (holding results.csv)",
    )
    ap.add_argument(
        "--out", default=str(REPO / "experiments" / "naiad" / "out" / "costmodel")
    )
    ap.add_argument("--models", nargs="*", default=["schedule", "bound"])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    rows = []
    for s in args.sweeps:
        d = REPO / "experiments" / "naiad" / "out" / s
        if (d / "results.csv").exists():
            rows += load_points(d, s)
    for d in args.extra:
        rows += load_points(Path(d))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} rows from {len(args.sweeps) + len(args.extra)} result sets")
    panels = []
    for model in args.models:
        preds, skipped = evaluate(rows, model)
        timing = _unique_timing(preds)
        sfx = "" if model == "schedule" else f"_{model}"
        summary, dec, xo = report(model, timing, skipped)
        write_csv(out / f"predictions{sfx}.csv", preds)
        write_csv(out / f"summary{sfx}.csv", summary)
        write_csv(out / f"decisions{sfx}.csv", dec)
        write_csv(out / f"crossovers{sfx}.csv", xo)
        panels.append((f"{model} model", timing))
    if not args.no_figure and panels:
        figure(panels, out / "costmodel_pred_vs_sim")
    return 0


def _fmt(x, status):
    return f"{x:.3g}" if x is not None and not math.isnan(x) else status


if __name__ == "__main__":
    sys.exit(main())
