"""``python -m experiments.naiad.run <sweep.yaml> [--jobs N] [--filter PAT] [--dry-run] [--quick]``.

Expands the sweep (:mod:`experiments.naiad.sweep`), writes each derived design
to ``out/<sweep>/designs/<hash>.yaml``, then builds and runs every point through
:func:`framework.naiad.dwarfs.run_point` in a pool of processes. Each finished
point is stored as ``out/<sweep>/points/<point_id>.json`` the moment it ends, so
an interrupted sweep resumes where it stopped: points with a stored result are
skipped (``--force`` reruns everything, ``--retry-failed`` reruns the failed
ones). Builds are cached by key in ``out/<sweep>/build/`` (or ``--build-root``).

Finally it writes ``results.csv`` (one row per study and point; columns in
:mod:`experiments.naiad.metrics`), ``crossover.csv`` (one row per crossover
group) and ``meta.json``. ``--no-run`` only rewrites those from stored results,
for instance after changing a study's ``ratio.vector`` model.
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from experiments.naiad import metrics
from experiments.naiad.sweep import (
    OUT,
    REPO,
    DesignFactory,
    Sweep,
    expand,
    load_sweep,
    size_str,
)

SWEEPS = Path(__file__).resolve().parent / "sweeps"
VECTOR_NOTE = (
    "The 'vector' baseline is the scalar F/D variant until RVV lands (Phase C); "
    "set a study's ratio.vector to 'rvv' (and its baseline.variant to the RVV variant) to switch."
)


def resolve_sweep(arg):
    path = Path(arg)
    if path.exists():
        return path
    for candidate in (SWEEPS / arg, SWEEPS / f"{arg}.yaml"):
        if candidate.exists():
            return candidate
    raise SystemExit(f"no sweep {arg!r} (not a file, nor in {SWEEPS})")


def git_commit():
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def label(point):
    where = f"{point.dwarf}/{point.variant}@{size_str(point.size)}"
    return f"{point.study}:{where}:{point.in_fmt}/{point.acc_fmt}#{point.design_hash}"


def matches(point, patterns):
    text = label(point)
    return all(
        fnmatch.fnmatch(text, p if any(ch in p for ch in "*?[") else f"*{p}*")
        for p in patterns
    )


# -- the worker ----------------------------------------------------------------------------

_SPECS = {}


def _spec(name, root):
    from framework.naiad.dwarfs import load_spec

    key = (name, root)
    if key not in _SPECS:
        _SPECS[key] = load_spec(name, root)
    return _SPECS[key]


def run_one(point_dict, result_path, build_root, commit):
    """Run one point and store its result JSON; returns the result. Never raises."""
    from framework.naiad.dwarfs import run_point

    start = time.time()
    result = {
        **point_dict,
        "size_str": size_str(point_dict["size"]),
        "design_file": os.path.relpath(
            point_dict["design_path"], Path(result_path).parents[1]
        ),
        "git_commit": commit,
        "status": "error",
        "message": "",
    }
    try:
        spec = _spec(point_dict["dwarf"], point_dict["dwarfs_root"])
        size = point_dict["size"]
        size = tuple(size) if isinstance(size, list) else size
        r = run_point(
            spec,
            point_dict["variant"],
            size,
            point_dict["in_fmt"],
            point_dict["acc_fmt"],
            design=point_dict["design_path"],
            seed=point_dict["seed"],
            build_root=build_root,
            count_all=point_dict["count_all"],
            extra_cflags=tuple(point_dict["extra_cflags"]),
        )
        result.update(
            status="ok" if r.exit_code == 0 else "exit",
            exit_code=r.exit_code,
            cycles=r.cycles,
            region_cycles=r.region_cycles,
            passed=bool(r.passed),
            error={k: {m: float(v) for m, v in e.items()} for k, e in r.error.items()},
            tolerance=float(
                spec.tolerance(point_dict["in_fmt"], point_dict["acc_fmt"])
            ),
            useful_flops=int(spec.useful_flops(size)),
            counters={k: v for k, v in r.counters.items() if not k.startswith("hart")},
        )
        if r.exit_code != 0:
            result["message"] = f"exit code {r.exit_code}: {r.console[-400:]}"
    except Exception as e:
        text = str(e) or type(e).__name__
        result["message"] = f"{type(e).__name__}: {text[-2000:]}"
        result["traceback"] = traceback.format_exc()[-4000:]
    result["wall_s"] = round(time.time() - start, 3)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=1, default=_json_default))
    tmp.replace(path)
    return result


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


# -- reporting -------------------------------------------------------------------------------


def collect(sweep, points, out_dir):
    """Rows (with speedups) and crossover records from every stored point result."""
    rows = []
    by_study = {s.name: s for s in sweep.studies}
    for p in points:
        path = out_dir / "points" / f"{p.point_id}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        result["design_path"] = str(out_dir / result["design_file"])
        rows.append(metrics.flatten(result, by_study[p.study]))
    cross = []
    for study in sweep.studies:
        metrics.add_speedups(rows, study)
        cross += metrics.crossovers(rows, study)
    return rows, cross


def write_reports(sweep, points, out_dir, commit):
    rows, cross = collect(sweep, points, out_dir)
    metrics.write_csv(out_dir / "results.csv", rows)
    cross_cols = [
        "study",
        "dwarf",
        "variant",
        "size",
        "in_fmt",
        "acc_fmt",
        "group",
        "crossover_ratio",
        "status",
        "points",
        "min_ratio",
        "max_ratio",
        "speedup_at_min",
        "speedup_at_max",
        "vector_baseline",
    ]
    metrics.write_csv(out_dir / "crossover.csv", cross, cross_cols)
    meta = {
        "sweep": sweep.name,
        "sweep_file": str(sweep.path),
        "git_commit": commit,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "points": len(points),
        "rows": len(rows),
        "note": VECTOR_NOTE,
        "studies": {
            s.name: {
                "vector_baseline": s.ratio.get("vector", "scalar"),
                "baseline": s.baseline,
                "figures": s.figures,
                "description": s.description,
            }
            for s in sweep.studies
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return rows, cross


# -- main ----------------------------------------------------------------------------------------


def run_sweep(
    sweep_path,
    *,
    jobs=1,
    filters=(),
    dry_run=False,
    quick=False,
    force=False,
    retry_failed=False,
    no_run=False,
    out_root=None,
    build_root=None,
    log=print,
):
    """Expand, run and report one sweep; returns ``(sweep, points, rows, crossovers)``."""
    sweep = load_sweep(sweep_path, quick=quick)
    if quick:
        sweep = Sweep(f"{sweep.name}-quick", sweep.path, sweep.studies)
    out_dir = Path(out_root or OUT) / sweep.name
    points = expand(sweep, DesignFactory(out_dir), log=log)
    points = [p for p in points if matches(p, filters)]
    by_id = {}
    for p in points:
        by_id.setdefault(p.point_id, p)
    commit = git_commit()
    if dry_run:
        for p in points:
            log(
                f"{p.point_id}  {label(p)}  {p.overrides or ''}{' ' + str(p.normalise) if p.normalise else ''}"
            )
        log(
            f"{len(points)} rows, {len(by_id)} unique points; designs in {out_dir / 'designs'}"
        )
        return sweep, points, [], []
    results_dir = out_dir / "points"
    todo = []
    for pid, p in by_id.items():
        path = results_dir / f"{pid}.json"
        if path.exists() and not force:
            if not retry_failed or json.loads(path.read_text()).get("status") == "ok":
                continue
        todo.append(p)
    if not no_run:
        log(
            f"{sweep.name}: {len(by_id)} points, {len(by_id) - len(todo)} already done, {len(todo)} to run ({jobs} jobs)"
        )
        build_root = Path(build_root or out_dir / "build")
        args = [
            (
                asdict(p),
                str(results_dir / f"{p.point_id}.json"),
                str(build_root),
                commit,
            )
            for p in todo
        ]
        done = 0

        def report(p, r):
            nonlocal done
            done += 1
            what = r["status"]
            if r["status"] == "ok":
                what = f"{'PASS' if r['passed'] else 'FAIL'} region={r['region_cycles']} cycles={r['cycles']}"
            else:
                what += f" {r['message'].splitlines()[0][:160] if r['message'] else ''}"
            log(f"[{done}/{len(todo)}] {label(p)}: {what} ({r['wall_s']} s)")

        if jobs <= 1:
            for p, a in zip(todo, args):
                report(p, run_one(*a))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_one, *a): p for p, a in zip(todo, args)}
                for f in as_completed(futures):
                    report(futures[f], f.result())
    rows, cross = write_reports(sweep, points, out_dir, commit)
    log(
        f"wrote {out_dir / 'results.csv'} ({len(rows)} rows) and {out_dir / 'crossover.csv'}"
    )
    return sweep, points, rows, cross


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m experiments.naiad.run", description=__doc__.split("\n\n")[1]
    )
    ap.add_argument(
        "sweep", help="sweep YAML (a path, or a name in experiments/naiad/sweeps/)"
    )
    ap.add_argument(
        "--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    ap.add_argument(
        "--filter",
        action="append",
        default=[],
        help="keep points whose label 'study:dwarf/variant@size:in/acc#design' matches (glob, or substring); repeatable",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list the points and write the designs, run nothing",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="the CI profile: smallest sizes (and each study's quick: keys)",
    )
    ap.add_argument(
        "--force", action="store_true", help="rerun points that already have a result"
    )
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="rerun points whose stored result is not ok",
    )
    ap.add_argument(
        "--no-run",
        action="store_true",
        help="only rewrite results.csv / crossover.csv from stored results",
    )
    ap.add_argument("--out", help=f"output root (default {OUT})")
    ap.add_argument(
        "--build-root", help="build cache directory (default out/<sweep>/build)"
    )
    args = ap.parse_args(argv)
    run_sweep(
        resolve_sweep(args.sweep),
        jobs=args.jobs,
        filters=args.filter,
        dry_run=args.dry_run,
        quick=args.quick,
        force=args.force,
        retry_failed=args.retry_failed,
        no_run=args.no_run,
        out_root=args.out,
        build_root=args.build_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
