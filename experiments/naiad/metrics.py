"""Rows, derived metrics and the crossover: from stored point results to ``results.csv``.

**Speedup** of a row is the baseline variant's ``region_cycles`` over the row's,
at the same study, dwarf, size and seed, the baseline's formats (the study's
``baseline.formats``, or ``same``) and every override the baseline also took
(keys that do not apply to the baseline, through ``only:``, are wildcards).

**The matrix-to-vector throughput ratio** is the matrix unit's peak MACs per
cycle *for the row's input format* (``m*n*k / mma_interval``, so it follows
``macs_per_cycle``, ``format_passes`` and ``mma_ii``) over the vector
baseline's peak FMAs per cycle. The baseline model is the study's
``ratio.vector``: ``scalar`` (the default, F/D: FMA is pipelined, so the peak is
``timing.scalar.issue_width`` FMAs a cycle) until RVV lands in Phase C; then
``rvv`` (one FMA per element of a ``vector.vlen``-bit register per cycle) or a
fixed number. Switching is that one key, and it needs no re-run: the ratio is
recomputed from the stored designs whenever the CSV is written.

**The crossover** of a group with at least two distinct ratios (same study, dwarf, variant, size, formats and
every override not in ``ratio_keys``) is the ratio at which speedup reaches 1,
interpolated linearly in speedup against log(ratio) between the first bracketing
pair of points sorted by ratio.
"""

import csv
import math
from functools import lru_cache
from pathlib import Path

from framework.naiad.design import load_design

FP64_UNIT_ROUNDOFF = 2.0**-53

FIXED_COLUMNS = [
    "study",
    "dwarf",
    "variant",
    "size",
    "in_fmt",
    "acc_fmt",
    "seed",
    "status",
    "passed",
    "exit_code",
    "cycles",
    "region_cycles",
    "speedup",
    "baseline_point",
    "mv_ratio",
    "matrix_peak_macs_per_cycle",
    "vector_peak_fmas_per_cycle",
    "vector_baseline",
    "useful_flops",
    "matrix_mma",
    "matrix_macs",
    "matrix_flops",
    "redundancy",
    "matrix_busy_cycles",
    "matrix_util",
    "macs_util",
    "normwise",
    "max_rel",
    "tolerance",
    "instret",
    "instret.scalar",
    "instret.fp",
    "instret.vector",
    "instret.matrix",
    "point_id",
    "design_hash",
    "design_file",
    "git_commit",
    "wall_s",
    "message",
]
#: Dynamic column groups, in CSV order after the fixed columns.
PREFIXES = ("ovr:", "norm:", "bytes:", "err:")


# -- peaks and the ratio ---------------------------------------------------------------


@lru_cache(maxsize=256)
def _design(path):
    return load_design(path)


def matrix_peak(design, in_fmt):
    """Peak MACs per cycle of the design's matrix unit for ``in_fmt`` (None if it has no timed unit)."""
    if design.matrix is None or design.timing is None or design.timing.matrix is None:
        return None
    m, n, k = design.matrix.m, design.matrix.n, design.matrix.k
    return m * n * k / design.timing.matrix.mma_interval(m, n, k, in_fmt)


def _scalar_peak(design, in_fmt):
    issue = 1 if design.timing is None else design.timing.scalar.issue_width
    return float(issue)


def _rvv_peak(design, in_fmt):
    from framework.matrix.formats import get_format

    vector = getattr(design, "vector", None)
    vlen = getattr(vector, "vlen", None)
    if vlen is None:
        raise ValueError(
            "ratio.vector: rvv needs a design with a vector unit (RVV is Phase C)"
        )
    return vlen / (8 * get_format(in_fmt).dtype.itemsize)


#: ``ratio.vector`` model name -> ``f(design, in_fmt) -> peak FMAs per cycle``.
VECTOR_PEAKS = {"scalar": _scalar_peak, "rvv": _rvv_peak}


def vector_peak(design, in_fmt, model="scalar"):
    if isinstance(model, (int, float)) and not isinstance(model, bool):
        return float(model)
    if model not in VECTOR_PEAKS:
        raise ValueError(
            f"ratio.vector: unknown model {model!r}; known: {sorted(VECTOR_PEAKS)} or a number"
        )
    return VECTOR_PEAKS[model](design, in_fmt)


# -- rows -------------------------------------------------------------------------------


def _num(v):
    return v if v is not None else ""


def byte_paths(counters):
    """``{path: bytes}``: memory<->tile/acc, per-block scalar bus bytes, and DMA, from a run's counters."""
    out = {}
    for key, value in counters.items():
        if key.startswith(("hart", "timing.")) or not key.endswith("bytes"):
            continue
        if key.startswith("matrix."):
            out[key[: -len("_bytes")]] = value
        elif key.startswith("bus."):
            out[key[: -len("_bytes")]] = value
        else:
            out[key] = (
                value  # an IP block's own byte counters (dma.bytes, dma.dram_to_scratchpad_bytes)
            )
    return out


def flatten(result, study):
    """One CSV row (a dict) from a stored point result and the study it belongs to."""
    c = result.get("counters") or {}
    row = {
        "study": study.name,
        "dwarf": result["dwarf"],
        "variant": result["variant"],
        "size": result["size_str"],
        "in_fmt": result["in_fmt"],
        "acc_fmt": result["acc_fmt"],
        "seed": result["seed"],
        "status": result["status"],
        "passed": result.get("passed", False),
        "exit_code": _num(result.get("exit_code")),
        "cycles": _num(result.get("cycles")),
        "region_cycles": _num(result.get("region_cycles")),
        "useful_flops": _num(result.get("useful_flops")),
        "tolerance": _num(result.get("tolerance")),
        "point_id": result["point_id"],
        "design_hash": result["design_hash"],
        "design_file": result["design_file"],
        "git_commit": result.get("git_commit", ""),
        "wall_s": _num(result.get("wall_s")),
        "message": result.get("message", ""),
        "vector_baseline": str(study.ratio.get("vector", "scalar")),
    }
    for key in study.overrides:
        value = result["overrides"].get(key)
        row[f"ovr:{key}"] = "" if value is None else _cell(value)
    for key, value in (result.get("normalise") or {}).items():
        row[f"norm:{key}"] = value
    for key, value in (result.get("normalised") or {}).items():
        row[f"norm:{key}"] = value
    for out, err in (result.get("error") or {}).items():
        row[f"err:{out}.normwise"] = err.get("normwise")
        row[f"err:{out}.max_rel"] = err.get("max_rel")
    errs = list((result.get("error") or {}).values())
    if errs:
        row["normwise"] = max(e["normwise"] for e in errs)
        row["max_rel"] = max(e["max_rel"] for e in errs)
    if c:
        macs = c.get("matrix.macs", 0)
        row["matrix_mma"] = c.get("matrix.mma", 0)
        row["matrix_macs"] = macs
        row["matrix_flops"] = 2 * macs
        uf = result.get("useful_flops")
        if macs and uf:
            row["redundancy"] = 2 * macs / uf
        row["instret"] = c.get("instret", "")
        for cls in ("scalar", "fp", "vector", "matrix"):
            row[f"instret.{cls}"] = c.get(f"instret.{cls}", "")
        busy = c.get("timing.matrix_busy_cycles")
        region = result.get("region_cycles")
        row["matrix_busy_cycles"] = _num(busy)
        if busy is not None and region:
            row["matrix_util"] = busy / region
        for path, value in byte_paths(c).items():
            row[f"bytes:{path}"] = value
    _add_ratio(row, result, study)
    return row


def _cell(value):
    if isinstance(value, (list, tuple, dict)):
        import json

        return json.dumps(value, separators=(",", ":"))
    return value


def _add_ratio(row, result, study):
    try:
        design = _design(result["design_path"])
        mp = matrix_peak(design, result["in_fmt"])
        vp = vector_peak(design, result["in_fmt"], study.ratio.get("vector", "scalar"))
    except (
        Exception
    ) as e:  # a missing design file or an unavailable model: leave the ratio blank
        row["message"] = (row.get("message") or "") + f" [ratio: {e}]"
        return
    if mp is None:
        return
    row["matrix_peak_macs_per_cycle"] = mp
    row["vector_peak_fmas_per_cycle"] = vp
    row["mv_ratio"] = mp / vp
    macs, region = row.get("matrix_macs"), result.get("region_cycles")
    if macs and region:
        row["macs_util"] = macs / (mp * region)


def add_speedups(rows, study):
    """Fill ``speedup`` and ``baseline_point`` on every non-baseline row of ``study`` (in place)."""
    base_variant = study.baseline.get("variant", "scalar")
    base_formats = study.baseline.get("formats", ["fp32", "fp32"])
    ovr_cols = [f"ovr:{k}" for k in study.overrides]
    baselines = [
        r
        for r in rows
        if r["study"] == study.name
        and r["variant"] == base_variant
        and r["status"] == "ok"
    ]
    for r in rows:
        if (
            r["study"] != study.name
            or r["variant"] == base_variant
            or not r.get("region_cycles")
        ):
            continue
        want = (
            (r["in_fmt"], r["acc_fmt"])
            if base_formats == "same"
            else tuple(base_formats)
        )
        for b in baselines:
            if (b["dwarf"], b["size"], b["seed"]) != (r["dwarf"], r["size"], r["seed"]):
                continue
            if (b["in_fmt"], b["acc_fmt"]) != want or not b.get("region_cycles"):
                continue
            if any(b.get(col, "") not in ("", r.get(col, "")) for col in ovr_cols):
                continue
            r["speedup"] = b["region_cycles"] / r["region_cycles"]
            r["baseline_point"] = b["point_id"]
            break
    return rows


def interpolate_crossover(ratios, speedups):
    """``(ratio, status)`` where speedup crosses 1, interpolated in log(ratio).

    ``status`` is ``bracketed`` (with the ratio), ``exact`` (a point sits at 1),
    ``below_range`` (speedup >= 1 at every ratio: the crossover is below the
    smallest), ``above_range`` (< 1 everywhere) or ``empty``.
    """
    pts = sorted(
        (r, s)
        for r, s in zip(ratios, speedups)
        if r and r > 0 and s is not None and s > 0
    )
    if not pts:
        return None, "empty"
    for r, s in pts:
        if s == 1:
            return r, "exact"
    for (r0, s0), (r1, s1) in zip(pts, pts[1:]):
        if (s0 - 1) * (s1 - 1) < 0:
            t = (1 - s0) / (s1 - s0)
            return math.exp(
                math.log(r0) + t * (math.log(r1) - math.log(r0))
            ), "bracketed"
    return None, ("below_range" if pts[0][1] > 1 else "above_range")


def crossovers(rows, study):
    """One record per crossover group of ``study`` (rows with a speedup and a ratio)."""
    ratio_cols = {f"ovr:{k}" for k in study.ratio_keys}
    ratio_cols |= {
        f"norm:{k.split('.', 1)[1]}"
        for k in study.ratio_keys
        if k.startswith("normalise.")
    }
    groups = {}
    for r in rows:
        if (
            r["study"] != study.name
            or r.get("speedup") in (None, "")
            or r.get("mv_ratio") in (None, "")
        ):
            continue
        rest = tuple(
            sorted(
                (k, str(v))
                for k, v in r.items()
                if k.startswith(("ovr:", "norm:"))
                and k not in ratio_cols
                and not _is_derived_norm(k)
            )
        )
        key = (r["dwarf"], r["variant"], r["size"], r["in_fmt"], r["acc_fmt"], rest)
        groups.setdefault(key, []).append(r)
    out = []
    for (dwarf, variant, size, in_fmt, acc_fmt, rest), members in groups.items():
        if len({float(r["mv_ratio"]) for r in members}) < 2:
            continue  # the ratio was not swept for this group
        members.sort(key=lambda r: float(r["mv_ratio"]))
        ratios = [float(r["mv_ratio"]) for r in members]
        speedups = [float(r["speedup"]) for r in members]
        value, status = interpolate_crossover(ratios, speedups)
        out.append(
            {
                "study": study.name,
                "dwarf": dwarf,
                "variant": variant,
                "size": size,
                "in_fmt": in_fmt,
                "acc_fmt": acc_fmt,
                "group": ";".join(f"{k}={v}" for k, v in rest),
                "crossover_ratio": "" if value is None else value,
                "status": status,
                "points": len(members),
                "min_ratio": ratios[0],
                "max_ratio": ratios[-1],
                "speedup_at_min": speedups[0],
                "speedup_at_max": speedups[-1],
                "vector_baseline": members[0].get("vector_baseline", ""),
            }
        )
    return out


def _is_derived_norm(col):
    # The normaliser's derived outputs (norm:d, norm:mma_ii, ...) follow its inputs; only inputs group.
    return col.startswith("norm:") and col[5:] not in (
        "operand_bytes",
        "mem_bytes_per_cycle",
        "tiles",
        "accs",
        "mode",
        "power_of_two",
    )


# -- CSV ----------------------------------------------------------------------------------


def columns(rows):
    seen = set()
    for r in rows:
        seen.update(r)
    cols = [c for c in FIXED_COLUMNS]
    for prefix in PREFIXES:
        cols += sorted(c for c in seen if c.startswith(prefix))
    cols += sorted(seen - set(cols))
    return cols


def write_csv(path, rows, cols=None):
    path = Path(path)
    cols = cols or columns(rows)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    tmp.replace(path)
    return path


def _parse(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path):
    with open(path, newline="") as f:
        return [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]
