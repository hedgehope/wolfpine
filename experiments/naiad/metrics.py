"""Rows, derived metrics and the crossover: from stored point results to ``results.csv``.

**Speedup** of a row is the baseline variant's ``region_cycles`` over the row's,
at the same study, dwarf, size and seed, the baseline's formats (the study's
``baseline.formats``, or ``same``) and every override the baseline also took
(keys that do not apply to the baseline, through ``only:``, are wildcards).

**The matrix-to-vector throughput ratio** is the matrix unit's peak MACs per
cycle *for the row's input format* (``m*n*k / mma_interval``, so it follows
``macs_per_cycle``, ``format_passes`` and ``mma_ii``) over the vector
baseline's peak FMAs per cycle. The baseline model is the study's
``ratio.vector``: ``scalar`` (F/D: FMA is pipelined, so the peak is
``timing.scalar.issue_width`` FMAs a cycle), ``rvv`` (the design's
``timing.vector.datapath_bits`` over the SEW the FMA runs at: the wider of the
row's input and accumulator formats, and at least 32 bits when either is bf16,
which RVV can only multiply-add widening into fp32; so 8 fp32 / 4 fp64 / 16
fp16 FMAs a cycle at 256 bits, and 8 for bf16 or fp16 inputs with an fp32
accumulator), or a fixed number. Switching is that one key, and it needs no
re-run: the ratio is recomputed from the stored designs whenever the CSV is
written.

**References** (the study's ``references:``) are further variants each row is
also compared with: ``speedup_vs_<variant>`` is matched exactly like the
baseline, and every crossover group gets a ``crossover_vs_<variant>`` on the
same ratio axis.

**Seeds**: a study with ``seeds:`` runs every point once per seed. Each row
carries ``normwise_median`` / ``normwise_max`` / ``n_seeds`` over the rows that
differ from it only in the seed, and :func:`precision_records` gives one record
per such group (``precision.csv``). Crossover groups are per seed.

**The crossover** of a group with at least two distinct ratios (same study, dwarf, variant, size, formats and
every override not in ``ratio_keys``) is the ratio at which speedup reaches 1,
interpolated linearly in speedup against log(ratio) between the first bracketing
pair of points sorted by ratio.
"""

import csv
import math
import statistics
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
    "normwise_median",
    "normwise_max",
    "n_seeds",
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
PREFIXES = ("speedup_vs_", "ovr:", "norm:", "bytes:", "err:")
#: Bits per element of each format, for the RVV peak model.
FORMAT_BITS = {
    "fp64": 64,
    "int32": 32,
    "fp32": 32,
    "tf32": 32,
    "fp16": 16,
    "bf16": 16,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "int8": 8,
}


# -- peaks and the ratio ---------------------------------------------------------------


@lru_cache(maxsize=256)
def _design(path):
    return load_design(path)


def matrix_peak(design, in_fmt, acc_fmt=None):
    """Peak MACs per cycle of the design's matrix unit for ``in_fmt`` (None if it has no timed unit).

    AME: one ``m x n x k`` mma per ``mma_interval``. IME (Zvvm): the reference
    mma of :func:`framework.matrix.ime.reference_geometry` (C at the accumulator's
    SEW, ``W = SEW / input width``, the design's lambdas; ``M x M x K_eff`` MACs at
    full VL) over its occupancy, the larger of compute and the VRF port time
    (IME adds no ports, so an mma's operand reads bound it), C re-read unless
    ``c_bypass``. ``acc_fmt`` defaults to ``in_fmt``.
    """
    if design.matrix is None or design.timing is None or design.timing.matrix is None:
        return None
    if getattr(design.matrix, "mode", "ame") == "ime":
        return _ime_peak(design, in_fmt, acc_fmt or in_fmt)
    m, n, k = design.matrix.m, design.matrix.n, design.matrix.k
    return m * n * k / design.timing.matrix.mma_interval(m, n, k, in_fmt)


def _ime_peak(design, in_fmt, acc_fmt):
    from framework.matrix.ime import reference_geometry

    sew = FORMAT_BITS[acc_fmt]
    w, rem = divmod(sew, FORMAT_BITS[in_fmt])
    if rem or w not in (1, 2, 4, 8):
        return None
    vlen = design.matrix.vlen
    geo = reference_geometry(vlen, sew, w, design.matrix.lambdas.get(sew) or None)
    if geo is None:
        return None
    spec = design.timing.matrix
    datapath = design.timing.vector.datapath_bits if design.timing.vector else vlen
    occ = spec.mma_occupancy(geo, in_fmt, datapath, c_read=not spec.c_bypass)[0]
    return geo.macs / occ


def _scalar_peak(design, in_fmt, acc_fmt=None):
    issue = 1 if design.timing is None else design.timing.scalar.issue_width
    return float(issue)


def rvv_sew(in_fmt, acc_fmt=None):
    """The element width an RVV FMA for this format pair runs at (see the module docstring)."""
    fmts = [f for f in (in_fmt, acc_fmt) if f]
    unknown = [f for f in fmts if f not in FORMAT_BITS]
    if unknown:
        raise ValueError(
            f"ratio.vector: rvv: no element width for format {unknown[0]!r}"
        )
    bits = max(FORMAT_BITS[f] for f in fmts)
    if "bf16" in fmts:  # Zvfbfwma only has the widening bf16 -> fp32 FMA
        bits = max(bits, 32)
    return bits


def _rvv_peak(design, in_fmt, acc_fmt=None):
    vector = None if design.timing is None else design.timing.vector
    if vector is None:
        raise ValueError(
            "ratio.vector: rvv needs a design whose cores have a vector unit and a timing section"
        )
    return vector.datapath_bits / rvv_sew(in_fmt, acc_fmt)


#: ``ratio.vector`` model name -> ``f(design, in_fmt, acc_fmt) -> peak FMAs per cycle``.
VECTOR_PEAKS = {"scalar": _scalar_peak, "rvv": _rvv_peak}


def vector_peak(design, in_fmt, model="scalar", acc_fmt=None):
    if isinstance(model, (int, float)) and not isinstance(model, bool):
        return float(model)
    if model not in VECTOR_PEAKS:
        raise ValueError(
            f"ratio.vector: unknown model {model!r}; known: {sorted(VECTOR_PEAKS)} or a number"
        )
    return VECTOR_PEAKS[model](design, in_fmt, acc_fmt)


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
        mp = matrix_peak(design, result["in_fmt"], result["acc_fmt"])
        vp = vector_peak(
            design,
            result["in_fmt"],
            study.ratio.get("vector", "scalar"),
            result["acc_fmt"],
        )
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


def _match_speedups(rows, study, base_variant, base_formats, col, point_col=None):
    """Set ``row[col]`` = base ``region_cycles`` / row's for every row of ``study`` that has a base."""
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
            if any(b.get(col_, "") not in ("", r.get(col_, "")) for col_ in ovr_cols):
                continue
            r[col] = b["region_cycles"] / r["region_cycles"]
            if point_col:
                r[point_col] = b["point_id"]
            break


def add_speedups(rows, study):
    """Fill ``speedup`` / ``baseline_point`` and every ``speedup_vs_<reference>`` of ``study`` (in place)."""
    base_formats = study.baseline.get("formats", ["fp32", "fp32"])
    _match_speedups(
        rows,
        study,
        study.baseline.get("variant", "scalar"),
        base_formats,
        "speedup",
        "baseline_point",
    )
    for ref in study.references or []:
        _match_speedups(rows, study, ref, base_formats, f"speedup_vs_{ref}")
    return rows


def _seed_group_key(r):
    return (
        r["study"],
        r["dwarf"],
        r["variant"],
        r["size"],
        r["in_fmt"],
        r["acc_fmt"],
        tuple(
            sorted(
                (k, str(v))
                for k, v in r.items()
                if k.startswith(("ovr:", "norm:")) and not _is_derived_norm(k)
            )
        ),
    )


def add_seed_stats(rows):
    """``normwise_median`` / ``normwise_max`` / ``n_seeds`` on every ok row, over its seed group (in place)."""
    groups = {}
    for r in rows:
        if r["status"] == "ok" and r.get("normwise") not in (None, ""):
            groups.setdefault(_seed_group_key(r), []).append(r)
    for members in groups.values():
        errs = sorted(float(r["normwise"]) for r in members)
        median = statistics.median(errs)
        for r in members:
            r["normwise_median"] = median
            r["normwise_max"] = errs[-1]
            r["n_seeds"] = len({r_["seed"] for r_ in members})
    return rows


def precision_records(rows):
    """One record per seed group (rows with an error): median / max / min normwise error and the seeds."""
    groups = {}
    for r in rows:
        if r["status"] == "ok" and r.get("normwise") not in (None, ""):
            groups.setdefault(_seed_group_key(r), []).append(r)
    out = []
    for (study, dwarf, variant, size, in_fmt, acc_fmt, rest), members in groups.items():
        members.sort(key=lambda r: r["seed"])
        errs = [float(r["normwise"]) for r in members]
        out.append(
            {
                "study": study,
                "dwarf": dwarf,
                "variant": variant,
                "size": size,
                "in_fmt": in_fmt,
                "acc_fmt": acc_fmt,
                "group": ";".join(f"{k}={v}" for k, v in rest),
                "n_seeds": len(members),
                "seeds": " ".join(str(r["seed"]) for r in members),
                "normwise_median": statistics.median(errs),
                "normwise_max": max(errs),
                "normwise_min": min(errs),
                "normwise_by_seed": " ".join(f"{e:.3e}" for e in errs),
                "passed_all": all(r.get("passed") for r in members),
                "tolerance": members[0].get("tolerance", ""),
            }
        )
    return out


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
        key = (
            r["dwarf"],
            r["variant"],
            r["size"],
            r["in_fmt"],
            r["acc_fmt"],
            r["seed"],
            rest,
        )
        groups.setdefault(key, []).append(r)
    out = []
    for (dwarf, variant, size, in_fmt, acc_fmt, seed, rest), members in groups.items():
        if len({float(r["mv_ratio"]) for r in members}) < 2:
            continue  # the ratio was not swept for this group
        members.sort(key=lambda r: float(r["mv_ratio"]))
        ratios = [float(r["mv_ratio"]) for r in members]
        speedups = [float(r["speedup"]) for r in members]
        value, status = interpolate_crossover(ratios, speedups)
        record = {
            "study": study.name,
            "dwarf": dwarf,
            "variant": variant,
            "size": size,
            "in_fmt": in_fmt,
            "acc_fmt": acc_fmt,
            "seed": seed,
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
        for ref in study.references or []:
            col = f"speedup_vs_{ref}"
            pts = [
                (float(r["mv_ratio"]), float(r[col]))
                for r in members
                if r.get(col) not in (None, "")
            ]
            ref_value, ref_status = interpolate_crossover(
                [p[0] for p in pts], [p[1] for p in pts]
            )
            record[f"crossover_vs_{ref}"] = "" if ref_value is None else ref_value
            record[f"status_vs_{ref}"] = ref_status
        out.append(record)
    return out


def _is_derived_norm(col):
    # The normaliser's derived outputs (norm:d, norm:mma_ii, ...) follow its inputs; only inputs group.
    return col.startswith("norm:") and col[5:] not in (
        "vlen",
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
