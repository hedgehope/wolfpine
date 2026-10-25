"""A compiler cost model for the dwarf reformulations: predict cycles per implementation, pick one.

Given a kernel (dwarf, problem size, input / accumulator formats) and a design
(its ``timing:`` section: scalar latencies, memory, the vector unit, the matrix
unit's kind, shape, peak and transfer paths), :func:`predict` returns the
cycles each implementation is expected to take over the kernel's timed region
-- the scalar F/D baseline, the RVV baseline, and every matrix reformulation the
design can run (AME, and the dwarf's alternative formulations such as
``ame_gauss`` / ``ame_fgemm``) -- and :func:`choose` the fastest, with the margin
to the runner-up. :func:`crossover` finds the matrix:vector peak ratio at which a
reformulation starts to pay.

**How it predicts.** Each dwarf's formula (``costmodel.dwarfs.<name>``) is its
kernel's loop nest, with trip counts from the size and, per body, the
instructions the kernel issues in program order (loads, FMAs, vector ops,
mmas, tile transfers) with their operands. :class:`~.engine.Engine` evaluates
that nest under the design's issue rules -- one instruction a cycle in order,
per-class latencies, the vector datapath / LSU occupancies, the matrix unit's
pipes and scoreboard -- replaying a few iterations of each loop and
extrapolating the steady state. Nothing is fitted to measured cycles: the
inputs are the loop bounds and bodies (what a compiler knows about the code it
emits) and the design's timing table. See ``docs`` in each dwarf module for
the closed-form bound each formula reduces to.

Typical use::

    from framework.naiad import costmodel
    costmodel.predict("jacobi", 130, "bf16", "fp32", "framework/naiad/designs/naiad-dwarfs.yaml")
    # {'scalar': ..., 'rvv': ..., 'ame': ...}
    costmodel.choose("jacobi", 130, "bf16", "fp32", design)  # Choice(variant='ame', ...)
"""

import importlib
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from framework.naiad.costmodel import engine as _engine
from framework.naiad.costmodel.dwarfs.common import Unsupported
from framework.naiad.costmodel.machine import Machine

#: Dwarfs with a formula (module ``costmodel.dwarfs.<name>``).
DWARFS = (
    "gemm",
    "jacobi",
    "fft",
    "nbody",
    "reduce",
    "scan",
    "tensorprod",
    "spmv",
    "apsp",
)
#: Variants that run on the matrix unit (the reformulations); the rest are baselines.
MATRIX_PREFIXES = ("ame", "ime", "vme")


def is_matrix_variant(variant):
    return variant.startswith(MATRIX_PREFIXES)


def dwarf_module(dwarf):
    if dwarf not in DWARFS:
        raise KeyError(
            f"no cost formula for dwarf {dwarf!r}; known: {', '.join(DWARFS)}"
        )
    return importlib.import_module(f"framework.naiad.costmodel.dwarfs.{dwarf}")


@lru_cache(maxsize=256)
def _machine_from_path(path):
    return Machine.from_design(path)


def machine(design):
    """A :class:`Machine` from a design path, a ``Design``, or a ``Machine``."""
    if isinstance(design, Machine):
        return design
    if isinstance(design, (str, Path)):
        return _machine_from_path(str(Path(design).resolve()))
    return Machine.from_design(design)


def _norm_size(size):
    return tuple(size) if isinstance(size, list) else size


@dataclass
class Estimate:
    """One variant's prediction: cycles plus the busy time of each unit and the instruction mix."""

    variant: str
    cycles: int
    busy: dict = field(default_factory=dict)
    instret: dict = field(default_factory=dict)


def explain(
    dwarf,
    size,
    in_fmt="fp32",
    acc_fmt="fp32",
    design=None,
    variants=None,
    switches=None,
    model="schedule",
):
    """``{variant: Estimate}`` for every variant the design can run (or those named).

    ``model`` is ``"schedule"`` (in-order issue with latencies, the default) or
    ``"bound"`` (throughput only: the busiest unit's occupancy, a roofline).
    """
    mod = dwarf_module(dwarf)
    m = machine(design if design is not None else _default_design())
    size = _norm_size(size)
    out = {}
    with _engine.model(model):
        for name, fn in mod.VARIANTS.items():
            if variants is not None and name not in variants:
                continue
            kwargs = dict(switches or {})
            try:
                e = (
                    fn(m, size, in_fmt, acc_fmt, **kwargs)
                    if kwargs
                    else fn(m, size, in_fmt, acc_fmt)
                )
            except Unsupported:
                continue
            out[name] = Estimate(name, int(e.cycles), dict(e.busy), dict(e.instret))
    return out


def predict(
    dwarf,
    size,
    in_fmt="fp32",
    acc_fmt="fp32",
    design=None,
    variants=None,
    switches=None,
    model="schedule",
):
    """``{variant: predicted region cycles}`` for every variant the design can run."""
    return {
        v: est.cycles
        for v, est in explain(
            dwarf, size, in_fmt, acc_fmt, design, variants, switches, model
        ).items()
    }


@dataclass
class Choice:
    """:func:`choose`'s answer."""

    #: The variant predicted fastest.
    variant: str
    cycles: int
    #: Runner-up cycles over the winner's (>= 1): how sure the choice is.
    margin: float
    #: The best matrix reformulation's predicted speedup over the best baseline (> 1: it pays).
    reformulation_speedup: float | None
    predictions: dict


def choose(
    dwarf,
    size,
    in_fmt="fp32",
    acc_fmt="fp32",
    design=None,
    variants=None,
    switches=None,
    model="schedule",
):
    """The variant predicted fastest, its margin, and whether reformulating pays."""
    m = machine(design if design is not None else _default_design())
    pred = predict(dwarf, size, in_fmt, acc_fmt, m, variants, switches, model)
    same = getattr(dwarf_module(dwarf), "redundant", lambda _m: set())(m)
    pred = {
        v: c
        for v, c in pred.items()
        if v not in same or (variants is not None and v in variants)
    }
    if not pred:
        raise Unsupported(f"the design can run no variant of {dwarf}")
    ranked = sorted(pred.items(), key=lambda kv: kv[1])
    best, cycles = ranked[0]
    margin = ranked[1][1] / cycles if len(ranked) > 1 else float("inf")
    mat = [c for v, c in pred.items() if is_matrix_variant(v)]
    base = [c for v, c in pred.items() if not is_matrix_variant(v)]
    speedup = min(base) / min(mat) if mat and base else None
    return Choice(best, cycles, margin, speedup, pred)


def with_matrix_peak(m, macs_per_cycle):
    """``m`` with the matrix unit's ``macs_per_cycle`` replaced (the crossover axis)."""
    if m.mode == "ame":
        return m.with_(ame=replace(m.ame, macs_per_cycle=macs_per_cycle))
    raise Unsupported(f"the matrix peak axis is modelled for AME designs, not {m.mode}")


def crossover(
    dwarf,
    size,
    in_fmt="fp32",
    acc_fmt="fp32",
    design=None,
    variant="ame",
    baseline="rvv",
    ratios=None,
    model="schedule",
):
    """The matrix:vector peak ratio at which ``variant`` matches ``baseline`` (log-interpolated).

    The ratio is the harness's (``experiments.naiad.metrics``): the matrix unit's
    peak MACs a cycle for ``in_fmt`` over the vector unit's FMAs a cycle
    (``datapath_bits / SEW``) for an RVV baseline, or 1 (one scalar FMA a cycle)
    for the scalar one. Returns ``(ratio | None, "ok" | "below_range" | "above_range")``.
    """
    import math

    m = machine(design if design is not None else _default_design())
    base = predict(dwarf, size, in_fmt, acc_fmt, m, variants=[baseline], model=model)[
        baseline
    ]
    if baseline == "scalar":
        vec_peak = 1.0
    else:
        sew = max(32, 8 * max(_fmt_bytes(in_fmt), _fmt_bytes(acc_fmt)))
        vec_peak = m.vector.datapath_bits / sew
    grid = ratios or [2 ** (i / 2) for i in range(-8, 19)]
    pts = []
    for r in grid:
        target_peak = r * vec_peak
        # invert peak(macs) = m*n*k / ceil(m*n*k*passes/macs) for the macs giving this peak
        x = m.ame
        macs = max(1, round(target_peak * x.format_passes.get(in_fmt, 1)))
        mm = with_matrix_peak(m, macs)
        ratio = mm.ame.peak_macs(in_fmt) / vec_peak
        c = predict(dwarf, size, in_fmt, acc_fmt, mm, variants=[variant], model=model)[
            variant
        ]
        pts.append((ratio, base / c))
    pts.sort()
    for (r0, s0), (r1, s1) in zip(pts, pts[1:]):
        if (s0 - 1) * (s1 - 1) <= 0 and s0 != s1:
            f = (1 - s0) / (s1 - s0)
            return math.exp(math.log(r0) + f * (math.log(r1) - math.log(r0))), "ok"
    return None, ("below_range" if pts and pts[0][1] > 1 else "above_range")


def _fmt_bytes(fmt):
    from framework.naiad.costmodel.machine import FORMAT_BYTES

    return FORMAT_BYTES[fmt]


def _default_design():
    return Path(__file__).resolve().parents[1] / "designs" / "naiad-dwarfs.yaml"


__all__ = [
    "DWARFS",
    "Choice",
    "Estimate",
    "Machine",
    "Unsupported",
    "choose",
    "crossover",
    "explain",
    "is_matrix_variant",
    "machine",
    "predict",
]
