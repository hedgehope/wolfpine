# ruff: noqa: INP001  (loaded by path by framework.naiad.dwarfs, not imported as a package)
"""MapReduce, reduction: the sum of a length-n vector as a product with a ones tile.

Reformulation (the ``ame`` variant). ``x`` is read as n / 256 consecutive 16 x 16
row-major tiles ``X_c``, and every one is multiplied by a ones tile ``J`` into a
single accumulator: ``acc = sum_c X_c J``. Each accumulator row then holds one
partial sum ``p_i`` (the sum of row i over every tile), replicated across the 16
columns. The 16 partials are finished either by a second mma
(``REDUCE_FINISH_MMA=1``, the default: store the accumulator, reload it as an
accumulator-format tile ``P`` and form ``J' P``, so every element is the total)
or by 16 scalar adds (``finish="scalar"``). Accumulation is in ``acc_fmt``
throughout; inputs are rounded to ``in_fmt`` once, on the host.

Redundancy. A ones *vector* on a tile unit uses one column of the 16 the mma
computes, so each input element costs 16 MACs: the matrix unit does
``4096 (n / 256 + finish)`` MACs for the ``n - 1`` adds of :func:`useful_flops`,
about 32x counting a MAC as 2 FLOPs. No optional AME feature helps a plain
sum (there is nothing to shift, transpose or scale); the one switch is the
finish, ``finish`` (``"mma"`` / ``"scalar"``).

Inputs are uniform on [0, 1), so the sum is well conditioned and the error is
the format's, not cancellation's.
"""

import importlib.util
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "naiad_dwarf_dw_formats", _here.parent / "jacobi" / "dw_formats.py"
)
dwf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dwf)

NAME = "reduce"
#: ``*_mc``: the same kernel split across the harts of a multi-hart design (``cores.count`` > 1;
#: the source says how); on one hart each is its single-hart variant plus no-op barriers.
#: ``ime`` / ``vme``: one call of the emitted GEMM driver against a stride-0 ones operand
#: (``gemm/reduce_gemm.c``; needs an IME or VME design, e.g. naiad-ime-timed / naiad-vme-timed).
VARIANTS = {
    "ame": "ame",
    "scalar": "scalar",
    "rvv": "rvv",
    "ame_mc": "ame_mc",
    "rvv_mc": "rvv_mc",
}
VARIANTS.update({"ime": "gemm", "vme": "gemm"})
SIZES = (4096, 16384, 65536)
FORMATS = dwf.FORMATS
#: The RVV baseline runs every pair (Zvfh / Zvfbfmin / Zvfbfwma widen the 16-bit inputs); scalar: F/D only.
VARIANT_FORMATS = {"rvv": FORMATS, "rvv_mc": FORMATS}
VARIANT_FORMATS.update({"ime": FORMATS, "vme": FORMATS})
#: ``finish``: how the 16 partial sums become one. ``mma`` (default): astore, reload as a
#: tile, one more mma. ``scalar``: 16 scalar adds. ``mmov``: as ``mma`` but the partials
#: move accumulator -> tile by ``amov.t.a`` (AME ``acc_move``; Ztt ``mmov.m.a``).
#: ``native``: ``amov.t.a`` then one ``treduce`` down the columns (AME ``reduce``;
#: Ztt ``mreduceadd.col``), no second mma.
SWITCHES = {"finish": "mma"}
#: ``finish`` value -> its cflags.
FINISHES = {
    "mma": [],
    "scalar": ["-DREDUCE_FINISH_MMA=0"],
    "mmov": ["-DREDUCE_FINISH_MMOV=1"],
    "native": ["-DREDUCE_FINISH_NATIVE=1"],
}


def _check_size(n):
    if n < 1024 or n % 1024:
        raise ValueError(f"reduce size must be a positive multiple of 1024, got {n}")


def switch_cflags(switches=None):
    """``-D`` flags for the switches that differ from :data:`SWITCHES` (usable as ``extra_cflags``)."""
    sw = {**SWITCHES, **(switches or {})}
    if set(sw) - set(SWITCHES) or sw["finish"] not in FINISHES:
        raise ValueError(
            f"bad reduce switches {switches}; known: finish = {' | '.join(FINISHES)}"
        )
    return list(FINISHES[sw["finish"]])


def cflags(size, in_fmt, acc_fmt, switches=None):
    _check_size(size)
    return [
        *dwf.fmt_cflags(in_fmt, acc_fmt),
        f"-DREDUCE_N={size}",
        # Format names for the ime / vme variants' GEMM driver (unused by the others).
        f"-DREDUCE_IN={in_fmt}",
        f"-DREDUCE_ACC={acc_fmt}",
        f"-DREDUCE_IN_UP={in_fmt.upper()}",
        f"-DREDUCE_ACC_UP={acc_fmt.upper()}",
        f"-DREDUCE_IN_{in_fmt.upper()}=1",
        f"-DREDUCE_ACC_{acc_fmt.upper()}=1",
        *switch_cflags(switches),
    ]


def make_inputs(size, rng):
    _check_size(size)
    return {"x": rng.random(size)}


def encode(inputs, in_fmt, acc_fmt):
    dwf.check_pair(in_fmt, acc_fmt)
    return {"x": dwf.encode(inputs["x"], in_fmt)}


def outputs(size, acc_fmt):
    return {"result": (dwf.ACC_DTYPE[acc_fmt], (1,))}


def decode(outputs, acc_fmt):
    return {"sum": np.asarray(outputs["result"], np.float64)}


def golden(inputs):
    return {"sum": np.array([np.sum(np.asarray(inputs["x"], np.float64))])}


def useful_flops(size):
    return size - 1


def tolerance(in_fmt, acc_fmt):
    return {"fp64": 1e-13, "fp32": 1e-5, "bf16": 5e-3, "fp16": 1e-3}[in_fmt]
