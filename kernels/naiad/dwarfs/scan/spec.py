# ruff: noqa: INP001  (loaded by path by experiments.naiad.dwarfs, not imported as a package)
"""MapReduce, scan: the inclusive prefix sum of a length-n vector, as products with triangular ones tiles.

Reformulation (the ``ame`` variant). ``x`` is read as n / 256 consecutive
16 x 16 row-major tiles ``X``. The in-tile scan is ``X U + (Ls X) J``: ``U`` is
the upper-triangular ones tile (row-wise inclusive scans), ``Ls`` the strictly
lower-triangular ones tile (column sums over the earlier rows) and ``J`` all
ones (which broadcasts each earlier-rows total along its row). ``Ls X`` lands in
an accumulator, which cannot be an mma operand, so it makes one round trip
through memory and is reloaded as an accumulator-format tile: the AME has no
accumulator-to-tile move, and this is what that costs.

Carry propagation across tiles is a **second mma** by default: a stride-0 load
of the previous tile's last output row (all 16 rows the same) times ``Sel``
(ones in row 15) adds that row's last element, the running total, to every
element. With ``epilogue=True`` (AME ``epilogue`` feature) the carry is instead
one scalar load of ``y[256 c - 1]`` and one ``aadds``. Either way the carry is
a serial dependence from each tile's store to the next tile's carry. The
running total is kept in ``acc_fmt``, as the scalar loop keeps it.

Redundancy. 4 mma per 256 outputs (3 with the epilogue) against the ``n - 1``
adds of :func:`useful_flops`: 16384 MACs per tile, about 128x counting a MAC
as 2 FLOPs (96x with the epilogue). Transposed loads and tshift do not help:
every operand is already in the orientation its mma needs.

Inputs are uniform on [0, 1), so every prefix is well conditioned.
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

NAME = "scan"
#: ``*_mc``: the same kernel split across the harts of a multi-hart design (``cores.count`` > 1;
#: the source says how); on one hart each is its single-hart variant plus no-op barriers.
VARIANTS = {"ame": "ame", "scalar": "scalar", "rvv": "rvv", "ame_mc": "ame_mc"}
SIZES = (4096, 16384, 65536)
FORMATS = dwf.FORMATS
#: The RVV baseline runs every pair (Zvfh / Zvfbfmin / Zvfbfwma widen the 16-bit inputs); scalar: F/D only.
VARIANT_FORMATS = {"rvv": FORMATS}
#: Feature switches (defaults reproduce the original kernel). ``mmov``: move ``Ls X``
#: accumulator -> tile with ``amov.t.a`` (AME ``acc_move``; Ztt ``mmov.m.a``) instead of
#: the memory round trip. ``prefix``: the in-tile scan from the native ``tprefix`` /
#: ``treduce`` (AME ``reduce``, plus ``acc_move``, ``shift`` and, for 16-bit inputs,
#: ``convert``; Ztt ``mprefixadd`` / ``mreduceadd``) instead of the triangular products.
SWITCHES = {"epilogue": False, "mmov": False, "prefix": False}


def _check_size(n):
    if n < 256 or n % 256:
        raise ValueError(f"scan size must be a positive multiple of 256, got {n}")


def switch_cflags(switches=None):
    """``-D`` flags for the switches that differ from :data:`SWITCHES` (usable as ``extra_cflags``)."""
    sw = {**SWITCHES, **(switches or {})}
    if set(sw) - set(SWITCHES):
        raise ValueError(
            f"unknown scan switch(es) {sorted(set(sw) - set(SWITCHES))}; known: {sorted(SWITCHES)}"
        )
    return [f"-DSCAN_{k.upper()}=1" for k in SWITCHES if sw[k] and not SWITCHES[k]]


def cflags(size, in_fmt, acc_fmt, switches=None):
    _check_size(size)
    return [
        *dwf.fmt_cflags(in_fmt, acc_fmt),
        f"-DSCAN_N={size}",
        *switch_cflags(switches),
    ]


def make_inputs(size, rng):
    _check_size(size)
    return {"x": rng.random(size)}


def encode(inputs, in_fmt, acc_fmt):
    dwf.check_pair(in_fmt, acc_fmt)
    return {"x": dwf.encode(inputs["x"], in_fmt)}


def outputs(size, acc_fmt):
    return {"y": (dwf.ACC_DTYPE[acc_fmt], (size,))}


def decode(outputs, acc_fmt):
    return {"y": np.asarray(outputs["y"], np.float64)}


def golden(inputs):
    return {"y": np.cumsum(np.asarray(inputs["x"], np.float64))}


def useful_flops(size):
    return size - 1


def tolerance(in_fmt, acc_fmt):
    return {"fp64": 1e-13, "fp32": 1e-5, "bf16": 5e-3, "fp16": 1e-3}[in_fmt]
