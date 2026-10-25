# ruff: noqa: INP001  (loaded by path by framework.naiad.dwarfs, not imported as a package)
"""Structured grid: JACOBI_ITERS sweeps of the 5-point Jacobi stencil, as shift / banded matrix products.

Problem. An n x n grid ``u`` with a fixed boundary ring; each sweep replaces every
interior point by the mean of its four neighbours,
``u'[i, j] = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]) / 4``, for
:data:`ITERS` sweeps. ``size`` is n, which must be 16 t + 2, so the interior is
exactly t x t output tiles of 16 x 16 and no tile ever writes the boundary.

Reformulation (the ``ame`` variant). The neighbour sum of the whole grid is
``T U + U T``, with ``T`` the tridiagonal matrix of ones off the diagonal. In
16 x 16 tiles ``T`` is block tridiagonal: ``T0`` (ones on the first sub- and
super-diagonals) on the block diagonal, and single-one corner tiles off it, so
one output tile ``C`` needs ``C T0 + T0 C`` plus four halo terms from its
neighbours. Each term is one mma into a single fp32/fp64 accumulator. Two forms
(``JACOBI_FORM``):

* ``window`` (0): ``Q Wup + Q Wdn + Wlt Q + Wrt Q``, ``Q = I``, where each W is
  the tile's 16 x 16 window moved by one row or column. The shift is in the
  load address, so the halo comes for free: 4 tile loads, 4 mma.
* ``banded`` (1, the default): the horizontal sum is ``U T`` for the block
  tridiagonal ``T``: ``C T0 + Lt Elt + Rt Ert``, with the left / centre / right
  tiles rotating through three tile registers along a tile row, so each tile
  costs one new load. The vertical sum depends on ``JACOBI_TSHIFT``:

  - off: ``Q Wup + Q Wdn``, two window loads;
  - on (default, AME ``shift`` feature): ``T0 C + Ah H``. ``H`` is *one* tile
    load with a 17-row stride, so its rows 0 and 1 are the grid rows just above
    and below the tile; ``tshift`` moves them to rows 14 and 15 and zeroes the
    14 junk rows, and ``Ah`` (ones at (0, 14) and (15, 15)) adds them to output
    rows 0 and 15. Both vertical halos cost one load, one tshift and one mma.

  So ``banded`` is 5 mma and 2 loads per tile with tshift, 5 mma and 3 loads
  without.

The 1/4 (``JACOBI_EPILOGUE``) is either one ``ascal`` per tile (AME
``epilogue`` feature) or folded into the constant tiles (default), which is
exact because 1/4 is a power of two: the epilogue cannot win here. Transposed
loads are not used: for a symmetric stencil on square tiles every operand is
already in the orientation its mma needs. The semiring is not needed.

Formats. The grid is stored in ``in_fmt`` between sweeps, so the matrix unit
always reads ``in_fmt`` operands: with a reduced ``in_fmt`` each intermediate
sweep's accumulator results are rounded back to it, on the matrix unit for
bf16 (an identity mma into a bf16 accumulator) and on the scalar side for fp16
(the design has no fp16 accumulator format). The output is the last sweep in
``acc_fmt``. The ``scalar`` variant does the same rounding.

Redundancy. :func:`useful_flops` counts 4 FLOPs (3 adds, 1 multiply) per
interior point per sweep. The matrix unit does 4096 MACs per mma, so 5 mma per
256 points is 80 MACs per point: about 40x the useful FLOPs (counting a MAC as
2 FLOPs); the run's ``matrix.macs`` counter shows the exact figure.

Feature switches (all compile-time; ``cflags(..., switches={...})``, or
``switch_cflags(switches)`` as a sweep's ``extra_cflags``): ``form``
(``"window"`` / ``"banded"``), ``tshift`` (bool, banded only) and ``epilogue``
(bool). :data:`ITERS` is not a switch: :func:`golden` assumes it.
"""

import importlib.util
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "naiad_dwarf_dw_formats", _here / "dw_formats.py"
)
dwf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dwf)

NAME = "jacobi"
#: ``*_mc``: the same kernel split across the harts of a multi-hart design (``cores.count`` > 1;
#: the source says how); on one hart each is its single-hart variant plus no-op barriers.
VARIANTS = {
    "ame": "ame",
    "scalar": "scalar",
    "rvv": "rvv",
    "ame_mc": "ame_mc",
    "rvv_mc": "rvv_mc",
}
SIZES = (34, 66, 130)
FORMATS = dwf.FORMATS
#: The RVV baseline runs every pair (Zvfh / Zvfbfmin / Zvfbfwma widen the 16-bit inputs); scalar: F/D only.
VARIANT_FORMATS = {"rvv": FORMATS, "rvv_mc": FORMATS}
ITERS = 8

#: Default switch values; ``cflags`` accepts overrides. ``mconv`` (AME ``acc_move`` +
#: ``convert``; Ztt ``mmov.m.a`` + ``mconv.ew``): with a 16-bit ``in_fmt``, each
#: intermediate sweep's tiles go accumulator -> tile -> in_fmt tile -> memory on the
#: matrix unit, replacing the identity-mma (bf16) or scalar (fp16) re-rounding pass; it
#: changes nothing for fp32 / fp64. ``tshift2`` (AME ``shift2``, a proposal, not in Ztt;
#: ``form="window"`` only): the four windows come from funnel shifts of the
#: centre / left / right tiles and one fused-halo load, so 2 loads per tile, not 4.
SWITCHES = {
    "form": "banded",
    "tshift": True,
    "epilogue": False,
    "mconv": False,
    "tshift2": False,
}
FORMS = {"window": 0, "banded": 1}


def _check_size(n):
    if n < 18 or (n - 2) % 16:
        raise ValueError(f"jacobi size must be 16 t + 2 (>= 18), got {n}")


def switch_cflags(switches=None):
    """``-D`` flags for the switches that differ from :data:`SWITCHES` (the C defaults match it).

    A sweep can pass the result as ``extra_cflags``: defaults are never emitted, so
    nothing is defined twice.
    """
    sw = {**SWITCHES, **(switches or {})}
    unknown = set(sw) - set(SWITCHES)
    if unknown:
        raise ValueError(
            f"unknown jacobi switch(es) {sorted(unknown)}; known: {sorted(SWITCHES)}"
        )
    if sw["form"] not in FORMS:
        raise ValueError(
            f"jacobi form must be one of {sorted(FORMS)}, got {sw['form']!r}"
        )
    if sw["tshift2"] and sw["form"] != "window":
        raise ValueError("the jacobi tshift2 switch applies to form='window' only")
    flags = ("tshift", "epilogue", "mconv", "tshift2")
    value = {"form": FORMS[sw["form"]], **{k: int(bool(sw[k])) for k in flags}}
    default = {
        "form": FORMS[SWITCHES["form"]],
        **{k: int(SWITCHES[k]) for k in flags},
    }
    return [f"-DJACOBI_{k.upper()}={v}" for k, v in value.items() if v != default[k]]


def cflags(size, in_fmt, acc_fmt, switches=None):
    _check_size(size)
    return [
        *dwf.fmt_cflags(in_fmt, acc_fmt),
        f"-DJACOBI_N={size}",
        f"-DJACOBI_ITERS={ITERS}",
        *switch_cflags(switches),
    ]


def make_inputs(size, rng):
    _check_size(size)
    return {"grid": rng.random((size, size))}


def encode(inputs, in_fmt, acc_fmt):
    dwf.check_pair(in_fmt, acc_fmt)
    return {"grid": dwf.encode(inputs["grid"], in_fmt)}


def outputs(size, acc_fmt):
    return {"out": (dwf.ACC_DTYPE[acc_fmt], (size, size))}


def decode(outputs, acc_fmt):
    return {"out": np.asarray(outputs["out"], np.float64)}


def golden(inputs, iters=ITERS):
    u = np.array(inputs["grid"], np.float64)
    for _ in range(iters):
        v = u.copy()
        v[1:-1, 1:-1] = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:])
        u = v
    return {"out": u}


def useful_flops(size, iters=ITERS):
    return 4 * (size - 2) ** 2 * iters


def tolerance(in_fmt, acc_fmt):
    return {"fp64": 1e-13, "fp32": 1e-6, "bf16": 2e-2, "fp16": 3e-3}[in_fmt]
