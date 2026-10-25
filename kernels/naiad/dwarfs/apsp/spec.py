# ruff: noqa: INP001  (loaded by path by framework.naiad.dwarfs, not imported as a package)
"""Graph traversal / dynamic programming: all-pairs shortest paths in the (min, +) semiring.

Problem. A directed graph on n nodes with integer edge weights in [1, 16]: a
ring ``i -> i+1 (mod n)`` (so every pair is connected and no distance is
infinite) plus every other edge with probability ``min(1, 8/n)`` (about eight
out-edges a node). The input is the distance matrix ``D`` (0 on the diagonal,
the weight on an edge, +inf elsewhere); the output is the shortest-path
distance matrix, computed in place. ``size`` is n, a multiple of 16.

Reformulation (the ``ame`` variant): **blocked Floyd–Warshall** on 16 x 16
blocks with the matrix unit's tropical mma (``mma.minplus``, AME ``semiring``
feature): ``acc[i, j] = min(acc[i, j], min_k a[i, k] + b[k, j])``. For each
diagonal block k:

1. ``D_kk = D_kk*``, its closure: ``aload`` D_kk once, then four squarings
   ``acc = min(acc, D_kk ⊗ D_kk)`` (paths of up to 16 edges inside the block;
   the zero diagonal makes ``D ⊗ D <= D``), the tile reloaded after each
   ``astore``. The same tile is both operands (square tiles: the a and b views
   coincide).
2. Row and column panels: ``D_kj = min(D_kj, D_kk* ⊗ D_kj)``,
   ``D_ik = min(D_ik, D_ik ⊗ D_kk*)``.
3. Everything else: ``D_ij = min(D_ij, D_ik ⊗ D_kj)``; D_ik is loaded once per
   block row, the accumulator starts as D_ij (``aload``, since ``azero``'s 0 is
   not min's identity) and is stored back.

That is ``(n/16)^2 + 3`` mma per k, so the unit issues about ``n^3`` (min, +)
pairs, the same as Floyd–Warshall's ``n^3`` (one add and one min each:
:func:`useful_flops` is ``2 n^3``, so the redundancy is about 1).

Baselines: ``scalar`` is the plain Floyd–Warshall triple loop with ``fmin``;
``rvv`` vectorises its j loop (``vfadd.vf`` + ``vfmin.vv``, two rows i per load
of row k).

Exactness: min is exact and every sum of integer weights below 2^24 is exact in
fp32 (2^8 in bf16, where distances here stay well below 256), so results match
the fp64 golden exactly: the tolerance is 0 for fp32/fp64 and the bf16/bf16
pair (a bf16 accumulator, AME only) allows for a distance above 256 rounding.
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

NAME = "apsp"
#: ``*_mc``: the same kernel split across the harts of a multi-hart design (``cores.count`` > 1;
#: the source says how); on one hart each is its single-hart variant plus no-op barriers.
VARIANTS = {"ame": "ame", "scalar": "scalar", "rvv": "rvv", "ame_mc": "ame_mc"}
SIZES = (32, 64, 128)
#: The tropical mma works in one format: the distances are re-read as tiles every
#: phase, so the accumulator format is the input format.
FORMATS = tuple(
    p
    for p in (("fp32", "fp32"), ("fp64", "fp64"), ("bf16", "bf16"))
    if p[0] != "bf16" or dwf.HAVE_ML_DTYPES
)
VARIANT_FORMATS = {"rvv": (("fp32", "fp32"), ("fp64", "fp64"))}
MAX_WEIGHT = 16
OUT_DEGREE = 8
_DTYPE = {"fp64": np.float64, "fp32": np.float32, "bf16": np.uint16}


def _check(n):
    n = int(n)
    if n < 16 or n % 16:
        raise ValueError(f"apsp size must be a positive multiple of 16, got {n}")
    return n


def cflags(size, in_fmt, acc_fmt):
    n = _check(size)
    if in_fmt != acc_fmt or in_fmt not in _DTYPE:
        raise ValueError(
            f"apsp supports fp32/fp32, fp64/fp64 and bf16/bf16, not {in_fmt}/{acc_fmt}"
        )
    return [
        f"-DAPSP_N={n}",
        f"-DAPSP_FMT={in_fmt}",
        f"-DAPSP_MFMT=NAIAD_MFMT_{in_fmt.upper()}",
        f"-DAPSP_FMT_{in_fmt.upper()}=1",
    ]


def make_inputs(size, rng):
    n = _check(size)
    p = min(1.0, OUT_DEGREE / n)
    w = rng.integers(1, MAX_WEIGHT + 1, size=(n, n)).astype(np.float64)
    edge = rng.random((n, n)) < p
    edge[np.arange(n), (np.arange(n) + 1) % n] = True
    D = np.where(edge, w, np.inf)
    np.fill_diagonal(D, 0.0)
    return {"D": D}


def encode(inputs, in_fmt, acc_fmt):
    D = np.asarray(inputs["D"], np.float64)
    return {"D": dwf.encode(D, in_fmt).reshape(-1)}


def outputs(size, acc_fmt):
    n = _check(size)
    return {"D": (_DTYPE[acc_fmt], (n, n))}


def decode(outs, acc_fmt):
    return {"D": dwf.as_f64(outs["D"], acc_fmt)}


def golden(inputs):
    D = np.array(inputs["D"], np.float64)
    for k in range(D.shape[0]):
        D = np.minimum(D, D[:, k, None] + D[None, k, :])
    return {"D": D}


def useful_flops(size):
    n = _check(size)
    return 2 * n**3


def tolerance(in_fmt, acc_fmt):
    return 1e-2 if in_fmt == "bf16" else 0.0
