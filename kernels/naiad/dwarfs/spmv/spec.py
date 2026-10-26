# ruff: noqa: INP001  (loaded by path by experiments.naiad.dwarfs, not imported as a package)
"""Sparse linear algebra: Block-Sparse-Row (BSR) sparse matrix times a block of vectors, Y = A X.

Problem. ``A`` is n x n, stored as BSR with 16 x 16 blocks (the AME tile shape):
``nnzb`` nonzero blocks at distinct, uniformly random block positions, each block
dense (standard normal values), so the *block density* ``nnzb / (n/16)^2`` is the
only sparsity parameter. ``X`` is n x nv (standard normal) and ``Y = A X`` is
n x nv, in the accumulator format.

``size`` is ``(n, density)`` or ``(n, density, nv)``; n is a multiple of 16,
``nnzb = max(1, round(density * (n/16)^2))`` (so the block count, and with it the
build, is fixed by the size and not by the seed), and ``nv`` defaults to 16.

**Why nv = 16 by default.** A single vector (nv = 1, a true SpMV) uses one of the
16 columns the mma computes, so 15/16 of the matrix unit's work is padding: the
redundancy is 16x by construction. With nv = 16 (SpMM against a block of
vectors, as block Krylov solvers, multiple right-hand sides and GNN feature
propagation do) one block is exactly one 16 x 16 x 16 mma and nothing is
wasted, so nv = 16 is the matrix unit's fair case and nv = 1 its worst. Both are
in the sweeps; any 1 <= nv <= 16 works.

Storage (identical for every variant): ``rowptr`` (int32, nb + 1) and ``colidx``
(int32, nnzb; block-column indices, sorted within a block row), ``vals``
(nnzb x 16 x 16 input-format elements, each block row-major and contiguous),
``X`` (n x nv row-major) and ``Y`` (n x nv row-major).

Variants:

* ``ame``: per block row one accumulator (two alternate, so a row's astore
  overlaps the next row's mma); each block is ``tload.a`` of the block, ``tload.b``
  of the 16 rows of X it touches (row stride nv elements: for nv < 16 the
  extra 16 - nv columns are neighbouring X rows, i.e. junk that only reaches
  output columns that are never stored), one mma. Loads run one block ahead
  of their mma, across block-row boundaries. nv = 16 stores the accumulator
  straight into Y; nv < 16 stores it to a staging tile and copies nv columns.
* ``scalar``: the same BSR walk with F/D, one running sum per (row, vector).
* ``rvv``: nv >= 2: the vector is over the nv right-hand sides (VL = nv):
  ``y_r += A[r][c] * X_c`` as ``vfmacc.vf``, four rows at a time so each X row
  is loaded once per four FMAs. nv = 1: the vector is over a block row's 16
  columns, ``acc_r += A_r * x_blk`` (``vfmacc.vv``), four rows at a time, one
  ``vfredusum`` per row at the end of the block row.

Useful FLOPs are ``2 * 256 * nnzb * nv`` (one MAC per stored element per vector).
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

NAME = "spmv"
#: ``*_mc``: the same kernel split across the harts of a multi-hart design (``cores.count`` > 1;
#: the source says how); on one hart each is its single-hart variant plus no-op barriers.
VARIANTS = {"ame": "ame", "scalar": "scalar", "rvv": "rvv", "ame_mc": "ame_mc"}
#: (n, block density): about four blocks per block row at every size.
SIZES = ((128, 0.5), (256, 0.25), (512, 0.125))
FORMATS = dwf.FORMATS
VARIANT_FORMATS = {"rvv": FORMATS}
BLOCK = 16
DEFAULT_NV = 16

_TOLERANCE = {"fp64": 1e-13, "fp32": 1e-5, "bf16": 1e-2, "fp16": 2e-3}


def dims(size):
    """``(n, density, nv, nnzb)`` for a size."""
    if not isinstance(size, (tuple, list)) or len(size) not in (2, 3):
        raise ValueError(f"spmv size is (n, density) or (n, density, nv), got {size!r}")
    n, density = int(size[0]), float(size[1])
    nv = int(size[2]) if len(size) == 3 else DEFAULT_NV
    if n < BLOCK or n % BLOCK:
        raise ValueError(f"spmv n must be a positive multiple of {BLOCK}, got {n}")
    if not 0 < density <= 1:
        raise ValueError(f"spmv block density must be in (0, 1], got {density}")
    if not 1 <= nv <= BLOCK:
        raise ValueError(f"spmv nv must be in 1..{BLOCK}, got {nv}")
    nb = n // BLOCK
    nnzb = max(1, min(nb * nb, round(density * nb * nb)))
    return n, density, nv, nnzb


def cflags(size, in_fmt, acc_fmt):
    n, _, nv, nnzb = dims(size)
    return [
        *dwf.fmt_cflags(in_fmt, acc_fmt),
        f"-DSPMV_N={n}",
        f"-DSPMV_NNZB={nnzb}",
        f"-DSPMV_NV={nv}",
    ]


def make_inputs(size, rng):
    n, _, nv, nnzb = dims(size)
    nb = n // BLOCK
    pos = np.sort(rng.choice(nb * nb, size=nnzb, replace=False))
    rows, cols = np.divmod(pos, nb)
    rowptr = np.zeros(nb + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=nb), out=rowptr[1:])
    return {
        "rowptr": rowptr,
        "colidx": cols.astype(np.int64),
        "vals": rng.standard_normal((nnzb, BLOCK, BLOCK)),
        "X": rng.standard_normal((n, nv)),
    }


def encode(inputs, in_fmt, acc_fmt):
    dwf.check_pair(in_fmt, acc_fmt)
    return {
        "rowptr": np.asarray(inputs["rowptr"], np.int32),
        "colidx": np.asarray(inputs["colidx"], np.int32),
        "vals": dwf.encode(inputs["vals"], in_fmt).reshape(-1),
        "X": dwf.encode(inputs["X"], in_fmt).reshape(-1),
    }


def outputs(size, acc_fmt):
    n, _, nv, _ = dims(size)
    return {"Y": (dwf.ACC_DTYPE[acc_fmt], (n, nv))}


def decode(outs, acc_fmt):
    return {"Y": np.asarray(outs["Y"], np.float64)}


def dense(inputs):
    """The dense fp64 matrix of the BSR ``inputs``."""
    rowptr, colidx, vals = inputs["rowptr"], inputs["colidx"], inputs["vals"]
    nb = len(rowptr) - 1
    A = np.zeros((nb * BLOCK, nb * BLOCK))
    for bi in range(nb):
        for p in range(rowptr[bi], rowptr[bi + 1]):
            bj = colidx[p]
            A[bi * BLOCK : (bi + 1) * BLOCK, bj * BLOCK : (bj + 1) * BLOCK] = vals[p]
    return A


def golden(inputs):
    return {"Y": dense(inputs) @ np.asarray(inputs["X"], np.float64)}


def useful_flops(size):
    _, _, nv, nnzb = dims(size)
    return 2 * BLOCK * BLOCK * nnzb * nv


def tolerance(in_fmt, acc_fmt):
    return _TOLERANCE.get(in_fmt, 5e-2)
