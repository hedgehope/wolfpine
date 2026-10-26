# ruff: noqa: INP001  (loaded by path by experiments.naiad.dwarfs, not imported as a package)
"""tensorprod: the spectral-element tensor-product kernel (Nekbone's local gradient).

For E elements of p x p x p data u[e][k][j][i] (i fastest) and a p x p 1D
operator D, compute the derivative along each direction:

    ur[e,k,j,i] = sum_l D[i,l] u[e,k,j,l]      (r: the contiguous index)
    us[e,k,j,i] = sum_l D[j,l] u[e,k,l,i]      (s: the middle index)
    ut[e,k,j,i] = sum_l D[k,l] u[e,l,j,i]      (t: the slowest index)

These are small matmuls (p = 8..16) with awkward shapes. The AME variant keeps a
block-diagonal operator, ``I_q (x) D`` (q = floor(T / p) copies, zero-padded to
the T x T tile), resident in one tile and streams u through the other operand,
so q slabs share one mma when p < T/2:

* t: ``[ut_e; ut_e+1; ..] = (I_q (x) D) . [U_e; U_e+1; ..]``, U_e = u_e as p rows of p^2;
* r: ``U' . (I_q (x) D^T)`` with U' = u viewed as rows of q*p (q consecutive i-rows);
* s: ``(I_q (x) D) . [u_ek; u_ek+1; ..]``, u_ek = slab (e, k) as p rows of p.

Unused tile rows/columns are zero in the operator, so they cost MACs but not
accuracy; that padding is what the per-p utilisation shows (t and r use q*p^2/T^2
of each mma, s only q*p^3/T^3, since its free index i is the short one).

Both variants write the same padded output layouts (the AME tiles' natural
ones; the scalar loop just indexes them), decoded here:

* ``out_t``: rows (e*p + k), row stride LDT = roundup(p^2, T), column (j*p + i);
* ``out_r``: rows of q i-rows, row stride T, column (c*p + i) for i-row q*row + c;
* ``out_s``: rows (e*p^2 + k*p + j), row stride T, column i;
* ``tp_dims``: (p, E), echoed by the kernel so ``decode`` can unpad.

``size`` is p, or (p, E); E defaults to 8. T is the matrix unit's tile side
(``tile`` below, 16 in the dwarfs' design; the kernel #errors if it differs).
"""

import numpy as np

NAME = "tensorprod"
VARIANTS = {"ame": "ame", "scalar": "scalar", "rvv": "rvv"}
SIZES = (8, 10, 12, 16)
FORMATS = (
    ("fp64", "fp64"),
    ("fp32", "fp32"),
    ("bf16", "fp32"),
    ("fp16", "fp32"),
)
#: The RVV baseline runs every pair (Zvfh / Zvfbfwma widening FMAs into fp32).
VARIANT_FORMATS = {"rvv": FORMATS}
DEFAULT_E = 8
TILE = 16

_TOLERANCE = {
    ("fp64", "fp64"): 1e-13,
    ("fp32", "fp32"): 1e-5,
    ("bf16", "fp32"): 1e-2,
    ("fp16", "fp32"): 2e-3,
}


def dims(size):
    """``(p, E)``."""
    if isinstance(size, (tuple, list)):
        p, e = (int(v) for v in size)
        return p, e
    return int(size), DEFAULT_E


def _roundup(x, m):
    return -(-x // m) * m


def layout(size, tile=TILE):
    """Every layout number the kernels use (they compute the same in C)."""
    p, e = dims(size)
    if not 1 <= p <= tile:
        raise ValueError(f"tensorprod needs 1 <= p <= {tile}, got {p}")
    q = tile // p
    ldt = _roundup(p * p, tile)
    groups_t = -(-e // q)
    rows_r = -(-(e * p * p) // q)
    groups_s = -(-(e * p) // q)
    return {
        "p": p,
        "E": e,
        "q": q,
        "T": tile,
        "LDT": ldt,
        "rows_t": groups_t * q * p + tile,
        "rows_r": _roundup(rows_r, tile),
        "rows_s": groups_s * q * p + tile,
        "jobs_t": groups_t * (ldt // tile),
        "jobs_r": _roundup(rows_r, tile) // tile,
        "jobs_s": groups_s,
    }


def _dtype(fmt):
    from framework.matrix.formats import get_format

    return get_format(fmt).dtype


def cflags(size, in_fmt, acc_fmt):
    p, e = dims(size)
    return [
        f"-DTP_P={p}",
        f"-DTP_E={e}",
        f"-DTP_IN={in_fmt}",
        f"-DTP_ACC={acc_fmt}",
        f"-DTP_IN_{in_fmt.upper()}=1",
        f"-DTP_ACC_{acc_fmt.upper()}=1",
        f"-DTP_MFMT_IN=NAIAD_MFMT_{in_fmt.upper()}",
        f"-DTP_MFMT_ACC=NAIAD_MFMT_{acc_fmt.upper()}",
    ]


def make_inputs(size, rng):
    p, e = dims(size)
    return {
        "D": rng.standard_normal((p, p)),
        "u": rng.standard_normal((e, p, p, p)),
    }


def encode(inputs, in_fmt, acc_fmt):
    from framework.matrix.formats import get_format

    f = get_format(in_fmt)
    return {
        "D": np.ascontiguousarray(f.round(inputs["D"])),
        "u": np.ascontiguousarray(f.round(inputs["u"])),
    }


def outputs(size, acc_fmt):
    lay = layout(size)
    dt = _dtype(acc_fmt)
    return {
        "out_t": (dt, (lay["rows_t"], lay["LDT"])),
        "out_r": (dt, (lay["rows_r"], lay["T"])),
        "out_s": (dt, (lay["rows_s"], lay["T"])),
        "tp_dims": (np.int64, (2,)),
    }


def decode(outs, acc_fmt):
    # decode() is not given the size, so the kernel echoes (p, E) in tp_dims.
    p, e = (int(v) for v in outs["tp_dims"])
    q = layout((p, e))["q"]
    t = np.asarray(outs["out_t"], np.float64)[: e * p, : p * p].reshape(e, p, p, p)
    r = np.asarray(outs["out_r"], np.float64)[:, : q * p].reshape(-1)[: e * p**3]
    s = np.asarray(outs["out_s"], np.float64)[: e * p * p, :p]
    return {"ur": r.reshape(e, p, p, p), "us": s.reshape(e, p, p, p), "ut": t}


def golden(inputs):
    D, u = inputs["D"], inputs["u"]
    return {
        "ur": np.einsum("il,ekjl->ekji", D, u),
        "us": np.einsum("jl,ekli->ekji", D, u),
        "ut": np.einsum("kl,elji->ekji", D, u),
    }


def useful_flops(size):
    p, e = dims(size)
    return 3 * 2 * e * p**4


def mma_count(size):
    lay = layout(size)
    return lay["jobs_t"] + lay["jobs_r"] + lay["jobs_s"]


def tolerance(in_fmt, acc_fmt):
    return _TOLERANCE.get((in_fmt, acc_fmt), 5e-2)
