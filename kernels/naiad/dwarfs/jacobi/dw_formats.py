# ruff: noqa: INP001  (loaded by path by experiments.naiad.dwarfs, not imported as a package)
"""Host-side format helpers shared by the jacobi, reduce and scan dwarf specs (numpy only).

The C side is ``dw_fmt.h`` next to this file. A spec loads this module by path
(``importlib.util.spec_from_file_location``), because the harness imports each spec by file location, not as
a package. Encodings match the matrix unit's (``framework/matrix/formats.py``)
without importing it: fp64 and fp32 are numpy's, fp16 is numpy's float16 (a
single rounding from fp64), and bf16 is raw ``uint16`` bits, rounded once from
fp64 (round-to-odd into fp32, then nearest-even into bf16), so ``ml_dtypes`` is
not needed on the host. The matrix unit itself needs ``ml_dtypes`` for bf16,
so :data:`FORMATS` lists the bf16 pair only when it is importable.
"""

import importlib.util

import numpy as np

#: (in_fmt, acc_fmt) pairs every kernel built on dw_fmt.h supports.
ALL_PAIRS = (("fp32", "fp32"), ("fp64", "fp64"), ("bf16", "fp32"), ("fp16", "fp32"))
HAVE_ML_DTYPES = importlib.util.find_spec("ml_dtypes") is not None
FORMATS = tuple(p for p in ALL_PAIRS if p[0] != "bf16" or HAVE_ML_DTYPES)

ACC_DTYPE = {"fp64": np.float64, "fp32": np.float32}
IN_DTYPE = {
    "fp64": np.float64,
    "fp32": np.float32,
    "fp16": np.float16,
    "bf16": np.uint16,
}
IN_BYTES = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2}


def check_pair(in_fmt, acc_fmt):
    if (in_fmt, acc_fmt) not in ALL_PAIRS:
        raise ValueError(
            f"unsupported format pair {in_fmt}/{acc_fmt}; supported: {ALL_PAIRS}"
        )


def fmt_cflags(in_fmt, acc_fmt):
    check_pair(in_fmt, acc_fmt)
    return [
        f"-DDW_IN_FMT=NAIAD_MFMT_{in_fmt.upper()}",
        f"-DDW_ACC_FMT=NAIAD_MFMT_{acc_fmt.upper()}",
    ]


def _f64_to_bf16_bits(x):
    x = np.asarray(x, np.float64)
    with np.errstate(over="ignore"):
        f = x.astype(np.float32)
    back = f.astype(np.float64)
    inexact = back != x
    u = f.view(np.uint32).astype(np.uint64)
    away = inexact & (np.abs(back) > np.abs(x))
    u[away] -= 1  # sign-magnitude: one step toward zero
    u[inexact] |= 1  # round to odd
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_bits_to_f64(b):
    return (
        (np.asarray(b, np.uint16).astype(np.uint32) << 16)
        .view(np.float32)
        .astype(np.float64)
    )


def encode(x, fmt):
    """fp64 ``x`` rounded once to ``fmt``, in the kernel's storage dtype."""
    if fmt == "bf16":
        return _f64_to_bf16_bits(x)
    return np.asarray(x, np.float64).astype(IN_DTYPE[fmt])


def as_f64(stored, fmt):
    """Storage-dtype values of ``fmt`` (as :func:`encode` makes them) back to fp64."""
    if fmt == "bf16":
        return _bf16_bits_to_f64(stored)
    return np.asarray(stored).astype(np.float64)


def round_to(x, fmt):
    """``x`` rounded to ``fmt``, as fp64."""
    return as_f64(encode(x, fmt), fmt)
