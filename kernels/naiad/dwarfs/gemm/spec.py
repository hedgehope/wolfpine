# ruff: noqa: INP001  (loaded by path by framework.naiad.dwarfs, not imported as a package)
"""gemm: dense linear algebra, the sanity anchor. C = A . B, A m x k, B k x n.

The AME variant is the emitted micro-kernel's blocked driver (naiad_gemm.h,
framework/naiad/emit/gemm.py) straight on the I/O arrays; the scalar variant is
a 4 x 4 register-blocked FP loop computing in the accumulator's precision (fp32
for 16-bit inputs, which it widens in software inside the timed region, since
the dwarfs' design has no Zfh/Zfbfmin).

``size`` is n (square) or (m, n, k). Sizes that are not multiples of the 32 x 32
output block exercise the driver's padded edges.
"""

import numpy as np

NAME = "gemm"
VARIANTS = {"ame": "ame", "scalar": "scalar"}
SIZES = (32, 64, 128)
FORMATS = (
    ("fp64", "fp64"),
    ("fp32", "fp32"),
    ("bf16", "fp32"),
    ("fp16", "fp32"),
    ("bf16", "bf16"),
)

_TOLERANCE = {
    ("fp64", "fp64"): 1e-13,
    ("fp32", "fp32"): 1e-5,
    ("bf16", "fp32"): 1e-2,
    ("fp16", "fp32"): 2e-3,
    ("bf16", "bf16"): 3e-2,
}


def dims(size):
    """``(m, n, k)`` for an int or a 3-tuple size."""
    if isinstance(size, (tuple, list)):
        m, n, k = (int(v) for v in size)
        return m, n, k
    return int(size), int(size), int(size)


def _dtype(fmt):
    from framework.matrix.formats import get_format

    return get_format(fmt).dtype


def cflags(size, in_fmt, acc_fmt):
    m, n, k = dims(size)
    return [
        f"-DGEMM_M={m}",
        f"-DGEMM_N={n}",
        f"-DGEMM_K={k}",
        f"-DGEMM_IN={in_fmt}",
        f"-DGEMM_ACC={acc_fmt}",
        f"-DGEMM_IN_{in_fmt.upper()}=1",
        f"-DGEMM_ACC_{acc_fmt.upper()}=1",
    ]


def make_inputs(size, rng):
    m, n, k = dims(size)
    return {"A": rng.standard_normal((m, k)), "B": rng.standard_normal((k, n))}


def encode(inputs, in_fmt, acc_fmt):
    from framework.matrix.formats import get_format

    f = get_format(in_fmt)
    return {name: np.ascontiguousarray(f.round(inputs[name])) for name in ("A", "B")}


def outputs(size, acc_fmt):
    m, n, _ = dims(size)
    return {"C": (_dtype(acc_fmt), (m, n))}


def decode(outs, acc_fmt):
    return {"C": np.asarray(outs["C"]).astype(np.float64)}


def golden(inputs):
    return {"C": inputs["A"] @ inputs["B"]}


def useful_flops(size):
    m, n, k = dims(size)
    return 2 * m * n * k


def tolerance(in_fmt, acc_fmt):
    return _TOLERANCE.get((in_fmt, acc_fmt), 5e-2)
