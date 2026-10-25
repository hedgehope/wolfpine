# ruff: noqa: INP001  (loaded by path by framework.naiad.dwarfs, not imported as a package)
"""gemm: dense linear algebra, the sanity anchor. C = A . B, A m x k, B k x n.

The AME variant is the emitted micro-kernel's blocked driver (naiad_gemm.h,
framework/naiad/emit/gemm.py) straight on the I/O arrays; the scalar variant is
a 4 x 4 register-blocked FP loop computing in the accumulator's precision (fp32
for 16-bit inputs, which it widens in software inside the timed region, since
the dwarfs' design has no Zfh/Zfbfmin).

The ``vme`` variant is the emitted Zvt driver (naiad_gemm.h from a VME design,
framework/naiad/emit/gemm_vme.py): A transposed through the tile state into a
panel, B in place, a block shape the emitter picks for the design.

The ``ame_mc`` variant is the AME driver with C's rows split across the harts
of a multi-hart design (``naiad-dwarfs-4c.yaml``; each hart has its own unit),
timed from a barrier to a barrier; on one hart it is the ame variant.

The ``tensix`` variant offloads to a Tensix tile on the bus (``naiad-tensix.yaml``,
kernels/naiad/include/tensix_accel.h): its region is the whole round trip (launch
image, DMA in with 2D-DMA tilize, doorbell, wait, DMA out), the tile's one-time
firmware boot excluded.

``size`` is n (square) or (m, n, k). Sizes that are not multiples of the 32 x 32
output block exercise the driver's padded edges.
"""

import numpy as np

NAME = "gemm"
VARIANTS = {
    "ame": "ame",
    "scalar": "scalar",
    "rvv": "rvv",
    "ime": "ime",
    "ame_mc": "ame_mc",
    "vme": "vme",
    "tensix": "tensix",
}
SIZES = (32, 64, 128)
FORMATS = (
    ("fp64", "fp64"),
    ("fp32", "fp32"),
    ("bf16", "fp32"),
    ("fp16", "fp32"),
    ("bf16", "bf16"),
)

#: The RVV baseline runs every pair (Zvfh / Zvfbfwma widening FMAs; bf16/bf16 rounds the fp32 sum once).
VARIANT_FORMATS = {
    "rvv": FORMATS,
    # IME (Zvvm; needs an IME design, e.g. naiad-ime-timed.yaml): Zvvfp64mm, Zvvfp32mm,
    # Zvvbf16fp32mm, Zvvfp16fp32mm and Zvvbf16mm (bf16 C rounded after every sub-dot-product).
    "ime": FORMATS,
    # VME (Zvt v0.4; needs a VME design, e.g. naiad-vme-timed.yaml): Zvtdmm, Zvtfmm, Zvtbf16fmm,
    # Zvtfp16fmm; bf16 -> bf16 accumulates in fp32 tiles and rounds once on store (no bf16 C in Zvt).
    "vme": FORMATS,
    # A Blackhole Tensix tile behind MMIO + DMA (needs naiad-tensix.yaml): the captured
    # tt-metal matmul_tiles image, bf16 in at HiFi2; whole 32x32 tiles only.
    "tensix": (("bf16", "fp32"), ("bf16", "bf16")),
}

#: Variants that need a block of this type in the design (the harness skips them elsewhere).
VARIANT_BLOCKS = {"tensix": "tensix"}

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
