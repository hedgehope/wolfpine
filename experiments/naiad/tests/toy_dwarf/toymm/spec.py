"""toymm: the experiment harness's own toy dwarf (C = A x B), so the harness tests need no real kernel.

Follows the dwarf-kernel contract: ``ame`` runs the product on the matrix unit
one tile at a time, ``scalar`` is the F/D triple loop. Sizes are compile-time
(``-DTOY_N``) and must be multiples of the tile shape.
"""

import numpy as np

from framework.matrix.formats import get_format

NAME = "toymm"
VARIANTS = {"ame": "ame", "scalar": "scalar"}
SIZES = [16, 32]
FORMATS = [("fp32", "fp32"), ("fp64", "fp64"), ("bf16", "fp32"), ("fp16", "fp32")]
#: Optional: the pairs a non-AME variant builds for (the harness skips the rest).
VARIANT_FORMATS = {"scalar": [("fp32", "fp32"), ("fp64", "fp64")]}

_CODES = {"fp64": 0, "fp32": 1, "fp16": 2, "bf16": 3}
_C_TYPES = {"fp64": "double", "fp32": "float", "fp16": "uint16_t", "bf16": "uint16_t"}


def cflags(size, in_fmt, acc_fmt):
    return [
        f"-DTOY_N={int(size)}",
        f"-DTOY_IN_T={_C_TYPES[in_fmt]}",
        f"-DTOY_ACC_T={_C_TYPES[acc_fmt]}",
        f"-DTOY_IN_FMT={_CODES[in_fmt]}",
        f"-DTOY_ACC_FMT={_CODES[acc_fmt]}",
    ]


def make_inputs(size, rng):
    return {
        "A": rng.standard_normal((size, size)),
        "B": rng.standard_normal((size, size)),
    }


def encode(inputs, in_fmt, acc_fmt):
    fmt = get_format(in_fmt)
    out = {name: fmt.round(value).astype(fmt.dtype) for name, value in inputs.items()}
    out["C"] = np.zeros_like(inputs["A"], dtype=get_format(acc_fmt).dtype)
    return out


def outputs(size, acc_fmt):
    return {"C": (get_format(acc_fmt).dtype, (size, size))}


def decode(outputs, acc_fmt):
    return {"C": np.asarray(outputs["C"]).astype(np.float64)}


def golden(inputs):
    return {"C": inputs["A"] @ inputs["B"]}


def useful_flops(size):
    return 2 * size**3


def tolerance(in_fmt, acc_fmt):
    return {"fp64": 1e-12, "fp32": 1e-5}.get(in_fmt, 2e-2)
