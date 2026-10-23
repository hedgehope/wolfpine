"""Numerics for Naiad's parametric RISC-V matrix unit: formats, tile MAC, accuracy.

Pure functions over numpy arrays, with no ISA wiring and no timing (that is
``timing.py``'s, later). ``ml_dtypes`` (the ``naiad`` extra) is only needed for
the bf16/fp8 formats and is imported on first use.
"""

from framework.matrix.accuracy import compare, evaluate_formats
from framework.matrix.formats import FORMATS, Format, format_of_dtype, get_format
from framework.matrix.mac import fma_round_once, tile_mac
from framework.matrix.semiring import SEMIRINGS, tropical_product

__all__ = [
    "FORMATS",
    "SEMIRINGS",
    "Format",
    "compare",
    "evaluate_formats",
    "fma_round_once",
    "format_of_dtype",
    "get_format",
    "tile_mac",
    "tropical_product",
]
