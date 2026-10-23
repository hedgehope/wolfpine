"""Number formats for the parametric matrix unit (Naiad WP5).

A registry of name -> :class:`Format`. Each format knows its width, its storage
dtype and how to ``round`` an array to the nearest value it can represent
(round-to-nearest-even). Nothing here is timing: these are pure functions over
numpy arrays.

The narrow float types (bf16, fp8) come from ``ml_dtypes``, which is an optional
extra (``pip install wolfpine[naiad]``). It is imported on first use, so
``framework`` itself still imports without it; only rounding to bf16/fp8 needs it.

Rounding goes through fp32 for every format narrower than fp32, and that is only
correct if the fp32 step does not itself round: fp64 -> fp32 -> bf16 under RNE
twice gets ties wrong (``1 + 2**-8 + 2**-40`` lands on 1.0, not ``1 + 2**-7`` --
and ``ml_dtypes``' own fp64 -> bf16 cast does exactly that). So wide inputs are
first rounded to fp32 *to odd*: truncate, then set the last bit if anything was
lost. With fp32's 23 fraction bits at least two more than any narrower format's,
the second (RNE) rounding is then the correctly rounded result.
"""

from dataclasses import dataclass, field

import numpy as np

_ML_DTYPES = None


def _ml_dtypes():
    global _ML_DTYPES
    if _ML_DTYPES is None:
        try:
            import ml_dtypes
        except ImportError as e:
            raise ImportError(
                "bf16/fp8 formats need ml_dtypes: pip install 'wolfpine[naiad]'"
            ) from e
        _ML_DTYPES = ml_dtypes
    return _ML_DTYPES


@dataclass(frozen=True)
class Format:
    """One number format. Float formats carry their field widths; ints only ``bits``.

    ``mantissa_bits`` counts the explicit fraction bits (23 for fp32), and
    ``emin`` is the exponent of the smallest normal. ``max`` is the largest
    finite value, which is not the IEEE formula for fp8 e4m3fn (no infinities,
    and only S.1111.111 is NaN, so its top binade is usable up to 448).
    """

    name: str
    bits: int
    kind: str  # "float" or "int"
    _dtype: str = field(repr=False)
    mantissa_bits: int = 0
    exponent_bits: int = 0
    has_inf: bool = True

    @property
    def is_float(self):
        return self.kind == "float"

    @property
    def dtype(self):
        """The numpy storage dtype. tf32 is stored in an fp32 container."""
        if self._dtype.startswith("ml:"):
            return np.dtype(getattr(_ml_dtypes(), self._dtype[3:]))
        return np.dtype(self._dtype)

    @property
    def emin(self):
        return 2 - (1 << (self.exponent_bits - 1)) if self.is_float else 0

    @property
    def max(self):
        if not self.is_float:
            return (1 << (self.bits - 1)) - 1
        emax = (1 << (self.exponent_bits - 1)) - 1
        if not self.has_inf:
            # e4m3fn: the all-ones exponent is a normal binade bar its NaN code
            return (2 - 2.0 ** (1 - self.mantissa_bits)) * 2.0 ** (emax + 1)
        return (2 - 2.0**-self.mantissa_bits) * 2.0**emax

    @property
    def min(self):
        return -self.max if self.is_float else -(1 << (self.bits - 1))

    @property
    def eps(self):
        """Spacing of the format at 1.0 (``2**-mantissa_bits``); 1 for ints."""
        return 2.0**-self.mantissa_bits if self.is_float else 1.0

    def ulp(self, x):
        """The spacing of this format at each element of ``x`` (subnormal spacing below emin)."""
        x = np.asarray(x, dtype=np.float64)
        if not self.is_float:
            return np.ones_like(x)
        _, e = np.frexp(x)  # x = f * 2**e, f in [0.5, 1): the binade exponent is e - 1
        e = np.where(
            x == 0, self.emin + 1, e
        )  # frexp(0) has e == 0; zero is in the bottom binade
        return np.ldexp(1.0, np.maximum(e - 1, self.emin) - self.mantissa_bits)

    def round(self, x, *, saturate=False):
        """Round ``x`` to this format (RNE), returning an array of :attr:`dtype`.

        Out-of-range values follow the format's own conversion: overflow goes to
        inf, or to NaN for e4m3fn (which has no inf). ``saturate=True`` clamps
        to ``[min, max]`` instead (OCP's "satfinite"), infinities included.
        Ints round half-to-even to an integer and then wrap two's-complement,
        or clamp with ``saturate``; NaN becomes 0.
        """
        if not self.is_float:
            return self._round_int(x, saturate)
        x = np.asarray(x)
        if self.name == "fp64":
            out = x.astype(np.float64)
            return np.clip(out, self.min, self.max) if saturate else out
        if self.name == "fp32":
            src = x.astype(np.float32) if _is_narrow_float(x.dtype) else _f64_to_f32(x)
        else:
            src = _to_f32_sticky(x)
        if saturate:
            src = np.clip(src, np.float32(self.min), np.float32(self.max))
        if self.name == "fp32":
            return src
        if self.name == "tf32":
            return _f32_to_tf32(src)
        return src.astype(self.dtype)

    def _round_int(self, x, saturate):
        x = np.asarray(x)
        lo, hi = self.min, self.max
        if x.dtype.kind in "iub":
            v = x.astype(np.int64)
            if saturate:
                v = np.clip(v, lo, hi)
            return v.astype(
                self.dtype
            )  # numpy integer narrowing wraps two's-complement
        v = np.rint(np.nan_to_num(x.astype(np.float64), nan=0.0))
        if saturate:
            return np.clip(v, lo, hi).astype(self.dtype)
        # Wrap in float64 (exact for integers) so huge values never hit a UB cast
        span = float(1 << self.bits)
        v = np.mod(v, span)
        v = np.where(v > hi, v - span, v)
        return v.astype(self.dtype)


def _is_narrow_float(dtype):
    """True for dtypes that widen to fp32 exactly (fp16/fp32 and the ml_dtypes types)."""
    return dtype in (np.float16, np.float32) or (
        dtype.kind == "V" and dtype.itemsize <= 2
    )


def _f64_to_f32(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(x, dtype=np.float64).astype(np.float32)


def _to_f32_sticky(x):
    """``x`` as fp32, rounded to odd if it was wider -- see the module docstring."""
    x = np.asarray(x)
    if _is_narrow_float(x.dtype):
        return x.astype(np.float32)
    x = np.asarray(x, dtype=np.float64)
    y = _f64_to_f32(x)
    back = y.astype(np.float64)
    inexact = (back != x) & ~np.isnan(x)
    # RNE may have rounded away from zero (to inf, even): step back to truncate
    y = np.where(
        inexact & (np.abs(back) > np.abs(x)), np.nextafter(y, np.float32(0)), y
    )
    bits = y.view(np.uint32) | inexact.astype(np.uint32)
    return bits.view(np.float32)


def _f32_to_tf32(x):
    """RNE fp32 -> tf32: keep 10 of the 23 fraction bits, stored back in fp32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(13)) & np.uint32(1)
    # Adding just under half an ulp plus the kept lsb is RNE; a carry out of the
    # mantissa bumps the exponent, which is right, and past max lands on inf.
    rounded = (bits + np.uint32(0xFFF) + lsb) & np.uint32(0xFFFFE000)
    out = rounded.view(np.float32)
    return np.where(np.isnan(x), x, out)


FORMATS = {
    f.name: f
    for f in (
        Format("fp64", 64, "float", "float64", 52, 11),
        Format("fp32", 32, "float", "float32", 23, 8),
        Format("tf32", 19, "float", "float32", 10, 8),
        Format("fp16", 16, "float", "float16", 10, 5),
        Format("bf16", 16, "float", "ml:bfloat16", 7, 8),
        Format("fp8_e4m3", 8, "float", "ml:float8_e4m3fn", 3, 4, has_inf=False),
        Format("fp8_e5m2", 8, "float", "ml:float8_e5m2", 2, 5),
        Format("int8", 8, "int", "int8"),
        Format("int32", 32, "int", "int32"),
    )
}

_ALIASES = {
    "float64": "fp64",
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
    "e4m3": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "e5m2": "fp8_e5m2",
    "float8_e5m2": "fp8_e5m2",
}


def get_format(fmt):
    """A :class:`Format` from itself, its name or an alias (``"bfloat16"``, ``"e4m3"`` ...)."""
    if isinstance(fmt, Format):
        return fmt
    name = _ALIASES.get(fmt, fmt)
    if name not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")
    return FORMATS[name]


def format_of_dtype(dtype):
    """The registry format whose storage dtype is ``dtype`` (fp32 for float32, not tf32), or None."""
    dtype = np.dtype(dtype)
    for f in FORMATS.values():
        if f.name == "tf32" or (f._dtype.startswith("ml:") and dtype.kind != "V"):
            continue
        if f.dtype == dtype:
            return f
    return None
