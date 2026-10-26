# ruff: noqa: INP001  (loaded by path by experiments.naiad.dwarfs, not imported as a package)
"""fft: spectral methods. Batched complex 1D FFT, DFT-16/32 blocks on the matrix unit.

Variants:

* ``ame``: Stockham-ordered mixed-radix FFT whose radix-16/32 butterflies are
  DFT matrices on the matrix unit (``F_r`` times a slab of the data), with
  the inter-stage twiddles applied on the scalar side in the accumulator
  format and the result rounded back to the input format for the next stage.
  Complex arithmetic is four real mma per tile pair (real-block form). See
  ``ame/fft_ame.c``.
* ``ame_gauss``: the same with the 3-mma Gauss complex product (``-DFFT_GAUSS=1``
  on the same source): 25% fewer MACs, one more rounding and a cancelling combine.
* ``ame_fused``: ``ame`` with the inter-stage twiddle multiply done on the
  accumulators (the ``ewise`` feature: ``aew.at`` / ``amacc.at`` with the
  twiddle block as two ``acc_fmt`` tiles) and the rounding back to ``in_fmt``
  done by ``amov.t.a`` + ``tconv`` (``acc_move``, ``convert``; a scalar pass
  without them), so the data never visits the scalar core. See
  ``ame_fused/fft_ame_fused.c``. Needs a design listing ``ewise``
  (:data:`VARIANT_FEATURES`).
* ``ame_rvv``: ``ame`` with the scalar twiddle pass replaced by an RVV one
  (the unfused hybrid: the data still goes accumulator -> memory -> vector
  registers -> memory -> tile between stages). See ``ame_rvv/fft_ame_rvv.c``.
* ``scalar``: iterative radix-2 Cooley-Tukey in fp32 (fp64 for an fp64
  accumulator) on the F/D unit.

``size`` is ``n`` or ``(n, batch)``: ``batch`` transforms of length ``n``
(default batch :data:`DEFAULT_BATCH`). The AME plan factors ``n`` into
radices 16 and 32 (:func:`plan`).

Twiddle precision: every constant (the DFT matrices, the inter-stage and the
radix-2 twiddles) is computed on the host in fp64, with the angles reduced
exactly so cos/sin of multiples of pi/2 are exact, then rounded once (RNE) to
the format the kernel reads it in: the DFT matrices to ``in_fmt`` (they are
matrix operands), the twiddles to ``acc_fmt`` (they multiply on the scalar
side). The kernel never evaluates a trigonometric function.

Numpy only. bf16 is produced with numpy bit manipulation (fp64 -> fp32 by
round-to-odd, then RNE to bf16), so ``ml_dtypes`` is not needed.
"""

import numpy as np

NAME = "fft"
VARIANTS = {
    "ame": "ame",
    "ame_gauss": "ame_gauss",
    "ame_fused": "ame_fused",
    "ame_rvv": "ame_rvv",
    "scalar": "scalar",
    "rvv": "rvv",
}
#: Built with the design's vector extensions (the hybrid's twiddle pass is RVV).
VECTOR_VARIANTS = ("rvv", "ame_rvv")
#: Matrix features a variant cannot build without (the harness skips it on a design lacking them).
VARIANT_FEATURES = {"ame_fused": ("ewise",)}
DEFAULT_BATCH = 4
SIZES = (256, 1024, 4096)
FORMATS = [("fp32", "fp32"), ("fp64", "fp64"), ("bf16", "fp32"), ("fp16", "fp32")]
#: The RVV baseline runs every pair (16-bit inputs widen in the first pass, then fp32).
VARIANT_FORMATS = {"rvv": FORMATS}

_PLANS = {
    256: (16, 16),
    512: (32, 16),
    1024: (32, 32),
    4096: (16, 16, 16),
    8192: (32, 16, 16),
}
_MFMT = {
    "fp64": "NAIAD_MFMT_FP64",
    "fp32": "NAIAD_MFMT_FP32",
    "fp16": "NAIAD_MFMT_FP16",
    "bf16": "NAIAD_MFMT_BF16",
}


def _shape(size):
    if isinstance(size, (tuple, list)):
        n, batch = size
    else:
        n, batch = size, DEFAULT_BATCH
    return int(n), int(batch)


def plan(n):
    """The AME variant's radices for length ``n`` (each 16 or 32)."""
    if n not in _PLANS:
        raise ValueError(
            f"fft: no radix-16/32 plan for n={n}; supported: {sorted(_PLANS)}"
        )
    return _PLANS[n]


def _twiddle_len(n):
    radices, total, b = plan(n), 0, 1
    for r in radices[:-1]:
        total += n // b  # r * m' = m entries for this stage
        b *= r
    return max(total, 1)


def cflags(size, in_fmt, acc_fmt):
    n, batch = _shape(size)
    radices = plan(n)
    flags = [
        f"-DDW_IN_FMT={_MFMT[in_fmt]}",
        f"-DDW_ACC_FMT={_MFMT[acc_fmt]}",
        f"-DFFT_N={n}",
        f"-DFFT_BATCH={batch}",
        f"-DFFT_S={len(radices)}",
        f"-DFFT_TW_LEN={_twiddle_len(n)}",
    ]
    flags += [f"-DFFT_R{i}={r}" for i, r in enumerate(radices)]
    return flags


def make_inputs(size, rng):
    n, batch = _shape(size)
    x = rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))
    return {"x": x}


# -- formats (numpy only) -----------------------------------------------------


def _round_to_odd_f32(x):
    """fp64 -> fp32 with round-to-odd, so a following narrowing rounds once."""
    x = np.asarray(x, np.float64)
    f = x.astype(np.float32)
    bits = f.view(np.uint32).copy()
    inexact = np.isfinite(x) & (f.astype(np.float64) != x)
    even = (bits & 1) == 0
    fix = inexact & even
    # Move one ulp towards x (the truncation), which makes the last bit odd.
    away = np.abs(f.astype(np.float64)) > np.abs(x)
    bits[fix & away] -= 1
    bits[fix & ~away] += 1
    return bits.view(np.float32)


def to_storage(x, fmt):
    """fp64 values -> the kernel's storage array for ``fmt`` (RNE, one rounding)."""
    x = np.asarray(x, np.float64)
    if fmt == "fp64":
        return x.copy()
    if fmt == "fp32":
        return x.astype(np.float32)
    if fmt == "fp16":
        return x.astype(np.float16).view(np.uint16)
    if fmt == "bf16":
        u = _round_to_odd_f32(x).view(np.uint32).astype(np.uint64)
        nan = (u & 0x7FFFFFFF) > 0x7F800000
        out = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        out[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
        return out
    raise ValueError(f"fft: unsupported format {fmt!r}")


def from_storage(a, fmt):
    a = np.asarray(a)
    if fmt in ("fp64", "fp32"):
        return a.astype(np.float64)
    if fmt == "fp16":
        return a.view(np.float16).astype(np.float64)
    if fmt == "bf16":
        return (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    raise ValueError(f"fft: unsupported format {fmt!r}")


def round_to(x, fmt):
    """``x`` rounded to ``fmt`` and back to fp64."""
    return from_storage(to_storage(x, fmt), fmt)


def _cis(p, q):
    """exp(-2 pi i p / q) in fp64, exact at multiples of pi/2."""
    p = np.asarray(p, np.int64) % q
    z = np.exp(-2j * np.pi * p / q)
    quarter = (4 * p) % q == 0
    exact = np.array([1, -1j, -1, 1j])[(4 * p[quarter]) // q]
    z[quarter] = exact
    return z


def dft_matrix(r):
    k = np.arange(r)
    return _cis(np.outer(k, k), r)


def stage_twiddles(n):
    """Concatenated W_m^(k j) tables for every AME stage but the last (complex fp64)."""
    out, b = [], 1
    for r in plan(n)[:-1]:
        m = n // b
        mp = m // r
        out.append(_cis(np.outer(np.arange(r), np.arange(mp)), m).reshape(-1))
        b *= r
    return np.concatenate(out) if out else np.ones(1, complex)


def encode(inputs, in_fmt, acc_fmt):
    x = np.asarray(inputs["x"])
    batch, n = x.shape
    enc = {
        "xr": to_storage(x.real.reshape(-1), in_fmt),
        "xi": to_storage(x.imag.reshape(-1), in_fmt),
    }
    tw = stage_twiddles(n)
    enc["twr"], enc["twi"] = to_storage(tw.real, acc_fmt), to_storage(tw.imag, acc_fmt)
    w2 = _cis(np.arange(n // 2), n)
    enc["w2r"], enc["w2i"] = to_storage(w2.real, acc_fmt), to_storage(w2.imag, acc_fmt)
    for r in (16, 32):
        f = dft_matrix(r)
        fi = to_storage(f.imag, in_fmt)
        enc[f"f{r}r"] = to_storage(f.real, in_fmt).reshape(-1)
        enc[f"f{r}i"] = fi.reshape(-1)
        enc[f"f{r}n"] = to_storage(-from_storage(fi, in_fmt), in_fmt).reshape(-1)
        enc[f"f{r}s"] = to_storage(f.real + f.imag, in_fmt).reshape(-1)
    return enc


def _acc_dtype(acc_fmt):
    return {"fp64": np.float64, "fp32": np.float32}[acc_fmt]


def outputs(size, acc_fmt):
    n, batch = _shape(size)
    dt = _acc_dtype(acc_fmt)
    return {"yr": (dt, (batch, n)), "yi": (dt, (batch, n))}


def decode(outputs, acc_fmt):
    yr = np.asarray(outputs["yr"], np.float64)
    yi = np.asarray(outputs["yi"], np.float64)
    return {"X": np.stack([yr, yi], axis=-1)}


def golden(inputs):
    """fp64 FFT of the unrounded inputs, as (batch, n, 2) [re, im]."""
    X = np.fft.fft(np.asarray(inputs["x"]), axis=-1)
    return {"X": np.stack([X.real, X.imag], axis=-1)}


def useful_flops(size):
    """5 n log2 n per transform (the standard radix-2 count), times the batch."""
    n, batch = _shape(size)
    return int(5 * n * np.log2(n)) * batch


def tolerance(in_fmt, acc_fmt):
    """Normwise pass threshold against the fp64 FFT of the unrounded input."""
    return {
        ("fp64", "fp64"): 1e-14,
        ("fp32", "fp32"): 1e-6,
        ("fp16", "fp32"): 2e-3,
        ("bf16", "fp32"): 1.5e-2,
    }.get((in_fmt, acc_fmt), 5e-2)


def model(x, radices):
    """Numpy model of the AME variant's Stockham decomposition (exact arithmetic, fp64)."""
    x = np.asarray(x, complex)
    n = x.shape[-1]
    cur, b = x.reshape(-1, n).copy(), 1
    for s, r in enumerate(radices):
        m = n // b
        mp = m // r
        y = np.einsum("kn,tbnj->tkbj", dft_matrix(r), cur.reshape(-1, b, r, mp))
        if s < len(radices) - 1:
            y = y * _cis(np.outer(np.arange(r), np.arange(mp)), m)[None, :, None, :]
        cur, b = y.reshape(-1, n), b * r
    return cur.reshape(x.shape)
