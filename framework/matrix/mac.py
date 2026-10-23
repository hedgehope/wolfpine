"""The functional tile multiply-accumulate core (Naiad WP3, numerics only).

``tile_mac`` is ``C + A @ B`` with the precision a matrix unit actually has:
inputs rounded to ``in_fmt``, products and sums carried in fp32 (fp64 when the
accumulator is fp64), and the result rounded to ``acc_fmt``. It returns a new
array and never touches its arguments; the ISA layer owns any state.

Products are exact in the compute dtype for every input format up to tf32
(bf16 x bf16 is 16 significant bits, tf32 x tf32 22, both under fp32's 24), so
the only roundings are in the sums and into the accumulator. numpy's matmul on
ml_dtypes arrays is a per-element loop rather than BLAS, which is why the
inputs are widened before multiplying.

``k_chunk`` is the bit-exact mode: K is split into pieces of that size, each
piece is summed in the compute dtype in K order (a defined order, independent
of BLAS), and the accumulator is rounded to ``acc_fmt`` after every piece --
which is what a datapath that writes a narrow accumulator back each pass does.
Without it the whole product is one BLAS call and one final rounding.
"""

import numpy as np

from framework.matrix.formats import get_format
from framework.matrix.semiring import SEMIRINGS, tropical_accumulate, tropical_product


def _check_shapes(C, A, B):
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2-D, got shapes {A.shape} and {B.shape}")
    (M, K), (K2, N) = A.shape, B.shape
    if K != K2:
        raise ValueError(f"inner dimensions differ: A is {A.shape}, B is {B.shape}")
    if C is not None and np.shape(C) != (M, N):
        raise ValueError(
            f"C must be ({M}, {N}) for A {A.shape} x B {B.shape}, got {np.shape(C)}"
        )
    return M, K, N


def tile_mac(C, A, B, in_fmt, acc_fmt, *, k_chunk=None, semiring="plus_times"):
    """Return ``C ⊕ (A ⊗ B)`` for (M, K) ``A``, (K, N) ``B`` and (M, N) ``C``.

    ``C=None`` is the zero accumulator (the semiring's identity for tropical
    products). Float formats pair with float formats; ints are int8/int32 in,
    int32 out, with two's-complement wraparound -- the sum is formed in int64
    and wrapped once, which is the same as wrapping after every add. The
    result is an array of ``acc_fmt``'s storage dtype.
    """
    in_f, acc_f = get_format(in_fmt), get_format(acc_fmt)
    A, B = np.asarray(A), np.asarray(B)
    M, K, N = _check_shapes(C, A, B)
    if semiring not in SEMIRINGS:
        raise ValueError(f"unknown semiring {semiring!r}; expected one of {SEMIRINGS}")
    if k_chunk is not None and (int(k_chunk) != k_chunk or k_chunk < 1):
        raise ValueError(f"k_chunk must be a positive integer, got {k_chunk!r}")
    if in_f.is_float != acc_f.is_float:
        raise ValueError(
            f"cannot mix float and int formats: {in_f.name} in, {acc_f.name} accumulate"
        )

    if not in_f.is_float:
        cdt = np.int64
    else:
        cdt = np.float64 if acc_f.name == "fp64" else np.float32
    a, b = in_f.round(A).astype(cdt), in_f.round(B).astype(cdt)

    with np.errstate(over="ignore", invalid="ignore"):
        if semiring != "plus_times":
            # An int result can only overflow at the identity (an empty K), and
            # saturating maps int64's extreme onto int32's -- still the identity
            sat = not acc_f.is_float
            if C is None:
                return acc_f.round(tropical_product(a, b, semiring), saturate=sat)
            return acc_f.round(
                tropical_accumulate(acc_f.round(C).astype(cdt), a, b, semiring),
                saturate=sat,
            )

        acc = np.zeros((M, N), cdt) if C is None else acc_f.round(C).astype(cdt)
        if k_chunk is None or not in_f.is_float:
            # Ints wrap modulo 2**32 however K is chunked, so chunking changes nothing
            return acc_f.round(acc + a @ b)

        out = acc_f.round(acc)
        for s in range(0, K, int(k_chunk)):
            part = np.zeros((M, N), cdt)
            for k in range(s, min(s + int(k_chunk), K)):
                part += np.multiply.outer(a[:, k], b[k, :])
            out = acc_f.round(out.astype(cdt) + part)
        return out


def fma_round_once(a, b, c, fmt):
    """``a * b + c`` rounded once to ``fmt`` (fp32 or narrower), elementwise.

    The fused multiply-add RVV's ``vfmacc`` and friends need. Operands are
    rounded to ``fmt`` first. fp64 is enough to emulate it, with one trick:
    for formats of p <= 24 significant bits the product of two operands has at
    most 48 and fits fp64's 53 exactly, so the add is the only rounding. That
    add's error is recovered exactly (Knuth's TwoSum) and folded into the last
    bit -- the sum is rounded *to odd* at 53 bits -- and 53 >= p + 2 makes the
    final RNE to ``fmt`` the correctly rounded result. Without the sticky bit
    fp64 is *not* enough: a sum that lands exactly on a midpoint of ``fmt`` after
    losing a tiny addend breaks the tie the wrong way.
    """
    f = get_format(fmt)
    if not f.is_float or f.mantissa_bits > 23:
        raise ValueError(
            f"fma_round_once supports fp32 and narrower float formats, not {f.name}"
        )
    a, b, c = (f.round(v).astype(np.float64) for v in (a, b, c))
    with np.errstate(over="ignore", invalid="ignore"):
        p = a * b
        s = p + c
        bv = s - p
        err = (p - (s - bv)) + (c - bv)
        inexact = (err != 0) & np.isfinite(s)
        # Truncate (step toward zero where RNE rounded away), then set the sticky bit
        s = np.where(inexact & ((err > 0) != (s > 0)), np.nextafter(s, 0.0), s)
        s = (s.view(np.uint64) | inexact.astype(np.uint64)).view(np.float64)
    return f.round(s)
