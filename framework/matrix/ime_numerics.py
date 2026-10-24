"""The ``"ime"`` summation strategy: Zvvm's FP accumulation model, bit for bit (``psm = 0``).

Registered in :data:`framework.matrix.mac.NUMERICS`, so
``tile_mac(C, A, Bt, in_fmt, acc_fmt, numerics="ime", group=G, rnd=..., w=W)``
is a Zvvm multiply-accumulate. The model is the draft's "Accumulation and
rounding model (floating-point)" (Zvvm @ ``6effbcae``):

* The ``K_eff`` products of each output element are cut into consecutive groups
  of ``G`` sub-dot-products, i.e. ``G * W`` products (a group never crosses an
  ``LMUL = 1`` step, which holds when ``G`` divides lambda; the caller checks).
* ``psm = 0``: each group's partial sum ``S`` is formed **exactly**.
* ``rnd``: ``frm`` rounds ``S`` to the C format (RNE; Naiad's frm), ``rto``
  rounds it to odd, ``xct`` keeps it exact.
* Then ``C = round_frm(C + S)``, group after group in K order.

So ``G = 1, psm = 0, rnd = frm`` (the default, the draft's systolic case) is
"each sub-dot-product rounded, then added", which for ``W = 1`` is an unfused
multiply-add per product; ``rnd = xct`` with ``G = 1`` is a fused one.

**Exactness.** Inputs are exact in fp64. A product of two inputs of at most 24
significant bits (fp32 and narrower) is exact in fp64; a group's sum is formed
in fp64 with an error-free TwoSum cascade, and any element whose cascade lost
bits is redone with rationals (:class:`fractions.Fraction`). A rounding of an
exact fp64 value to a narrower format is one correct rounding; a sum of two
exact values goes through TwoSum, round-to-odd at 53 bits, then the target
rounding (53 >= p + 2 makes it the correctly rounded sum). fp64 inputs take the
rational path except under the default (``G * W = 1``, ``rnd = frm``), where
numpy's correctly rounded fp64 product and sum *are* the model. Elements with
an infinite or NaN input are computed in plain fp64 (their result is Inf or
NaN either way). Every NaN result is the format's canonical NaN, as the draft
requires. fflags are not produced.
"""

from fractions import Fraction

import numpy as np

from framework.matrix.formats import get_format
from framework.matrix.mac import register_numerics

_CANONICAL_NAN = {
    "fp64": np.uint64(0x7FF8000000000000),
    "fp32": np.uint32(0x7FC00000),
    "fp16": np.uint16(0x7E00),
    "bf16": np.uint16(0x7FC0),
}
_UINT = {8: np.uint64, 4: np.uint32, 2: np.uint16, 1: np.uint8}


def _twosum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _sticky53(s, err):
    """``s`` (the RNE sum) and its exact error -> the sum rounded to odd at 53 bits."""
    inexact = (err != 0) & np.isfinite(s)
    if not inexact.any():
        return s
    s = np.where(inexact & ((err > 0) != (s > 0)), np.nextafter(s, 0.0), s)
    return (s.view(np.uint64) | inexact.astype(np.uint64)).view(np.float64)


def round_rne(x, fmt):
    """Exact fp64 values ``x`` rounded (RNE) to ``fmt``, as ``fmt``'s storage dtype."""
    with np.errstate(over="ignore", invalid="ignore"):
        return fmt.round(np.asarray(x, dtype=np.float64))


def round_odd(x, fmt):
    """Exact fp64 values ``x`` rounded to odd in ``fmt`` (truncate, then set the last bit if inexact)."""
    x = np.asarray(x, dtype=np.float64)
    if fmt.name == "fp64":
        return x.copy()
    q = round_rne(x, fmt)
    back = q.astype(np.float64)
    inexact = (back != x) & np.isfinite(x)
    if not inexact.any():
        return q
    ut = _UINT[q.dtype.itemsize]
    bits = q.view(ut).copy()
    away = inexact & (np.abs(back) > np.abs(x))
    bits = np.where(away, bits - ut(1), bits)  # one step toward zero in magnitude
    bits = bits | inexact.astype(ut)
    return bits.view(q.dtype)


def add_round(c, x, fmt):
    """``c + x`` correctly rounded (RNE) to ``fmt``; ``c`` and ``x`` are exact fp64 values."""
    with np.errstate(over="ignore", invalid="ignore"):
        if fmt.name == "fp64":
            return c + x
        s, err = _twosum(c, x)
        finite = np.isfinite(s)
        s = np.where(finite, _sticky53(s, np.where(finite, err, 0.0)), s)
        return fmt.round(s)


def canonicalise_nan(out, fmt):
    """Replace every NaN in ``out`` (``fmt``'s storage dtype) by the format's canonical NaN."""
    if not fmt.is_float or fmt.name not in _CANONICAL_NAN:
        return out
    nan = np.isnan(out.astype(np.float64))
    if nan.any():
        out = out.copy()
        out.view(_UINT[out.dtype.itemsize])[nan] = _CANONICAL_NAN[fmt.name]
    return out


# -- the exact (rational) path, per element ------------------------------------------------------


def _frac_round(q, fmt, mode):
    """A rational ``q`` rounded to ``fmt`` (``"rne"`` or ``"rto"``), as an fp64 holding the result."""
    try:
        x = float(q)  # correctly rounded (int / int true division)
    except OverflowError:
        x = float("inf") if q > 0 else float("-inf")
    if fmt.name == "fp64" and mode == "rne":
        return x
    if np.isfinite(x) and Fraction(x) != q:
        # Round to odd at 53 bits: truncate, then set the last bit.
        if abs(Fraction(x)) > abs(q):
            x = float(np.nextafter(x, 0.0))
        x = _set_odd(x)
    xv = np.array([x], dtype=np.float64)
    if mode == "rto":
        return float(round_odd(xv, fmt).astype(np.float64)[0])
    return float(round_rne(xv, fmt).astype(np.float64)[0])


def _set_odd(x):
    a = np.array([x], dtype=np.float64)
    return float((a.view(np.uint64) | np.uint64(1)).view(np.float64)[0])


def _exact_element(c, a_row, b_row, g, rnd, fmt):
    """One output element by rationals: ``c`` + the groups of ``a_row . b_row``, per the model."""
    cv = float(c)
    for q in range(0, len(a_row), g):
        if not np.isfinite(cv):
            cv = cv + float(np.dot(a_row[q:], b_row[q:]))
            break
        S = sum(
            (
                Fraction(float(a)) * Fraction(float(b))
                for a, b in zip(a_row[q : q + g], b_row[q : q + g])
            ),
            Fraction(0),
        )
        if rnd == "xct":
            x = S
        else:
            xf = _frac_round(S, fmt, "rne" if rnd == "frm" else "rto")
            if not np.isfinite(xf):
                cv = cv + xf
                continue
            x = Fraction(xf)
        cv = _frac_round(Fraction(cv) + x, fmt, "rne")
    return cv


# -- the strategy --------------------------------------------------------------------------------------


def ime_tile_mac(
    C,
    A,
    B,
    in_f,
    acc_f,
    *,
    k_chunk=None,
    fmt_b=None,
    group=1,
    psm=0,
    rnd="frm",
    w=1,
    step=None,
):
    """``C + A @ B`` (``A`` M x K, ``B`` K x N) under Zvvm's accumulation model.

    ``in_f`` is A's format and ``fmt_b`` B's (default ``in_f``; mixed fp16 x bf16
    is legal); ``group`` is ``G``, ``w`` the widening factor ``W``, ``step`` the K
    of one ``LMUL = 1`` step (``lambda * W``; checked to be a multiple of ``G * W``
    when given). Returns ``acc_f``'s storage dtype.
    """
    if k_chunk is not None:
        raise ValueError("ime numerics groups K itself (G, W); k_chunk does not apply")
    if psm != 0:
        raise ValueError(
            "ime numerics models psm = 0 only (psm = 1 is implementation-defined)"
        )
    if rnd not in ("frm", "rto", "xct"):
        raise ValueError(f"rnd must be frm, rto or xct, got {rnd!r}")
    fb = in_f if fmt_b is None else get_format(fmt_b)
    if not (in_f.is_float and fb.is_float and acc_f.is_float):
        raise ValueError(
            "ime numerics is the floating-point model; integers wrap exactly"
        )
    A, B = np.asarray(A), np.asarray(B)
    M, K = A.shape
    N = B.shape[1]
    g = int(group) * int(w)
    if g < 1 or K % g or (step is not None and step % g):
        raise ValueError(
            f"K {K} (step {step}) is not a whole number of groups of G * W = {g}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        a = A.astype(np.float64)
        b = B.astype(np.float64).T  # N x K: B_tile's rows
        c = np.zeros((M, N)) if C is None else np.asarray(C).astype(np.float64)
        kg = K // g
        P = a.reshape(M, 1, kg, g) * b.reshape(1, N, kg, g)
        special = ~(
            np.isfinite(a).all(axis=1)[:, None]
            & np.isfinite(b).all(axis=1)[None, :]
            & np.isfinite(c)
        )
        exact_products = in_f.mantissa_bits + fb.mantissa_bits + 2 <= 53
        slow = np.zeros((M, N), dtype=bool)
        if exact_products:
            S = P[..., 0]
            for t in range(1, g):
                S, err = _twosum(S, P[..., t])
                slow |= (err != 0).any(axis=-1)
        elif g == 1 and rnd == "frm":
            S = P[..., 0]  # numpy's product is the correctly rounded (RNE) one
        else:
            S = P.sum(axis=-1)
            slow[:] = True
        slow &= ~special

        fast_fp32 = acc_f.name == "fp32" and rnd != "xct"
        out = c.astype(np.float32) if fast_fp32 else c
        for q in range(kg):
            s = S[:, :, q]
            if rnd == "xct":
                x = s
            elif rnd == "frm":
                x = round_rne(s, acc_f)
            else:
                x = round_odd(s, acc_f)
            if fast_fp32:
                out = out + x  # both fp32: one IEEE (RNE) rounding
            else:
                out = add_round(out, x.astype(np.float64), acc_f).astype(np.float64)
        out = acc_f.round(np.asarray(out, dtype=np.float64)) if not fast_fp32 else out
        out = np.asarray(out).astype(acc_f.dtype)

        if special.any():
            naive = c + (a[:, None, :] * b[None, :, :]).sum(axis=-1)
            out[special] = acc_f.round(naive[special])
        for i, j in zip(*np.nonzero(slow)):
            out[i, j] = acc_f.round(
                np.float64(_exact_element(c[i, j], a[i], b[j], g, rnd, acc_f))
            )
    return canonicalise_nan(out, acc_f)


register_numerics("ime", ime_tile_mac)


def int_mac(C, A, B, acc_bits):
    """``C + A @ B`` for integer tiles, wrapping modulo ``2**acc_bits`` (A, B signed or unsigned arrays)."""
    a = (
        A.astype(np.int64).view(np.uint64)
        if A.dtype.kind == "i"
        else A.astype(np.uint64)
    )
    b = (
        B.astype(np.int64).view(np.uint64)
        if B.dtype.kind == "i"
        else B.astype(np.uint64)
    )
    c = C.astype(np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        total = (
            c + a @ b
        )  # uint64 arithmetic wraps modulo 2**64, so modulo 2**acc_bits too
    return total.astype(np.dtype(f"<i{acc_bits // 8}"))
