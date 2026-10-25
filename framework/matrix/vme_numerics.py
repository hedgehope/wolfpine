"""The ``"zvt"`` summation strategy: Zvt v0.4's floating-point accumulation, bit for bit.

Registered in :data:`framework.matrix.mac.NUMERICS`, so ``tile_mac(C, A, B,
in_fmt, acc_fmt, numerics="zvt", fmt_b=..., kmax=...)`` is one Zvt multiply
(``A`` is the draft's ``A^T``: ``tm x tk``; ``B`` is ``tk x tn``). The rule is
the draft's (12.1.1.8):

* **SEW >= 32** (fp32 -> fp32, fp64 -> fp64): each product is computed in IEEE
  754 arithmetic and rounded to TEW, then added to the accumulator with the
  same rounding mode, in increasing k: two roundings per term, never fused.
  Only ``frm = RNE`` is modelled (the ISA layer refuses the others).
* **SEW <= 16** (fp16 / bf16 / OFP8 -> fp32): the products of each output
  element are bulk-normalised and summed in fixed point, then converted to fp32
  rounding to odd, by the RISC-V Bulk Normalization Algorithm
  (``riscv-isa-manual`` ``src/unpriv/rvbna.adoc`` @ ``a38e1213``), with ``n =
  KMAX`` products (2 for 16-bit, 4 for 8-bit inputs; the draft fixes ``n`` per
  format, and a missing ``k >= tk`` product is a zero, which never moves
  ``max_exp``), ``q = 24``, ``g = o = log2(n)``. The result is then added to the
  accumulator in IEEE arithmetic **rounding to odd**, with RVBNA's RTO variant:
  an overflow is an infinity, not the largest finite value.

**RVBNA as implemented.** :func:`rvbna_dot` follows the pseudocode (reference
exponents from the biased exponents, subnormals at ``emin_normal``, zero
products at reference exponent 0, alignment to ``q + 1 + g`` bits with every
discarded bit jammed into the LSB, exact signed sum) except for one place,
where the pseudocode cannot be what is meant: its final ``jamMask = rawJamMask
>> (g + o + 1 - lzc)`` selects only bits that the left shift by ``lzc`` has just
zeroed, so ``jamSig`` would always be 0 and the final rounding would be a
truncation, not the round-to-odd its own prose (and the Zvt text) asks for; its
subnormal branch shifts the unnormalised ``accAbs`` by ``q - 1 + resExp``,
which is not the denormalisation either. Here the final step is the exact
accumulator value (an integer of at most ``q + 1 + g + o`` bits times a power of
two, so exact in fp64) rounded to odd at fp32 with gradual underflow, which is
what the prose describes. Specials: any NaN input, ``inf * 0`` or ``inf - inf``
among the products gives the canonical NaN; otherwise an infinite product gives
that infinity. A zero sum is ``+0``. Flags (invalid, overflow) are not produced.
"""

import numpy as np

from framework.matrix.formats import get_format
from framework.matrix.ime_numerics import (
    _sticky53,
    _twosum,
    canonicalise_nan,
    round_odd,
)
from framework.matrix.mac import register_numerics

_Q, _F_BIAS = 24, 127
_F32_MAX = float(np.finfo(np.float32).max)
_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _fields(x, f):
    """``(sign, significand, reference exponent, zero, value)`` of an array in format ``f``."""
    x = np.asarray(x)
    bits = x.view(_UINT[x.dtype.itemsize]).astype(np.int64)
    m, e = f.mantissa_bits, f.exponent_bits
    exp = (bits >> m) & ((1 << e) - 1)
    mant = bits & ((1 << m) - 1)
    sign = (bits >> (m + e)) & 1
    sub = exp == 0
    sig = np.where(sub, mant, mant | (1 << m))
    ref = np.where(sub, 1, exp)
    return sign, sig, ref, sig == 0, x.astype(np.float64)


def rto_fp32(x):
    """Exact fp64 values rounded to odd at fp32, RVBNA style: overflow is an infinity."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        out = round_odd(x, get_format("fp32")).astype(np.float32)
        big = np.isfinite(x) & (np.abs(x) > _F32_MAX)
        if big.any():
            out = np.where(
                big, np.copysign(np.float32(np.inf), x).astype(np.float32), out
            )
    return out


def rvbna_dot(A, B, fmt_a, fmt_b, n):
    """RVBNA: ``A (M x K) . B (K x N)`` per element, ``K <= n`` products, as fp32 (round to odd)."""
    fa, fb = get_format(fmt_a), get_format(fmt_b)
    M, K = A.shape
    N = B.shape[1]
    if K > n:
        raise ValueError(f"RVBNA over {K} products with n = {n}")
    g = (n - 1).bit_length()  # g = o = log2(n); o (overflow bits) only bounds |acc|
    sa, siga, refa, za, va = _fields(A, fa)
    sb, sigb, refb, zb, vb = _fields(np.ascontiguousarray(B.T), fb)  # N x K
    if K == 0:
        return np.zeros((M, N), np.float32)
    psign = sa[:, None, :] ^ sb[None, :, :]
    psig = siga[:, None, :] * sigb[None, :, :]
    zero = za[:, None, :] | zb[None, :, :]
    ref = np.where(zero, 0, refa[:, None, :] + refb[None, :, :])
    max_exp = ref.max(axis=-1)
    pl, pr = fa.mantissa_bits + 1, fb.mantissa_bits + 1
    pad = _Q + 1 + g - (pl + pr)
    v = psig << pad
    shift = np.minimum(max_exp[..., None] - ref, 62)
    aligned = v >> shift
    jam = (v & ((np.int64(1) << shift) - 1)) != 0
    aligned = aligned | jam.astype(np.int64)
    acc = np.where(psign == 1, -aligned, aligned).sum(axis=-1)
    bias = (1 << (fa.exponent_bits - 1)) - 1 + (1 << (fb.exponent_bits - 1)) - 1
    scale = max_exp - bias - (pl - 1) - (pr - 1) - pad
    exact = np.ldexp(acc.astype(np.float64), scale.astype(np.int64))
    out = rto_fp32(exact)
    out = np.where(acc == 0, np.float32(0.0), out)  # a zero sum is +0
    # Specials, from the values: NaN in, inf * 0 or inf - inf -> NaN; else an infinite product.
    finite = np.isfinite(va).all(axis=1)[:, None] & np.isfinite(vb).all(axis=1)[None, :]
    if not finite.all():
        with np.errstate(invalid="ignore", over="ignore"):
            s = (va[:, None, :] * vb[None, :, :]).sum(axis=-1)
        out = np.where(finite, out, s.astype(np.float32))
    return out


def add_rto_fp32(c, s):
    """``c + s`` (fp32 arrays) in IEEE arithmetic rounded to odd, RVBNA's RTO (overflow -> inf)."""
    c64, s64 = c.astype(np.float64), s.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        tot, err = _twosum(c64, s64)
        finite = np.isfinite(tot)
        tot = np.where(finite, _sticky53(tot, np.where(finite, err, 0.0)), c64 + s64)
    return rto_fp32(tot)


def zvt_tile_mac(C, A, B, in_f, acc_f, *, k_chunk=None, fmt_b=None, kmax=None):
    """``C + A @ B`` (``A`` M x K, ``B`` K x N) under Zvt's rule; returns ``acc_f``'s dtype."""
    if k_chunk is not None:
        raise ValueError(
            "zvt numerics fixes its own order (k by k, or one RVBNA group); k_chunk does not apply"
        )
    fb = in_f if fmt_b is None else get_format(fmt_b)
    if not (in_f.is_float and fb.is_float and acc_f.is_float):
        raise ValueError(
            "zvt numerics is the floating-point model; integers wrap exactly"
        )
    A, B = np.asarray(A), np.asarray(B)
    M, K = A.shape
    N = B.shape[1]
    c = (
        np.zeros((M, N), acc_f.dtype)
        if C is None
        else np.asarray(C).astype(acc_f.dtype)
    )
    if max(in_f.bits, fb.bits) >= 32:
        if in_f.name != acc_f.name or fb.name != acc_f.name:
            raise ValueError(
                f"zvt: {in_f.name} x {fb.name} -> {acc_f.name} is not a Zvt type"
            )
        a, b = A.astype(acc_f.dtype), B.astype(acc_f.dtype)
        with np.errstate(over="ignore", invalid="ignore"):
            for k in range(K):
                c = (
                    c + a[:, k : k + 1] * b[k : k + 1, :]
                )  # product rounded to TEW, then the add
        return canonicalise_nan(c.astype(acc_f.dtype), acc_f)
    if acc_f.name != "fp32":
        raise ValueError(
            f"zvt: SEW <= 16 inputs accumulate into fp32, not {acc_f.name}"
        )
    n = kmax if kmax is not None else (2 if in_f.bits == 16 else 4)
    s = rvbna_dot(A, B, in_f.name, fb.name, n)
    return canonicalise_nan(add_rto_fp32(c, s), acc_f)


register_numerics("zvt", zvt_tile_mac)

__all__ = ["add_rto_fp32", "rto_fp32", "rvbna_dot", "zvt_tile_mac"]
