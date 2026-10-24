"""Vector floating point: exact IEEE results on numpy arrays of bit patterns.

Every function takes element bit patterns (unsigned numpy arrays, one SEW) and
returns ``(bits, fflags)``. The reference semantics are :mod:`fd_softfloat`'s
(exact maths, one rounding, RISC-V NaN and flag rules, every rounding mode);
each operation here is either a call to it per element (the **slow path**) or
a numpy computation that gives the same bits and flags (the **fast path**).
An element the fast path cannot vouch for goes to the slow path, so the
answer is always softfloat's; ``fp_test.py`` checks the two against each
other on random and adversarial operands.

Fast paths, and why each is exact:

* **SEW 16 / 32** (any rounding mode). Operands are widened to fp64 exactly.
  The fp64 result ``hi`` of ``+ - * / sqrt`` and of the fused multiply-add is
  paired with the exact sign of its error: TwoSum for sums (exact for any
  finite fp64 inputs), none for products (24 + 24 bits fit in 53), and a
  Dekker TwoProduct remainder for ``/`` and ``sqrt`` (``a - q*b`` is exact by
  Sterbenz, the operands being fp32-ranged). ``hi`` is then nudged to
  **round-to-odd** at 53 bits, and since 53 >= p + 2 a single rounding of
  that to the target precision under any mode is the correctly rounded
  result (Boldo and Melquiond, "When double rounding is odd", 2005). An FMA
  is ``a*b`` (exact) then TwoSum with ``c``, so it is single-rounded too.
* **SEW 64, round to nearest even.** numpy's fp64 arithmetic is IEEE RNE, so
  the result bits are right; only the inexact flag needs work: TwoSum /
  TwoProduct error terms for ``+ - *``, the exact remainder for ``/`` and
  ``sqrt``. The FMA is Boldo and Melquiond's emulation with round-to-odd
  ("Emulation of FMA and correctly rounded sums: proved algorithms using
  rounding to odd", IEEE TC 2008): ``uh + ul = a*b`` exactly, ``th + tl =
  c + uh`` (TwoSum), ``v = RO(tl + ul)``, result ``RN(th + v)``.
* Everywhere else (SEW 64 under RTZ/RDN/RUP/RMM; NaN, infinite, zero-sum
  under RDN, subnormal, tiny or overflowing elements; ranges where an error
  term could leave fp64's normal range) the element takes the slow path.

The only flags the fast paths raise are NX; anything that could raise NV, DZ,
OF or UF is a slow-path element by construction.
"""

import struct

import numpy as np

from framework.pe.rv.isa import fd_softfloat as sf

F16 = sf.FloatFormat("h", 5, 10)
FMT = {16: F16, 32: sf.F32, 64: sf.F64}
#: Round-to-odd, a pseudo rounding mode for ``vfncvt.rod.f.f.w``.
ROD = 8

_U = {16: np.uint16, 32: np.uint32, 64: np.uint64}
_FL = {16: np.float16, 32: np.float32, 64: np.float64}
_FRAC52 = np.uint64((1 << 52) - 1)
_SPLIT = 134217729.0  # 2**27 + 1
_TWO_995 = 2.0**995
_TINY_SAFE = 2.0**-900
_MIN_NORMAL64 = 2.0**-1022
_MIN_NORMAL = {16: 2.0**-14, 32: 2.0**-126}


# -- bit-pattern helpers -----------------------------------------------------


def _fields(bits, fmt):
    """``(exp field, frac field, sign)`` of a bit array as int64 arrays."""
    b = bits.astype(np.uint64)
    frac = (b & np.uint64((1 << fmt.frac_bits) - 1)).astype(np.int64)
    exp = ((b >> np.uint64(fmt.frac_bits)) & np.uint64((1 << fmt.exp_bits) - 1)).astype(
        np.int64
    )
    sign = (b >> np.uint64(fmt.bits - 1)).astype(np.int64)
    return exp, frac, sign


def nonfinite(bits, fmt):
    return (
        (bits.astype(np.uint64) >> np.uint64(fmt.frac_bits))
        & np.uint64((1 << fmt.exp_bits) - 1)
    ) == ((1 << fmt.exp_bits) - 1)


def isnan(bits, fmt):
    exp, frac, _ = _fields(bits, fmt)
    return (exp == (1 << fmt.exp_bits) - 1) & (frac != 0)


def issnan(bits, fmt):
    exp, frac, _ = _fields(bits, fmt)
    return (
        (exp == (1 << fmt.exp_bits) - 1)
        & (frac != 0)
        & ((frac >> (fmt.frac_bits - 1)) == 0)
    )


def to_f64(bits, sew):
    """Bit patterns of one SEW as exact fp64 values (NaNs keep no payload guarantee)."""
    if sew == 64:
        return bits.view(np.float64)
    return bits.view(_FL[sew]).astype(np.float64)


def bf16_to_f32_bits(bits):
    return bits.astype(np.uint32) << np.uint32(16)


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _round_odd(hi, err):
    """RO(x) as fp64, given ``hi = RN(x)`` and ``err`` with the sign of ``x - hi``."""
    bits = hi.view(np.uint64)
    need = (err != 0) & ((bits & np.uint64(1)) == 0)
    if not need.any():
        return hi
    up = np.signbit(err) == np.signbit(hi)
    one = np.uint64(1)
    out = np.where(need, np.where(up, bits + one, bits - one), bits)
    return out.view(np.float64)


def round_f64(x, fmt, rm):
    """Round fp64 values (exact, or round-to-odd at 53 bits) to a format of precision <= 51.

    Returns ``(bits, inexact, bad)``: ``bad`` marks the elements this cannot
    round (NaN/infinite inputs, results that overflow or are subnormal); zeros
    come back exact with their sign.

    Under RNE this is numpy's own fp64 -> fp32 / fp16 cast (IEEE RNE, and
    exact here because ``x`` is exact or round-to-odd at 53 >= p + 2 bits),
    with the inexact and range checks done on the result.
    """
    if rm == sf.RNE and fmt in (F16, sf.F32):
        dt = _FL[fmt.bits]
        r = x.astype(dt)
        back = r.astype(np.float64)
        bad = ~np.isfinite(back) | ((np.abs(x) < _MIN_NORMAL[fmt.bits]) & (x != 0))
        return r.view(_U[fmt.bits]).astype(np.uint64), back != x, bad
    b = x.view(np.uint64)
    sign = b >> np.uint64(63)
    e = ((b >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64) - 1023
    mant = (b & _FRAC52) | np.uint64(1 << 52)
    p = fmt.precision
    shift = np.uint64(53 - p)
    keep = mant >> shift
    rem = mant & np.uint64((1 << (53 - p)) - 1)
    half = np.uint64(1 << (52 - p))
    inexact = rem != 0
    if rm == sf.RNE:
        inc = (rem > half) | ((rem == half) & ((keep & np.uint64(1)) == 1))
    elif rm == sf.RTZ or rm == ROD:
        inc = None
    elif rm == sf.RDN:
        inc = inexact & (sign == 1)
    elif rm == sf.RUP:
        inc = inexact & (sign == 0)
    else:  # RMM
        inc = rem >= half
    if rm == ROD:
        keep = keep | inexact.astype(np.uint64)
    elif inc is not None:
        keep = keep + inc.astype(np.uint64)
    carry = keep >> np.uint64(p)
    keep = keep >> carry
    biased = e + carry.astype(np.int64) + fmt.bias
    zero = (b & np.uint64((1 << 63) - 1)) == 0
    bad = ((biased >= (1 << fmt.exp_bits) - 1) | (biased <= 0)) & ~zero
    bad |= ~np.isfinite(x)
    biased = np.where(bad | zero, 0, biased).astype(np.uint64)
    out = (
        (sign << np.uint64(fmt.bits - 1))
        | (biased << np.uint64(fmt.frac_bits))
        | (np.where(zero, np.uint64(0), keep) & np.uint64((1 << fmt.frac_bits) - 1))
    )
    return out, inexact & ~zero, bad


def _finish(out, flags, bad, slow):
    """Run ``slow(i) -> (bits, flags)`` for each ``bad`` element; return ``(out, flags)``."""
    for i in np.flatnonzero(bad):
        r, f = slow(int(i))
        out[i] = r
        flags |= f
    return out, flags


# -- arithmetic ----------------------------------------------------------------


def _narrow_exact(op, xa, xb, xc):
    """fp64 ``(hi, err)`` for an op on fp32/fp16-valued fp64 operands; err has the error's sign."""
    if op == "add":
        return _two_sum(xa, xb)
    if op == "sub":
        return _two_sum(xa, -xb)
    if op == "mul":
        return xa * xb, None
    if op == "div":
        q = xa / xb
        p, pe = _two_prod(q, xb)
        r = (xa - p) - pe
        return q, r * np.sign(xb)
    if op == "sqrt":
        q = np.sqrt(xa)
        p, pe = _two_prod(q, q)
        return q, (xa - p) - pe
    raise ValueError(op)


def _sf_binary(op, a, b, fmt, rm):
    if op == "add":
        return sf.add(a, b, fmt, rm)
    if op == "sub":
        return sf.add(a, b, fmt, rm, negate_b=True)
    if op == "mul":
        return sf.mul(a, b, fmt, rm)
    if op == "div":
        return sf.div(a, b, fmt, rm)
    raise ValueError(op)


def binary(op, a, b, sew, rm):
    """Elementwise ``a <op> b`` (``op`` in add/sub/mul/div) on bit arrays of one SEW."""
    fmt = FMT[sew]
    out = np.empty(len(a), dtype=_U[sew])
    if len(a) == 0:
        return out, 0
    with np.errstate(all="ignore"):
        xa, xb = to_f64(a, sew), to_f64(b, sew)
        bad = ~(np.isfinite(xa) & np.isfinite(xb))
        if op == "div":
            bad |= xb == 0
        if sew == 64:
            if rm != sf.RNE:
                bad[:] = True
                res = np.zeros(len(a))
                inexact = np.zeros(len(a), bool)
            else:
                res, inexact, more = _f64_binary(op, xa, xb)
                bad |= more
            out[:] = res.view(np.uint64)
        else:
            hi, err = _narrow_exact(op, xa, xb, None)
            ro = hi if err is None else _round_odd(hi, err)
            bits, inexact, more = round_f64(ro, fmt, rm)
            if err is not None:
                inexact = inexact | (err != 0)
            bad |= more
            if rm == sf.RDN and op in ("add", "sub"):
                bad |= hi == 0
            out[:] = bits
    flags = sf.NX if (inexact & ~bad).any() else 0
    return _finish(
        out, flags, bad, lambda i: _sf_binary(op, int(a[i]), int(b[i]), fmt, rm)
    )


def _subnormal(x):
    ax = np.abs(x)
    return (ax < _MIN_NORMAL64) & (ax != 0)


def _f64_binary(op, a, b):
    """RNE fp64 ``(result, inexact, bad)``."""
    if op in ("add", "sub"):
        s, err = _two_sum(a, b if op == "add" else -b)
        bad = ~np.isfinite(s) | _subnormal(s)
        return s, err != 0, bad
    big = (np.abs(a) >= _TWO_995) | (np.abs(b) >= _TWO_995)
    tiny_in = _subnormal(a) | _subnormal(b)
    if op == "mul":
        p, err = _two_prod(a, b)
        ap = np.abs(p)
        bad = (
            big
            | tiny_in
            | ~np.isfinite(p)
            | ((ap < _TINY_SAFE) & ((a != 0) & (b != 0)))
        )
        return p, err != 0, bad
    if op == "div":
        q = a / b
        p, pe = _two_prod(q, b)
        r = (a - p) - pe
        aq = np.abs(q)
        bad = big | tiny_in | (np.abs(q) >= _TWO_995) | ~np.isfinite(q)
        bad |= ((aq < _TINY_SAFE) | (np.abs(a) < _TINY_SAFE)) & (a != 0)
        return q, r != 0, bad
    raise ValueError(op)


def sqrt(a, sew, rm):
    fmt = FMT[sew]
    out = np.empty(len(a), dtype=_U[sew])
    if len(a) == 0:
        return out, 0
    with np.errstate(all="ignore"):
        xa = to_f64(a, sew)
        bad = ~np.isfinite(xa) | (xa < 0)
        if sew == 64:
            q = np.sqrt(xa)
            p, pe = _two_prod(q, q)
            r = (xa - p) - pe
            inexact = r != 0
            bad |= _subnormal(xa) | ((xa < _TINY_SAFE) & (xa != 0))
            if rm != sf.RNE:
                bad[:] = True
            out[:] = q.view(np.uint64)
        else:
            hi, err = _narrow_exact("sqrt", xa, None, None)
            bits, inexact, more = round_f64(_round_odd(hi, err), fmt, rm)
            inexact = inexact | (err != 0)
            bad |= more
            out[:] = bits
    flags = sf.NX if (inexact & ~bad).any() else 0
    return _finish(out, flags, bad, lambda i: sf.sqrt(int(a[i]), fmt, rm))


def fma(a, b, c, sew, rm, neg_product=False, neg_addend=False):
    """``(+/-)(a*b) (+/-) c`` with one rounding, elementwise."""
    fmt = FMT[sew]
    out = np.empty(len(a), dtype=_U[sew])
    if len(a) == 0:
        return out, 0
    with np.errstate(all="ignore"):
        xa, xb, xc = to_f64(a, sew), to_f64(b, sew), to_f64(c, sew)
        if neg_product:
            xa = -xa
        if neg_addend:
            xc = -xc
        bad = ~(np.isfinite(xa) & np.isfinite(xb) & np.isfinite(xc))
        if sew == 64:
            if rm != sf.RNE:
                bad[:] = True
                z = np.zeros(len(a))
                inexact = np.zeros(len(a), bool)
            else:
                z, inexact, more = _f64_fma(xa, xb, xc)
                bad |= more
            out[:] = z.view(np.uint64)
        else:
            hi, err = _two_sum(xa * xb, xc)
            bits, inexact, more = round_f64(_round_odd(hi, err), fmt, rm)
            inexact = inexact | (err != 0)
            bad |= more
            if rm == sf.RDN:
                bad |= hi == 0
            out[:] = bits
    flags = sf.NX if (inexact & ~bad).any() else 0
    return _finish(
        out,
        flags,
        bad,
        lambda i: sf.fma(
            int(a[i]), int(b[i]), int(c[i]), fmt, rm, neg_product, neg_addend
        ),
    )


def _f64_fma(a, b, c):
    """RNE fp64 FMA ``(z, inexact, bad)`` by round-to-odd emulation (see the module docs)."""
    uh, ul = _two_prod(a, b)
    th, tl = _two_sum(c, uh)
    s, se = _two_sum(tl, ul)
    v = _round_odd(s, se)
    v_inexact = se != 0
    z, ze = _two_sum(th, v)
    inexact = v_inexact | (ze != 0)
    big = (np.abs(a) >= _TWO_995) | (np.abs(b) >= _TWO_995) | (np.abs(c) >= _TWO_995)
    bad = big | _subnormal(a) | _subnormal(b) | _subnormal(c) | (a == 0) | (b == 0)
    bad |= ~np.isfinite(z) | ~np.isfinite(th)
    bad |= np.abs(uh) < _TINY_SAFE
    bad |= (np.abs(z) < _TINY_SAFE) & ((z != 0) | v_inexact)
    bad |= (s != 0) & (np.abs(s) < _MIN_NORMAL64 * 2.0**60)
    return z, inexact, bad


# -- non-arithmetic ------------------------------------------------------------


def _key(bits, fmt):
    """Total-order key for non-NaN bit patterns (-0 and +0 equal)."""
    b = bits.astype(np.uint64)
    mag = (b & np.uint64((1 << (fmt.bits - 1)) - 1)).astype(np.int64)
    return np.where((b >> np.uint64(fmt.bits - 1)) == 1, -mag, mag)


def minmax(a, b, sew, is_max):
    fmt = FMT[sew]
    na, nb = isnan(a, fmt), isnan(b, fmt)
    flags = sf.NV if (issnan(a, fmt).any() or issnan(b, fmt).any()) else 0
    ka, kb = _key(a, fmt), _key(b, fmt)
    a_neg = (a.astype(np.uint64) >> np.uint64(fmt.bits - 1)) == 1
    if is_max:
        pick_a = (ka > kb) | ((ka == kb) & ~a_neg)
    else:
        pick_a = (ka < kb) | ((ka == kb) & a_neg)
    out = np.where(pick_a, a, b)
    out = np.where(na & ~nb, b, out)
    out = np.where(nb & ~na, a, out)
    out = np.where(na & nb, a.dtype.type(fmt.canonical_nan), out)
    return out.astype(a.dtype), flags


def compare(a, b, sew, op):
    """``op`` in eq/ne/lt/le/gt/ge -> (bool array, flags). eq/ne are quiet, the rest signalling."""
    fmt = FMT[sew]
    nan = isnan(a, fmt) | isnan(b, fmt)
    if op in ("eq", "ne"):
        flags = sf.NV if (issnan(a, fmt).any() or issnan(b, fmt).any()) else 0
    else:
        flags = sf.NV if nan.any() else 0
    ka, kb = _key(a, fmt), _key(b, fmt)
    if op == "eq":
        r = ka == kb
    elif op == "ne":
        return (ka != kb) | nan, flags
    elif op == "lt":
        r = ka < kb
    elif op == "le":
        r = ka <= kb
    elif op == "gt":
        r = ka > kb
    else:
        r = ka >= kb
    return r & ~nan, flags


def classify(a, sew):
    fmt = FMT[sew]
    exp, frac, sign = _fields(a, fmt)
    top = (1 << fmt.exp_bits) - 1
    neg = sign == 1
    inf = (exp == top) & (frac == 0)
    nan = (exp == top) & (frac != 0)
    quiet = (frac >> (fmt.frac_bits - 1)) == 1
    zero = (exp == 0) & (frac == 0)
    sub = (exp == 0) & (frac != 0)
    normal = (exp != 0) & (exp != top)
    out = np.zeros(len(a), dtype=np.int64)
    for bit, cond in (
        (0, inf & neg),
        (1, normal & neg),
        (2, sub & neg),
        (3, zero & neg),
        (4, zero & ~neg),
        (5, sub & ~neg),
        (6, normal & ~neg),
        (7, inf & ~neg),
        (8, nan & ~quiet),
        (9, nan & quiet),
    ):
        out |= cond.astype(np.int64) << bit
    return out


def sgnj(a, b, sew, kind):
    """``kind`` 0 = vfsgnj, 1 = vfsgnjn, 2 = vfsgnjx."""
    top = a.dtype.type(1 << (sew - 1))
    if kind == 0:
        s = b & top
    elif kind == 1:
        s = (b ^ top) & top
    else:
        s = (a ^ b) & top
    return (a & ~top) | s


# -- conversions -----------------------------------------------------------------


def convert(a, src, dst, rm):
    """Float -> float, bit arrays (``src``/``dst`` are formats; ``dst`` is FMT[...] or BF16)."""
    out = np.empty(len(a), dtype=np.uint64)
    if len(a) == 0:
        return out, 0
    with np.errstate(all="ignore"):
        if src.bits == 16 and src.exp_bits == 8:  # bf16
            x = bf16_to_f32_bits(a).view(np.float32).astype(np.float64)
        else:
            x = to_f64(a.astype(_U[src.bits]), src.bits)
        bad = nonfinite(a, src)
        if dst.bits == 64:
            out[:] = x.view(np.uint64)  # widening to fp64: exact
            inexact = np.zeros(len(a), bool)
            bad |= _subnormal(x) & (src.bits == 64)
        else:
            bits, inexact, more = round_f64(x, dst, rm)
            bad |= more
            out[:] = bits
    flags = sf.NX if (inexact & ~bad).any() else 0

    def slow(i):
        if rm == ROD:
            bits, f = sf.convert(int(a[i]), src, dst, sf.RTZ)
            if f & sf.NX and not bits & 1:
                exp = (bits >> dst.frac_bits) & ((1 << dst.exp_bits) - 1)
                if exp != (1 << dst.exp_bits) - 1:  # not inf / NaN
                    bits |= 1
            return bits, f
        return sf.convert(int(a[i]), src, dst, rm)

    return _finish(out, flags, bad, slow)


def _int_bounds(bits, signed):
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def to_int(a, src, rm, bits, signed):
    """Float -> integer, saturating; returns ``(uint64 two's-complement patterns, flags)``."""
    n = len(a)
    if n == 0:
        return np.empty(0, np.uint64), 0
    with np.errstate(all="ignore"):
        x = to_f64(a.astype(_U[src.bits]), src.bits)
        if rm == sf.RNE:
            r = np.rint(x)
        elif rm == sf.RTZ:
            r = np.trunc(x)
        elif rm == sf.RDN:
            r = np.floor(x)
        elif rm == sf.RUP:
            r = np.ceil(x)
        else:  # RMM
            t = np.trunc(x)
            r = np.where(np.abs(x - t) >= 0.5, t + np.sign(x), t)
        lo, hi = _int_bounds(bits, signed)
        nan = np.isnan(x)
        over = (r > float(hi)) | (r >= 2.0 ** (bits - (1 if signed else 0)))
        under = r < float(lo)
        invalid = nan | over | under
        flags = sf.NV if invalid.any() else 0
        inexact = (r != x) & ~invalid
        if inexact.any():
            flags |= sf.NX
        safe = np.where(invalid, 0.0, r)
        if signed or bits < 64:
            vals = safe.astype(np.int64).astype(np.uint64)
        else:
            big = safe >= 2.0**63
            vals = np.where(
                big,
                (safe - 2.0**63).astype(np.int64).astype(np.uint64)
                + np.uint64(1 << 63),
                safe.astype(np.int64).astype(np.uint64),
            )
        sat_hi = np.uint64(hi & ((1 << 64) - 1))
        sat_lo = np.uint64(lo & ((1 << 64) - 1))
        neg_inf_like = under | (over & False)
        vals = np.where(nan | over, sat_hi, vals)
        vals = np.where(neg_inf_like, sat_lo, vals)
    return vals, flags


def from_int(v, signed, src_bits, dst, rm):
    """Integer (uint64 patterns of ``src_bits`` width) -> float bits of ``dst``."""
    n = len(v)
    out = np.empty(n, dtype=np.uint64)
    if n == 0:
        return out, 0
    v = v.astype(np.uint64)
    if src_bits < 64:
        v = v & np.uint64((1 << src_bits) - 1)
    if signed:
        iv = v.astype(np.int64)
        if src_bits < 64:
            iv = np.where(iv >= (1 << (src_bits - 1)), iv - (1 << src_bits), iv)
    with np.errstate(all="ignore"):
        if src_bits <= 32:
            x = (iv if signed else v).astype(np.float64)  # exact
            err = None
            bad = np.zeros(n, bool)
        else:
            x = (iv if signed else v).astype(np.float64)  # RNE
            bad = np.abs(x) >= 2.0**63
            safe = np.where(bad, 0.0, x)
            if signed:
                err = iv - safe.astype(np.int64)
            else:
                back = safe.astype(np.int64).astype(np.uint64)
                err = (v.astype(np.int64) - back.astype(np.int64)).astype(np.int64)
                bad |= v >= np.uint64(1 << 63)
            err = np.sign(err).astype(np.float64)
        if dst.bits == 64:
            out[:] = x.view(np.uint64)
            inexact = np.zeros(n, bool) if err is None else err != 0
            if rm != sf.RNE and err is not None:
                bad |= err != 0
        else:
            ro = x if err is None else _round_odd(x, err)
            bits, inexact, more = round_f64(ro, dst, rm)
            if err is not None:
                inexact = inexact | (err != 0)
            bad |= more
            out[:] = bits
    flags = sf.NX if (inexact & ~bad).any() else 0

    def slow(i):
        value = int(v[i])
        if signed and value >> (src_bits - 1):
            value -= 1 << src_bits
        return sf.from_int(value, dst, rm)

    return _finish(out, flags, bad, slow)


# -- reductions ---------------------------------------------------------------------

_PACK = {16: struct.Struct("<e"), 32: struct.Struct("<f")}


def ordered_sum(start, values, sew, rm):
    """Sequential ``(((start + v0) + v1) + ...)`` with a rounding per step, as Spike's vfred[ou]sum."""
    fmt = FMT[sew]
    if len(values) == 0:
        return start, 0
    if rm == sf.RNE:
        fast = _ordered_sum_rne(start, values, sew, fmt)
        if fast is not None:
            return fast
    acc, flags = start, 0
    for x in values.tolist():
        acc, f = sf.add(acc, x, fmt, rm)
        flags |= f
    return acc, flags


def _ordered_sum_rne(start, values, sew, fmt):
    """Python-float loop for RNE sums; None if any step leaves the easy cases."""
    vals = np.concatenate([np.array([start], dtype=values.dtype), values])
    if nonfinite(vals, fmt).any():
        return None
    xs = to_f64(vals, sew).tolist()
    acc = xs[0]
    inexact = False
    if sew == 64:
        for x in xs[1:]:
            s = acc + x
            bb = s - acc
            if (acc - (s - bb)) + (x - bb) != 0:
                inexact = True
            acc = s
            if (
                s != s
                or abs(s) == float("inf")
                or (s != 0 and abs(s) < 2.2250738585072014e-308)
            ):
                return None
        bits = struct.unpack("<Q", struct.pack("<d", acc))[0]
        return bits, sf.NX if inexact else 0
    pack = _PACK[sew]
    tiny = 2.0 ** (1 - fmt.bias)
    for x in xs[1:]:
        s = acc + x
        bb = s - acc
        if (acc - (s - bb)) + (x - bb) != 0:
            inexact = True
        try:
            r = pack.unpack(pack.pack(s))[0]
        except OverflowError:
            return None
        if r != s:
            inexact = True
        if r != 0 and abs(r) < tiny:
            return None
        if s == 0 and r == 0 and acc != 0:
            pass
        acc = r
    raw = pack.pack(acc)
    bits = int.from_bytes(raw, "little")
    return bits, sf.NX if inexact else 0


def reduce_minmax(start, values, sew, is_max):
    """``vfredmax`` / ``vfredmin``: minimumNumber / maximumNumber over ``start`` and ``values``."""
    fmt = FMT[sew]
    if len(values) == 0:
        return int(start), 0  # Spike: no active element leaves the start value as it is
    vals = np.concatenate([np.array([start], dtype=values.dtype), values])
    flags = sf.NV if issnan(vals, fmt).any() else 0
    keep = vals[~isnan(vals, fmt)]
    if len(keep) == 0:
        return fmt.canonical_nan, flags
    k = _key(keep, fmt)
    best = k.max() if is_max else k.min()
    cands = keep[k == best]
    neg = (cands.astype(np.uint64) >> np.uint64(fmt.bits - 1)) == 1
    if is_max:
        pick = cands[~neg][0] if (~neg).any() else cands[0]
    else:
        pick = cands[neg][0] if neg.any() else cands[0]
    return int(pick), flags


# -- estimates (vfrec7 / vfrsqrt7), after Spike's tables ------------------------------

_RSQRT7 = [
    52, 51, 50, 48, 47, 46, 44, 43, 42, 41, 40, 39, 38, 36, 35, 34,
    33, 32, 31, 30, 30, 29, 28, 27, 26, 25, 24, 23, 23, 22, 21, 20,
    19, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12, 11, 10, 10, 9,
    9, 8, 7, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0,
    127, 125, 123, 121, 119, 118, 116, 114, 113, 111, 109, 108, 106, 105, 103, 102,
    100, 99, 97, 96, 95, 93, 92, 91, 90, 88, 87, 86, 85, 84, 83, 82,
    80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 70, 69, 68, 67, 66,
    65, 64, 63, 63, 62, 61, 60, 59, 59, 58, 57, 56, 56, 55, 54, 53,
]  # fmt: skip

_REC7 = [
    127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100,
    99, 97, 96, 94, 93, 91, 90, 88, 87, 85, 84, 83, 81, 80, 79, 77,
    76, 75, 74, 72, 71, 70, 69, 68, 66, 65, 64, 63, 62, 61, 60, 59,
    58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43,
    42, 41, 40, 40, 39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 31, 30,
    29, 28, 28, 27, 26, 25, 25, 24, 23, 23, 22, 21, 21, 20, 19, 19,
    18, 17, 17, 16, 15, 15, 14, 14, 13, 12, 12, 11, 11, 10, 9, 9,
    8, 8, 7, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0,
]  # fmt: skip


def _rsqrte7(val, e, s, sub):
    exp = (val >> s) & ((1 << e) - 1)
    sig = val & ((1 << s) - 1)
    sign = val >> (e + s)
    if sub:
        while not (sig & (1 << (s - 1))):
            exp -= 1
            sig <<= 1
        sig = (sig << 1) & ((1 << s) - 1)
    idx = ((exp & 1) << 6) | (sig >> (s - 6))
    out_sig = _RSQRT7[idx] << (s - 7)
    bias = (1 << (e - 1)) - 1
    out_exp = (3 * bias - 1 - exp) // 2
    return (sign << (s + e)) | (out_exp << s) | out_sig


def _recip7(val, e, s, rm, sub):
    exp = (val >> s) & ((1 << e) - 1)
    sig = val & ((1 << s) - 1)
    sign = val >> (e + s)
    if sub:
        while not (sig & (1 << (s - 1))):
            exp -= 1
            sig <<= 1
        sig = (sig << 1) & ((1 << s) - 1)
        if exp != 0 and exp != -1:  # result overflows
            sig_max = (1 << s) - 1
            exp_max = (1 << e) - 2
            if rm == sf.RTZ or (rm == sf.RDN and not sign) or (rm == sf.RUP and sign):
                return (sign << (s + e)) | (exp_max << s) | sig_max, sf.NX | sf.OF
            return (sign << (s + e)) | (((1 << e) - 1) << s), sf.NX | sf.OF
    idx = sig >> (s - 7)
    out_sig = _REC7[idx] << (s - 7)
    bias = (1 << (e - 1)) - 1
    out_exp = 2 * bias - 1 - exp
    if out_exp == 0 or out_exp == -1:
        out_sig = (out_sig >> 1) | (1 << (s - 1))
        if out_exp == -1:
            out_sig >>= 1
            out_exp = 0
    return (sign << (s + e)) | ((out_exp & ((1 << e) - 1)) << s) | out_sig, 0


def rsqrt7(a, sew):
    """``vfrsqrt7.v``: 7-bit reciprocal square-root estimate (Spike's f*_rsqrte7)."""
    fmt = FMT[sew]
    e, s = fmt.exp_bits, fmt.frac_bits
    out = np.empty(len(a), dtype=_U[sew])
    flags = 0
    cls = classify(a, sew)
    for i, (v, c) in enumerate(zip(a.tolist(), cls.tolist())):
        if c & 0b111:  # negative inf / normal / subnormal
            r, f = fmt.canonical_nan, sf.NV
        elif c & 0x100:
            r, f = fmt.canonical_nan, sf.NV
        elif c & 0x200:
            r, f = fmt.canonical_nan, 0
        elif c & 0x8:  # -0
            r, f = (1 << (fmt.bits - 1)) | (((1 << e) - 1) << s), sf.DZ
        elif c & 0x10:  # +0
            r, f = ((1 << e) - 1) << s, sf.DZ
        elif c & 0x80:  # +inf
            r, f = 0, 0
        else:
            r, f = _rsqrte7(v, e, s, bool(c & 0x20)), 0
        out[i] = r
        flags |= f
    return out, flags


def rec7(a, sew, rm):
    """``vfrec7.v``: 7-bit reciprocal estimate (Spike's f*_recip7)."""
    fmt = FMT[sew]
    e, s = fmt.exp_bits, fmt.frac_bits
    out = np.empty(len(a), dtype=_U[sew])
    flags = 0
    cls = classify(a, sew)
    sign_bit = 1 << (fmt.bits - 1)
    inf = ((1 << e) - 1) << s
    for i, (v, c) in enumerate(zip(a.tolist(), cls.tolist())):
        if c & 0x1:
            r, f = sign_bit, 0
        elif c & 0x80:
            r, f = 0, 0
        elif c & 0x8:
            r, f = sign_bit | inf, sf.DZ
        elif c & 0x10:
            r, f = inf, sf.DZ
        elif c & 0x100:
            r, f = fmt.canonical_nan, sf.NV
        elif c & 0x200:
            r, f = fmt.canonical_nan, 0
        else:
            r, f = _recip7(v, e, s, rm, bool(c & 0x24))
        out[i] = r
        flags |= f
    return out, flags
