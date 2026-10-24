"""Exact IEEE-754 binary arithmetic for the RISC-V F / D / Zfbfmin ISAs.

Every operation works on bit patterns and returns ``(bits, fflags)``. A finite
operand is decoded to an exact ``(sign, m, e)`` with value ``(-1)**sign * m *
2**e`` (``m`` a non-negative Python int), the operation is done exactly in
integers, and :func:`round_pack` rounds the exact result once, under any of the
five RISC-V rounding modes, raising the flags the spec asks for. Python ints are
unbounded, so a product, a fused multiply-add or an add of operands 2000
binades apart is held exactly with no sticky-bit bookkeeping; only division and
square root, whose exact results need not be finite binary fractions, compute a
few guard bits and fold the remainder into a sticky bit.

**Host-float fast path.** Under RNE, the common case of fp32 / fp64 add, sub,
mul, div, sqrt and fused multiply-add is computed with host doubles instead
(fp64 fma, which would need ``math.fma`` from Python 3.13, with a specialised
integer path), where the result is provably the exact path's, bits and flags alike
(see :func:`_fast_add64` and below for the argument per operation). It covers
normal operands with a result whose biased exponent is at least 2 (so it is
neither tiny before nor after rounding) and below the overflow threshold, in
which case the only flag is NX, decided by an exact check. Anything else (other
rounding modes, zeros, subnormals, infinities, NaNs, a tiny or overflowing
result, an fp32 sum that lands exactly on a rounding midpoint) falls through
to the exact path, which stays the definition: ``add_soft`` and friends are that path
alone, for cross-checking.

RISC-V conventions modelled (unprivileged spec, chapters F and D):

* tininess is detected **after** rounding, and underflow is raised only when
  the tiny result is also inexact;
* every NaN result is the canonical quiet NaN; a signalling NaN input raises NV
  (invalid), a quiet one does not, except where the operation says otherwise
  (``flt`` / ``fle``);
* min / max are IEEE 754-2019 minimumNumber / maximumNumber with -0 < +0;
* float -> integer conversions saturate, and NaN converts to the largest
  integer; an out-of-range input raises NV and not NX.
"""

import struct
from math import isqrt
from math import sqrt as _host_sqrt
from typing import NamedTuple

# fflags bits.
NX = 0x01  # inexact
UF = 0x02  # underflow
OF = 0x04  # overflow
DZ = 0x08  # divide by zero
NV = 0x10  # invalid operation

# Rounding modes, as encoded in an instruction's rm field and in frm.
RNE, RTZ, RDN, RUP, RMM = 0, 1, 2, 3, 4
DYN = 7
ROUNDING_MODE_NAMES = ("rne", "rtz", "rdn", "rup", "rmm", None, None, "dyn")

# Operand classes from :func:`decode`.
FINITE, INF, QNAN, SNAN = 0, 1, 2, 3


class FloatFormat(NamedTuple):
    name: str
    exp_bits: int
    frac_bits: int

    @property
    def bits(self):
        return 1 + self.exp_bits + self.frac_bits

    @property
    def bias(self):
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def precision(self):
        """Significant bits, hidden bit included."""
        return self.frac_bits + 1

    @property
    def canonical_nan(self):
        return (((1 << self.exp_bits) - 1) << self.frac_bits) | (
            1 << (self.frac_bits - 1)
        )


F32 = FloatFormat("s", 8, 23)
F64 = FloatFormat("d", 11, 52)
BF16 = FloatFormat("bf16", 8, 7)


def decode(bits, fmt):
    """``(sign, m, e, cls)`` for a bit pattern; ``m`` / ``e`` are 0 unless finite."""
    frac_bits = fmt.frac_bits
    sign = bits >> (fmt.bits - 1)
    exp = (bits >> frac_bits) & ((1 << fmt.exp_bits) - 1)
    frac = bits & ((1 << frac_bits) - 1)
    if exp == (1 << fmt.exp_bits) - 1:
        if frac == 0:
            return sign, 0, 0, INF
        return sign, 0, 0, (QNAN if frac >> (frac_bits - 1) else SNAN)
    if exp == 0:
        return sign, frac, 1 - fmt.bias - frac_bits, FINITE
    return sign, frac | (1 << frac_bits), exp - fmt.bias - frac_bits, FINITE


def _shift_round(m, shift, sign, rm):
    """``m / 2**shift`` (``shift`` > 0) rounded to an integer under ``rm``.

    Returns ``(mantissa, inexact)``.
    """
    mant = m >> shift
    rem = m & ((1 << shift) - 1)
    if rem == 0:
        return mant, False
    if rm == RNE:
        half = 1 << (shift - 1)
        inc = rem > half or (rem == half and mant & 1)
    elif rm == RTZ:
        inc = False
    elif rm == RDN:
        inc = sign
    elif rm == RUP:
        inc = not sign
    else:  # RMM
        inc = rem >= 1 << (shift - 1)
    return mant + 1 if inc else mant, True


def _overflow(sign, fmt, rm):
    """The overflowed result: infinity, or the largest finite value when ``rm``
    rounds towards zero for this sign."""
    to_inf = rm in (RNE, RMM) or (rm == RDN and sign) or (rm == RUP and not sign)
    top = (1 << fmt.exp_bits) - 1
    bits = (top << fmt.frac_bits) if to_inf else ((top << fmt.frac_bits) - 1)
    return (sign << (fmt.bits - 1)) | bits, OF | NX


def round_pack(sign, m, e, fmt, rm):
    """The exact value ``(-1)**sign * m * 2**e`` rounded to ``fmt``.

    Returns ``(bits, fflags)``. ``m == 0`` gives a zero of the given sign.
    """
    sign_bit = sign << (fmt.bits - 1)
    if m == 0:
        return sign_bit, 0
    p = fmt.precision
    emin = 1 - fmt.bias
    n = m.bit_length()
    top = e + n - 1  # unbiased exponent of the leading bit
    # Exponent of the result's unit in the last place: subnormals share emin's.
    q = (top if top >= emin else emin) - (p - 1)
    shift = q - e
    if shift <= 0:
        mant, inexact = m << -shift, False
    else:
        mant, inexact = _shift_round(m, shift, sign, rm)
        if mant >> p:  # rounded up to the next binade
            mant >>= 1
            q += 1
    flags = 0
    if inexact:
        flags = NX
        if top < emin:
            # Tiny before rounding. Tiny after rounding too, unless the value
            # rounded to p bits with an unbounded exponent reaches 2**emin,
            # which only a value in the binade just below can do.
            tiny = True
            if top == emin - 1 and n > p:
                full, _ = _shift_round(m, n - p, sign, rm)
                tiny = not full >> p
            if tiny:
                flags |= UF
    if mant >> (p - 1):
        biased = q + (p - 1) + fmt.bias
        if biased >= (1 << fmt.exp_bits) - 1:
            return _overflow(sign, fmt, rm)
        return (
            sign_bit | (biased << fmt.frac_bits) | (mant & ((1 << fmt.frac_bits) - 1))
        ), flags
    return sign_bit | mant, flags  # subnormal, or zero


def _nan_result(fmt, *classes):
    """Canonical NaN, with NV if any operand is a signalling NaN."""
    return fmt.canonical_nan, (NV if SNAN in classes else 0)


def _inf(sign, fmt):
    return (sign << (fmt.bits - 1)) | (((1 << fmt.exp_bits) - 1) << fmt.frac_bits)


def _add_exact(sa, ma, ea, sb, mb, eb, fmt, rm):
    """Round the exact sum of two finite values."""
    e = ea if ea < eb else eb
    a = ma << (ea - e)
    b = mb << (eb - e)
    if sa == sb:
        total = a + b
        sign = sa
    else:
        total = a - b if sb else b - a  # the positive one minus the negative one
        sign = 0
        if total < 0:
            total, sign = -total, 1
    if total == 0:
        # An exact zero: the operands' sign if they agree (both zeros), else
        # +0, or -0 when rounding down.
        return ((sa if sa == sb else int(rm == RDN)) << (fmt.bits - 1)), 0
    return round_pack(sign, total, e, fmt, rm)


def add_soft(a, b, fmt, rm, negate_b=False):
    """``a + b`` (``a - b`` with ``negate_b``)."""
    sa, ma, ea, ca = decode(a, fmt)
    sb, mb, eb, cb = decode(b, fmt)
    if negate_b:
        sb ^= 1
    if ca >= QNAN or cb >= QNAN:
        return _nan_result(fmt, ca, cb)
    if ca == INF or cb == INF:
        if ca == INF and cb == INF and sa != sb:
            return fmt.canonical_nan, NV
        return _inf(sa if ca == INF else sb, fmt), 0
    return _add_exact(sa, ma, ea, sb, mb, eb, fmt, rm)


def mul_soft(a, b, fmt, rm):
    sa, ma, ea, ca = decode(a, fmt)
    sb, mb, eb, cb = decode(b, fmt)
    sign = sa ^ sb
    if ca >= QNAN or cb >= QNAN:
        return _nan_result(fmt, ca, cb)
    if ca == INF or cb == INF:
        if (ca == FINITE and ma == 0) or (cb == FINITE and mb == 0):
            return fmt.canonical_nan, NV
        return _inf(sign, fmt), 0
    return round_pack(sign, ma * mb, ea + eb, fmt, rm)


def div_soft(a, b, fmt, rm):
    sa, ma, ea, ca = decode(a, fmt)
    sb, mb, eb, cb = decode(b, fmt)
    sign = sa ^ sb
    if ca >= QNAN or cb >= QNAN:
        return _nan_result(fmt, ca, cb)
    if ca == INF:
        return (fmt.canonical_nan, NV) if cb == INF else (_inf(sign, fmt), 0)
    if cb == INF:
        return sign << (fmt.bits - 1), 0
    if mb == 0:
        if ma == 0:
            return fmt.canonical_nan, NV
        return _inf(sign, fmt), DZ
    if ma == 0:
        return sign << (fmt.bits - 1), 0
    # Enough quotient bits that the round and sticky positions are both below
    # the result's last place, then the remainder as one more sticky bit.
    k = max(0, fmt.precision + 3 + mb.bit_length() - ma.bit_length())
    q, r = divmod(ma << k, mb)
    return round_pack(sign, (q << 1) | (r != 0), ea - eb - k - 1, fmt, rm)


def sqrt_soft(a, fmt, rm):
    sa, ma, ea, ca = decode(a, fmt)
    if ca >= QNAN:
        return _nan_result(fmt, ca)
    if ca == FINITE and ma == 0:
        return a, 0  # sqrt(-0) is -0
    if sa:
        return fmt.canonical_nan, NV
    if ca == INF:
        return a, 0
    if ea & 1:
        ma <<= 1
        ea -= 1
    k = max(0, fmt.precision + 4 - ma.bit_length() // 2)
    scaled = ma << (2 * k)
    s = isqrt(scaled)
    return round_pack(0, (s << 1) | (s * s != scaled), (ea >> 1) - k - 1, fmt, rm)


def fma_soft(a, b, c, fmt, rm, negate_product=False, negate_addend=False):
    """``(+/-)(a * b) (+/-) c`` with one rounding (fmadd / fmsub / fnmsub / fnmadd)."""
    sa, ma, ea, ca = decode(a, fmt)
    sb, mb, eb, cb = decode(b, fmt)
    sc, mc, ec, cc = decode(c, fmt)
    sp = sa ^ sb ^ negate_product
    if negate_addend:
        sc ^= 1
    # inf * 0 is invalid even when the addend is a quiet NaN.
    if (ca == INF and cb == FINITE and mb == 0) or (
        cb == INF and ca == FINITE and ma == 0
    ):
        return fmt.canonical_nan, NV
    if ca >= QNAN or cb >= QNAN or cc >= QNAN:
        return _nan_result(fmt, ca, cb, cc)
    if ca == INF or cb == INF:
        if cc == INF and sc != sp:
            return fmt.canonical_nan, NV
        return _inf(sp, fmt), 0
    if cc == INF:
        return _inf(sc, fmt), 0
    return _add_exact(sp, ma * mb, ea + eb, sc, mc, ec, fmt, rm)


# -- the host-float fast path ---------------------------------------------------
#
# Python floats are IEEE binary64 with round-to-nearest-even (CPython never
# changes the host's rounding mode or enables flush-to-zero), and +, -, *, /
# and math.sqrt are correctly rounded. Every helper returns ``(bits, fflags)``
# exactly as the exact path would, or None to defer to it.

_D = struct.Struct("<d")
_Q = struct.Struct("<Q")
_F = struct.Struct("<f")
_I = struct.Struct("<I")
_unpack_d = _D.unpack
_pack_d = _D.pack
_unpack_q = _Q.unpack
_unpack_f = _F.unpack
_pack_f = _F.pack
_unpack_i = _I.unpack

_FRAC52 = (1 << 52) - 1
_HIDDEN52 = 1 << 52


def _f64(bits):
    return _unpack_d(bits.to_bytes(8, "little"))[0]


def _bits64(x):
    return _unpack_q(_pack_d(x))[0]


def _fast_add64(a, b, negate_b):
    """fp64 ``a + b`` under RNE, or None.

    Operands must be normal with biased exponent <= 0x7FC (|x| < 2**1022), so
    no step of TwoSum below can overflow; the result must have a biased
    exponent in [2, 0x7FE] (|s| >= 2**-1021: not tiny, and finite). Then ``s``
    is the correctly rounded sum (RNE) and TwoSum (Knuth) gives the exact
    rounding error ``err``, so NX is exactly ``err != 0``. A zero result
    (exact cancellation) is excluded by the exponent bound, so the sign-of-zero
    rule is the exact path's.
    """
    ea = (a >> 52) & 0x7FF
    eb = (b >> 52) & 0x7FF
    if not (0 < ea < 0x7FD and 0 < eb < 0x7FD):
        return None
    x = _f64(a)
    y = _f64(b)
    if negate_b:
        y = -y
    s = x + y
    r = _bits64(s)
    er = (r >> 52) & 0x7FF
    if er < 2 or er == 0x7FF:
        return None
    z = s - x
    err = (x - (s - z)) + (y - z)
    return r, (NX if err else 0)


def _fast_mul64(a, b):
    """fp64 ``a * b`` under RNE, or None.

    Normal operands and a result with biased exponent in [2, 0x7FE], so the
    exact product (>= the result times 1 - 2**-53) is in the normal range too,
    where it is representable exactly when its integer significand
    ``ma * mb`` has at most 53 significant bits. That decides NX.
    """
    ea = (a >> 52) & 0x7FF
    eb = (b >> 52) & 0x7FF
    if not (0 < ea < 0x7FF and 0 < eb < 0x7FF):
        return None
    r = _bits64(_f64(a) * _f64(b))
    er = (r >> 52) & 0x7FF
    if er < 2 or er == 0x7FF:
        return None
    m = ((a & _FRAC52) | _HIDDEN52) * ((b & _FRAC52) | _HIDDEN52)
    m >>= (m & -m).bit_length() - 1
    return r, (NX if m.bit_length() > 53 else 0)


def _fast_div64(a, b):
    """fp64 ``a / b`` under RNE, or None.

    Normal operands and a normal, finite, not-tiny result ``q`` (as for mul).
    The quotient is exact iff ``q * b == a`` exactly, which is checked in
    integers: ``mq * mb * 2**t == ma`` with ``t`` from the three exponents.
    """
    ea = (a >> 52) & 0x7FF
    eb = (b >> 52) & 0x7FF
    if not (0 < ea < 0x7FF and 0 < eb < 0x7FF):
        return None
    r = _bits64(_f64(a) / _f64(b))
    er = (r >> 52) & 0x7FF
    if er < 2 or er == 0x7FF:
        return None
    prod = ((r & _FRAC52) | _HIDDEN52) * ((b & _FRAC52) | _HIDDEN52)
    ma = (a & _FRAC52) | _HIDDEN52
    t = er + eb - ea - 1075
    exact = (prod << t) == ma if t >= 0 else prod == (ma << -t)
    return r, (0 if exact else NX)


def _fast_fma64(a, b, c, negate_product, negate_addend):
    """fp64 fused multiply-add under RNE, in integers, or None.

    There is no host fma before Python 3.13, so this is the exact path
    specialised to its common case: normal operands whose aligned terms lie
    within 160 bits of each other (so the integers stay small), a nonzero sum,
    and a result whose biased exponent is in [2, 0x7FE]. The exact sum is
    formed in integers and rounded once, to nearest-even, exactly as
    :func:`round_pack` does for a normal result; NX is a nonzero remainder.
    """
    ea = (a >> 52) & 0x7FF
    eb = (b >> 52) & 0x7FF
    ec = (c >> 52) & 0x7FF
    if not (0 < ea < 0x7FF and 0 < eb < 0x7FF and 0 < ec < 0x7FF):
        return None
    p = ((a & _FRAC52) | _HIDDEN52) * ((b & _FRAC52) | _HIDDEN52)
    ep = ea + eb - 2150  # p's value is p * 2**ep
    m = (c & _FRAC52) | _HIDDEN52
    em = ec - 1075
    d = ep - em
    if d > 160 or d < -160:
        return None
    if d >= 0:
        p <<= d
        e = em
    else:
        m <<= -d
        e = ep
    if ((a ^ b) >> 63) ^ negate_product:
        p = -p
    if (c >> 63) ^ negate_addend:
        m = -m
    total = p + m
    if total == 0:
        return None
    sign = 0
    if total < 0:
        sign = 1
        total = -total
    n = total.bit_length()
    biased = e + n - 1 + 1023
    if biased < 2 or biased > 0x7FE:
        return None
    shift = n - 53
    if shift <= 0:
        return (sign << 63) | (biased << 52) | ((total << -shift) & _FRAC52), 0
    mant = total >> shift
    rem = total & ((1 << shift) - 1)
    if rem:
        half = 1 << (shift - 1)
        if rem > half or (rem == half and mant & 1):
            mant += 1
            if mant >> 53:
                mant >>= 1
                biased += 1
                if biased > 0x7FE:
                    return None
        return (sign << 63) | (biased << 52) | (mant & _FRAC52), NX
    return (sign << 63) | (biased << 52) | (mant & _FRAC52), 0


def _fast_sqrt64(a):
    """fp64 ``sqrt(a)`` under RNE, or None.

    A positive normal operand (a negative one is NV, a zero keeps its sign:
    both the exact path's). The root of a normal double is normal, and
    ``math.sqrt`` is correctly rounded; it is exact iff ``s * s == a``,
    checked in integers.
    """
    ea = a >> 52  # includes the sign bit: a negative operand fails the bound
    if not 0 < ea < 0x7FF:
        return None
    r = _bits64(_host_sqrt(_f64(a)))
    er = r >> 52
    ms = (r & _FRAC52) | _HIDDEN52
    ma = (a & _FRAC52) | _HIDDEN52
    t = 2 * er - ea - 1075
    sq = ms * ms
    exact = (sq << t) == ma if t >= 0 else sq == (ma << -t)
    return r, (0 if exact else NX)


# fp32 goes through binary64: a single is exactly a double, the product of two
# singles is exact in a double (48 significant bits, exponents far inside the
# double range), and for +, -, *, /, sqrt rounding the double result to single
# gives the correctly rounded single result, because 53 >= 2 * 24 + 2 (double
# rounding is innocuous; Figueroa, "When is double rounding innocuous?", 1995).
# Sums are handled separately (they are not one double operation away when the
# exact sum needs more than 53 bits): see _fast_sum32.


def _f32(bits):
    return _unpack_f(bits.to_bytes(4, "little"))[0]


def _round32(x):
    """``x`` rounded to single (RNE) as ``(bits, value)``, or None if it overflows or is tiny.

    ``struct``'s ``f`` packing is the C cast (RNE), and raises on overflow.
    A biased exponent below 2 (|result| < 2**-125) is left to the exact path,
    which decides tininess after rounding.
    """
    try:
        packed = _pack_f(x)
    except OverflowError:
        return None
    bits = _unpack_i(packed)[0]
    e = (bits >> 23) & 0xFF
    if e < 2 or e == 0xFF:
        return None
    return bits, _unpack_f(packed)[0]


def _normal32(bits):
    return 0 < (bits >> 23) & 0xFF < 0xFF


def _fast_sum32(x, y):
    """Single RNE of the exact ``x + y`` (doubles, each exact in at most 48 bits), or None.

    ``s = x + y`` in double with the TwoSum error ``err`` (exact; nothing here
    can overflow a double). If ``err == 0`` the sum is ``s`` exactly, so its
    single rounding is ``round32(s)``, inexact iff that differs from ``s``. If
    ``err != 0`` the exact sum is ``s + err`` with ``|err| <= ulp53(s) / 2``,
    which rounds to single as ``s`` does unless ``s`` is exactly a single
    rounding midpoint (every midpoint is a double, at least one ulp53 from any
    other double, so ``s + err`` cannot reach or cross one ``s`` is not on);
    a midpoint defers to the exact path, otherwise the result is inexact.
    ``s``'s binade is a single-normal one whenever the rounded result is, so a
    midpoint is a double whose low 29 significand bits are ``1 << 28``.
    """
    s = x + y
    rounded = _round32(s)
    if rounded is None:
        return None
    bits, value = rounded
    z = s - x
    err = (x - (s - z)) + (y - z)
    if err == 0:
        return bits, (0 if value == s else NX)
    if _bits64(s) & 0x1FFFFFFF == 0x10000000:
        return None
    return bits, NX


def _fast_add32(a, b, negate_b):
    if not (_normal32(a) and _normal32(b)):
        return None
    y = _f32(b)
    return _fast_sum32(_f32(a), -y if negate_b else y)


def _fast_mul32(a, b):
    if not (_normal32(a) and _normal32(b)):
        return None
    p = _f32(a) * _f32(b)  # exact
    rounded = _round32(p)
    if rounded is None:
        return None
    bits, value = rounded
    return bits, (0 if value == p else NX)


def _fast_div32(a, b):
    """Single quotient: double rounding is innocuous; exact iff ``q * b == a`` (exact in double)."""
    if not (_normal32(a) and _normal32(b)):
        return None
    x = _f32(a)
    y = _f32(b)
    rounded = _round32(x / y)
    if rounded is None:
        return None
    bits, value = rounded
    return bits, (0 if value * y == x else NX)


def _fast_sqrt32(a):
    """Single root: double rounding is innocuous; exact iff ``s * s == a`` (exact in double)."""
    if not 0 < a >> 23 < 0xFF:  # positive normal (the sign bit fails the bound)
        return None
    x = _f32(a)
    rounded = _round32(_host_sqrt(x))
    if rounded is None:
        return None
    bits, value = rounded
    return bits, (0 if value * value == x else NX)


def _fast_fma32(a, b, c, negate_product, negate_addend):
    """Single fused multiply-add: the product is exact in double, then one :func:`_fast_sum32`."""
    if not (_normal32(a) and _normal32(b) and _normal32(c)):
        return None
    p = _f32(a) * _f32(b)
    z = _f32(c)
    return _fast_sum32(-p if negate_product else p, -z if negate_addend else z)


def add(a, b, fmt, rm, negate_b=False):
    """``a + b`` (``a - b`` with ``negate_b``)."""
    if rm == RNE:
        if fmt is F64:
            fast = _fast_add64(a, b, negate_b)
        elif fmt is F32:
            fast = _fast_add32(a, b, negate_b)
        else:
            fast = None
        if fast is not None:
            return fast
    return add_soft(a, b, fmt, rm, negate_b)


def mul(a, b, fmt, rm):
    if rm == RNE:
        if fmt is F64:
            fast = _fast_mul64(a, b)
        elif fmt is F32:
            fast = _fast_mul32(a, b)
        else:
            fast = None
        if fast is not None:
            return fast
    return mul_soft(a, b, fmt, rm)


def div(a, b, fmt, rm):
    if rm == RNE:
        if fmt is F64:
            fast = _fast_div64(a, b)
        elif fmt is F32:
            fast = _fast_div32(a, b)
        else:
            fast = None
        if fast is not None:
            return fast
    return div_soft(a, b, fmt, rm)


def sqrt(a, fmt, rm):
    if rm == RNE:
        if fmt is F64:
            fast = _fast_sqrt64(a)
        elif fmt is F32:
            fast = _fast_sqrt32(a)
        else:
            fast = None
        if fast is not None:
            return fast
    return sqrt_soft(a, fmt, rm)


def fma(a, b, c, fmt, rm, negate_product=False, negate_addend=False):
    """``(+/-)(a * b) (+/-) c`` with one rounding (fmadd / fmsub / fnmsub / fnmadd)."""
    if rm == RNE:
        if fmt is F64:
            fast = _fast_fma64(a, b, c, negate_product, negate_addend)
        elif fmt is F32:
            fast = _fast_fma32(a, b, c, negate_product, negate_addend)
        else:
            fast = None
        if fast is not None:
            return fast
    return fma_soft(a, b, c, fmt, rm, negate_product, negate_addend)


def _order_key(bits, fmt):
    """A total-order key for a non-NaN value; -0 and +0 compare equal."""
    magnitude = bits & ((1 << (fmt.bits - 1)) - 1)
    return -magnitude if bits >> (fmt.bits - 1) else magnitude


def minmax(a, b, fmt, is_max):
    """IEEE 754-2019 minimumNumber / maximumNumber, with -0 < +0."""
    _, _, _, ca = decode(a, fmt)
    _, _, _, cb = decode(b, fmt)
    flags = NV if SNAN in (ca, cb) else 0
    a_nan, b_nan = ca >= QNAN, cb >= QNAN
    if a_nan and b_nan:
        return fmt.canonical_nan, flags
    if a_nan:
        return b, flags
    if b_nan:
        return a, flags
    ka, kb = _order_key(a, fmt), _order_key(b, fmt)
    if ka == kb:
        # Equal values differ only as +0 / -0: min takes the negative one.
        a_neg = a >> (fmt.bits - 1)
        return (b if a_neg else a) if is_max else (a if a_neg else b), flags
    return (a if ka > kb else b) if is_max else (a if ka < kb else b), flags


def compare(a, b, fmt, op):
    """``feq`` / ``flt`` / ``fle`` (``op`` = "eq" / "lt" / "le"): ``(0 or 1, fflags)``.

    ``feq`` is a quiet comparison (NV only for a signalling NaN); ``flt`` and
    ``fle`` are signalling (NV for any NaN).
    """
    _, _, _, ca = decode(a, fmt)
    _, _, _, cb = decode(b, fmt)
    if ca >= QNAN or cb >= QNAN:
        if op == "eq":
            return 0, (NV if SNAN in (ca, cb) else 0)
        return 0, NV
    ka, kb = _order_key(a, fmt), _order_key(b, fmt)
    if op == "eq":
        return int(ka == kb), 0
    if op == "lt":
        return int(ka < kb), 0
    return int(ka <= kb), 0


def fclass(a, fmt):
    """The ``fclass`` one-hot mask (bits 0-9)."""
    sign, m, _, cls = decode(a, fmt)
    if cls == INF:
        return 1 << (0 if sign else 7)
    if cls == SNAN:
        return 1 << 8
    if cls == QNAN:
        return 1 << 9
    if m == 0:
        return 1 << (3 if sign else 4)
    if (a >> fmt.frac_bits) & ((1 << fmt.exp_bits) - 1) == 0:
        return 1 << (2 if sign else 5)  # subnormal
    return 1 << (1 if sign else 6)


def convert(a, src, dst, rm):
    """Float -> float (``fcvt.s.d``, ``fcvt.d.s``, ``fcvt.bf16.s``, ...)."""
    sign, m, e, cls = decode(a, src)
    if cls >= QNAN:
        return _nan_result(dst, cls)
    if cls == INF:
        return _inf(sign, dst), 0
    return round_pack(sign, m, e, dst, rm)


def to_int(a, fmt, rm, bits, signed):
    """Float -> integer, saturating: ``(python int, fflags)``.

    The result is in ``[-2**(bits-1), 2**(bits-1))`` when ``signed``, else in
    ``[0, 2**bits)``; the caller reduces it to a register pattern.
    """
    lo, hi = (
        (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    )
    sign, m, e, cls = decode(a, fmt)
    if cls >= QNAN:
        return hi, NV
    if cls == INF:
        return (lo if sign else hi), NV
    if e >= 0:
        value, inexact = m << e, False
    else:
        value, inexact = _shift_round(m, -e, sign, rm)
    if sign:
        value = -value
    if value > hi:
        return hi, NV
    if value < lo:
        return lo, NV
    return value, (NX if inexact else 0)


def from_int(value, fmt, rm):
    """Integer (any Python int) -> float."""
    if value < 0:
        return round_pack(1, -value, 0, fmt, rm)
    return round_pack(0, value, 0, fmt, rm)
