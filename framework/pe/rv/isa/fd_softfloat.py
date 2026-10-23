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

This is the slow but correct path: there is no host-float fast path, because
the host FPU knows only round-to-nearest-even and reports no flags, and the
cases where it would agree (RNE, exact results) are the ones a cross-check
covers rather than the ones that go wrong. ``math.fma`` (Python 3.13) would not
help either, for the same reason.

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

from math import isqrt
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


def add(a, b, fmt, rm, negate_b=False):
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


def mul(a, b, fmt, rm):
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


def div(a, b, fmt, rm):
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


def sqrt(a, fmt, rm):
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


def fma(a, b, c, fmt, rm, negate_product=False, negate_addend=False):
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
