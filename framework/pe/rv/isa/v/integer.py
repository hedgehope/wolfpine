"""Integer and fixed-point vector arithmetic: one numpy expression per instruction.

Elements are held as unsigned numpy arrays of the SEW (``state.UINT``) and
reinterpreted as signed views where an operation is signed, so wrap-around is
numpy's own. Products wider than 64 bits (``vmulh*``, ``vsmul``, ``vwmul*`` at
SEW 64) go through :func:`mulhu64`, a 32-bit-limb high product.

Fixed point follows the spec's ``roundoff`` with ``vxrm`` (rnu / rne / rdn /
rod) and sets ``vxsat`` on saturation.
"""

import numpy as np

from framework.pe.rv.isa.v.common import (
    aligned,
    alu_ok,
    body_mask,
    count,
    nregs,
    overlapped,
    overlapped_widen,
    simm5,
    size,
    write_back,
    xscalar,
)
from framework.pe.rv.isa.v.state import SINT, UINT

_M32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)


def mulhu64(a, b):
    """High 64 bits of the unsigned 128-bit product of two uint64 arrays."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    alo, ahi = a & _M32, a >> _S32
    blo, bhi = b & _M32, b >> _S32
    ll = alo * blo
    lh = alo * bhi
    hl = ahi * blo
    hh = ahi * bhi
    mid = (ll >> _S32) + (lh & _M32) + (hl & _M32)
    return hh + (lh >> _S32) + (hl >> _S32) + (mid >> _S32)


def _mulh(a, b, sew, kind):
    """High half of a SEW x SEW product: kind 'uu', 'ss' or 'su' (vs2 signed, vs1 unsigned)."""
    u = UINT[sew]
    if sew == 64:
        hi = mulhu64(a, b)
        a_neg = (np.asarray(a, np.uint64) >> np.uint64(63)) == 1
        b_neg = (np.asarray(b, np.uint64) >> np.uint64(63)) == 1
        if kind in ("ss", "su"):
            hi = hi - np.where(a_neg, np.asarray(b, np.uint64), np.uint64(0))
        if kind == "ss":
            hi = hi - np.where(b_neg, np.asarray(a, np.uint64), np.uint64(0))
        return hi.astype(u)
    if kind == "uu":
        p = a.astype(np.uint64) * np.asarray(b).astype(np.uint64)
        return (p >> np.uint64(sew)).astype(u)
    sa = a.view(SINT[sew]).astype(np.int64)
    sb = (np.asarray(b).view(SINT[sew]) if kind == "ss" else np.asarray(b)).astype(
        np.int64
    )
    return ((sa * sb) >> sew).astype(np.int64).astype(u)


def _div(a, b, sew, kind):
    """vdivu / vdiv / vremu / vrem with the spec's divide-by-zero and overflow results."""
    u, s = UINT[sew], SINT[sew]
    b = np.broadcast_to(b, a.shape)
    zero = b == 0
    if kind in ("divu", "remu"):
        safe = np.where(zero, u.type(1), b)
        if kind == "divu":
            return np.where(zero, u.type((1 << sew) - 1), a // safe).astype(u)
        return np.where(zero, a, a % safe).astype(u)
    sa, sb = a.view(s), b.view(s)
    ovf = (sa == s.type(-(1 << (sew - 1)))) & (sb == -1)
    safe = np.where(zero | ovf, s.type(1), sb)
    r = np.fmod(sa, safe)
    if kind == "div":
        q = (sa - r) // safe
        q = np.where(zero, s.type(-1), np.where(ovf, sa, q))
        return q.astype(s).view(u)
    r = np.where(zero, sa, np.where(ovf, s.type(0), r))
    return r.astype(s).view(u)


def _sh(b, sew):
    return (np.asarray(b) & UINT[sew].type(sew - 1)).astype(UINT[sew])


def _sra(a, b, sew):
    s = SINT[sew]
    return (a.view(s) >> _sh(b, sew).astype(s)).view(UINT[sew])


#: name -> f(a, b, sew) for SEW-wide results. ``a`` is vs2, ``b`` vs1 / rs1 / imm.
INT_BINARY = {
    "add": lambda a, b, sew: a + b,
    "sub": lambda a, b, sew: a - b,
    "rsub": lambda a, b, sew: b - a,
    "and": lambda a, b, sew: a & b,
    "or": lambda a, b, sew: a | b,
    "xor": lambda a, b, sew: a ^ b,
    "minu": lambda a, b, sew: np.minimum(a, b),
    "maxu": lambda a, b, sew: np.maximum(a, b),
    "min": lambda a, b, sew: np.minimum(
        a.view(SINT[sew]), np.asarray(b).view(SINT[sew])
    ).view(UINT[sew]),
    "max": lambda a, b, sew: np.maximum(
        a.view(SINT[sew]), np.asarray(b).view(SINT[sew])
    ).view(UINT[sew]),
    "sll": lambda a, b, sew: a << _sh(b, sew),
    "srl": lambda a, b, sew: a >> _sh(b, sew),
    "sra": _sra,
    "mul": lambda a, b, sew: a * b,
    "mulh": lambda a, b, sew: _mulh(a, b, sew, "ss"),
    "mulhu": lambda a, b, sew: _mulh(a, b, sew, "uu"),
    "mulhsu": lambda a, b, sew: _mulh(a, b, sew, "su"),
    "divu": lambda a, b, sew: _div(a, b, sew, "divu"),
    "div": lambda a, b, sew: _div(a, b, sew, "div"),
    "remu": lambda a, b, sew: _div(a, b, sew, "remu"),
    "rem": lambda a, b, sew: _div(a, b, sew, "rem"),
}

#: name -> f(a, b, sew) -> bool array, for the mask-producing compares.
INT_COMPARE = {
    "seq": lambda a, b, sew: a == b,
    "sne": lambda a, b, sew: a != b,
    "sltu": lambda a, b, sew: a < b,
    "slt": lambda a, b, sew: a.view(SINT[sew]) < np.asarray(b).view(SINT[sew]),
    "sleu": lambda a, b, sew: a <= b,
    "sle": lambda a, b, sew: a.view(SINT[sew]) <= np.asarray(b).view(SINT[sew]),
    "sgtu": lambda a, b, sew: a > b,
    "sgt": lambda a, b, sew: a.view(SINT[sew]) > np.asarray(b).view(SINT[sew]),
}


def operand(vs, rf, kind, src, sew, n, vl):
    """vs1 elements, or the x-register / immediate scalar, for an OPI / OPM instruction."""
    if kind == "v":
        return vs.group(src, n, UINT[sew])[:vl]
    if kind == "x":
        return xscalar(rf, src, sew)
    if kind == "i":
        return simm5(src, sew)
    return UINT[sew].type(src)  # 'u': the zero-extended uimm5


# -- single-width ------------------------------------------------------------------


def binary(vs, rf, mem, func, vd, vs2, kind, src, vm):
    """vd = f(vs2, src): VI_CHECK_SSS."""
    if not alu_ok(vs) or (not vm and vd == 0):
        return False
    lg = vs.lmul_log2
    if not (aligned(vd, lg) and aligned(vs2, lg) and (kind != "v" or aligned(src, lg))):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(lg)
    if vl:
        u = UINT[sew]
        a = vs.group(vs2, n, u)[:vl]
        b = operand(vs, rf, kind, src, sew, n, vl)
        m = body_mask(vs, vm, vl)
        write_back(vs.group(vd, n, u), vl, func(a, b, sew), m)
        count(vs, vl, m)
    return True


def compare(vs, rf, mem, func, vd, vs2, kind, src, vm):
    """Mask vd = f(vs2, src) on the body, undisturbed elsewhere: VI_CHECK_MSS."""
    if not alu_ok(vs):
        return False
    lg = vs.lmul_log2
    if not aligned(vs2, lg) or (vd != vs2 and overlapped(vd, 1, vs2, size(lg))):
        return False
    if kind == "v" and (
        not aligned(src, lg) or (vd != src and overlapped(vd, 1, src, size(lg)))
    ):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(lg)
    if vl:
        a = vs.group(vs2, n, UINT[sew])[:vl]
        b = operand(vs, rf, kind, src, sew, n, vl)
        res = func(a, b, sew)
        m = body_mask(vs, vm, vl)
        _write_mask(vs, vd, vl, res, m)
        count(vs, vl, m)
    return True


def _write_mask(vs, vd, vl, bits, m):
    cur = vs.mask(vd)
    if m is None:
        cur[:vl] = bits
    else:
        np.copyto(cur[:vl], bits, where=m)
    vs.vrf[vd] = np.packbits(cur, bitorder="little")


def multiply_add(vs, rf, mem, name, vd, vs2, kind, src, vm):
    """vmacc / vnmsac / vmadd / vnmsub: three-operand with vd as the accumulator or multiplicand."""
    if not alu_ok(vs) or (not vm and vd == 0):
        return False
    lg = vs.lmul_log2
    if not (aligned(vd, lg) and aligned(vs2, lg) and (kind != "v" or aligned(src, lg))):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(lg)
    if vl:
        u = UINT[sew]
        a = vs.group(vs2, n, u)[:vl]
        b = operand(vs, rf, kind, src, sew, n, vl)
        dst = vs.group(vd, n, u)
        d = dst[:vl]
        if name == "macc":
            res = b * a + d
        elif name == "nmsac":
            res = d - b * a
        elif name == "madd":
            res = b * d + a
        else:  # nmsub
            res = a - b * d
        m = body_mask(vs, vm, vl)
        write_back(dst, vl, res, m)
        count(vs, vl, m)
    return True


def merge(vs, rf, mem, vd, vs2, kind, src, vm):
    """vmerge.v[vxi]m (vm = 0) and vmv.v.[vxi] (vm = 1, vs2 = 0)."""
    if not alu_ok(vs):
        return False
    lg = vs.lmul_log2
    if vm and vs2 != 0:
        return False
    if not vm and vd == 0:
        return False
    if not (
        aligned(vd, lg)
        and (vm or aligned(vs2, lg))
        and (kind != "v" or aligned(src, lg))
    ):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(lg)
    if vl:
        u = UINT[sew]
        b = operand(vs, rf, kind, src, sew, n, vl)
        dst = vs.group(vd, n, u)
        if vm:
            dst[:vl] = b
        else:
            a = vs.group(vs2, n, u)[:vl]
            dst[:vl] = np.where(vs.mask(0, vl), b, a)
        vs.elements += vl
    return True


def carry(vs, rf, mem, name, vd, vs2, kind, src, vm):
    """vadc / vsbc (vm must be 0: v0 is the carry/borrow in) and vmadc / vmsbc (mask out)."""
    if not alu_ok(vs):
        return False
    lg = vs.lmul_log2
    produces_mask = name in ("madc", "msbc")
    if not produces_mask and (vm or vd == 0):
        return False
    if produces_mask:
        if not aligned(vs2, lg) or (vd != vs2 and overlapped(vd, 1, vs2, size(lg))):
            return False
        if kind == "v" and (
            not aligned(src, lg) or (vd != src and overlapped(vd, 1, src, size(lg)))
        ):
            return False
    elif not (
        aligned(vd, lg) and aligned(vs2, lg) and (kind != "v" or aligned(src, lg))
    ):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(lg)
    if vl:
        u = UINT[sew]
        a = vs.group(vs2, n, u)[:vl]
        b = np.broadcast_to(operand(vs, rf, kind, src, sew, n, vl), a.shape)
        cin = vs.mask(0, vl).astype(u) if not vm else u.type(0)
        if name in ("adc", "madc"):
            s = a + b
            res = s + cin
            cout = (s < a) | (res < s)
        else:
            d = a - b
            res = d - cin
            cout = (a < b) | (d < cin)
        if produces_mask:
            _write_mask(vs, vd, vl, cout, None)
        else:
            vs.group(vd, n, u)[:vl] = res
        vs.elements += vl
    return True


# -- fixed point ---------------------------------------------------------------------


def round_inc(v, d, vxrm):
    """The spec's roundoff increment ``r`` for ``v >> d`` (uint64 arrays; ``d`` may be 0)."""
    v = np.asarray(v, dtype=np.uint64)
    d = np.broadcast_to(np.asarray(d, dtype=np.uint64), v.shape)
    one = np.uint64(1)
    pos = d > 0
    dm1 = np.where(pos, d - one, np.uint64(0))
    bit = np.where(pos, (v >> dm1) & one, np.uint64(0)).astype(bool)
    rest_mask = np.where(
        d > 1, (one << np.minimum(dm1, np.uint64(63))) - one, np.uint64(0)
    )
    rest = (v & rest_mask) != 0
    vd_bit = ((v >> np.minimum(d, np.uint64(63))) & one).astype(bool) & (d < 64)
    if vxrm == 0:  # rnu
        r = bit
    elif vxrm == 1:  # rne
        r = bit & (rest | vd_bit)
    elif vxrm == 2:  # rdn
        r = np.zeros(v.shape, bool)
    else:  # rod
        r = ~vd_bit & (bit | rest)
    return r.astype(np.uint64)


def _sat_add(a, b, sew, kind, vs):
    u, s = UINT[sew], SINT[sew]
    b = np.broadcast_to(b, a.shape)
    if kind == "addu":
        r = a + b
        sat = r < a
        out = np.where(sat, u.type((1 << sew) - 1), r)
    elif kind == "subu":
        sat = a < b
        out = np.where(sat, u.type(0), a - b)
    else:
        sa, sb = a.view(s), b.view(s)
        r = (a + b if kind == "add" else a - b).view(s)
        if kind == "add":
            sat = ((sa < 0) == (sb < 0)) & ((r < 0) != (sa < 0))
        else:
            sat = ((sa < 0) != (sb < 0)) & ((r < 0) != (sa < 0))
        lim = np.where(sa < 0, s.type(-(1 << (sew - 1))), s.type((1 << (sew - 1)) - 1))
        out = np.where(sat, lim, r).view(u)
    return out.astype(u), sat


def _avg(a, b, sew, kind, vxrm):
    """vaadd(u) / vasub(u): (a +/- b) >> 1 with rounding, computed without overflow."""
    u = UINT[sew]
    b = np.broadcast_to(b, a.shape)
    one = u.type(1)
    if kind in ("aaddu", "asubu"):
        x, y = a, b
        shr = lambda t: t >> one  # noqa: E731
    else:
        s = SINT[sew]
        x, y = a.view(s), b.view(s)
        shr = lambda t: t >> s.type(1)  # noqa: E731
    if kind in ("aaddu", "aadd"):
        half = shr(x) + shr(y) + (x & y & 1)
    else:
        half = shr(x) - shr(y) - ((~x) & y & 1)
    lsb = ((a ^ b) & one).astype(bool)
    hb = (half.view(u) & one).astype(bool)
    if vxrm == 0:
        r = lsb
    elif vxrm == 1:
        r = lsb & hb
    elif vxrm == 2:
        r = np.zeros(a.shape, bool)
    else:
        r = lsb & ~hb
    return (half.view(u) + r.astype(u)).astype(u)


def _smul(a, b, sew, vxrm):
    """vsmul: signed (a*b) >> (SEW-1), rounded, saturating; returns (result, saturated)."""
    u, s = UINT[sew], SINT[sew]
    b = np.broadcast_to(b, a.shape)
    lo_lim = s.type(-(1 << (sew - 1)))
    sat = (a.view(s) == lo_lim) & (b.view(s) == lo_lim)
    if sew < 64:
        p = a.view(s).astype(np.int64) * b.view(s).astype(np.int64)
        d = sew - 1
        r = round_inc(p.view(np.uint64), np.uint64(d), vxrm)
        res = ((p >> d) + r.view(np.int64)).astype(s)
    else:
        a64, b64 = a.astype(np.uint64), b.astype(np.uint64)
        hi = mulhu64(a64, b64)
        hi = hi - np.where((a64 >> np.uint64(63)) == 1, b64, np.uint64(0))
        hi = hi - np.where((b64 >> np.uint64(63)) == 1, a64, np.uint64(0))
        lo = a64 * b64
        shifted = (hi << np.uint64(1)) | (lo >> np.uint64(63))
        bit = ((lo >> np.uint64(62)) & np.uint64(1)).astype(bool)
        rest = (lo & np.uint64((1 << 62) - 1)) != 0
        vd_bit = ((lo >> np.uint64(63)) & np.uint64(1)).astype(bool)
        if vxrm == 0:
            r = bit
        elif vxrm == 1:
            r = bit & (rest | vd_bit)
        elif vxrm == 2:
            r = np.zeros(a.shape, bool)
        else:
            r = ~vd_bit & (bit | rest)
        res = (shifted + r.astype(np.uint64)).view(np.int64)
    res = np.where(sat, s.type((1 << (sew - 1)) - 1), res).astype(s)
    return res.view(u), sat


def _scale_shift(a, b, sew, signed, vxrm):
    """vssrl / vssra."""
    u = UINT[sew]
    d = _sh(b, sew).astype(np.uint64)
    if signed:
        v = a.view(SINT[sew]).astype(np.int64)
        res = v >> d.astype(np.int64)
        bits = v.view(np.uint64) if sew == 64 else v.astype(np.uint64)
    else:
        v = a.astype(np.uint64)
        res = v >> d
        bits = v
    r = round_inc(bits, d, vxrm)
    return (res.astype(np.uint64) + r).astype(u)


def fixed(vs, rf, mem, name, vd, vs2, kind, src, vm):
    """vsadd(u) / vssub(u) / vaadd(u) / vasub(u) / vsmul / vssrl / vssra: VI_CHECK_SSS."""
    if not alu_ok(vs) or (not vm and vd == 0):
        return False
    lg = vs.lmul_log2
    if not (aligned(vd, lg) and aligned(vs2, lg) and (kind != "v" or aligned(src, lg))):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(lg)
    if vl:
        u = UINT[sew]
        a = vs.group(vs2, n, u)[:vl]
        b = operand(vs, rf, kind, src, sew, n, vl)
        m = body_mask(vs, vm, vl)
        sat = None
        if name in ("saddu", "sadd", "ssubu", "ssub"):
            res, sat = _sat_add(
                a,
                b,
                sew,
                {"saddu": "addu", "sadd": "add", "ssubu": "subu", "ssub": "sub"}[name],
                vs,
            )
        elif name in ("aaddu", "aadd", "asubu", "asub"):
            res = _avg(a, b, sew, name, vs.vxrm)
        elif name == "smul":
            res, sat = _smul(a, b, sew, vs.vxrm)
        else:
            res = _scale_shift(a, b, sew, name == "ssra", vs.vxrm)
        if sat is not None and (sat if m is None else sat & m).any():
            vs.vxsat = 1
        write_back(vs.group(vd, n, u), vl, res, m)
        count(vs, vl, m)
    return True


# -- widening ------------------------------------------------------------------------


def _ext(x, sew, sign):
    """SEW elements (or a scalar) extended to 2*SEW, as the unsigned 2*SEW type."""
    w = UINT[2 * sew]
    x = np.asarray(x)
    if sign:
        return x.view(SINT[sew]).astype(SINT[2 * sew]).view(w)
    return x.astype(w)


def widen_check(vs, vd, vs2, kind, src, vm, wide_vs2=False):
    """VI_CHECK_DSS / VI_CHECK_DDS."""
    lg = vs.lmul_log2
    if (
        lg > 2
        or vs.sew * 2 > vs.elen
        or not aligned(vd, lg + 1)
        or (not vm and vd == 0)
    ):
        return False
    if wide_vs2:
        if not aligned(vs2, lg + 1):
            return False
    else:
        if not aligned(vs2, lg):
            return False
        if (overlapped if lg < 0 else overlapped_widen)(
            vd, size(lg + 1), vs2, size(lg)
        ):
            return False
    if kind == "v":
        if not aligned(src, lg):
            return False
        if (overlapped if lg < 0 else overlapped_widen)(
            vd, size(lg + 1), src, size(lg)
        ):
            return False
    return True


#: name -> (vs2 signed, src signed, op, vs2 already wide)
_WIDEN = {
    "waddu": (False, False, "add", False),
    "wadd": (True, True, "add", False),
    "wsubu": (False, False, "sub", False),
    "wsub": (True, True, "sub", False),
    "waddu.w": (False, False, "add", True),
    "wadd.w": (True, True, "add", True),
    "wsubu.w": (False, False, "sub", True),
    "wsub.w": (True, True, "sub", True),
    "wmulu": (False, False, "mul", False),
    "wmulsu": (True, False, "mul", False),
    "wmul": (True, True, "mul", False),
    "wmaccu": (False, False, "macc", False),
    "wmacc": (True, True, "macc", False),
    "wmaccsu": (False, True, "macc", False),  # vd += signed(vs1) * unsigned(vs2)
    "wmaccus": (True, False, "macc", False),  # vd += unsigned(x) * signed(vs2)
}


def widening(vs, rf, mem, name, vd, vs2, kind, src, vm):
    a_signed, b_signed, op, wide = _WIDEN[name]
    if not alu_ok(vs) or not widen_check(vs, vd, vs2, kind, src, vm, wide):
        return False
    sew, vl, lg = vs.sew, vs.vl, vs.lmul_log2
    if vl:
        w = UINT[2 * sew]
        n = nregs(lg)
        if wide:
            a = vs.group(vs2, nregs(lg + 1), w)[:vl]
        else:
            a = _ext(vs.group(vs2, n, UINT[sew])[:vl], sew, a_signed)
        b = _ext(operand(vs, rf, kind, src, sew, n, vl), sew, b_signed)
        dst = vs.group(vd, nregs(lg + 1), w)
        if op == "add":
            res = a + b
        elif op == "sub":
            res = a - b
        elif op == "mul":
            res = a * b
        else:
            res = dst[:vl] + a * b
        m = body_mask(vs, vm, vl)
        write_back(dst, vl, res.astype(w), m)
        count(vs, vl, m)
    return True


# -- narrowing -----------------------------------------------------------------------


def narrow_check(vs, vd, vs2, kind, src, vm):
    """VI_CHECK_SDS."""
    lg = vs.lmul_log2
    if (
        lg > 2
        or vs.sew * 2 > vs.elen
        or not aligned(vs2, lg + 1)
        or not aligned(vd, lg)
    ):
        return False
    if not vm and vd == 0:
        return False
    if vd != vs2 and overlapped(vd, size(lg), vs2, size(lg + 1)):
        return False
    return kind != "v" or aligned(src, lg)


def narrowing(vs, rf, mem, name, vd, vs2, kind, src, vm):
    """vnsrl / vnsra / vnclipu / vnclip (.wv / .wx / .wi)."""
    if not alu_ok(vs) or not narrow_check(vs, vd, vs2, kind, src, vm):
        return False
    sew, vl, lg = vs.sew, vs.vl, vs.lmul_log2
    if vl:
        u, w = UINT[sew], UINT[2 * sew]
        a = vs.group(vs2, nregs(lg + 1), w)[:vl]
        b = operand(vs, rf, kind, src, sew, nregs(lg), vl)
        d = (np.asarray(b).astype(np.uint64)) & np.uint64(2 * sew - 1)
        m = body_mask(vs, vm, vl)
        if name in ("nsrl", "nsra"):
            if name == "nsrl":
                res = a.astype(np.uint64) >> d
            else:
                res = (
                    a.view(SINT[2 * sew]).astype(np.int64) >> d.astype(np.int64)
                ).view(np.uint64)
            res = res.astype(u)
        else:
            if name == "nclipu":
                v = a.astype(np.uint64)
                full = (v >> d) + round_inc(
                    v, d, vs.vxrm
                )  # cannot wrap: d = 0 rounds by 0
                over = full > np.uint64((1 << sew) - 1)
                res = np.where(over, np.uint64((1 << sew) - 1), full).astype(u)
            else:
                v = a.view(SINT[2 * sew]).astype(np.int64)
                full = (v >> d.astype(np.int64)) + round_inc(
                    v.view(np.uint64), d, vs.vxrm
                ).view(np.int64)
                lo, hi = -(1 << (sew - 1)), (1 << (sew - 1)) - 1
                over = (full > hi) | (full < lo)
                res = np.clip(full, lo, hi).astype(SINT[sew]).view(u)
            if (over if m is None else over & m).any():
                vs.vxsat = 1
        write_back(vs.group(vd, nregs(lg), u), vl, res, m)
        count(vs, vl, m)
    return True


def extension(vs, rf, mem, vd, vs2, factor, signed, vm):
    """vzext.vf2/4/8 and vsext.vf2/4/8."""
    if not alu_ok(vs) or (not vm and vd == 0) or vd == vs2:
        return False
    sew, lg = vs.sew, vs.lmul_log2
    shift = {2: 1, 4: 2, 8: 3}[factor]
    src_sew = sew // factor
    if src_sew < 8 or lg - shift < -3:
        return False
    if not aligned(vd, lg) or not aligned(vs2, lg - shift):
        return False
    if lg - shift < 0:
        if overlapped(vd, size(lg), vs2, size(lg - shift)):
            return False
    elif overlapped_widen(vd, size(lg), vs2, size(lg - shift)):
        return False
    vl = vs.vl
    if vl:
        src = vs.group(vs2, nregs(lg - shift), UINT[src_sew])[:vl]
        if signed:
            res = src.view(SINT[src_sew]).astype(SINT[sew]).view(UINT[sew])
        else:
            res = src.astype(UINT[sew])
        m = body_mask(vs, vm, vl)
        write_back(vs.group(vd, nregs(lg), UINT[sew]), vl, res, m)
        count(vs, vl, m)
    return True


# -- reductions ----------------------------------------------------------------------


def reduction(vs, rf, mem, name, vd, vs2, vs1, vm):
    """vred* (single width) and vwredsum(u) (2*SEW accumulator)."""
    wide = name in ("wsumu", "wsum")
    if vs.vill or vs.vstart != 0:
        return False
    if wide and vs.sew * 2 > vs.elen:
        return False
    lg = vs.lmul_log2
    if not aligned(vs2, lg):
        return False
    sew, vl = vs.sew, vs.vl
    if vl == 0:
        return True
    u = UINT[sew]
    a = vs.group(vs2, nregs(lg), u)[:vl]
    if not vm:
        m = vs.mask(0, vl)
        a = a[m]
    if wide:
        w = UINT[2 * sew]
        start = vs.group(vs1, 1, w)[0]
        ext = _ext(a, sew, name == "wsum")
        res = np.add(start, ext.sum(dtype=w), dtype=w)
        vs.group(vd, 1, w)[0] = res
        vs.elements += len(a)
        return True
    start = vs.group(vs1, 1, u)[0]
    if name == "sum":
        res = np.add(start, a.sum(dtype=u), dtype=u)
    elif name == "and":
        res = np.bitwise_and.reduce(a, initial=start)
    elif name == "or":
        res = np.bitwise_or.reduce(a, initial=start)
    elif name == "xor":
        res = np.bitwise_xor.reduce(a, initial=start)
    elif name in ("minu", "maxu"):
        f = np.minimum if name == "minu" else np.maximum
        res = f.reduce(a, initial=start)
    else:
        s = SINT[sew]
        f = np.minimum if name == "min" else np.maximum
        res = f.reduce(a.view(s), initial=start.view(s)).view(u) if len(a) else start
    vs.group(vd, 1, u)[0] = u.type(res)
    vs.elements += len(a)
    return True
