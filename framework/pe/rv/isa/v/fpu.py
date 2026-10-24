"""Vector floating-point instructions (OPFVV / OPFVF), Zvfh at SEW 16, and the bf16 Zvfb* ops.

Arithmetic is :mod:`framework.pe.rv.isa.v.fp` (softfloat-exact, numpy fast
paths). Every instruction needs a legal ``frm`` (Spike's ``VFP_RM``), SEW 16
needs Zvfh, and SEW 8 is never a floating-point width. Only active body
elements are computed, so a masked-off element never raises a flag. Widening
instructions widen their operands exactly first (with the sNaN -> NV the
conversion raises) and then operate once at 2*SEW, as Spike does, so the
result is rounded once.
"""

import numpy as np

from framework.pe.rv.isa import fd_softfloat as sf
from framework.pe.rv.isa.v import fp
from framework.pe.rv.isa.v.common import (
    aligned,
    alu_ok,
    body_mask,
    flags,
    frm,
    fscalar,
    nregs,
    overlapped,
    size,
)
from framework.pe.rv.isa.v.integer import _write_mask, narrow_check, widen_check
from framework.pe.rv.isa.v.state import UINT


def fp_ok(vs, sew=None):
    """Is ``sew`` (default the current SEW) a vector FP width on this hart?"""
    sew = vs.sew if sew is None else sew
    if sew == 16:
        return vs.zvfh
    return sew == 32 or (sew == 64 and vs.elen >= 64)


def _active(m, vl):
    return slice(None) if m is None else m


def _src(vs, rf, kind, src, sew, n, vl, bf16=False):
    if kind == "v":
        return vs.group(src, n, UINT[sew])[:vl]
    return np.full(vl, fscalar(rf, src, sew, bf16), dtype=UINT[sew])


def _common(vs, rf, vd, vm, sew=None):
    """``(rm)`` for an FP instruction's shared preconditions, or None when it is illegal."""
    if not alu_ok(vs) or not fp_ok(vs, sew) or (not vm and vd == 0):
        return None
    return frm(rf)


def _apply(vs, dst, vl, m, compute):
    """Compute on the active elements only and write them back; returns the flags."""
    if m is None:
        res, f = compute(slice(None))
        dst[:vl] = res
    else:
        idx = np.flatnonzero(m)
        res, f = compute(idx)
        dst[idx] = res
    vs.elements += vl if m is None else len(np.flatnonzero(m))
    return f


def _sss(vs, vd, vs2, kind, src):
    lg = vs.lmul_log2
    return aligned(vd, lg) and aligned(vs2, lg) and (kind != "v" or aligned(src, lg))


# -- single width ----------------------------------------------------------------------


def binary(vs, rf, mem, op, vd, vs2, kind, src, vm, reverse=False):
    """vfadd / vfsub / vfrsub / vfmul / vfdiv / vfrdiv / vfmin / vfmax / vfsgnj[nx]."""
    rm = _common(vs, rf, vd, vm)
    if rm is None or not _sss(vs, vd, vs2, kind, src):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(vs.lmul_log2)
    if vl == 0:
        return True
    a = vs.group(vs2, n, UINT[sew])[:vl]
    b = _src(vs, rf, kind, src, sew, n, vl)
    if reverse:
        a, b = b, a
    m = body_mask(vs, vm, vl)

    def compute(sel):
        x, y = a[sel], b[sel]
        if op in ("add", "sub", "mul", "div"):
            return fp.binary(op, x, y, sew, rm)
        if op in ("min", "max"):
            return fp.minmax(x, y, sew, op == "max")
        return fp.sgnj(x, y, sew, ("sgnj", "sgnjn", "sgnjx").index(op)), 0

    flags(rf, _apply(vs, vs.group(vd, n, UINT[sew]), vl, m, compute))
    return True


#: name -> (product uses vd (madd forms), negate product, negate addend)
_FMA = {
    "macc": (False, False, False),
    "nmacc": (False, True, True),
    "msac": (False, False, True),
    "nmsac": (False, True, False),
    "madd": (True, False, False),
    "nmadd": (True, True, True),
    "msub": (True, False, True),
    "nmsub": (True, True, False),
}


def fma(vs, rf, mem, name, vd, vs2, kind, src, vm):
    rm = _common(vs, rf, vd, vm)
    if rm is None or not _sss(vs, vd, vs2, kind, src):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(vs.lmul_log2)
    if vl == 0:
        return True
    vd_form, neg_p, neg_a = _FMA[name]
    a = vs.group(vs2, n, UINT[sew])[:vl]
    b = _src(vs, rf, kind, src, sew, n, vl)
    dst = vs.group(vd, n, UINT[sew])
    d = dst[:vl].copy()
    m = body_mask(vs, vm, vl)

    def compute(sel):
        if vd_form:  # vd = +/-(vs1 * vd) +/- vs2
            return fp.fma(b[sel], d[sel], a[sel], sew, rm, neg_p, neg_a)
        return fp.fma(b[sel], a[sel], d[sel], sew, rm, neg_p, neg_a)

    flags(rf, _apply(vs, dst, vl, m, compute))
    return True


def compare(vs, rf, mem, op, vd, vs2, kind, src, vm):
    """vmfeq / vmfne / vmflt / vmfle / vmfgt / vmfge -> mask."""
    if not alu_ok(vs) or not fp_ok(vs) or frm(rf) is None:
        return False
    lg = vs.lmul_log2
    if not aligned(vs2, lg) or (vd != vs2 and overlapped(vd, 1, vs2, size(lg))):
        return False
    if kind == "v" and (
        not aligned(src, lg) or (vd != src and overlapped(vd, 1, src, size(lg)))
    ):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(lg)
    if vl == 0:
        return True
    a = vs.group(vs2, n, UINT[sew])[:vl]
    b = _src(vs, rf, kind, src, sew, n, vl)
    m = body_mask(vs, vm, vl)
    sel = slice(None) if m is None else np.flatnonzero(m)
    res, f = fp.compare(a[sel], b[sel], sew, op)
    bits = np.zeros(vl, bool)
    bits[sel] = res
    _write_mask(vs, vd, vl, bits, m)
    flags(rf, f)
    vs.elements += vl
    return True


def merge(vs, rf, mem, vd, vs2, rs1, vm):
    """vfmerge.vfm (vm = 0) / vfmv.v.f (vm = 1, vs2 = 0)."""
    if not alu_ok(vs) or not fp_ok(vs) or frm(rf) is None:
        return False
    if (vm and vs2 != 0) or (not vm and vd == 0):
        return False
    lg = vs.lmul_log2
    if not aligned(vd, lg) or not aligned(vs2, lg):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(lg)
    if vl:
        scalar = fscalar(rf, rs1, sew)
        dst = vs.group(vd, n, UINT[sew])
        if vm:
            dst[:vl] = scalar
        else:
            dst[:vl] = np.where(
                vs.mask(0, vl), scalar, vs.group(vs2, n, UINT[sew])[:vl]
            )
        vs.elements += vl
    return True


def unary(vs, rf, mem, op, vd, vs2, vm):
    """VFUNARY1: vfsqrt / vfrsqrt7 / vfrec7 / vfclass."""
    rm = _common(vs, rf, vd, vm)
    if rm is None or not _sss(vs, vd, vs2, "x", 0):
        return False
    sew, vl, n = vs.sew, vs.vl, nregs(vs.lmul_log2)
    if vl == 0:
        return True
    a = vs.group(vs2, n, UINT[sew])[:vl]
    m = body_mask(vs, vm, vl)

    def compute(sel):
        x = a[sel]
        if op == "sqrt":
            return fp.sqrt(x, sew, rm)
        if op == "rsqrt7":
            return fp.rsqrt7(x, sew)
        if op == "rec7":
            return fp.rec7(x, sew, rm)
        return fp.classify(x, sew).astype(UINT[sew]), 0

    flags(rf, _apply(vs, vs.group(vd, n, UINT[sew]), vl, m, compute))
    return True


# -- conversions -----------------------------------------------------------------------

#: VFUNARY0 vs1 code -> (name, shape, kind, signed, rounding override)
#: shape: "s" single width, "w" widening, "n" narrowing; kind: f2i / i2f / f2f / bf.
CONVERSIONS = {
    0x00: ("vfcvt.xu.f.v", "s", "f2i", False, None),
    0x01: ("vfcvt.x.f.v", "s", "f2i", True, None),
    0x02: ("vfcvt.f.xu.v", "s", "i2f", False, None),
    0x03: ("vfcvt.f.x.v", "s", "i2f", True, None),
    0x06: ("vfcvt.rtz.xu.f.v", "s", "f2i", False, sf.RTZ),
    0x07: ("vfcvt.rtz.x.f.v", "s", "f2i", True, sf.RTZ),
    0x08: ("vfwcvt.xu.f.v", "w", "f2i", False, None),
    0x09: ("vfwcvt.x.f.v", "w", "f2i", True, None),
    0x0A: ("vfwcvt.f.xu.v", "w", "i2f", False, None),
    0x0B: ("vfwcvt.f.x.v", "w", "i2f", True, None),
    0x0C: ("vfwcvt.f.f.v", "w", "f2f", False, None),
    0x0D: ("vfwcvtbf16.f.f.v", "w", "bf", False, None),
    0x0E: ("vfwcvt.rtz.xu.f.v", "w", "f2i", False, sf.RTZ),
    0x0F: ("vfwcvt.rtz.x.f.v", "w", "f2i", True, sf.RTZ),
    0x10: ("vfncvt.xu.f.w", "n", "f2i", False, None),
    0x11: ("vfncvt.x.f.w", "n", "f2i", True, None),
    0x12: ("vfncvt.f.xu.w", "n", "i2f", False, None),
    0x13: ("vfncvt.f.x.w", "n", "i2f", True, None),
    0x14: ("vfncvt.f.f.w", "n", "f2f", False, None),
    0x15: ("vfncvt.rod.f.f.w", "n", "f2f", False, fp.ROD),
    0x16: ("vfncvt.rtz.xu.f.w", "n", "f2i", False, sf.RTZ),
    0x17: ("vfncvt.rtz.x.f.w", "n", "f2i", True, sf.RTZ),
    0x1D: ("vfncvtbf16.f.f.w", "n", "bf", False, None),
}


def _cvt_widths(vs, shape, kind):
    """``(source width, destination width, float width)`` or None if illegal at this SEW."""
    sew = vs.sew
    if shape == "s":
        return (sew, sew, sew) if fp_ok(vs) else None
    if shape == "w":
        src, dst = sew, 2 * sew
        if dst > vs.elen:
            return None
        if kind == "bf":
            return (16, 32, 32) if vs.zvfbfmin and sew == 16 else None
        if kind == "i2f":  # int SEW -> float 2*SEW
            return (src, dst, dst) if fp_ok(vs, dst) and dst >= 16 else None
        return (
            (src, dst, src) if fp_ok(vs, src) and (kind != "f2f" or src >= 16) else None
        )
    # narrowing: source 2*SEW, destination SEW
    src, dst = 2 * sew, sew
    if src > vs.elen:
        return None
    if kind == "bf":
        return (32, 16, 32) if vs.zvfbfmin and sew == 16 else None
    if kind == "f2i":  # float 2*SEW -> int SEW
        return (src, dst, src) if fp_ok(vs, src) else None
    return (src, dst, dst) if fp_ok(vs, dst) else None


def convert(vs, rf, mem, code, vd, vs2, vm):
    name, shape, kind, signed, override = CONVERSIONS[code]
    if not alu_ok(vs) or (not vm and vd == 0):
        return False
    rm = frm(rf)
    if rm is None:
        return False
    widths = _cvt_widths(vs, shape, kind)
    if widths is None:
        return False
    src_w, dst_w, _ = widths
    lg = vs.lmul_log2
    if shape == "s" and not _sss(vs, vd, vs2, "x", 0):
        return False
    if shape == "w" and not widen_check(vs, vd, vs2, "x", 0, vm):
        return False
    if shape == "n" and not narrow_check(vs, vd, vs2, "x", 0, vm):
        return False
    if override is not None:
        rm = override
    vl = vs.vl
    if vl == 0:
        return True
    src_lg = lg + 1 if shape == "n" else lg
    dst_lg = lg + 1 if shape == "w" else lg
    a = vs.group(vs2, nregs(src_lg), UINT[src_w])[:vl]
    dst = vs.group(vd, nregs(dst_lg), UINT[dst_w])
    m = body_mask(vs, vm, vl)

    def compute(sel):
        x = a[sel]
        if kind == "f2i":
            bits, f = fp.to_int(x, fp.FMT[src_w], rm, dst_w, signed)
        elif kind == "i2f":
            bits, f = fp.from_int(x, signed, src_w, fp.FMT[dst_w], rm)
        elif kind == "bf":
            if shape == "w":
                bits, f = fp.convert(x, sf.BF16, sf.F32, rm)
            else:
                bits, f = fp.convert(x, sf.F32, sf.BF16, rm)
        else:
            bits, f = fp.convert(x, fp.FMT[src_w], fp.FMT[dst_w], rm)
        return bits.astype(UINT[dst_w]), f

    flags(rf, _apply(vs, dst, vl, m, compute))
    return True


# -- widening arithmetic ------------------------------------------------------------------

#: name -> (op, vs2 already wide)
_WIDEN = {
    "wadd": ("add", False),
    "wsub": ("sub", False),
    "wadd.w": ("add", True),
    "wsub.w": ("sub", True),
    "wmul": ("mul", False),
    "wmacc": ("macc", False),
    "wnmacc": ("nmacc", False),
    "wmsac": ("msac", False),
    "wnmsac": ("nmsac", False),
    "wmaccbf16": ("macc", False),
}


def _widen_bits(x, sew, bf16=False):
    """SEW float bits -> exact 2*SEW float bits, with the conversion's flags."""
    if bf16:
        return fp.convert(x, sf.BF16, sf.F32, sf.RNE)
    return fp.convert(x, fp.FMT[sew], fp.FMT[2 * sew], sf.RNE)


def widening(vs, rf, mem, name, vd, vs2, kind, src, vm):
    op, wide = _WIDEN[name]
    bf16 = name == "wmaccbf16"
    if not alu_ok(vs):
        return False
    sew = vs.sew
    if bf16:
        if not (vs.zvfbfwma and sew == 16):
            return False
    elif not fp_ok(vs) or not fp_ok(vs, 2 * sew):
        return False
    rm = frm(rf)
    if rm is None or not widen_check(vs, vd, vs2, kind, src, vm, wide):
        return False
    vl, lg = vs.vl, vs.lmul_log2
    if vl == 0:
        return True
    w = 2 * sew
    if wide:
        a = vs.group(vs2, nregs(lg + 1), UINT[w])[:vl]
    else:
        a = vs.group(vs2, nregs(lg), UINT[sew])[:vl]
    b = _src(vs, rf, kind, src, sew, nregs(lg), vl, bf16)
    dst = vs.group(vd, nregs(lg + 1), UINT[w])
    d = dst[:vl].copy()
    m = body_mask(vs, vm, vl)

    def compute(sel):
        f = 0
        if wide:
            x = a[sel]
        else:
            x, f = _widen_bits(a[sel], sew, bf16)
            x = x.astype(UINT[w])
        y, fy = _widen_bits(b[sel], sew, bf16)
        y = y.astype(UINT[w])
        f |= fy
        if op in ("add", "sub", "mul"):
            r, fr = fp.binary(op, x, y, w, rm)
        else:
            neg_p, neg_a = _FMA[op][1:]
            r, fr = fp.fma(y, x, d[sel], w, rm, neg_p, neg_a)
        return r, f | fr

    flags(rf, _apply(vs, dst, vl, m, compute))
    return True


# -- reductions --------------------------------------------------------------------------


def reduction(vs, rf, mem, name, vd, vs2, vs1, vm):
    """vfred{u,o}sum / vfredmax / vfredmin and vfwred{u,o}sum.

    Both sums are **sequential** in element order (Spike's order, which is a
    legal order for the unordered form too); NaN, flag and signed-zero results
    are therefore the ordered sum's.
    """
    wide = name.startswith("w")
    if not alu_ok(vs) or not fp_ok(vs):
        return False
    sew = vs.sew
    if wide and (2 * sew > vs.elen or not fp_ok(vs, 2 * sew)):
        return False
    rm = frm(rf)
    if rm is None or not aligned(vs2, vs.lmul_log2):
        return False
    vl = vs.vl
    if vl == 0:
        return True
    a = vs.group(vs2, nregs(vs.lmul_log2), UINT[sew])[:vl]
    if not vm:
        a = a[vs.mask(0, vl)]
    acc_w = 2 * sew if wide else sew
    start = int(vs.group(vs1, 1, UINT[acc_w])[0])
    f = 0
    if wide:
        a, f = _widen_bits(a, sew)
        a = a.astype(UINT[acc_w])
    if name in ("max", "min"):
        res, f2 = fp.reduce_minmax(
            np.uint64(start).astype(UINT[sew]), a, sew, name == "max"
        )
    else:
        if name in ("usum", "wusum") and len(a) == 0:
            # Spike: an unordered sum over no active elements still canonicalises a NaN start.
            res, f2 = start, 0
            fmt = fp.FMT[acc_w]
            if fp.isnan(np.array([start], dtype=UINT[acc_w]), fmt)[0]:
                f2 = (
                    sf.NV
                    if fp.issnan(np.array([start], dtype=UINT[acc_w]), fmt)[0]
                    else 0
                )
                res = fmt.canonical_nan
        else:
            res, f2 = fp.ordered_sum(start, a, acc_w, rm)
    vs.group(vd, 1, UINT[acc_w])[0] = res
    flags(rf, f | f2)
    vs.elements += len(a)
    return True
