"""Mask instructions, element moves and permutes: vm*, vcpop/vfirst, vmsbf/sif/of, viota, vid,
slides, gathers, vcompress, scalar moves and whole-register moves."""

import numpy as np

from framework.pe.rv.isa.fd_isa import write_f
from framework.pe.rv.isa.i64_isa import write_reg
from framework.pe.rv.isa.v.common import (
    M64,
    aligned,
    alu_ok,
    body_mask,
    count,
    fscalar,
    nregs,
    overlapped,
    sext_to_xlen,
    size,
    write_back,
    xscalar,
)
from framework.pe.rv.isa.v.state import UINT

_MASK_LOGICAL = {
    "andn": lambda a, b: a & ~b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "orn": lambda a, b: a | ~b,
    "nand": lambda a, b: ~(a & b),
    "nor": lambda a, b: ~(a | b),
    "xnor": lambda a, b: ~(a ^ b),
}


def mask_logical(vs, rf, mem, name, vd, vs2, vs1):
    """vmand.mm and friends: vd.mask[i] = op(vs2.mask[i], vs1.mask[i]) for i < vl."""
    if not alu_ok(vs):
        return False
    vl = vs.vl
    if vl:
        res = _MASK_LOGICAL[name](vs.mask(vs2, vl), vs.mask(vs1, vl))
        cur = vs.mask(vd)
        cur[:vl] = res
        vs.vrf[vd] = np.packbits(cur, bitorder="little")
        vs.elements += vl
    return True


def cpop_first(vs, rf, mem, name, rd, vs2, vm):
    """vcpop.m / vfirst.m -> x[rd]."""
    if vs.vill or vs.vstart != 0:
        return False
    vl = vs.vl
    bits = vs.mask(vs2, vl)
    if not vm:
        bits = bits & vs.mask(0, vl)
    if name == "cpop":
        value = int(np.count_nonzero(bits))
    else:
        hits = np.flatnonzero(bits)
        value = int(hits[0]) if len(hits) else M64
    if rd:
        write_reg(rf, rd, value)
    vs.elements += vl
    return True


def set_first(vs, rf, mem, name, vd, vs2, vm):
    """vmsbf / vmsif / vmsof: set-before-, set-including-, set-only-first."""
    if vs.vill or vs.vstart != 0 or vd == vs2 or (not vm and vd == 0):
        return False
    vl = vs.vl
    if vl:
        src = vs.mask(vs2, vl)
        m = None if vm else vs.mask(0, vl)
        active = src if m is None else src & m
        hits = np.flatnonzero(active)
        first = int(hits[0]) if len(hits) else vl
        idx = np.arange(vl)
        if name == "sbf":
            res = idx < first
        elif name == "sif":
            res = idx <= first
        else:
            res = idx == first
        cur = vs.mask(vd)
        if m is None:
            cur[:vl] = res
        else:
            np.copyto(cur[:vl], res, where=m)
        vs.vrf[vd] = np.packbits(cur, bitorder="little")
        vs.elements += vl
    return True


def iota(vs, rf, mem, vd, vs2, vm):
    """viota.m: vd[i] = number of set (active) mask bits of vs2 below i."""
    if vs.vill or vs.vstart != 0 or (not vm and vd == 0):
        return False
    lg = vs.lmul_log2
    if not aligned(vd, lg) or overlapped(vd, size(lg), vs2, 1):
        return False
    vl, sew = vs.vl, vs.sew
    if vl:
        m = body_mask(vs, vm, vl)
        src = vs.mask(vs2, vl)
        if m is not None:
            src = src & m
        csum = np.cumsum(src, dtype=np.uint64) - src
        write_back(vs.group(vd, nregs(lg), UINT[sew]), vl, csum.astype(UINT[sew]), m)
        count(vs, vl, m)
    return True


def vid(vs, rf, mem, vd, vm):
    if not alu_ok(vs) or (not vm and vd == 0) or not aligned(vd, vs.lmul_log2):
        return False
    vl, sew = vs.vl, vs.sew
    if vl:
        m = body_mask(vs, vm, vl)
        write_back(
            vs.group(vd, nregs(vs.lmul_log2), UINT[sew]),
            vl,
            np.arange(vl, dtype=UINT[sew]),
            m,
        )
        count(vs, vl, m)
    return True


# -- scalar moves ----------------------------------------------------------------------


def mv_x_s(vs, rf, mem, rd, vs2):
    """vmv.x.s: x[rd] = sext(vs2[0]); ignores vl."""
    if not alu_ok(vs):
        return False
    sew = vs.sew
    if rd:
        write_reg(rf, rd, sext_to_xlen(int(vs.group(vs2, 1, UINT[sew])[0]), sew))
    return True


def mv_s_x(vs, rf, mem, vd, rs1):
    """vmv.s.x: vd[0] = x[rs1] when vl > 0 (vstart < vl); the rest undisturbed."""
    if not alu_ok(vs):
        return False
    if vs.vstart < vs.vl:
        sew = vs.sew
        vs.group(vd, 1, UINT[sew])[0] = xscalar(rf, rs1, sew)
        vs.elements += 1
    return True


def fmv_f_s(vs, rf, mem, rd, vs2, fp_ok):
    """vfmv.f.s: f[rd] = vs2[0], NaN-boxed to FLEN."""
    if not alu_ok(vs) or not fp_ok(vs):
        return False
    sew = vs.sew
    raw = int(vs.group(vs2, 1, UINT[sew])[0])
    write_f(rf, rd, raw if sew == 64 else (M64 ^ ((1 << sew) - 1)) | raw)
    return True


def fmv_s_f(vs, rf, mem, vd, rs1, fp_ok):
    if not alu_ok(vs) or not fp_ok(vs):
        return False
    if vs.vstart < vs.vl:
        sew = vs.sew
        vs.group(vd, 1, UINT[sew])[0] = fscalar(rf, rs1, sew)
        vs.elements += 1
    return True


# -- slides ------------------------------------------------------------------------------


def _slide_check(vs, vd, vs2, vm, overlap_forbidden):
    lg = vs.lmul_log2
    if (
        not alu_ok(vs)
        or not aligned(vs2, lg)
        or not aligned(vd, lg)
        or (not vm and vd == 0)
    ):
        return False
    return not (overlap_forbidden and vd == vs2)


def slide(vs, rf, mem, name, vd, vs2, kind, src, vm, fp_ok=None):
    """vslideup / vslidedown (.vx/.vi) and v[f]slide1up / v[f]slide1down."""
    up = name in ("up", "1up")
    if not _slide_check(vs, vd, vs2, vm, up):
        return False
    if kind == "f" and not fp_ok(vs):
        return False
    sew, vl, vlmax = vs.sew, vs.vl, vs.vlmax
    if vl == 0:
        return True
    u = UINT[sew]
    n = nregs(vs.lmul_log2)
    srcv = vs.group(vs2, n, u)
    dst = vs.group(vd, n, u)
    m = body_mask(vs, vm, vl)
    if name in ("up", "down"):
        off = rf[src].read_uint() if kind == "x" else src
        if name == "up":
            if off >= vl:
                return True
            lo = off
            res = srcv[: vl - off].copy()
            if m is None:
                dst[lo:vl] = res
            else:
                np.copyto(dst[lo:vl], res, where=m[lo:vl])
            count(vs, vl - lo, None if m is None else m[lo:])
            return True
        idx = np.arange(vl, dtype=np.uint64) + np.uint64(off & M64)
        valid = (idx < vlmax) & (idx >= np.uint64(off & M64))  # no wrap-around
        res = np.where(valid, srcv[np.where(valid, idx, 0).astype(np.int64)], u.type(0))
        write_back(dst, vl, res, m)
        count(vs, vl, m)
        return True
    scalar = xscalar(rf, src, sew) if kind == "x" else fscalar(rf, src, sew)
    res = np.empty(vl, dtype=u)
    if name == "1up":
        res[0] = scalar
        res[1:] = srcv[: vl - 1]
    else:
        res[: vl - 1] = srcv[1:vl]
        res[vl - 1] = scalar
    write_back(dst, vl, res, m)
    count(vs, vl, m)
    return True


# -- gathers and compress -------------------------------------------------------------------


def gather(vs, rf, mem, vd, vs2, kind, src, vm, ei16=False):
    """vrgather.vv / .vx / .vi and vrgatherei16.vv: vd[i] = idx >= VLMAX ? 0 : vs2[idx]."""
    if not alu_ok(vs) or (not vm and vd == 0):
        return False
    sew, vl, lg, vlmax = vs.sew, vs.vl, vs.lmul_log2, vs.vlmax
    if (
        not aligned(vd, lg)
        or not aligned(vs2, lg)
        or overlapped(vd, size(lg), vs2, size(lg))
    ):
        return False
    idx_lg = lg
    if kind == "v":
        if ei16:
            idx_lg = lg + {8: 1, 16: 0, 32: -1, 64: -2}[sew]
            if not -3 <= idx_lg <= 3:
                return False
        if not aligned(src, idx_lg) or overlapped(vd, size(lg), src, size(idx_lg)):
            return False
    if vl == 0:
        return True
    u = UINT[sew]
    table = vs.group(vs2, nregs(lg), u)
    if kind == "v":
        idx = vs.group(src, nregs(idx_lg), UINT[16] if ei16 else u)[:vl].astype(
            np.uint64
        )
    elif kind == "x":
        idx = np.full(vl, rf[src].read_uint(), dtype=np.uint64)
    else:
        idx = np.full(vl, src, dtype=np.uint64)
    valid = idx < np.uint64(vlmax)
    res = np.where(valid, table[np.where(valid, idx, 0).astype(np.int64)], u.type(0))
    m = body_mask(vs, vm, vl)
    write_back(vs.group(vd, nregs(lg), u), vl, res, m)
    count(vs, vl, m)
    return True


def compress(vs, rf, mem, vd, vs2, vs1):
    """vcompress.vm: pack the elements of vs2 whose vs1 mask bit is set into vd[0..]."""
    if not alu_ok(vs):
        return False
    lg = vs.lmul_log2
    if not aligned(vd, lg) or not aligned(vs2, lg) or vd == vs2:
        return False
    if overlapped(vd, size(lg), vs1, 1) or overlapped(vd, size(lg), vs2, size(lg)):
        return False
    vl = vs.vl
    if vl:
        u = UINT[vs.sew]
        n = nregs(lg)
        picked = vs.group(vs2, n, u)[:vl][vs.mask(vs1, vl)]
        vs.group(vd, n, u)[: len(picked)] = picked
        vs.elements += vl
    return True


def move_whole(vs, rf, mem, vd, vs2, nr):
    """vmv<nr>r.v: copy nr whole registers (vtype and vl are ignored)."""
    if nr not in (1, 2, 4, 8) or vd % nr or vs2 % nr:
        return False
    if vd != vs2:
        vs.vrf[vd : vd + nr] = vs.vrf[vs2 : vs2 + nr].copy()
    return True
