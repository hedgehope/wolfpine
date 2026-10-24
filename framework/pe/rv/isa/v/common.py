"""Helpers shared by the RVV instruction handlers: operand checks, the body mask, write-back.

Every handler is ``handler(vs, rf, mem, ...) -> bool``: ``vs`` the hart's
:class:`~framework.pe.rv.isa.v.state.VectorState`, ``rf`` its register file,
``mem`` its memory, the rest decoded instruction fields. Returning False means
the instruction is illegal in the current configuration (the core then reports
an unknown instruction, Wolfpine's stand-in for the trap); returning True
means it executed, and the dispatcher clears ``vstart``.

The legality rules are Spike's (``riscv/v_ext_macros.h``): register-group
alignment, the overlap rules for widening, narrowing and mask-producing
instructions, ``vd != v0`` for a masked instruction that writes SEW-wide
elements, and **``vstart == 0`` for every arithmetic instruction** (Spike's
default; only loads and stores honour a nonzero ``vstart``).
"""

import numpy as np

from framework.pe.rv.isa import fd_softfloat as sf
from framework.pe.rv.isa.fd_isa import raise_flags, read_f, resolve_rm
from framework.pe.rv.isa.v.state import SINT, UINT

M64 = (1 << 64) - 1


def nregs(log2):
    """Registers a group of EMUL = 2**log2 occupies (a fractional group is one register)."""
    return 1 << log2 if log2 > 0 else 1


def aligned(reg, log2):
    return log2 <= 0 or reg & ((1 << log2) - 1) == 0


def overlapped(a, asize, b, bsize):
    """Spike's ``is_overlapped`` (sizes in registers; 0 counts as 1)."""
    asize = asize or 1
    bsize = bsize or 1
    return max(a + asize, b + bsize) - min(a, b) < asize + bsize


def overlapped_widen(a, asize, b, bsize):
    """Spike's ``is_overlapped_widen``: the source may sit in the destination group's top half."""
    asize = asize or 1
    bsize = bsize or 1
    if (
        a < b
        and overlapped(a, asize, b, bsize)
        and not overlapped(a, asize, b + bsize, bsize)
    ):
        return False
    return max(a + asize, b + bsize) - min(a, b) < asize + bsize


def size(log2):
    """Spike's ``(int) vflmul`` scaled by 2**log2: 0 for fractional groups."""
    return 1 << log2 if log2 >= 0 else 0


def alu_ok(vs):
    """The common arithmetic precondition: a legal vtype and vstart == 0."""
    return not vs.vill and vs.vstart == 0


def body_mask(vs, vm, vl):
    """The active-element mask for elements ``[0, vl)``, or None when every element is active."""
    if vm:
        return None
    return vs.mask(0, vl)


def write_back(dst, vl, values, m):
    """Store ``values`` (elements ``0..vl-1``) into ``dst``; masked-off elements are undisturbed."""
    if m is None:
        dst[:vl] = values
    else:
        np.copyto(dst[:vl], values, where=m)


def count(vs, n, m):
    """Add the active elements among the first ``n`` to the unit's element counter."""
    vs.elements += n if m is None else int(np.count_nonzero(m[:n]))


def xscalar(rf, rs1, sew):
    """x[rs1] truncated to SEW, as an unsigned numpy scalar."""
    v = rf[rs1].read_uint()
    return UINT[sew].type(v & ((1 << sew) - 1))


def simm5(field, sew):
    """The 5-bit immediate sign-extended to SEW, as an unsigned numpy scalar."""
    v = field - 32 if field & 0x10 else field
    return UINT[sew].type(v & ((1 << sew) - 1))


def fscalar(rf, rs1, sew, bf16=False):
    """f[rs1] as a SEW-bit pattern: NaN-boxing checked below FLEN = 64.

    An unboxed value reads as the canonical NaN of the operand's format (bf16's
    for the Zvfbfwma ``.vf`` form, which reads its scalar as bf16).
    """
    raw = read_f(rf, rs1)
    if sew == 64:
        return np.uint64(raw)
    box = M64 ^ ((1 << sew) - 1)
    if raw & box != box:
        return UINT[sew].type(0x7FC0 if bf16 else {16: 0x7E00, 32: 0x7FC00000}[sew])
    return UINT[sew].type(raw & ((1 << sew) - 1))


def frm(rf):
    """The dynamic rounding mode, or None when frm holds a reserved value (an illegal instruction)."""
    return resolve_rm(rf, sf.DYN)


def flags(rf, f):
    if f:
        raise_flags(rf, f)


def signed(a):
    return a.view(SINT[a.dtype.itemsize * 8])


def sext_to_xlen(value, sew):
    value &= (1 << sew) - 1
    if value >> (sew - 1):
        value -= 1 << sew
    return value & M64
