"""Ztt (RISC-V AME draft v0.6) as an RV64 ISA executor: custom-1 words, the draft's state and CSRs.

Definitions (parameters, descriptors, encodings) are :mod:`framework.matrix.ztt`'s;
this module is the execution side. :class:`RV64_ZTT_ISA` claims every custom-1
word on a hart whose ``register_file.matrix`` is a :class:`ZttHart`; a custom-1
word that decodes to nothing is an illegal instruction.

**State.** :class:`ZttBackend` holds the architectural state: ``AME_NUM_M_REGS``
M registers of ``AME_NELEM * AME_UNIT_DATATYPE_SIZE`` bits with their ``Md``
descriptors, ``AME_NUM_ACC_REGS`` accumulators (each one square of the widest
supported type) with ``Ad``, and the context CSRs (``amestype``,
``amefflags``, ``amexsat``, ``amestatus``). A square's elements are stored
row-major and contiguous across its register group, so ``mss.1r`` of one group
member is that member's bytes and the layout ``mls.rm`` defines is the storage
layout. A backend is dedicated to one hart, or shared by all (``backend:
shared``); each hart sees it through a :class:`ZttHart` (its ``MS`` field, its
ownership, its counters).

**Checks, in the draft's order** (``src/instruction_semantics.adoc``): ``MS`` Off,
or (for all but ``ame.acquire``/``ame.release``) the backend not owned, is an
illegal instruction; then every encoded M / Acc index (illegal past the
implemented registers); then the datatype tuple (unsupported: ``amestatus.UN``
is set and nothing else changes); then the register-group alignment and span
(illegal); then the base address's alignment to the element size (address
misaligned) and every byte of the transfer (access fault, ``mtval`` = the
faulting segment's address); only then are registers or memory written. A
store's segments are all checked before the first byte is written, so a fault
writes nothing. Traps are delivered by an armed Naiad core
(:mod:`framework.pe.rv.trap` reads ``rv_cause`` / ``rv_tval`` from
:class:`ZttTrap`); on a core without ``mtvec`` they end the run, as other
illegal instructions do.

**Implemented tuples.** Every one of the draft's 138 instructions decodes. These
execute when their datatypes are supported (the design's ``datatypes`` and
``matmul``):

* ``ame.acquire``, ``ame.release``; ``msettyp``, ``mgettyp``, ``asettyp``, ``agettyp``
* ``mls`` / ``mss`` ``.rm .cm .st .tst .1r``; ``mzero.2d.m``, ``mzero.2d.acc``
* ``mmov.m.a``, ``mmov.a.m``, ``mmov.m.m``, ``mbcast.m.x``, ``mmove{8,16,32,64}.{m.x,x.m}``
* ``mmulacc.2d`` and its negated and transposed forms (six)
* ``mconv.ew``; ``madd``, ``msub``, ``mmul``, ``mmax``, ``mmin`` (``.ew`` and ``.ew.x``, one
  datatype for every tensor operand); ``mrowshift.ew.x``, ``mcolshift.ew.x``
* ``mreduce{add,max,min}.{row,col}``, ``mprefix{add,max}.{row,col}``

Every other instruction passes the index checks and then sets ``amestatus.UN``,
which the draft allows for any tuple an implementation does not support.

**Numerics, as the draft asks an implementation to document them.**

* Matrix multiplies: each output element starts from the accumulator and adds
  (or subtracts) the N products ``q = 0 .. N-1`` in ascending order, each product
  rounded to the accumulator's format and then added and rounded (unfused);
  products of fp16/bf16/fp8 inputs are exact in fp32, so for them this equals a
  fused fold. Integer accumulators wrap (two's complement); saturating
  accumulators are unsupported tuples. Multiplies do not update ``amefflags``.
* Floating-point element results round to nearest even (the only rounding mode
  implemented: a descriptor with another is unsupported); NaN results are the
  format's canonical quiet NaN; OCP E4M3, which has no infinity, saturates.
  ``amefflags`` accrues NV, OF, UF (tininess after rounding) and NX.
* Integer element results are exact and then wrap, or clamp and set ``amexsat``
  for a saturating destination. A float-to-integer conversion rounds by the
  destination's rounding-mode field; NaN, infinity and out-of-range set NV and
  give zero (wrapping destination) or the clamped bound (saturating).
* Reductions and prefixes fold in ascending index order in the destination type,
  rounding (or saturating) after every step.

**Not modelled:** packed datatypes (narrower than the unit size), custom
datatypes, the remaining element-wise operations (they report UN), address
translation (Naiad is machine mode), and any timing (a Ztt design is untimed).
"""

from fractions import Fraction

import numpy as np

from framework.matrix import ztt
from framework.matrix.formats import get_format
from framework.matrix.mac import tile_mac
from framework.matrix.unit import IllegalMatrixInstruction, MatrixAccessFault
from framework.memory.memory import resolve_plain_ram_span
from framework.pe.rv.isa.i64_isa import write_reg
from framework.pe.rv.isa.rv_isa import RV_ISA
from framework.pe.rv.isa.zicsr_isa import CSRError

_M64 = (1 << 64) - 1
CAUSE_ILLEGAL = 2
CAUSE_LOAD_MISALIGNED, CAUSE_LOAD_ACCESS = 4, 5
CAUSE_STORE_MISALIGNED, CAUSE_STORE_ACCESS = 6, 7
CSR_MSTATUS = 0x300

#: The counters (``hart<N>.matrix.<key>``), in report order.
COUNTER_NAMES = (
    "instructions",
    "unsupported",
    "mma",
    "macs",
    "mem_to_m_bytes",
    "m_to_mem_bytes",
    "ewise_elems",
    "acquire_granted",
    "acquire_refused",
)
_M_FIELDS = ("md", "ms", "ms1", "ms2")
_ACC_FIELDS = ("acc", "ad")
_FF_NV, _FF_DZ, _FF_OF, _FF_UF, _FF_NX = 16, 8, 4, 2, 1
#: Canonical quiet NaN bit patterns.
_QNAN = {
    "fp16": 0x7E00,
    "bf16": 0x7FC0,
    "fp32": 0x7FC00000,
    "fp64": 0x7FF8000000000000,
    "fp8_e4m3": 0x7F,
    "fp8_e5m2": 0x7E,
}
_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


class ZttTrap(Exception):
    """A Ztt instruction's synchronous exception: ``rv_cause`` and ``rv_tval`` (None: the instruction bits)."""

    def __init__(self, message, cause, tval=None):
        super().__init__(message)
        self.rv_cause, self.rv_tval = cause, tval


class ZttIllegal(ZttTrap, IllegalMatrixInstruction):
    def __init__(self, message):
        super().__init__(message, CAUSE_ILLEGAL)


class ZttMemoryFault(ZttTrap, MatrixAccessFault):
    pass


class _Unsupported(Exception):
    """An unsupported datatype tuple: ``amestatus.UN``, no other change."""


def np_dtype(dt):
    if dt.is_float:
        return get_format(dt.name).dtype
    return np.dtype(f"<{'i' if dt.signed else 'u'}{dt.bytes}")


def _int_bounds(dt):
    if dt.signed:
        return -(1 << (dt.bits - 1)), (1 << (dt.bits - 1)) - 1
    return 0, (1 << dt.bits) - 1


def _to_int(values, dt):
    """Exact integers (an object or int64 array) -> ``dt``'s storage, and whether any saturated."""
    v = np.asarray(values, dtype=object)
    lo, hi = _int_bounds(dt)
    if dt.sat:
        clipped = np.minimum(np.maximum(v, lo), hi)
        sat = bool(np.any(clipped != v))
        v = clipped
    else:
        sat = False
        span = 1 << dt.bits
        v = (v - lo) % span + lo
    return np.array(v.tolist(), dtype=np_dtype(dt)).reshape(np.shape(values)), sat


def _ints(raw):
    """A stored integer array as exact Python integers (an object array)."""
    return np.asarray(raw).astype(object)


def _floats(raw):
    return np.asarray(raw).astype(np.float64)


def _is_snan(raw, dt):
    if dt.name == "fp8_e4m3":
        return np.zeros(np.shape(raw), bool)
    f = get_format(dt.name)
    bits = np.asarray(raw).view(_UINT[dt.bytes]).astype(np.uint64)
    man = f.mantissa_bits
    exp_all = ((bits >> np.uint64(man)) & np.uint64((1 << f.exponent_bits) - 1)) == (
        1 << f.exponent_bits
    ) - 1
    frac = bits & np.uint64((1 << man) - 1)
    quiet = (bits >> np.uint64(man - 1)) & np.uint64(1)
    return exp_all & (frac != 0) & (quiet == 0)


def _round_fp(wide, dt, *, exact=None, invalid=None):
    """``wide`` (fp64) rounded to ``dt`` (RNE) -> (storage array, amefflags bits).

    ``exact``: where ``wide`` is the exact result (default: everywhere, true for
    every fp32-or-narrower op here). ``invalid``: where the operation itself was
    invalid (NV). NaN results become the canonical quiet NaN.
    """
    f = get_format(dt.name)
    wide = np.asarray(wide, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if not f.has_inf:  # E4M3: saturate what IEEE would send to infinity
            src = np.where(np.isnan(wide), wide, np.clip(wide, f.min, f.max))
            out = f.round(src)
        else:
            out = f.round(wide)
        back = out.astype(np.float64)
    nan = np.isnan(wide)
    flags = 0
    if invalid is not None and np.any(invalid):
        flags |= _FF_NV
    finite = np.isfinite(wide)
    overflow = finite & ((~np.isfinite(back)) | (np.abs(wide) > f.max))
    if np.any(overflow):
        flags |= _FF_OF | _FF_NX
    inexact = finite & ~nan & (back != wide)
    if exact is not None:
        inexact |= finite & ~np.asarray(exact, bool)
    if np.any(inexact):
        flags |= _FF_NX
        tiny = 2.0**f.emin
        if np.any(inexact & (np.abs(back) < tiny)):
            flags |= _FF_UF
    if np.any(nan):
        bits = out.view(_UINT[dt.bytes]).copy()
        bits[nan] = _QNAN[dt.name]
        out = bits.view(out.dtype)
    return out, flags


def _float_to_int(x, dt):
    """fp64 values -> ``dt`` (an integer type) by its rounding-mode field -> (array, NV, saturated)."""
    x = np.asarray(x, np.float64)
    lo, hi = _int_bounds(dt)
    bad = ~np.isfinite(x)
    with np.errstate(invalid="ignore"):
        fl = np.floor(np.where(bad, 0.0, x))
        frac = np.where(bad, 0.0, x) - fl
        if dt.rnd == ztt.INT_RNU:
            r = np.floor(np.where(bad, 0.0, x) + 0.5)
        elif dt.rnd == ztt.INT_RNE:
            r = np.rint(np.where(bad, 0.0, x))
        elif dt.rnd == ztt.INT_RDN:
            r = fl
        else:  # ROD: truncate, then jam the LSB when anything was lost
            r = fl + ((frac != 0) & (np.mod(fl, 2) == 0))
    vals = np.array([int(v) for v in r.reshape(-1)], dtype=object).reshape(x.shape)
    out_of_range = (vals < lo) | (vals > hi)
    nv = bool(np.any(bad) or np.any(out_of_range))
    if dt.sat:
        pos_inf = np.isposinf(x) | (np.isnan(x) & False)
        vals = np.where(bad, np.where(pos_inf, hi, np.where(np.isnan(x), 0, lo)), vals)
        vals = np.minimum(np.maximum(vals, lo), hi)
        sat = bool(np.any(out_of_range) or np.any(np.isinf(x)))
    else:
        vals = np.where(bad | out_of_range, 0, vals)
        sat = False
    arr, _ = _to_int(vals, dt)
    return arr, nv, sat


def convert(raw, src, dst):
    """``raw`` elements of ``src`` converted to ``dst`` -> (array, amefflags bits, saturated)."""
    if not src.is_float and not dst.is_float:
        arr, sat = _to_int(_ints(raw), dst)
        return arr, 0, sat
    if not src.is_float:  # int -> float: exact in fp64 for <= 53-bit magnitudes
        vals = _ints(raw)
        wide = np.array([float(v) for v in vals.reshape(-1)], np.float64).reshape(
            vals.shape
        )
        exact = np.array([Fraction(float(v)) == v for v in vals.reshape(-1)]).reshape(
            vals.shape
        )
        arr, flags = _round_fp(wide, dst, exact=exact)
        return arr, flags, False
    wide = _floats(raw)
    snan = _is_snan(raw, src)
    if not dst.is_float:
        arr, nv, sat = _float_to_int(wide, dst)
        inexact = np.isfinite(wide) & (np.floor(wide) != wide)
        flags = (_FF_NV if nv or np.any(snan) else 0) | (
            _FF_NX if np.any(inexact) and not nv else 0
        )
        return arr, flags, sat
    arr, flags = _round_fp(wide, dst, invalid=snan)
    return arr, flags, False


def conversion_supported(src, dst):
    """Every int/float pairing, except a 64-bit integer into a float narrower than fp64."""
    if not src.is_float and dst.is_float:
        return src.bits <= 32 or dst.name == "fp64"
    return True


def _exact_fp64(op, a, b, result):
    """Where an fp64 ``op`` on ``a``, ``b`` gave the exact value (``result`` is numpy's rounded one)."""
    out = np.ones(result.shape, bool)
    fa, fb, fr = a.reshape(-1), b.reshape(-1), result.reshape(-1)
    flat = out.reshape(-1)
    for i in range(fr.size):
        x, y, r = float(fa[i]), float(fb[i]), float(fr[i])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(r)):
            continue
        X, Y = Fraction(x), Fraction(y)
        exact = X + Y if op == "add" else X - Y if op == "sub" else X * Y
        flat[i] = Fraction(r) == exact
    return out


def _ieee_max(a, b, neg_zero_wins=False):
    out = np.where(a > b, a, b) if not neg_zero_wins else np.where(a < b, a, b)
    zeros = (a == 0) & (b == 0)
    if neg_zero_wins:
        out = np.where(zeros & (np.signbit(a) | np.signbit(b)), -0.0, out)
    else:
        out = np.where(zeros & ~(np.signbit(a) & np.signbit(b)), 0.0, out)
    return np.where(np.isnan(a) | np.isnan(b), np.nan, out)


def _fmax(a, b):
    return _ieee_max(a, b)


def _fmin(a, b):
    return _ieee_max(a, b, neg_zero_wins=True)


class ZttBackend:
    """One AME backend's architectural state (registers, descriptors, context CSRs) and owner."""

    def __init__(self, spec):
        self.spec = spec
        self.n = spec.n
        self.nelem = spec.nelem
        self.unit = spec.unit_bits
        self.supported = frozenset(spec.datatypes)
        self.matmul = frozenset(spec.matmul)
        widest = max(ztt.classify(ztt.descriptor(n)).bits for n in spec.datatypes)
        self.acc_bytes = self.nelem * widest // 8
        self.m = np.zeros((spec.m_regs, spec.mreg_bytes), np.uint8)
        self.acc = np.zeros((spec.acc_regs, self.acc_bytes), np.uint8)
        self.md = [0] * spec.m_regs
        self.ad = [0] * spec.acc_regs
        self.owner = None
        self.reset()

    def reset(self):
        self.clear()
        self.owner = None

    def clear(self):
        """The initial state a new grant binds: zero registers and descriptors, reset context CSRs."""
        self.m.fill(0)
        self.acc.fill(0)
        self.md = [0] * self.spec.m_regs
        self.ad = [0] * self.spec.acc_regs
        self.amestype = self.amefflags = self.amexsat = self.amestatus = 0

    def dtype(self, desc):
        """The supported :class:`~framework.matrix.ztt.Dtype` ``desc`` names, or None."""
        dt = ztt.classify(desc)
        if dt is None or dt.name not in self.supported:
            return None
        if dt.is_float and dt.rnd != ztt.FP_RNE:
            return None
        return dt

    def group(self, dt):
        return max(1, dt.bits // self.unit)


class ZttHart:
    """One hart's view of a backend: its ``MS`` field, ownership, CSRs and counters."""

    mode = "ztt"

    def __init__(self, backend, hart_id, csrs):
        self.backend = backend
        self.spec = backend.spec
        self.hart_id = hart_id
        self.csrs = csrs
        self.ms_shift = self.spec.ms_shift
        self.stats = dict.fromkeys(COUNTER_NAMES, 0)

    def reset(self):
        self.backend.reset()
        self.stats = dict.fromkeys(COUNTER_NAMES, 0)

    def counters(self):
        return dict(self.stats)

    # -- mstatus.MS and ownership -------------------------------------------------------------

    @property
    def owned(self):
        return self.backend.owner == self.hart_id

    def ms(self):
        return (
            (self.csrs.read(CSR_MSTATUS) >> self.ms_shift) & 3
            if self.csrs is not None
            else ztt.MS_DIRTY
        )

    def set_ms(self, value):
        if self.csrs is None:
            return
        store = self.csrs.store
        status = store.get(CSR_MSTATUS, 0) & ~(3 << self.ms_shift)
        status |= value << self.ms_shift
        if value == ztt.MS_DIRTY:
            status |= ztt.MSTATUS_SD
        store[CSR_MSTATUS] = status

    def dirty(self):
        if self.ms() != ztt.MS_DIRTY:
            self.set_ms(ztt.MS_DIRTY)

    def check_mstatus_write(self, value):
        """A write that turns MS Off while this hart owns the backend is refused (illegal)."""
        if self.owned and ((value >> self.ms_shift) & 3) == ztt.MS_OFF:
            raise CSRError(
                "Ztt: mstatus.MS cannot be set Off while the hart owns the AME backend (ame.release first)"
            )

    # -- CSRs ------------------------------------------------------------------------------------

    def _csr_access(self, name, write):
        if self.ms() == ztt.MS_OFF:
            raise CSRError(f"Ztt CSR {name} accessed with mstatus.MS Off")
        if name in ztt.WRITABLE_CSRS and not self.owned:
            raise CSRError(f"Ztt CSR {name} accessed without owning the AME backend")
        if write and name not in ztt.WRITABLE_CSRS:
            raise CSRError(f"Ztt CSR {name} is read-only")

    def read_csr(self, name):
        self._csr_access(name, False)
        b = self.backend
        return {
            "amenlen": self.spec.n,
            "ameudsz": self.spec.unit_bits,
            "amestype": b.amestype,
            "ameown": int(self.owned),
            "amefflags": b.amefflags,
            "amexsat": b.amexsat,
            "amestatus": b.amestatus,
        }[name]

    def write_csr(self, name, value):
        self._csr_access(name, True)
        b = self.backend
        if name == "amestype":
            b.amestype = value & 0xFFFFFFFF  # WARL-free: any encoding is kept
        elif name == "amefflags":
            b.amefflags = value & 0x1F
        elif name == "amexsat":
            b.amexsat = value & 1
        else:  # amestatus: writing 0 clears a bit, writing 1 leaves it as it was
            b.amestatus &= value & 1
        self.dirty()

    # -- shared helpers ------------------------------------------------------------------------

    def illegal(self, instr, why):
        return ZttIllegal(f"illegal Ztt instruction {instr:#010x}: {why}")

    def _group(self, r, dt, insn, instr):
        g = self.backend.group(dt)
        if r % g or r + g > self.spec.m_regs:
            raise self.illegal(
                instr, f"{insn.name}: m{r} cannot start a {g}-register {dt.name} group"
            )
        return g

    def _m_dt(self, r):
        dt = self.backend.dtype(self.backend.md[r])
        if dt is None:
            raise _Unsupported()
        return dt

    def _acc_dt(self, a):
        dt = self.backend.dtype(self.backend.ad[a])
        if dt is None:
            raise _Unsupported()
        return dt

    def square(self, r, dt):
        """The square in the group at ``m{r}`` as an (N, N) writable view of ``dt``."""
        g = self.backend.group(dt)
        raw = self.backend.m[r : r + g].reshape(-1)[: self.backend.nelem * dt.bytes]
        return raw.view(np_dtype(dt)).reshape(self.backend.n, self.backend.n)

    def acc_square(self, a, dt):
        raw = self.backend.acc[a, : self.backend.nelem * dt.bytes]
        return raw.view(np_dtype(dt)).reshape(self.backend.n, self.backend.n)

    def _flags(self, fflags=0, sat=False):
        if fflags:
            self.backend.amefflags |= fflags
        if sat:
            self.backend.amexsat |= 1

    # -- execution ----------------------------------------------------------------------------

    def execute(self, insn, instr, register_file, memory):
        if self.ms() == ztt.MS_OFF:
            raise self.illegal(instr, f"{insn.name} with mstatus.MS Off")
        if not self.owned and insn.name not in ("ame.acquire", "ame.release"):
            raise self.illegal(
                instr, f"{insn.name} without owning the AME backend (ame.acquire)"
            )
        ops = {name: insn.field(instr, name) for name in insn.operands}
        for name, value in ops.items():
            if name in _M_FIELDS and value >= self.spec.m_regs:
                raise self.illegal(
                    instr, f"{insn.name}: m{value} (this unit has {self.spec.m_regs})"
                )
            if name in _ACC_FIELDS and value >= self.spec.acc_regs:
                raise self.illegal(
                    instr,
                    f"{insn.name}: acc{value} (this unit has {self.spec.acc_regs})",
                )
        if insn.name in _PAIRED and ops["ms1"] == ops["ms2"]:
            raise self.illegal(
                instr, f"{insn.name}: ms1 and ms2 must name different registers"
            )
        self.stats["instructions"] += 1
        handler = HANDLERS.get(insn.name)
        try:
            if handler is None:
                raise _Unsupported()
            handler(self, insn, ops, register_file, memory, instr)
        except _Unsupported:
            self.backend.amestatus |= 1
            self.stats["unsupported"] += 1
            self.dirty()


def _x(rf, i):
    return rf[i].read_uint()


# -- control ------------------------------------------------------------------------------------


def _acquire(h, insn, ops, rf, mem, instr):
    desc = _x(rf, ops["rs1"])
    wait, tclass = desc & 3, (desc >> 2) & 0xF
    b = h.backend
    if desc >> 6 or wait > 1 or (wait == 1 and tclass not in (0, 1, 2, 3, 0xF)):
        status = ztt.ACQ_BAD_DESC
    elif h.owned:
        status = ztt.ACQ_GRANTED
    elif b.owner is None:
        b.owner = h.hart_id
        b.clear()
        h.set_ms(ztt.MS_INITIAL)
        status = ztt.ACQ_GRANTED
    else:  # no simulated wait: a bounded wait on a busy backend times out at once
        status = ztt.ACQ_TIMEOUT if wait == 1 else ztt.ACQ_BUSY
    granted = status == ztt.ACQ_GRANTED
    h.stats["acquire_granted" if granted else "acquire_refused"] += 1
    if ops["rd"]:
        write_reg(rf, ops["rd"], int(granted) | (status << 1))


def _release(h, insn, ops, rf, mem, instr):
    if h.owned:
        h.backend.owner = None
        h.set_ms(ztt.MS_INITIAL)


def _msettyp(h, insn, ops, rf, mem, instr):
    desc = _x(rf, ops["rs1"]) & 0xFFFFFFFF
    dt = h.backend.dtype(desc)
    if dt is None:
        raise _Unsupported()
    md = ops["md"]
    g = h._group(md, dt, insn, instr)
    h.backend.md[md] = desc
    h.backend.m[md : md + g] = 0
    h.dirty()


def _mgettyp(h, insn, ops, rf, mem, instr):
    if ops["rd"]:
        write_reg(rf, ops["rd"], h.backend.md[ops["ms1"]])


def _asettyp(h, insn, ops, rf, mem, instr):
    desc = _x(rf, ops["rs1"]) & 0xFFFFFFFF
    if h.backend.dtype(desc) is None:
        raise _Unsupported()
    h.backend.ad[ops["ad"]] = desc
    h.backend.acc[ops["ad"]] = 0
    h.dirty()


def _agettyp(h, insn, ops, rf, mem, instr):
    if ops["rd"]:
        write_reg(rf, ops["rd"], h.backend.ad[ops["ad"]])


def _mzero_m(h, insn, ops, rf, mem, instr):
    md = ops["md"]
    dt = h._m_dt(md)
    g = h._group(md, dt, insn, instr)
    h.backend.m[md : md + g] = 0
    h.dirty()


def _mzero_acc(h, insn, ops, rf, mem, instr):
    h._acc_dt(ops["acc"])
    h.backend.acc[ops["acc"]] = 0
    h.dirty()


# -- memory -------------------------------------------------------------------------------------


def _check_mapped(memory, addr, size):
    """The first unmapped address of ``[addr, addr + size)``, or None."""
    end = addr + size
    if end > 1 << 64:
        return 1 << 64 if addr < 1 << 64 else addr
    top = getattr(memory, "memory_map", None)
    a = addr
    while a < end:
        span = resolve_plain_ram_span(memory, a)
        if span is not None:
            a = span[1] + 1
            continue
        if top is None:
            return None
        rng, target = top.locate(a)
        if rng is None or target is None:
            return a
        a = rng.high + 1  # MMIO: its own accesses decide
    return None


def _segments(h, form, base, stride, ebytes):
    """``[(address, bytes)]`` of a transfer, in ascending segment order."""
    n = h.backend.n
    if form in ("rm", "cm", "1r"):
        return [(base, h.backend.nelem * ebytes)]
    return [((base + i * stride) & _M64, n * ebytes) for i in range(n)]


def _mem_operands(h, insn, ops, rf, instr, store):
    form = insn.name.split(".")[1]
    r = ops["ms1" if store else "md"]
    if form == "1r":
        ebytes, g, dt = h.spec.unit_bits // 8, 1, None
    else:
        dt = h._m_dt(r)
        g = h._group(r, dt, insn, instr)
        ebytes = dt.bytes
    base = _x(rf, ops["rs1"])
    stride = _x(rf, ops["rs2"]) if "rs2" in ops else 0
    if base % ebytes:
        cause = CAUSE_STORE_MISALIGNED if store else CAUSE_LOAD_MISALIGNED
        raise ZttTrap(
            f"{insn.name}: base {base:#x} is not aligned to {ebytes} bytes", cause, base
        )
    return form, r, g, dt, ebytes, base, stride


def _register_image(h, r, g):
    return h.backend.m[r : r + g].reshape(-1)


def _mls(h, insn, ops, rf, mem, instr):
    form, r, g, dt, eb, base, stride = _mem_operands(h, insn, ops, rf, instr, False)
    n = h.backend.n
    chunks = []
    for addr, size in _segments(h, form, base, stride, eb):
        bad = _check_mapped(mem, addr, size)
        try:
            data = bytes(mem.read(addr, size)) if bad is None else b""
        except IndexError:
            data = b""
        if len(data) != size:
            raise ZttMemoryFault(
                f"{insn.name}: {size} bytes at {addr:#x} reach unmapped memory",
                CAUSE_LOAD_ACCESS,
                addr if bad is None else bad,
            )
        chunks.append(np.frombuffer(data, np.uint8))
    raw = np.concatenate(chunks).reshape(n, n, eb)  # [segment][element][byte]
    if form in ("cm", "tst"):  # segments are the square's columns
        raw = raw.transpose(1, 0, 2)
    image = _register_image(h, r, g)
    image[: n * n * eb] = raw.reshape(-1)
    h.stats["mem_to_m_bytes"] += n * n * eb
    h.dirty()


def _mss(h, insn, ops, rf, mem, instr):
    form, r, g, dt, eb, base, stride = _mem_operands(h, insn, ops, rf, instr, True)
    n = h.backend.n
    raw = _register_image(h, r, g)[: n * n * eb].reshape(n, n, eb)
    if form in ("cm", "tst"):
        raw = raw.transpose(1, 0, 2)
    segs = _segments(h, form, base, stride, eb)
    for addr, size in segs:  # every segment is checked before the first byte is written
        bad = _check_mapped(mem, addr, size)
        if bad is not None:
            raise ZttMemoryFault(
                f"{insn.name}: {size} bytes at {addr:#x} reach unmapped memory",
                CAUSE_STORE_ACCESS,
                bad,
            )
    flat = np.ascontiguousarray(raw).reshape(len(segs), -1)
    for (addr, size), data in zip(segs, flat):
        mem.write(addr, data[:size].tobytes())
    h.stats["m_to_mem_bytes"] += n * n * eb


# -- moves ----------------------------------------------------------------------------------------


def _mmov_m_a(h, insn, ops, rf, mem, instr):
    md, a = ops["md"], ops["acc"]
    dm, da = h._m_dt(md), h._acc_dt(a)
    if dm.desc != da.desc:
        raise _Unsupported()
    h._group(md, dm, insn, instr)
    h.square(md, dm)[...] = h.acc_square(a, da)
    h.dirty()


def _mmov_a_m(h, insn, ops, rf, mem, instr):
    ms, a = ops["ms"], ops["acc"]
    dm, da = h._m_dt(ms), h._acc_dt(a)
    if dm.desc != da.desc:
        raise _Unsupported()
    h._group(ms, dm, insn, instr)
    h.acc_square(a, da)[...] = h.square(ms, dm)
    h.dirty()


def _mmov_m_m(h, insn, ops, rf, mem, instr):
    md, ms = ops["md"], ops["ms"]
    dd, ds = h._m_dt(md), h._m_dt(ms)
    if dd.desc != ds.desc:
        raise _Unsupported()
    h._group(md, dd, insn, instr)
    h._group(ms, ds, insn, instr)
    src = h.square(ms, ds).copy()
    h.square(md, dd)[...] = src
    h.dirty()


def _scalar(bits, dt):
    """The low ``dt.bits`` of an X register as one ``dt`` element (a 0-d storage array)."""
    raw = np.array([bits & ((1 << dt.bits) - 1)], dtype=np.uint64).astype(
        _UINT[dt.bytes]
    )
    return raw.view(np_dtype(dt))


def _mbcast(h, insn, ops, rf, mem, instr):
    md = ops["md"]
    src = h.backend.dtype(_x(rf, ops["rs2"]) & 0xFFFFFFFF)
    if src is None:
        raise _Unsupported()
    dst = h._m_dt(md)
    if not conversion_supported(src, dst):
        raise _Unsupported()
    h._group(md, dst, insn, instr)
    value, fflags, sat = convert(_scalar(_x(rf, ops["rs1"]), src), src, dst)
    h.square(md, dst)[...] = value[0]
    h._flags(fflags, sat)
    h.dirty()


def _mmove_geometry(h, insn, instr, ebits):
    row_bytes = h.backend.n * h.spec.unit_bits // 8
    if row_bytes < ebits // 8:
        raise h.illegal(
            instr,
            f"{insn.name}: a row of {row_bytes} bytes is narrower than the element",
        )
    return h.backend.n * (row_bytes * 8 // ebits)


def _mmove_to_m(h, insn, ops, rf, mem, instr):
    ebits = int(insn.name[5:].split(".")[0])
    count = _mmove_geometry(h, insn, instr, ebits)
    pos = _x(rf, ops["rs1"]) & (count - 1)
    eb = ebits // 8
    value = _x(rf, ops["rs2"]) & ((1 << ebits) - 1)
    h.backend.m[ops["md"], pos * eb : (pos + 1) * eb] = np.frombuffer(
        value.to_bytes(eb, "little"), np.uint8
    )
    h.dirty()


def _mmove_to_x(h, insn, ops, rf, mem, instr):
    ebits = int(insn.name[5:].split(".")[0])
    count = _mmove_geometry(h, insn, instr, ebits)
    pos = _x(rf, ops["rs1"]) & (count - 1)
    eb = ebits // 8
    value = int.from_bytes(
        h.backend.m[ops["ms2"], pos * eb : (pos + 1) * eb].tobytes(), "little"
    )
    if ops["rd"]:
        write_reg(rf, ops["rd"], value)


# -- matrix multiply ------------------------------------------------------------------------------


def _matmul(h, insn, ops, rf, mem, instr):
    a_r, b_r, c_r = ops["ms1"], ops["ms2"], ops["acc"]
    da, db, dc = h._m_dt(a_r), h._m_dt(b_r), h._acc_dt(c_r)
    if (da.name, db.name, dc.name) not in h.backend.matmul or dc.sat:
        raise _Unsupported()
    h._group(a_r, da, insn, instr)
    h._group(b_r, db, insn, instr)
    name = insn.name.split(".")[0]
    neg = name.endswith("neg")
    A, B = h.square(a_r, da), h.square(b_r, db)
    if name.startswith("mmulat"):
        A = A.T
    elif name.startswith("mmulbt"):
        B = B.T
    C = h.acc_square(c_r, dc)
    if dc.is_float:
        Bf = -B.astype(np.float64) if neg else B.astype(np.float64)
        Af = A.astype(np.float64)
        fmt_in = da.name
        out = tile_mac(C.astype(np.float64), Af, Bf, fmt_in, dc.name, k_chunk=1)
        out = np.asarray(out)
        nan = np.isnan(out.astype(np.float64))
        if np.any(nan):
            bits = out.view(_UINT[dc.bytes]).copy()
            bits[nan] = _QNAN[dc.name]
            out = bits.view(out.dtype)
        result = out
    else:
        prod = _ints(A).dot(_ints(B))
        acc = _ints(C) - prod if neg else _ints(C) + prod
        result, _ = _to_int(acc, dc)
    C[...] = result
    n = h.backend.n
    h.stats["mma"] += 1
    h.stats["macs"] += n * n * n
    h.dirty()


# -- element-wise ---------------------------------------------------------------------------------

_EW_OPS = {"madd", "msub", "mmul", "mmax", "mmin"}


def _ewise_result(op, a, b, dt):
    """``op`` on two elements arrays of ``dt`` (``a`` is A or c, ``b`` is B) -> (storage, fflags, sat)."""
    if not dt.is_float:
        x, y = _ints(a), _ints(b)
        if op == "madd":
            v = x + y
        elif op == "msub":
            v = y - x
        elif op == "mmul":
            v = x * y
        elif op == "mmax":
            v = np.where(x > y, x, y)
        else:
            v = np.where(x < y, x, y)
        out, sat = _to_int(v, dt)
        return out, 0, sat
    x, y = _floats(a), _floats(b)
    snan = _is_snan(a, dt) | _is_snan(b, dt)
    with np.errstate(over="ignore", invalid="ignore"):
        if op == "madd":
            v = x + y
        elif op == "msub":
            v = y - x
        elif op == "mmul":
            v = x * y
        elif op == "mmax":
            v = _fmax(x, y)
        else:
            v = _fmin(x, y)
    invalid = snan | (np.isnan(v) & ~np.isnan(x) & ~np.isnan(y))
    exact = None
    if dt.name == "fp64" and op in ("madd", "msub", "mmul"):
        exact = _exact_fp64(
            {"madd": "add", "msub": "sub", "mmul": "mul"}[op],
            y if op == "msub" else x,
            x if op == "msub" else y,
            v,
        )
    out, fflags = _round_fp(v, dt, exact=exact, invalid=invalid)
    return out, fflags, False


def _ewise(h, insn, ops, rf, mem, instr):
    op = insn.name.split(".")[0]
    scalar = insn.name.endswith(".x")
    md, ms2 = ops["md"], ops["ms2"]
    dd, d2 = h._m_dt(md), h._m_dt(ms2)
    if scalar:
        st = h.backend.dtype(h.backend.amestype)
        if st is None or dd.desc != d2.desc or not conversion_supported(st, d2):
            raise _Unsupported()
    else:
        d1 = h._m_dt(ops["ms1"])
        if not dd.desc == d1.desc == d2.desc:
            raise _Unsupported()
        h._group(ops["ms1"], d1, insn, instr)
    h._group(md, dd, insn, instr)
    h._group(ms2, d2, insn, instr)
    B = h.square(ms2, d2).copy()
    fflags = 0
    sat = False
    if scalar:
        c, fflags, sat = convert(_scalar(_x(rf, ops["rs1"]), st), st, d2)
        A = np.broadcast_to(c[0], B.shape)
    else:
        A = h.square(ops["ms1"], d1).copy()
    out, f2, s2 = _ewise_result(op, A, B, dd)
    h.square(md, dd)[...] = out
    h._flags(fflags | f2, sat or s2)
    h.stats["ewise_elems"] += B.size
    h.dirty()


def _shift(h, insn, ops, rf, mem, instr):
    md, ms2 = ops["md"], ops["ms2"]
    dd, d2 = h._m_dt(md), h._m_dt(ms2)
    if dd.desc != d2.desc:
        raise _Unsupported()
    h._group(md, dd, insn, instr)
    h._group(ms2, d2, insn, instr)
    c = _x(rf, ops["rs1"]) & 0xFFFFFFFF
    c = c - (1 << 32) if c >> 31 else c
    A = h.square(ms2, d2).copy()
    n = h.backend.n
    out = np.zeros_like(A)
    if abs(c) < n:
        if insn.name.startswith("mrow"):  # D[i, j] = A[i + c, j]
            src = slice(max(c, 0), n + min(c, 0))
            dst = slice(max(-c, 0), n - max(c, 0))
            out[dst, :] = A[src, :]
        else:  # D[i, j] = A[i, j + c]
            src = slice(max(c, 0), n + min(c, 0))
            dst = slice(max(-c, 0), n - max(c, 0))
            out[:, dst] = A[:, src]
    h.square(md, dd)[...] = out
    h.dirty()


# -- reductions and prefixes ----------------------------------------------------------------------


def _fold_step(kind, prev, x, dt):
    """One fold step in ``dt``: ``prev`` and ``x`` storage arrays -> (storage, fflags, sat)."""
    if kind == "add":
        return _ewise_result("madd", prev, x, dt)
    return _ewise_result("mmax" if kind == "max" else "mmin", prev, x, dt)


def _reduce(h, insn, ops, rf, mem, instr):
    base, axis_name = insn.name.split(".")
    prefix = base.startswith("mprefix")
    kind = base[len("mprefix") :] if prefix else base[len("mreduce") :]
    md, ms1 = ops["md"], ops["ms1"]
    dd, ds = h._m_dt(md), h._m_dt(ms1)
    if not conversion_supported(ds, dd):
        raise _Unsupported()
    h._group(md, dd, insn, instr)
    h._group(ms1, ds, insn, instr)
    A, fflags, sat = convert(h.square(ms1, ds).copy(), ds, dd)
    if axis_name == "col":  # fold down each column: work on the transpose
        A = A.T
    n = h.backend.n
    out = np.empty_like(A)
    acc = A[:, 0].copy()
    out[:, 0] = acc
    for k in range(1, n):
        acc, f, s = _fold_step(kind, acc, A[:, k].copy(), dd)
        fflags |= f
        sat = sat or s
        out[:, k] = acc
    if not prefix:
        out = np.repeat(acc[:, None], n, axis=1)
    if axis_name == "col":
        out = out.T
    h.square(md, dd)[...] = out
    h._flags(fflags, sat)
    h.stats["ewise_elems"] += A.size
    h.dirty()


def _mconv(h, insn, ops, rf, mem, instr):
    md, ms1 = ops["md"], ops["ms1"]
    dd, ds = h._m_dt(md), h._m_dt(ms1)
    if not conversion_supported(ds, dd):
        raise _Unsupported()
    h._group(md, dd, insn, instr)
    h._group(ms1, ds, insn, instr)
    out, fflags, sat = convert(h.square(ms1, ds).copy(), ds, dd)
    h.square(md, dd)[...] = out
    h._flags(fflags, sat)
    h.stats["ewise_elems"] += out.size
    h.dirty()


#: Instructions whose two operands must be distinct registers (an encoded constraint: illegal).
_PAIRED = frozenset({"mcolunzip.ew", "mcolzip.ew", "mrowunzip.ew", "mrowzip.ew"})

HANDLERS = {
    "ame.acquire": _acquire,
    "ame.release": _release,
    "msettyp": _msettyp,
    "mgettyp": _mgettyp,
    "asettyp": _asettyp,
    "agettyp": _agettyp,
    "mzero.2d.m": _mzero_m,
    "mzero.2d.acc": _mzero_acc,
    "mmov.m.a": _mmov_m_a,
    "mmov.a.m": _mmov_a_m,
    "mmov.m.m": _mmov_m_m,
    "mbcast.m.x": _mbcast,
    "mconv.ew": _mconv,
    "mrowshift.ew.x": _shift,
    "mcolshift.ew.x": _shift,
}
for _form in ("rm", "cm", "st", "tst", "1r"):
    HANDLERS[f"mls.{_form}"] = _mls
    HANDLERS[f"mss.{_form}"] = _mss
for _e in (8, 16, 32, 64):
    HANDLERS[f"mmove{_e}.m.x"] = _mmove_to_m
    HANDLERS[f"mmove{_e}.x.m"] = _mmove_to_x
for _name in (
    "mmulacc",
    "mmulaccneg",
    "mmulatacc",
    "mmulataccneg",
    "mmulbtacc",
    "mmulbtaccneg",
):
    HANDLERS[f"{_name}.2d"] = _matmul
for _op in _EW_OPS:
    HANDLERS[f"{_op}.ew"] = _ewise
    HANDLERS[f"{_op}.ew.x"] = _ewise
for _k in ("add", "max", "min"):
    for _ax in ("row", "col"):
        HANDLERS[f"mreduce{_k}.{_ax}"] = _reduce
for _k in ("add", "max"):
    for _ax in ("row", "col"):
        HANDLERS[f"mprefix{_k}.{_ax}"] = _reduce
del _form, _e, _name, _op, _k, _ax

#: Every instruction this model executes for a supported tuple (the rest report UN).
IMPLEMENTED = frozenset(HANDLERS)
missing = sorted(IMPLEMENTED - set(ztt.TABLE))
if missing:
    raise ztt.ZttError(f"handlers for instructions the table does not have: {missing}")
del missing


def _disasm(insn, ops):
    return f"{insn.name} " + ", ".join(f"{k}={v}" for k, v in ops.items())


class RV64_ZTT_ISA(RV_ISA):
    """Ztt on the hart's :class:`ZttHart` (``register_file.matrix``): every custom-1 word."""

    OPCODES = frozenset({ztt.OPCODE})
    #: The counter hub's instruction class.
    INSTR_CLASS = "matrix"
    _decoded = {}

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        unit = getattr(register_file, "matrix", None)
        if (
            unit is None
            or getattr(unit, "mode", None) != "ztt"
            or instr & 0x7F != ztt.OPCODE
        ):
            return False
        insn = cls._decoded.get(instr, 0)
        if insn == 0:
            insn = cls._decoded[instr] = ztt.decode(instr)
        if insn is None:
            raise ZttIllegal(
                f"illegal Ztt instruction {instr:#010x}: no Ztt v0.6 instruction has this encoding"
            )
        unit.execute(insn, instr, register_file, memory_space)
        if snoop:
            ops = {name: insn.field(instr, name) for name in insn.operands}
            cls.print_snoop(snoop, _disasm(insn, ops))
        return True


def attach(core, hart):
    """Give ``core`` the Ztt ``hart`` view: ``register_file.matrix``, the ISA, the CSRs and the MS guard."""
    core.matrix = hart
    core.register_file.matrix = hart
    csrs = getattr(core, "csrs", None)
    hart.csrs = csrs
    if csrs is not None:
        for addr, name in ztt.CSRS.items():
            csrs.ext_csrs[addr] = lambda name=name: hart.read_csr(name)
            if name in ztt.WRITABLE_CSRS or addr >> 10 != 0x3:
                csrs.ext_csr_writers[addr] = lambda value, name=name: hart.write_csr(
                    name, value
                )
        csrs.write_guards[CSR_MSTATUS] = hart.check_mstatus_write
    core.add_isa(RV64_ZTT_ISA)
    return hart


def make_harts(spec, cores):
    """One :class:`ZttHart` per core, over one backend each or one shared backend."""
    shared = ZttBackend(spec) if spec.backend == "shared" else None
    return [
        ZttHart(
            shared if shared is not None else ZttBackend(spec),
            core.core_id,
            getattr(core, "csrs", None),
        )
        for core in cores
    ]


__all__ = [
    "COUNTER_NAMES",
    "HANDLERS",
    "IMPLEMENTED",
    "RV64_ZTT_ISA",
    "ZttBackend",
    "ZttHart",
    "ZttIllegal",
    "ZttMemoryFault",
    "ZttTrap",
    "attach",
    "convert",
    "make_harts",
]
