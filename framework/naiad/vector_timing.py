"""The vector unit's cycle model: a design's ``timing.vector`` section and its issue logic.

Deliberately simple, and separate from the functional RVV code (which never
reads it). :class:`VectorTiming` is driven by ``NaiadCostState.can_issue`` for
every vector word, after the scalar scoreboard has cleared the instruction's
x/f sources, and answers the cycle its scalar result (if any) is ready, or -1
to stall.

* **One arithmetic datapath** of ``datapath_bits`` (the lanes: ``datapath_bits
  / SEW`` elements a cycle). An instruction occupies it for ``ceil(VL * EEW /
  datapath_bits)`` cycles, where EEW is the widest element the instruction
  touches (2*SEW for widening and narrowing forms), so LMUL enters through VL
  (and whole-register moves charge ``NR * VLEN / datapath_bits``). Its
  results are ready ``startup`` cycles after it leaves the datapath (``startup
  + occupancy`` after issue). Divides and square roots occupy it ``div_ii``
  times longer per element group; reductions add ``log2(lanes)`` cycles of
  tree latency. A second arithmetic instruction waits for the datapath
  (``vector_busy``).
* **No chaining** (``chaining: false``, the default): a consumer waits until
  its producer's *whole* result is ready (``vector_raw``), tracked per vector
  register (a group is ready when its last register is).
* **Chaining** (``chaining: true``): element-group chaining, Cray / RVV style,
  tracked per vector register. An op processes its ``n`` elements in ``beats``
  equal element groups over its occupancy (the datapath's, or the LSU transfer
  for a load or store), element ``x`` in beat ``x * beats // n``. A producer's
  element is readable ``startup`` (a load: ``mem_startup`` + region latency)
  after its beat leaves the unit, so its *first* group is readable at ``issue
  + beat + startup`` and its last at the unchained ready time; each register
  records when its first and its last element are readable. A consumer may
  start once every source register's first element is ready when the
  consumer's beat reaches it, and the register's last element likewise (both
  streams are linear, so the two ends suffice): ``start >= max(first[r] -
  offset(first element in r), ready[r] - offset(last element in r))``. Chaining
  only lifts the RAW wait: the datapath and the LSU are still one instruction
  at a time, a load still honours memory RAW and the region port, and a
  chained store is still complete (for :mod:`~framework.naiad.store_hazards`)
  when its transfer ends, after its last element arrived. Loads chain into
  arithmetic and arithmetic into stores the same way. Not chained: sources an
  op reads out of order (``vrgather``'s and a slide-down's ``vs2``) and a
  reduction's scalar ``vs1`` wait for the whole register group; reductions
  and ``vcompress`` results are readable only when the op completes; a
  register holding only tail elements (at or past ``vl``) is not waited for. With VLEN = ``datapath_bits`` and LMUL 1 an op is
  one beat, so chaining saves nothing: the gain per dependence is the
  producer's occupancy less one beat, and the ``startup`` stays exposed.
* **One vector load/store unit**, independent of the datapath. A unit-stride
  access moves ``bytes`` in ``ceil(bytes / min(mem_bytes_per_cycle, region
  bandwidth))`` cycles; strided, indexed and segment accesses take at least
  ``VL * NF / indexed_elements_per_cycle`` (one address per element). The
  access is charged against the scalar model's memory region for its base
  address (latency, port occupancy, counts), as a scalar access would be. A
  load's data is ready ``mem_startup + region latency + transfer`` cycles
  after issue; stores are posted.
* **Memory RAW.** A store registers the bytes it writes with the SoC's
  stores-in-flight table (:mod:`framework.naiad.store_hazards`), complete when
  the LSU frees (``issue + transfer``); a load overlapping one still in flight
  waits for it (``mem_raw``). A vector load after a vector store already waits
  for the LSU, so this bites the core's scalar loads and other harts' loads.
  Strided accesses are tracked byte-exactly, indexed ones by the bounding
  interval of their element addresses.
* ``vset{i}vl{i}`` cost one cycle and write rd like an ALU op. Scalar results
  (``vmv.x.s``, ``vcpop``, ``vfirst``, ``vfmv.f.s``) are ready when the
  instruction completes.

Element counts use the current ``vl`` (all elements, active or not: a masked-off
lane is still a lane cycle), so the model needs the hart's ``VectorState``.
"""

import math
from dataclasses import dataclass

from framework.naiad.design import DesignError
from framework.naiad.store_hazards import footprint
from framework.pe.rv.isa.v import fpu, integer, permute
from framework.pe.rv.isa.v import isa as visa
from framework.pe.rv.isa.v import mem as vmem
from framework.pe.rv.isa.v.common import nregs
from framework.pe.rv.isa.v.state import UINT

VECTOR_BUSY = "vector_busy"
VECTOR_RAW = "vector_raw"
MEM = "mem"
MEM_RAW = "mem_raw"


@dataclass(frozen=True)
class VectorTimingSpec:
    """``timing.vector``."""

    #: Arithmetic datapath width in bits: datapath_bits / SEW elements per cycle.
    datapath_bits: int = 256
    #: Cycles from leaving the datapath to the result being readable.
    startup: int = 4
    #: Occupancy multiplier per element group for vdiv/vrem/vfdiv/vfsqrt.
    div_ii: int = 8
    #: Cycles a load adds to its region's latency.
    mem_startup: int = 2
    #: Vector load/store unit width, bytes per cycle (also capped by the region's bandwidth).
    mem_bytes_per_cycle: float = 32
    #: Element addresses generated per cycle for strided / indexed / segment accesses.
    indexed_elements_per_cycle: float = 1
    #: Element-group chaining between dependent vector instructions (off: wait for the whole result).
    chaining: bool = False


def parse_vector(raw, design, parse_fields, number, flag):
    """``timing.vector`` (defaults when absent) for a design with a vector unit, else None."""
    if not design.cores.has_vector:
        if raw is not None:
            raise DesignError(
                "timing.vector: the design's cores have no vector unit to time"
            )
        return None
    rules = {
        "startup": lambda v, w: number(v, w, minimum=0),
        "mem_startup": lambda v, w: number(v, w, minimum=0),
        "mem_bytes_per_cycle": lambda v, w: number(v, w, minimum=1, allow_float=True),
        "indexed_elements_per_cycle": lambda v, w: number(
            v, w, minimum=1, allow_float=True
        ),
        "chaining": flag,
    }
    spec = parse_fields(raw, VectorTimingSpec, "timing.vector", rules)
    if spec.datapath_bits % 8 or spec.datapath_bits < 8:
        raise DesignError(
            f"timing.vector.datapath_bits must be a multiple of 8, got {spec.datapath_bits}"
        )
    return spec


# -- static decode: what each vector word reads and writes ------------------------------

#: Operand roles: the register count of each is resolved at issue from vtype.
_S, _W, _M, _N2 = (
    "s",
    "w",
    "m",
    "n2",
)  # SEW group, 2*SEW group, one mask register, (narrow source = w)

ARITH, DIV, RED, LOAD, STORE, CFG = range(6)
_F = 32  # f-register i is scalar scoreboard entry 32 + i


@dataclass(frozen=True)
class VOp:
    kind: int
    #: ((base, role), ...) vector sources; role in s / w / m, or ("n", count) / ("e", shift).
    vsrc: tuple
    #: (base, role) or None.
    vdst: tuple | None
    #: Scalar scoreboard entries read / written (-1: none).
    ssrc: tuple
    sdst: int
    #: True when any operand is 2*SEW wide.
    wide: bool = False
    masked: bool = False
    #: Memory: (mode, eew, nf) / whole-register count.
    mem: tuple | None = None
    #: Chaining only: vector source bases the op cannot chain on (it reads them out of order).
    whole_src: tuple = ()
    #: Chaining only: False when the result is only readable once the op completes.
    streams: bool = True


def _scalar_src(kind, src):
    if kind == "x":
        return (src,)
    if kind == "f":
        return (_F + src,)
    return ()


def _vsrc(kind, src, role=_S):
    return ((src, role),) if kind == "v" else ()


def classify(instr):
    """The :class:`VOp` for a vector word, or None if the functional decoder does not claim it."""
    decoded = visa.decode(instr)
    if decoded is None:
        return None
    handler, args, _ = decoded
    h = getattr(handler, "func", handler)
    if h is visa._vset:
        form, rd, rs1, rs2, _ = args
        ssrc = (rs1, rs2) if form == "vsetvl" else ((rs1,) if form == "vsetvli" else ())
        return VOp(CFG, (), None, ssrc, rd or -1)
    if h is vmem.access:
        is_load, mode, eew, nf, vd, rs1, rs2, vm = args[:8]
        vs = ((vd, ("d", eew, nf)),) if not is_load else ()
        if mode == "indexed":
            vs += ((rs2, ("i", eew)),)
        ssrc = (rs1, rs2) if mode == "strided" else (rs1,)
        return VOp(
            LOAD if is_load else STORE,
            vs + (((0, _M),) if not vm else ()),
            (vd, ("d", eew, nf)) if is_load else None,
            ssrc,
            -1,
            masked=not vm,
            mem=(mode, eew, nf),
        )
    if h is vmem.whole:
        is_load, nreg, eew, vd, rs1 = args
        role = ("n", nreg)
        return VOp(
            LOAD if is_load else STORE,
            () if is_load else ((vd, role),),
            (vd, role) if is_load else None,
            (rs1,),
            -1,
            mem=("whole", 8, nreg),
        )
    if h is vmem.mask_access:
        is_load, vd, rs1 = args
        return VOp(
            LOAD if is_load else STORE,
            () if is_load else ((vd, _M),),
            (vd, _M) if is_load else None,
            (rs1,),
            -1,
            mem=("mask", 8, 1),
        )
    mask_src = lambda vm: ((0, _M),) if not vm else ()  # noqa: E731
    if h in (integer.binary, integer.fixed, integer.multiply_add, fpu.binary, fpu.fma):
        extra, vd, vs2, kind, src, vm = args
        acc = ((vd, _S),) if h in (integer.multiply_add, fpu.fma) else ()
        is_div = (
            h is integer.binary
            and extra in (integer.INT_BINARY[n] for n in ("div", "divu", "rem", "remu"))
        ) or (h is fpu.binary and extra == "div")
        return VOp(
            DIV if is_div else ARITH,
            ((vs2, _S),) + _vsrc(kind, src) + acc + mask_src(vm),
            (vd, _S),
            _scalar_src(kind, src),
            -1,
            masked=not vm,
        )
    if h in (integer.compare, fpu.compare, integer.carry):
        extra, vd, vs2, kind, src, vm = args
        out = (
            (vd, _M)
            if h is not integer.carry or extra in ("madc", "msbc")
            else (vd, _S)
        )
        return VOp(
            ARITH,
            ((vs2, _S),) + _vsrc(kind, src) + mask_src(vm),
            out,
            _scalar_src(kind, src),
            -1,
        )
    if h is integer.merge:
        vd, vs2, kind, src, vm = args
        srcs = _vsrc(kind, src) + (() if vm else ((vs2, _S), (0, _M)))
        return VOp(ARITH, srcs, (vd, _S), _scalar_src(kind, src), -1)
    if h in (integer.widening, fpu.widening):
        extra, vd, vs2, kind, src, vm = args
        wide_vs2 = extra.endswith(".w")
        acc = ((vd, _W),) if "macc" in extra or "msac" in extra else ()
        return VOp(
            ARITH,
            ((vs2, _W if wide_vs2 else _S),) + _vsrc(kind, src) + acc + mask_src(vm),
            (vd, _W),
            _scalar_src(kind, src),
            -1,
            wide=True,
        )
    if h is integer.narrowing:
        extra, vd, vs2, kind, src, vm = args
        return VOp(
            ARITH,
            ((vs2, _W),) + _vsrc(kind, src) + mask_src(vm),
            (vd, _S),
            _scalar_src(kind, src),
            -1,
            wide=True,
        )
    if h is integer.extension:
        vd, vs2, factor, _, vm = args
        return VOp(ARITH, ((vs2, ("e", factor)),) + mask_src(vm), (vd, _S), (), -1)
    if h in (integer.reduction, fpu.reduction):
        name, vd, vs2, vs1, vm = args
        wide = name.startswith("w")
        return VOp(
            RED,
            ((vs2, _S), (vs1, _M)) + mask_src(vm),
            (vd, _M),
            (),
            -1,
            wide=wide,
            whole_src=(vs1,),  # the scalar operand, not a stream
        )
    if h is fpu.merge:
        vd, vs2, rs1, vm = args
        return VOp(
            ARITH, (() if vm else ((vs2, _S), (0, _M))), (vd, _S), (_F + rs1,), -1
        )
    if h is fpu.unary:
        op, vd, vs2, vm = args
        return VOp(
            DIV if op == "sqrt" else ARITH,
            ((vs2, _S),) + mask_src(vm),
            (vd, _S),
            (),
            -1,
        )
    if h is fpu.convert:
        code, vd, vs2, vm = args
        shape = fpu.CONVERSIONS[code][1]
        src_role = _W if shape == "n" else _S
        dst_role = _W if shape == "w" else _S
        return VOp(
            ARITH,
            ((vs2, src_role),) + mask_src(vm),
            (vd, dst_role),
            (),
            -1,
            wide=shape != "s",
        )
    if h is permute.mask_logical:
        _, vd, vs2, vs1 = args
        return VOp(ARITH, ((vs2, _M), (vs1, _M)), (vd, _M), (), -1)
    if h is permute.cpop_first:
        _, rd, vs2, vm = args
        return VOp(RED, ((vs2, _M),) + mask_src(vm), None, (), rd or -1)
    if h is permute.set_first:
        _, vd, vs2, vm = args
        return VOp(ARITH, ((vs2, _M),) + mask_src(vm), (vd, _M), (), -1)
    if h is permute.iota:
        vd, vs2, vm = args
        return VOp(ARITH, ((vs2, _M),) + mask_src(vm), (vd, _S), (), -1)
    if h is permute.vid:
        vd, vm = args
        return VOp(ARITH, mask_src(vm), (vd, _S), (), -1)
    if h is permute.mv_x_s:
        rd, vs2 = args
        return VOp(ARITH, ((vs2, _M),), None, (), rd or -1)
    if h is permute.fmv_f_s:
        rd, vs2, _ = args
        return VOp(ARITH, ((vs2, _M),), None, (), _F + rd)
    if h is permute.mv_s_x:
        vd, rs1 = args
        return VOp(ARITH, (), (vd, _M), (rs1,), -1)
    if h is permute.fmv_s_f:
        vd, rs1, _ = args
        return VOp(ARITH, (), (vd, _M), (_F + rs1,), -1)
    if h in (permute.slide, permute.gather):
        if h is permute.slide:
            name, vd, vs2, kind, src, vm = args
            # Element i of a slide-down reads vs2[i + off]: it cannot chain on vs2.
            whole = (vs2,) if name in ("down", "1down") else ()
        else:
            vd, vs2, kind, src, vm = args
            whole = (vs2,)  # vrgather reads vs2 in data-dependent order
        return VOp(
            ARITH,
            ((vs2, _S),) + _vsrc(kind, src) + mask_src(vm),
            (vd, _S),
            _scalar_src(kind, src),
            -1,
            whole_src=whole,
        )
    if h is permute.compress:
        vd, vs2, vs1 = args
        return VOp(ARITH, ((vs2, _S), (vs1, _M)), (vd, _S), (), -1, streams=False)
    if h is permute.move_whole:
        vd, vs2, nr = args
        return VOp(
            ARITH, ((vs2, ("n", nr)),), (vd, ("n", nr)), (), -1, mem=("whole", 8, nr)
        )
    raise AssertionError(f"vector_timing.classify: no rule for {h.__name__}")


# -- the issue logic ---------------------------------------------------------------------


class VectorTiming:
    """One hart's vector timing state."""

    #: The hart's XLEN (the cost state sets it): strides are signed XLEN values.
    xlen = 64

    def __init__(self, spec, vector_state, memory):
        self.spec = spec
        self.vs = vector_state
        self.memory = memory
        self._ops = {}
        self.reset()

    def reset(self):
        self.ready = [0] * 32
        # Chaining only: when each register's first element is readable, and the
        # ``ready`` value that was current when it was set (another writer, such as
        # an IME or VME mma, updates ``ready`` alone, making the register whole).
        self.first = [0] * 32
        self.first_of = [0] * 32
        self.chained = 0
        self.unit_free = 0
        self.lsu_free = 0
        self.stall_until = 0
        self.stall_reason = VECTOR_BUSY
        self.busy_cycles = 0
        self.mem_busy_cycles = 0

    def counters(self):
        out = {
            "vector_busy_cycles": self.busy_cycles,
            "vector_mem_busy_cycles": self.mem_busy_cycles,
        }
        if self.spec.chaining:
            # Vector instructions that started before a source was whole.
            out["vector_chained"] = self.chained
        return out

    def decode(self, instr):
        op = self._ops.get(instr, 0)
        if op == 0:
            op = self._ops[instr] = classify(instr)
        return op

    def _regs(self, base, role):
        """The registers an operand covers under the current vtype."""
        lg = self.vs.lmul_log2
        if role == _S:
            n = nregs(lg)
        elif role == _W:
            n = nregs(lg + 1)
        elif role == _M:
            n = 1
        elif role[0] == "n":
            n = role[1]
        elif role[0] == "e":
            n = nregs(lg - {2: 1, 4: 2, 8: 3}[role[1]])
        elif role[0] == "i":  # index EEW over SEW
            n = nregs(lg + role[1].bit_length() - self.vs.sew.bit_length())
        else:  # ("d", eew, nf): data EEW
            n = nregs(lg + role[1].bit_length() - self.vs.sew.bit_length()) * role[2]
        return range(base, min(32, base + n))

    def _stall(self, until, reason):
        self.stall_until, self.stall_reason = until, reason
        return -1

    def _footprint(self, op, addr, nbytes, register_file):
        """The bytes a vector access touches (:func:`~framework.naiad.store_hazards.footprint`)."""
        mode, eew, nf = op.mem
        vs = self.vs
        if mode == "strided":
            xlen = self.xlen
            stride = register_file[op.ssrc[1]].read_uint()
            if stride >> (xlen - 1):
                stride -= 1 << xlen
            return footprint(addr, stride, vs.vl, nf * eew // 8)
        if mode == "indexed" and vs.vl:
            # Gather / scatter: the bounding interval of the element addresses.
            rs2 = next(b for b, role in op.vsrc if role[0] == "i")
            idx = vs.group(rs2, len(self._regs(rs2, ("i", eew))), UINT[eew])[: vs.vl]
            lo, hi = int(idx.min()), int(idx.max())
            return footprint(addr + lo, 0, 1, hi - lo + nf * vs.sew // 8)
        return footprint(addr, 0, 1, nbytes)

    def issue(self, op, cycle, register_file):
        """Issue ``op`` at ``cycle``: the cycle its scalar result is ready, or -1 (stall)."""
        spec = self.spec
        if op.kind == CFG:
            return cycle + 1
        if spec.chaining:
            return self._issue_chained(op, cycle, register_file)
        ready = self.ready
        for base, role in op.vsrc:
            for r in self._regs(base, role):
                if ready[r] > cycle:
                    return self._stall(ready[r], VECTOR_RAW)
        vs = self.vs
        if op.kind in (LOAD, STORE):
            if self.lsu_free > cycle:
                return self._stall(self.lsu_free, VECTOR_BUSY)
            mode, eew, nf = op.mem
            if mode == "whole":
                nbytes = nf * vs.vlenb
            elif mode == "mask":
                nbytes = (vs.vl + 7) // 8
            else:
                data_eew = vs.sew if mode == "indexed" else eew
                nbytes = vs.vl * nf * data_eew // 8
            addr = register_file[(op.ssrc[0])].read_uint()
            region = self.memory.lookup(addr)
            if region.port_free > cycle:
                return self._stall(region.port_free, MEM)
            inflight = self.memory.inflight
            if op.kind == LOAD and inflight.until > cycle:
                at = inflight.wait(
                    self._footprint(op, addr, nbytes, register_file), cycle
                )
                if at:  # an earlier store is still writing these bytes
                    return self._stall(at, MEM_RAW)
            bw = spec.mem_bytes_per_cycle
            if region.bandwidth is not None:
                bw = min(bw, region.bandwidth)
            t = max(1, math.ceil(nbytes / bw))
            if mode in ("strided", "indexed") or (mode == "unit" and nf > 1):
                t = max(t, math.ceil(vs.vl * nf / spec.indexed_elements_per_cycle))
            latency = region.access(addr, nbytes, op.kind == STORE, cycle, t)
            self.lsu_free = cycle + t
            self.mem_busy_cycles += t
            if op.kind == STORE and nbytes:
                inflight.add(
                    self._footprint(op, addr, nbytes, register_file), cycle + t, cycle
                )
            if op.kind == LOAD and op.vdst is not None:
                done = cycle + spec.mem_startup + latency + t
                for r in self._regs(*op.vdst):
                    ready[r] = done
            return cycle + 1
        if self.unit_free > cycle:
            return self._stall(self.unit_free, VECTOR_BUSY)
        if op.mem is not None:  # vmv<nr>r.v
            occupancy = max(1, math.ceil(op.mem[2] * vs.vlen / spec.datapath_bits))
        else:
            width = 2 * vs.sew if op.wide else vs.sew
            occupancy = max(1, math.ceil(vs.vl * width / spec.datapath_bits))
            if op.kind == DIV:
                occupancy *= spec.div_ii
        done = cycle + occupancy + spec.startup
        if op.kind == RED:
            lanes = max(1, spec.datapath_bits // vs.sew)
            done += math.ceil(math.log2(lanes)) if lanes > 1 else 0
        self.unit_free = cycle + occupancy
        self.busy_cycles += occupancy
        if op.vdst is not None:
            for r in self._regs(*op.vdst):
                ready[r] = done
        return done if op.sdst >= 0 else cycle + 1

    # -- chaining (``timing.vector.chaining: true``) -------------------------------------

    def _eew(self, role):
        """Element width in bits of an operand role (a mask is one bit per element)."""
        if role == _S:
            return self.vs.sew
        if role == _W:
            return 2 * self.vs.sew
        if role == _M:
            return 1
        tag = role[0]
        if tag == "n":
            return 8
        if tag == "e":
            return self.vs.sew // role[1]
        return role[1]  # "i" / "d": the access's EEW

    def _spans(self, base, role, n):
        """``(reg, first, last)`` element indices (below ``n``) each register of an operand holds."""
        regs = self._regs(base, role)
        per = max(1, self.vs.vlen // self._eew(role))
        fields = role[2] if isinstance(role, tuple) and role[0] == "d" else 1
        emul = max(1, len(regs) // fields)
        out = []
        for i, r in enumerate(regs):
            lo = (i % emul) * per
            out.append((r, lo, min(lo + per, n) - 1))
        return out

    def _shape(self, op):
        """``(n, occupancy, beats)`` of a datapath op: its elements, cycles and element groups."""
        spec, vs = self.spec, self.vs
        if op.mem is not None:  # vmv<nr>r.v: whole registers as bytes
            occ = max(1, math.ceil(op.mem[2] * vs.vlen / spec.datapath_bits))
            return op.mem[2] * vs.vlenb, occ, occ
        width = 2 * vs.sew if op.wide else vs.sew
        beats = max(1, math.ceil(vs.vl * width / spec.datapath_bits))
        occ = beats * spec.div_ii if op.kind == DIV else beats
        return vs.vl, occ, beats

    def _chain_at(self, op, n, occ, beats):
        """The first cycle ``op`` may start, by its vector sources, and that bound unchained.

        Element ``x`` of ``n`` is processed in beat ``k = x * beats // n``, which
        starts ``k * occ // beats`` cycles into the op. A source register may be
        read once its first element is ready when the op reaches that element, and
        its last element likewise; between the two both stream linearly.
        """
        ready, first, first_of = self.ready, self.first, self.first_of
        at = whole = 0
        for base, role in op.vsrc:
            if n == 0 or base in op.whole_src:
                for r in self._regs(base, role):
                    at = max(at, ready[r])
                    whole = max(whole, ready[r])
                continue
            for r, lo, hi in self._spans(base, role, n):
                if hi < lo:
                    continue  # holds only tail elements: not read
                done = ready[r]
                whole = max(whole, done)
                f = first[r] if first_of[r] == done else done
                start_lo = (lo * beats // n) * occ // beats
                start_hi = (hi * beats // n) * occ // beats
                at = max(at, f - start_lo, done - start_hi)
        return at, whole

    def _stream_out(self, op, cycle, n, occ, beats, latency, streams=True):
        """Record when each destination register's first and last elements are readable."""
        ready, first, first_of = self.ready, self.first, self.first_of
        done = cycle + occ + latency
        if not streams or n == 0:
            for r in self._regs(*op.vdst):
                ready[r] = first[r] = first_of[r] = done
            return
        for r, lo, hi in self._spans(*op.vdst, n):
            if hi < lo:
                ready[r] = first[r] = first_of[r] = done
                continue
            end_lo = -(-((lo * beats // n) + 1) * occ // beats)
            end_hi = -(-((hi * beats // n) + 1) * occ // beats)
            first[r] = cycle + end_lo + latency
            ready[r] = first_of[r] = cycle + end_hi + latency

    def _issue_chained(self, op, cycle, register_file):
        """:meth:`issue` with element-group chaining (see the module docstring)."""
        spec, vs = self.spec, self.vs
        if op.kind in (LOAD, STORE):
            mode, eew, nf = op.mem
            if mode == "whole":
                nbytes = nf * vs.vlenb
                n = nbytes
            elif mode == "mask":
                nbytes = (vs.vl + 7) // 8
                n = vs.vl
            else:
                data_eew = vs.sew if mode == "indexed" else eew
                nbytes = vs.vl * nf * data_eew // 8
                n = vs.vl
            addr = register_file[(op.ssrc[0])].read_uint()
            region = self.memory.lookup(addr)
            bw = spec.mem_bytes_per_cycle
            if region.bandwidth is not None:
                bw = min(bw, region.bandwidth)
            t = max(1, math.ceil(nbytes / bw))
            if mode in ("strided", "indexed") or (mode == "unit" and nf > 1):
                t = max(t, math.ceil(vs.vl * nf / spec.indexed_elements_per_cycle))
            at, whole = self._chain_at(op, n, t, t)
            if at > cycle:
                return self._stall(at, VECTOR_RAW)
            if self.lsu_free > cycle:
                return self._stall(self.lsu_free, VECTOR_BUSY)
            if region.port_free > cycle:
                return self._stall(region.port_free, MEM)
            inflight = self.memory.inflight
            if op.kind == LOAD and inflight.until > cycle:
                at_mem = inflight.wait(
                    self._footprint(op, addr, nbytes, register_file), cycle
                )
                if at_mem:  # an earlier store is still writing these bytes
                    return self._stall(at_mem, MEM_RAW)
            if whole > cycle:
                self.chained += 1
            latency = region.access(addr, nbytes, op.kind == STORE, cycle, t)
            self.lsu_free = cycle + t
            self.mem_busy_cycles += t
            if op.kind == STORE and nbytes:
                # Complete when the LSU frees: by then the chained data has all arrived.
                inflight.add(
                    self._footprint(op, addr, nbytes, register_file), cycle + t, cycle
                )
            if op.kind == LOAD and op.vdst is not None:
                self._stream_out(op, cycle, n, t, t, spec.mem_startup + latency)
            return cycle + 1
        n, occ, beats = self._shape(op)
        at, whole = self._chain_at(op, n, occ, beats)
        if at > cycle:
            return self._stall(at, VECTOR_RAW)
        if self.unit_free > cycle:
            return self._stall(self.unit_free, VECTOR_BUSY)
        if whole > cycle:
            self.chained += 1
        done = cycle + occ + spec.startup
        if op.kind == RED:
            lanes = max(1, spec.datapath_bits // vs.sew)
            done += math.ceil(math.log2(lanes)) if lanes > 1 else 0
        self.unit_free = cycle + occ
        self.busy_cycles += occ
        if op.vdst is not None:
            # A reduction's result exists only after its tree: it does not stream.
            self._stream_out(
                op,
                cycle,
                n,
                occ,
                beats,
                done - cycle - occ,
                streams=op.streams and op.kind != RED,
            )
        return done if op.sdst >= 0 else cycle + 1
