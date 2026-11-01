"""An abstract in-order issue model for loop bodies: the cost model's arithmetic.

A dwarf's cost formula (``framework.naiad.costmodel.dwarfs``) describes its
kernel as a loop nest whose bodies are short sequences of abstract
instructions -- "a load into ``a``", "an fma reading ``a`` and ``b``", "an mma
into accumulator 0 from tiles 1 and 2", "a 128-byte vector load". The
:class:`Engine` issues them under the same rules the Naiad cycle model applies
to real instructions (``framework.pe.rv.timing.core``,
``framework.pe.rv.timing.vector``, ``framework.matrix.timing``):

* one instruction issues per cycle (two at ``Scalar.issue_width`` 2, when the pair needs different
  pipes: see :data:`P_SINGLE`), in program order, once every operand is
  ready and the unit it needs is free (an instruction that waits holds up
  everything behind it);
* per-class result latencies; divides and FP divide/sqrt unpipelined; a taken
  branch holds the next instruction for ``branch_taken`` cycles; loads pay the
  data region's latency;
* the vector unit: one datapath (occupancy ``ceil(VL * EEW / datapath_bits)``,
  results ``startup`` later; no chaining unless the machine's
  ``vector.chaining`` is set, see below) and one load/store unit
  (``ceil(bytes / mem_bytes_per_cycle)``, at least one element a cycle when
  strided);
* the AME unit: mma / tile-load / tile-store pipes, a tile and accumulator
  scoreboard (RAW, WAW, WAR), accumulator forwarding between mmas;
* memory RAW: a load named as reading what an asynchronous (tile / vector)
  store wrote waits for that store to complete.

**Chaining.** With ``Machine.vector.chaining`` (the design's
``timing.vector.chaining: true``) a vector instruction does not wait for its
producer's whole result. The rule is ``framework.pe.rv.timing.vector``'s,
restated per register name instead of per architectural register: an op
processes its ``n`` elements in ``beats`` equal groups over its occupancy
(element ``x`` in beat ``x * beats // n``); a producer records, per
destination register, when its first and its last element become readable
(``issue + beat end + startup``, a load's ``mem_startup`` plus the region
latency); a consumer starts once, for every source register, the first
element is ready when the consumer's beat reaches it and the last likewise.
Not chained: sources read out of order (``vrgather``'s and a slide-down's
``vs2``, a reduction's scalar ``vs1``), reduction and ``vcompress`` results.
The datapath and the LSU stay one instruction at a time. A name passed
alone covers all ``n`` of the consumer's elements (a register group seen at
its two ends, which is all the linear streams need); the assembly-listing
interpreter passes :class:`VGroup` to give each register its own element
range, as the hardware's decode does.

So the engine is the closed form ``cycles = f(loop bounds, body, timing)``
evaluated by replay rather than by algebra: no constant in it is fitted.

**Loops.** :meth:`Engine.loop` replays a body for a few warm-up and measured
iterations, then fast-forwards the remaining ones by the measured steady-state
period (shifting every in-flight time by the same amount), so the cost of a
loop is ``O(warm + measure)`` replays whatever its trip count. A body may
nest further loops. Bodies whose schedule repeats with a period above one
(ping-pong buffers) pass ``period`` so the measured window is a whole number
of periods.
"""

import math
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

#: Scalar instruction classes and the ``Scalar`` field holding their latency.
_LATENCY_FIELD = {
    "alu": "alu",
    "mul": "mul",
    "div": "div",
    "csr": "csr",
    "fadd": "fp_add",
    "fmul": "fp_mul",
    "fma": "fp_fma",
    "fdiv": "fp_div",
    "fsqrt": "fp_sqrt",
    "fmisc": "fp_misc",
}


#: Issue pipes (``framework.pe.rv.timing.core``): at ``issue_width`` 2 two instructions pair
#: unless both need one of the single pipes, the first ends the issue group, or either issues alone.
P_MEM, P_MULDIV, P_FP, P_VEC = 1, 2, 4, 8
P_SINGLE = P_MEM | P_MULDIV | P_FP | P_VEC
P_CTRL = 16  # a branch or jump: nothing issues after it in its cycle
P_ALONE = 32  # a CSR access: issues alone


@dataclass(frozen=True)
class VGroup:
    """A vector register group as a chained operand: its registers in order, and how they are read.

    ``eew`` is the operand's element width in bits (1 for a mask), ``whole`` a source
    read out of order (waited for in full), ``fields`` the segment count of a
    segment access's data group.
    """

    names: tuple
    eew: int
    whole: bool = False
    fields: int = 1


def ceil_div(a, b):
    return -(-a // b)


def transfer(nbytes, *bandwidths):
    """Cycles to move ``nbytes`` at the narrowest of ``bandwidths`` (None = unlimited), at least 1."""
    lim = [b for b in bandwidths if b is not None]
    if not lim:
        return 1
    return max(1, math.ceil(nbytes / min(lim)))


class Engine:
    """In-order issue over abstract instructions for one :class:`~framework.naiad.costmodel.machine.Machine`.

    Register names are any hashable values chosen by the caller; scalar,
    vector, tile and accumulator names live in separate namespaces.
    """

    def __init__(self, machine):
        self.m = machine
        s = machine.scalar
        self.lat = {k: getattr(s, f) for k, f in _LATENCY_FIELD.items()}
        #: The next issue cycle; 0 is the opening cycle read of the timed region.
        self.t = 1
        self.ready = {}
        self.unit = {}
        self.lsu = 0
        self.vready = {}
        #: Chaining only: per vector register name, when its first element is readable, and the
        #: ``vready`` value that was current when it was set.
        self.vfirst, self.vfirst_of = {}, {}
        self.chaining = bool(machine.vector is not None and machine.vector.chaining)
        self.vunit = 0
        self.vlsu = 0
        self.tile_ready, self.tile_rfree = {}, {}
        self.acc_ready, self.acc_mready, self.acc_rfree = {}, {}, {}
        self.pipe = {"mma": 0, "load": 0, "store": 0, "ewise": 0}
        self.mem = {}
        self.busy = Counter()
        self.instret = Counter()
        #: Set to a list to record ``(issue cycle, op)`` for every instruction (debugging).
        self.log = None
        self.label = None
        #: Dual issue (``Machine.scalar.issue_width`` 2): the cycle and pipes of the last instruction
        #: if it issued in slot 0 and slot 1 of its cycle is still free, else None.
        self.width = machine.scalar.issue_width
        self._open = None

    # -- helpers ----------------------------------------------------------------

    def _at(self, table, names):
        c = 0
        for n in names:
            v = table.get(n, 0)
            if v > c:
                c = v
        return c

    def _issue_at(self, *times, pipes=0):
        """Issue one instruction needing ``pipes`` once every ``times`` has passed; returns its cycle.

        Single issue: the first cycle at or after ``self.t``. Dual issue: also slot 1 of the previous
        instruction's cycle when it pairs (the simulator's ``DualIssueCostState`` rules) and
        everything it waits for is already there.
        """
        c = self.t
        for v in times:
            if v > c:
                c = v
        if self.width > 1:
            prev, self._open = self._open, None
            if prev is not None and c <= prev[0] + 1 and self.t <= prev[0] + 1:
                at = prev[0]
                if (
                    all(v <= at for v in times)
                    and not (pipes | prev[1]) & P_ALONE
                    and not prev[1] & P_CTRL
                    and not prev[1] & pipes & P_SINGLE
                ):
                    self.t = at + 1
                    self._log(at)
                    return at
            if not pipes & (P_CTRL | P_ALONE):
                self._open = (c, pipes)
        self.t = c + 1
        self._log(c)
        return c

    def _log(self, c):
        if self.log is not None:
            import sys

            self.log.append((c, self.label or sys._getframe(2).f_code.co_name))

    # -- scalar -----------------------------------------------------------------

    def alu(self, n=1, dst=None, srcs=()):
        """``n`` single-cycle integer instructions (the last writes ``dst``)."""
        if n <= 0:
            return
        c = self._issue_at(self._at(self.ready, srcs))
        last = c
        if self.width > 1:
            # two ALU pipes: the rest pair up (slot 1 of this cycle first, if it is free)
            rest = n - 1
            if self._open is not None and rest:
                rest -= 1
                self._open = None
            if rest:
                last = c + (rest + 1) // 2
                if rest % 2 == 0:
                    self._open = None
                else:
                    self._open = (last, 0)
                self.t = last + 1
        else:
            last = c + n - 1
            self.t = c + n
        self.instret["scalar"] += n
        if dst is not None:
            self.ready[dst] = last + self.lat["alu"]

    def op(self, kind, dst=None, srcs=()):
        """One scalar instruction of class ``kind`` (see ``_LATENCY_FIELD``)."""
        lat = self.lat[kind]
        unit = None
        if kind == "div" and not self.m.scalar.div_pipelined:
            unit = "div"
        elif kind in ("fdiv", "fsqrt") and not self.m.scalar.fp_div_pipelined:
            unit = "fdiv"
        pipes = {"mul": P_MULDIV, "div": P_MULDIV, "csr": P_ALONE}.get(kind, P_FP)
        c = self._issue_at(
            self._at(self.ready, srcs), self.unit.get(unit, 0), pipes=pipes
        )
        if unit is not None:
            self.unit[unit] = c + lat
        if dst is not None:
            self.ready[dst] = c + lat
        self.instret["fp" if kind[0] == "f" else "scalar"] += 1
        return c

    def load(self, dst=None, srcs=(), after=(), fp=False):
        """A scalar load; ``after`` names memory tokens of asynchronous stores it must wait for."""
        c = self._issue_at(
            self._at(self.ready, srcs),
            self.lsu,
            self._at(self.mem, after),
            pipes=P_MEM,
        )
        if dst is not None:
            self.ready[dst] = c + self.m.load_latency
        self.instret["fp" if fp else "scalar"] += 1
        return c

    def store(self, srcs=(), fp=False):
        c = self._issue_at(self._at(self.ready, srcs), self.lsu, pipes=P_MEM)
        self.lsu = c + self.m.scalar.store
        self.instret["fp" if fp else "scalar"] += 1
        return c

    def branch(self, taken=True, srcs=()):
        c = self._issue_at(self._at(self.ready, srcs), pipes=P_CTRL)
        if taken and self.m.scalar.branch_taken:
            self.t = c + 1 + self.m.scalar.branch_taken
        self.instret["scalar"] += 1
        return c

    def jump(self):
        return self.branch(True)

    # -- vector -------------------------------------------------------------------

    def vset(self, dst=None):
        c = self._issue_at(pipes=P_VEC)
        if dst is not None:
            self.ready[dst] = c + 1
        self.instret["vector"] += 1

    # -- chaining (``Machine.vector.chaining``) ----------------------------------------

    def _operands(self, srcs, sgroups, n):
        """``(name, first element, last element, whole)`` for each vector source register."""
        if sgroups is None:
            return [(s, 0, n - 1, False) for s in srcs]
        vlen = self.m.vector.vlen
        out = []
        for g in sgroups:
            per = max(1, vlen // g.eew)
            emul = max(1, len(g.names) // g.fields)
            for i, nm in enumerate(g.names):
                lo = (i % emul) * per
                out.append((nm, lo, min(lo + per, n) - 1, g.whole))
        return out

    def _chain_at(self, srcs, sgroups, n, occ, beats):
        """The first cycle an op of ``n`` elements may start by its vector sources (see the module docstring)."""
        ready, first, first_of = self.vready, self.vfirst, self.vfirst_of
        at = 0
        for nm, lo, hi, whole in self._operands(srcs, sgroups, n):
            done = ready.get(nm, 0)
            if n == 0 or whole:
                at = max(at, done)
                continue
            if hi < lo:
                continue  # holds only tail elements: not read
            f = first.get(nm, 0) if first_of.get(nm, 0) == done else done
            start_lo = (lo * beats // n) * occ // beats
            start_hi = (hi * beats // n) * occ // beats
            at = max(at, f - start_lo, done - start_hi)
        return at

    def _stream_out(self, dst, dgroup, c, n, occ, beats, latency, streams=True):
        """Record when each destination register's first and last elements are readable.

        ``dgroup`` is ``(eew, fields)`` when ``dst`` lists a group's registers in order;
        otherwise every name covers all ``n`` elements.
        """
        names = _names(dst)
        done = c + occ + latency
        if dgroup is None:
            spans = [(nm, 0, n - 1) for nm in names]
        else:
            eew, fields = dgroup
            per = max(1, self.m.vector.vlen // eew)
            emul = max(1, len(names) // fields)
            spans = []
            for i, nm in enumerate(names):
                lo = (i % emul) * per
                spans.append((nm, lo, min(lo + per, n) - 1))
        for nm, lo, hi in spans:
            if not streams or n == 0 or hi < lo:
                self.vready[nm] = self.vfirst[nm] = self.vfirst_of[nm] = done
                continue
            end_lo = -(-((lo * beats // n) + 1) * occ // beats)
            end_hi = -(-((hi * beats // n) + 1) * occ // beats)
            self.vfirst[nm] = c + end_lo + latency
            self.vready[nm] = self.vfirst_of[nm] = c + end_hi + latency

    def vop(
        self,
        dst,
        srcs=(),
        vl=None,
        sew=32,
        wide=False,
        kind="arith",
        sdst=None,
        xsrcs=(),
        regs=None,
        sgroups=None,
        dgroup=None,
        streams=True,
    ):
        """A vector arithmetic instruction over ``vl`` elements of width ``sew`` (2*SEW when ``wide``).

        ``kind``: ``arith``, ``div`` (``div_ii`` times longer) or ``red`` (plus the
        lane-tree latency). ``regs`` overrides the occupancy with a whole-register
        count (``vmv<nr>r``). ``sdst`` names a scalar result (``vfmv.f.s``, ``vcpop``).
        ``sgroups`` / ``dgroup`` / ``streams`` refine the chaining rule only (see :class:`VGroup`;
        ``streams=False``: the result is readable only when the op completes).
        """
        v = self.m.vector
        if regs is not None:
            beats = max(1, ceil_div(regs * v.vlen, v.datapath_bits))
            occ = beats
            n = regs * v.vlenb
        else:
            width = 2 * sew if wide else sew
            beats = max(1, ceil_div(vl * width, v.datapath_bits))
            occ = beats * v.div_ii if kind == "div" else beats
            n = vl
        if self.chaining:
            vat = self._chain_at(srcs, sgroups, n, occ, beats)
        else:
            vat = self._at(self.vready, srcs)
        c = self._issue_at(vat, self._at(self.ready, xsrcs), self.vunit, pipes=P_VEC)
        done = c + occ + v.startup
        if kind == "red":
            lanes = max(1, v.datapath_bits // sew)
            done += math.ceil(math.log2(lanes)) if lanes > 1 else 0
        self.vunit = c + occ
        self.busy["vector"] += occ
        if self.chaining:
            self._stream_out(
                dst,
                dgroup,
                c,
                n,
                occ,
                beats,
                done - c - occ,
                streams=streams and kind != "red",
            )
        else:
            for d in _names(dst):
                self.vready[d] = done
        if sdst is not None:
            self.ready[sdst] = done
        self.instret["vector"] += 1
        return c

    def vload(
        self,
        dst,
        nbytes,
        xsrcs=(),
        elems=0,
        after=(),
        n=0,
        srcs=(),
        sgroups=None,
        dgroup=None,
    ):
        """A vector load of ``nbytes``; ``elems`` > 0 for a strided / indexed / segment access.

        ``n`` is the element count and ``srcs`` / ``sgroups`` the index vector of an indexed
        load; both matter to chaining only.
        """
        v = self.m.vector
        t = transfer(nbytes, v.mem_bytes_per_cycle, self.m.memory.bandwidth)
        if elems:
            t = max(t, math.ceil(elems / v.indexed_elements_per_cycle))
        vat = 0
        if srcs or sgroups:
            if self.chaining:
                vat = self._chain_at(srcs, sgroups, n or t, t, t)
            else:
                vat = self._at(self.vready, srcs)
        c = self._issue_at(
            vat,
            self._at(self.ready, xsrcs),
            self.vlsu,
            self._at(self.mem, after),
            pipes=P_VEC | P_MEM,
        )
        self.vlsu = c + t
        self.busy["vector_mem"] += t
        lat = v.mem_startup + self.m.memory.latency
        if self.chaining:
            self._stream_out(dst, dgroup, c, n or t, t, t, lat)
        else:
            for d in _names(dst):
                self.vready[d] = c + lat + t
        self.instret["vector"] += 1
        return c

    def vstore(
        self,
        srcs,
        nbytes,
        xsrcs=(),
        elems=0,
        token=None,
        n=0,
        sgroups=None,
    ):
        v = self.m.vector
        t = transfer(nbytes, v.mem_bytes_per_cycle, self.m.memory.bandwidth)
        if elems:
            t = max(t, math.ceil(elems / v.indexed_elements_per_cycle))
        names = _names(srcs)
        if self.chaining:
            vat = self._chain_at(names, sgroups, n or t, t, t)
        else:
            vat = self._at(self.vready, names)
        c = self._issue_at(
            vat, self._at(self.ready, xsrcs), self.vlsu, pipes=P_VEC | P_MEM
        )
        self.vlsu = c + t
        self.busy["vector_mem"] += t
        if token is not None:
            self.mem[token] = c + t
        self.instret["vector"] += 1
        return c

    # -- AME --------------------------------------------------------------------

    def _matrix_done(self, done):
        if not self.m.ame.overlap and done > self.t:
            self.t = done

    def mma(self, acc, a, b, ii, accumulate=True, macs=0):
        """``acc (+)= a x b`` with initiation interval ``ii`` (``AME.ii``)."""
        x = self.m.ame
        acc_wait = (
            (self.acc_mready if x.acc_forwarding else self.acc_ready).get(acc, 0)
            if accumulate
            else 0
        )
        c = self._issue_at(
            self._at(self.tile_ready, (a, b)),
            acc_wait,
            self.acc_rfree.get(acc, 0),
            self.pipe["mma"],
            pipes=P_VEC,
        )
        self.pipe["mma"] = c + ii
        self.busy["mma"] += ii
        done = c + ii + x.pipeline_depth - 1
        self.acc_ready[acc] = done
        self.acc_mready[acc] = c + ii
        for t in (a, b):
            self.tile_rfree[t] = max(self.tile_rfree.get(t, 0), c + ii)
        self.instret["matrix"] += 1
        self.busy["mmas"] += 1
        self.busy["macs"] += macs
        self._matrix_done(done)
        return c

    def mop(
        self,
        ii,
        latency=1,
        tiles_read=(),
        tiles_written=(),
        accs_read=(),
        accs_written=(),
        pipe="mma",
    ):
        """A register-to-register matrix op holding ``pipe`` for ``ii`` (epilogue, amov, tconv, treduce)."""
        c = self._issue_at(
            self._at(self.tile_ready, tiles_read),
            self._at(self.acc_ready, accs_read),
            *(
                max(self.tile_ready.get(t, 0), self.tile_rfree.get(t, 0))
                for t in tiles_written
            ),
            *(
                max(self.acc_ready.get(a, 0), self.acc_rfree.get(a, 0))
                for a in accs_written
            ),
            self.pipe[pipe] if pipe else 0,
            pipes=P_VEC,
        )
        if pipe:
            self.pipe[pipe] = c + ii
            self.busy[pipe] += ii
        done = c + ii + latency - 1
        for t in tiles_written:
            self.tile_ready[t] = done
        for a in accs_written:
            self.acc_ready[a] = self.acc_mready[a] = done
        for t in tiles_read:
            self.tile_rfree[t] = max(self.tile_rfree.get(t, 0), c + ii)
        for a in accs_read:
            self.acc_rfree[a] = max(self.acc_rfree.get(a, 0), c + ii)
        self.instret["matrix"] += 1
        self._matrix_done(done)
        return c

    def scratch_tile(self, acc):
        """``MX_SCRATCH_TILE(acc)`` of ``matrix.h``: the tile an accumulator moves through without ``acc_mem``."""
        x = self.m.ame
        n = 4 if x.tiles >= 16 else 1
        return ("scratch", x.tiles - 1 - (acc % n))

    def tload(self, tile, nbytes, xsrcs=(), after=(), acc=False):
        """``tload.a/.b`` (or ``aload`` with ``acc``) of ``nbytes`` into ``tile``.

        Without ``acc_mem`` an ``aload`` is ``matrix.h``'s lowering: a tile load into the
        accumulator's scratch tile, then ``amov.a.t``.
        """
        x = self.m.ame
        if acc and not x.acc_mem:
            st = self.scratch_tile(tile)
            c = self.tload(st, nbytes, xsrcs, after)
            self.mop(
                math.ceil(x.m / x.tile_to_acc_rows_per_cycle),
                x.tile_to_acc_latency,
                tiles_read=(st,),
                accs_written=(tile,),
            )
            return c
        if acc:
            dst_wait = max(self.acc_ready.get(tile, 0), self.acc_rfree.get(tile, 0))
        else:
            dst_wait = max(self.tile_ready.get(tile, 0), self.tile_rfree.get(tile, 0))
        c = self._issue_at(
            self._at(self.ready, xsrcs),
            dst_wait,
            self.pipe["load"],
            self._at(self.mem, after),
            pipes=P_VEC | P_MEM,
        )
        t = transfer(nbytes, x.tile_load_bandwidth, self.m.memory.bandwidth)
        self.pipe["load"] = c + t
        self.busy["load"] += t
        done = c + x.tile_load_latency + self.m.memory.latency + t - 1
        if acc:
            self.acc_ready[tile] = self.acc_mready[tile] = done
        else:
            self.tile_ready[tile] = done
        self.instret["matrix"] += 1
        self._matrix_done(done)
        return c

    def tstore(self, src, nbytes, xsrcs=(), acc=False, token=None):
        """``tstore`` (or ``astore`` with ``acc``) of ``nbytes`` from ``src``.

        Without ``acc_mem`` an ``astore`` is ``amov.t.a`` into the scratch tile, then a tile store.
        """
        x = self.m.ame
        if acc and not x.acc_mem:
            st = self.scratch_tile(src)
            self.mop(
                math.ceil(x.m / x.acc_to_tile_rows_per_cycle),
                x.acc_to_tile_latency,
                accs_read=(src,),
                tiles_written=(st,),
            )
            return self.tstore(st, nbytes, xsrcs, token=token)
        src_wait = (self.acc_ready if acc else self.tile_ready).get(src, 0)
        c = self._issue_at(
            self._at(self.ready, xsrcs),
            src_wait,
            self.pipe["store"],
            pipes=P_VEC | P_MEM,
        )
        t = transfer(nbytes, x.tile_store_bandwidth, self.m.memory.bandwidth)
        self.pipe["store"] = c + t
        self.busy["store"] += t
        done = c + x.tile_store_latency + self.m.memory.latency + t - 1
        table = self.acc_rfree if acc else self.tile_rfree
        table[src] = max(table.get(src, 0), c + t)
        if token is not None:
            self.mem[token] = c + x.tile_store_latency + t - 1
        self.instret["matrix"] += 1
        self._matrix_done(done)
        return c

    def azero(self, acc):
        x = self.m.ame
        c = self._issue_at(
            self.acc_rfree.get(acc, 0), self.acc_ready.get(acc, 0), pipes=P_VEC
        )
        self.acc_ready[acc] = self.acc_mready[acc] = c + x.azero
        self.instret["matrix"] += 1
        return c

    def tshift(self, dst, src, latency=None):
        x = self.m.ame
        lat = x.tshift if latency is None else latency
        c = self._issue_at(
            self.tile_ready.get(src, 0),
            max(self.tile_ready.get(dst, 0), self.tile_rfree.get(dst, 0)),
            pipes=P_VEC,
        )
        self.tile_ready[dst] = c + lat
        self.tile_rfree[src] = max(self.tile_rfree.get(src, 0), c + lat)
        self.instret["matrix"] += 1
        return c

    # -- loops ------------------------------------------------------------------

    def _shift(self, d, since):
        if d <= 0:
            return
        self.t += d
        if self._open is not None and self._open[0] >= since:
            self._open = (self._open[0] + d, self._open[1])
        for table in (
            self.ready,
            self.unit,
            self.vready,
            self.vfirst,
            self.vfirst_of,
            self.tile_ready,
            self.tile_rfree,
            self.acc_ready,
            self.acc_mready,
            self.acc_rfree,
            self.pipe,
            self.mem,
        ):
            for k, v in table.items():
                if v >= since:
                    table[k] = v + d
        if self.lsu >= since:
            self.lsu += d
        if self.vunit >= since:
            self.vunit += d
        if self.vlsu >= since:
            self.vlsu += d

    def loop(self, trips, body, period=1, warm=None, measure=None):
        """Run ``body(i)`` for ``i`` in ``range(trips)``, fast-forwarding the steady state."""
        trips = int(trips)
        if trips <= 0:
            return
        warm = warm if warm is not None else max(3, 2 * period)
        measure = measure if measure is not None else max(4, 2 * period)
        measure = ceil_div(measure, period) * period
        warm = ceil_div(warm, period) * period
        if trips <= warm + 2 * measure:
            for i in range(trips):
                body(i)
            return
        for i in range(warm):
            body(i)
        t0 = self.t
        busy0, ret0 = Counter(self.busy), Counter(self.instret)
        for i in range(warm, warm + measure):
            body(i)
        delta = self.t - t0
        dbusy = self.busy - busy0
        dret = self.instret - ret0
        left = trips - warm - measure
        periods, rest = divmod(left, measure)
        self._shift(periods * delta, t0)
        for k, v in dbusy.items():
            self.busy[k] += v * periods
        for k, v in dret.items():
            self.instret[k] += v * periods
        start = warm + measure + periods * measure
        for i in range(start, start + rest):
            body(i)

    # -- finishing --------------------------------------------------------------

    def drain(self):
        """Wait for everything in flight (the kernels' ``naiad_mx_drain`` / the closing cycle read)."""
        c = self.t
        for table in (self.acc_ready, self.tile_ready):
            for v in table.values():
                c = max(c, v)
        for v in (*self.pipe.values(), self.vlsu, self.vunit):
            c = max(c, v)
        self.t = c
        self._open = None

    def add_cycles(self, n):
        """Serial cycles a formula computes outside the engine (an emitter's own schedule model)."""
        self.t += n
        self._open = None

    @property
    def cycles(self):
        return self.t


#: Which engine the formulas build: "schedule" (in-order issue, the default) or "bound" (throughput).
_MODEL = ContextVar("costmodel_engine", default="schedule")
MODELS = ("schedule", "bound")


def new_engine(machine):
    """An engine of the current model (see :func:`model`)."""
    return BoundEngine(machine) if _MODEL.get() == "bound" else Engine(machine)


@contextmanager
def model(name):
    """``with model("bound"): ...`` evaluates the formulas with :class:`BoundEngine`."""
    if name not in MODELS:
        raise ValueError(f"unknown cost model {name!r}; expected one of {MODELS}")
    token = _MODEL.set(name)
    try:
        yield
    finally:
        _MODEL.reset(token)


def _names(x):
    if x is None:
        return ()
    if isinstance(x, (tuple, list)):
        return x
    return (x,)


class BoundEngine(Engine):
    """The same instruction interface, timed by throughput alone: the roofline a simple compiler uses.

    Every instruction adds its occupancy to the unit it needs -- one issue slot,
    the mma pipe's interval, the tile load / store path's transfer time, the
    vector datapath's and LSU's occupancy, an unpipelined divider's latency --
    and the predicted cycles are the busiest unit's total. Dependencies, result
    latencies and in-order issue are ignored, so this is a lower bound on the
    schedule engine's answer; the gap between the two is what latency and
    program order cost on an in-order, chain-free machine.
    """

    def __init__(self, machine):
        super().__init__(machine)
        self.demand = Counter()

    def _issue_at(self, *times):
        c = self.t
        self.t += 1
        self.demand["issue"] += 1
        return c

    def _use(self, unit, cycles):
        self.demand[unit] += cycles

    def alu(self, n=1, dst=None, srcs=()):
        if n <= 0:
            return
        self.demand["issue"] += n
        self.t += n
        self.instret["scalar"] += n

    def op(self, kind, dst=None, srcs=()):
        c = self._issue_at()
        if kind == "div" and not self.m.scalar.div_pipelined:
            self._use("div", self.lat["div"])
        elif kind in ("fdiv", "fsqrt") and not self.m.scalar.fp_div_pipelined:
            self._use("fdiv", self.lat[kind])
        self.instret["fp" if kind[0] == "f" else "scalar"] += 1
        return c

    def load(self, dst=None, srcs=(), after=(), fp=False):
        self.instret["fp" if fp else "scalar"] += 1
        return self._issue_at()

    def store(self, srcs=(), fp=False):
        self.instret["fp" if fp else "scalar"] += 1
        return self._issue_at()

    def branch(self, taken=True, srcs=()):
        self.instret["scalar"] += 1
        c = self._issue_at()
        if taken and self.m.scalar.branch_taken:
            self.demand["issue"] += self.m.scalar.branch_taken
        return c

    def vset(self, dst=None):
        self.instret["vector"] += 1
        self._issue_at()

    def vop(
        self,
        dst,
        srcs=(),
        vl=None,
        sew=32,
        wide=False,
        kind="arith",
        sdst=None,
        xsrcs=(),
        regs=None,
        sgroups=None,
        dgroup=None,
        streams=True,
    ):
        v = self.m.vector
        if regs is not None:
            occ = max(1, ceil_div(regs * v.vlen, v.datapath_bits))
        else:
            occ = max(1, ceil_div(vl * (2 * sew if wide else sew), v.datapath_bits))
            if kind == "div":
                occ *= v.div_ii
        self._use("vector", occ)
        self.instret["vector"] += 1
        return self._issue_at()

    def _vmem(self, nbytes, elems):
        v = self.m.vector
        t = transfer(nbytes, v.mem_bytes_per_cycle, self.m.memory.bandwidth)
        if elems:
            t = max(t, math.ceil(elems / v.indexed_elements_per_cycle))
        self._use("vector_mem", t)
        self.instret["vector"] += 1
        return self._issue_at()

    def vload(self, dst, nbytes, xsrcs=(), elems=0, after=(), **_):
        return self._vmem(nbytes, elems)

    def vstore(self, srcs, nbytes, xsrcs=(), elems=0, token=None, **_):
        return self._vmem(nbytes, elems)

    def mma(self, acc, a, b, ii, accumulate=True, macs=0):
        self._use("mma", ii)
        self.instret["matrix"] += 1
        return self._issue_at()

    def mop(
        self,
        ii,
        latency=1,
        tiles_read=(),
        tiles_written=(),
        accs_read=(),
        accs_written=(),
        pipe="mma",
    ):
        if pipe:
            self._use(pipe if pipe != "mma" else "mma", ii)
        self.instret["matrix"] += 1
        return self._issue_at()

    def tload(self, tile, nbytes, xsrcs=(), after=(), acc=False):
        x = self.m.ame
        if acc and not x.acc_mem:
            self.mop(math.ceil(x.m / x.tile_to_acc_rows_per_cycle))
        self._use(
            "load", transfer(nbytes, x.tile_load_bandwidth, self.m.memory.bandwidth)
        )
        self.instret["matrix"] += 1
        return self._issue_at()

    def tstore(self, src, nbytes, xsrcs=(), acc=False, token=None):
        x = self.m.ame
        if acc and not x.acc_mem:
            self.mop(math.ceil(x.m / x.acc_to_tile_rows_per_cycle))
        self._use(
            "store", transfer(nbytes, x.tile_store_bandwidth, self.m.memory.bandwidth)
        )
        self.instret["matrix"] += 1
        return self._issue_at()

    def azero(self, acc):
        self.instret["matrix"] += 1
        return self._issue_at()

    def tshift(self, dst, src, latency=None):
        self.instret["matrix"] += 1
        return self._issue_at()

    def loop(self, trips, body, period=1, warm=None, measure=None):
        trips = int(trips)
        if trips <= 0:
            return
        k = max(period, min(trips, 2 * period))
        k = ceil_div(k, period) * period
        if trips <= k:
            for i in range(trips):
                body(i)
            return
        d0, r0, t0 = Counter(self.demand), Counter(self.instret), self.t
        for i in range(k):
            body(i)
        scale = trips / k
        dd = self.demand - d0
        dr = self.instret - r0
        for key, val in dd.items():
            self.demand[key] += val * (scale - 1)
        for key, val in dr.items():
            self.instret[key] += val * (scale - 1)
        self.t += (self.t - t0) * (scale - 1)

    def drain(self):
        pass

    def add_cycles(self, n):
        self.demand["serial"] += n

    @property
    def cycles(self):
        par = max(
            (
                v / self.width if k == "issue" else v
                for k, v in self.demand.items()
                if k != "serial"
            ),
            default=0,
        )
        return int(round(par + self.demand["serial"])) or 1

    @property
    def busy(self):
        return Counter({k: int(v) for k, v in self.demand.items()})

    @busy.setter
    def busy(self, value):
        pass
