"""The VME (Zvt v0.4) mode of the GEMM emitter: ``naiad_gemm.h`` for a vector-coupled unit.

Called by :func:`framework.naiad.emit.gemm.emit_gemm_header` for a design in
``mode: vme``. The API is the AME emitter's driver half: ``naiad_gemm_<in>_<acc>``
(``C = A . B``), ``..._acc`` (``C += A . B``) and, under ``NAIAD_GEMM_IN`` /
``NAIAD_GEMM_ACC``, the unsuffixed ``naiad_gemm``, ``naiad_gemm_acc`` and
``naiad_gemm_in_t`` / ``naiad_gemm_acc_t``. Any m, n, k; row-major, strides in
elements. (There is no ``_ukernel``: a VME micro-kernel needs A as a transposed
panel, which only the driver builds.)

**What a Zvt multiply needs.** ``vtfmm`` (and ``.alt``, ``vtmmu``/``vtmms``) is
``C[tm, tn] += A[tk, tm]^T . B[tk, tn]`` with ``tk <= KMAX`` (1 for 32/64-bit
inputs, 2 for 16-bit, 4 for 8-bit): K-rows of A and B in vector register
groups, one LMUL group per K-row, rows ``8 / KMAX`` registers apart, and C in an
accumulator tile (``E = TE``, or ``TE / 2`` at TEW 64). A row of B is a run of B
in memory, but a K-row of A is a column of A, so the driver transposes A
through the tile state first (the draft's transpose: rows in with ``vtle``,
columns out with ``vtse``, two tiles alternating so the column stores of one
sub-block overlap the row loads of the next).

**The A panel is the register image.** The transposed block row of A is
written in exactly the order a whole-register load (``vl8re<SEW>.v``) puts it
in registers: one 8-register window holds ``S = NP / BM`` K steps of the
``BM`` A operand groups (``NP = 8 / (KMAX * LMUL)`` groups fit a window), or one
step spans ``BM / NP`` windows, so **one scalar-addressed load fetches several
operand groups** (or several K steps of them). B is read in place, one
whole-register load per K-row (or per group) of each step, when its columns
are a multiple of the tile edge E and the edge fills the group (``LMUL * VLEN /
SEW == E``); a block cut by the N edge runs a narrower K loop (BN/2, BN/4, ...,
the same A panel). Otherwise B is first copied (RVV) to a zero-padded buffer
in the same layout. The K loop is a modulo schedule over operand rings: each
load goes after the last multiply that reads its registers' old contents (plus
an optional one-multiply lag, so it does not stall issue on the read), loads
and their pointer advances are separate ``asm volatile`` ops placed ``occ - 1``
to a gap between multiplies (the body's last gap keeps room for the loop
branch), the body is unrolled to the rings' period (literal register numbers
repeat; a power of two, at most ``MAX_KQ`` K elements), and the last body
drops the next body's prefetches.

**The block shape** (``BM x BN`` accumulator tiles, ring depths, load
granularity, multiply order, lag, latency lead, body length) is chosen per
(in, acc) pair by running every candidate through a small model of the
design's timing (the matrix unit's occupancy and operand-read time, the vector
LSU and load latency, one scalar instruction per cycle; one instruction per
cycle when the design is untimed) over a ``ref``-cubed problem, and taking the
fewest modelled cycles (then the fewest instructions). ``block="BMxBN"``
forces a shape.

**Dual issue.** For a design with ``timing.scalar.issue_width: 2`` the two tile-move loops that
are bound by issue rather than by a unit, the A transpose (``vtle`` rows in, ``vtse`` columns out) and
the full-tile C store, are scheduled for the front end instead of one instruction a cycle
(:mod:`framework.naiad.emit.dual_sched`): each tile op needs a pointer advance and an index advance,
and written in program order the advance a tile op waits on sits right behind its producer, so the
second slot stays empty (4 cycles for a row load plus a column store). The loops keep one pointer and
one index register per stream (per C column in the store) and the scheduler interleaves the
independent advances with the tile ops, which pair with them: 3 cycles for the same pair. A zero A
stride (the ones-tile reductions) is tested at run time and drops the row pointer advance, as the
compiler folds it in the single-issue text. Single issue emits the same text as before.

**Edges.** An M edge leaves stale rows in the panel's padding columns, which
only reach C rows that are never stored; a K edge is zeroed in the panel (and
in the padded B copy), so padded K steps add exact zeros; C is stored with
``tm``/``tn`` clipped. K beyond ``KC`` is done in chunks that reload C.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

from framework.matrix import vme
from framework.naiad.emit import dual_sched

#: Format -> (C storage type from dwarf_util.h, bytes); only these pairs are emitted.
C_TYPES = {
    "fp64": ("naiad_fp64_t", 8),
    "fp32": ("naiad_fp32_t", 4),
    "fp16": ("naiad_fp16_t", 2),
    "bf16": ("naiad_bf16_t", 2),
    "int8": ("naiad_int8_t", 1),
    "int32": ("naiad_int32_t", 4),
}
#: Zvt's multiplies and the vme.h macro for each.
_MMA = (
    ("vtfmm.tvv", "vme_vtfmm"),
    ("vtfmm.alt.tvv", "vme_vtfmm_alt"),
    ("vtmmu.tvv", "vme_vtmmu"),
    ("vtmms.tvv", "vme_vtmms"),
)
#: Vector register windows (8 registers each) the K loop's operand rings may use.
WINDOWS = 4
#: K elements per chunk (rounded to the K loop's granularity); C is reloaded between chunks.
KC_TARGET = 256
#: Bytes of the zero-padded B copy (ragged N or K); its column chunk is sized to fit.
BPAD_BYTES = 128 * 1024
#: The most K elements one loop body may take (a power of two): K a multiple of it runs in place.
MAX_KQ = 32
#: The dual-issue C store keeps a pointer and an index register per block column: at most this many.
MAX_DUAL_BN = 4
#: Default reference problem edge for choosing the block shape (the gemm dwarf's largest size).
REF_SIZE = 128


# -- geometry ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """One (in, acc) pair on a VME unit: element widths, KMAX, tile edge, the msetmtype LMUL."""

    in_fmt: str
    acc_fmt: str
    vlen: int
    te: int
    sew: int
    tw: int
    kmax: int
    lmul: int
    mma: str  # the vme.h macro
    alt: int  # msetmtype altfmt

    @property
    def tew(self):
        return self.sew * self.tw

    @property
    def e(self):
        """Accumulator tile edge (elements)."""
        return vme.ete(self.te, self.tew)

    @property
    def eve(self):
        return self.vlen // self.sew

    @property
    def sw(self):
        """Elements an operand group holds (LMUL registers); at least ``e``."""
        return self.lmul * self.eve

    @property
    def rs(self):
        """Registers between an operand's K-rows."""
        return 8 // self.kmax

    @property
    def np(self):
        """Operand groups per K-row slot of a window."""
        return self.rs // self.lmul

    @property
    def vlenb(self):
        return self.vlen // 8

    @property
    def ies(self):
        return self.sew // 8

    @property
    def aes(self):
        return self.tew // 8

    @property
    def ntiles(self):
        """Accumulator tiles at TEW."""
        return vme.NTILES[self.tew]

    @property
    def acc_step(self):
        return vme.tile_step(self.tew)

    @property
    def t_step(self):
        """Tile-number step at TEW = SEW (the transpose's tiles)."""
        return vme.tile_step(self.sew)

    @property
    def inplace_ok(self):
        """B can be read in place: a tile row exactly fills its operand group."""
        return self.sw == self.e

    @classmethod
    def for_pair(cls, spec, in_fmt, acc_fmt):
        if in_fmt not in C_TYPES or acc_fmt not in C_TYPES:
            raise ValueError(f"{in_fmt} -> {acc_fmt}: no C type for the GEMM emitter")
        sew, tew = 8 * C_TYPES[in_fmt][1], 8 * C_TYPES[acc_fmt][1]
        if tew % sew:
            raise ValueError(
                f"{in_fmt} -> {acc_fmt}: accumulator narrower than its inputs"
            )
        tw = tew // sew
        if (sew, tw) not in vme.KMAX:
            raise ValueError(
                f"{in_fmt} -> {acc_fmt}: no Zvt (SEW, TWIDEN) = ({sew}, {tw})"
            )
        found = None
        for name, macro in _MMA:
            for alt in (0, 1):
                row = vme.arith_row(name, sew, tw, alt)
                if (
                    row
                    and row.fmt_a == in_fmt
                    and row.fmt_b == in_fmt
                    and row.fmt_c == acc_fmt
                ):
                    found = found or (macro, alt, row.ext)
        if found is None or not spec.has(found[2]):
            raise ValueError(
                f"{in_fmt} -> {acc_fmt}: the design has no Zvt multiply for this pair"
            )
        for tew_needed in {sew, tew}:
            if tew_needed not in spec.tews or tew_needed > spec.elen:
                raise ValueError(
                    f"{in_fmt} -> {acc_fmt}: needs Zvt moves at TEW {tew_needed}"
                )
        geo = cls(
            in_fmt,
            acc_fmt,
            spec.vlen,
            spec.te,
            sew,
            tw,
            vme.KMAX[(sew, tw)],
            vme.config_lmul(sew, tw, spec.te, spec.vlen),
            found[0],
            found[1],
        )
        if vme.ete(spec.te, sew) != geo.e:
            raise ValueError(
                f"{in_fmt} -> {acc_fmt}: transpose tile edge differs from the accumulator's"
            )
        return geo


def format_pairs(spec):
    """The design's (in, acc) pairs the emitter can type (OFP8 has no C type)."""
    return [p for p in spec.format_pairs if p[0] in C_TYPES and p[1] in C_TYPES]


# -- timing model for the shape choice -----------------------------------------------------------


@dataclass(frozen=True)
class Costs:
    """What the shape choice models: ``None`` everywhere = untimed (one instruction per cycle)."""

    timed: bool = False
    mma_occ: int = 1
    mma_read: int = 1
    mma_drain: int = 0  # pipeline_depth - 1 (to the tile being readable)
    forwarding: bool = True
    mem_bw: float = 32
    load_latency: int = 0  # mem_startup + the program memory's latency

    def load_time(self, nbytes):
        return max(1, math.ceil(nbytes / self.mem_bw)) if self.timed else 1


def costs_for(design, geo):
    timing = getattr(design, "timing", None)
    mt = getattr(timing, "matrix", None)
    vt = getattr(timing, "vector", None)
    if mt is None or vt is None or not hasattr(mt, "mma_occupancy"):
        return Costs()
    macs = geo.e * geo.e * geo.kmax
    nbytes = geo.kmax * 2 * geo.e * geo.ies
    occ, _, t_read = mt.mma_occupancy(macs, nbytes, geo.in_fmt, vt.datapath_bits)
    prog = design.program_memory
    latency = vt.mem_startup + timing.region(prog.name, prog.type).latency
    return Costs(
        True,
        occ,
        t_read,
        mt.pipeline_depth - 1,
        mt.acc_forwarding,
        vt.mem_bytes_per_cycle,
        latency,
    )


# -- the K-loop plan ---------------------------------------------------------------------------


class _Ring:
    """Where one operand's groups live: ``depth`` windows from ``wbase``, ``nblk`` groups a step."""

    def __init__(self, geo, nblk, depth, wbase):
        self.geo, self.nblk, self.depth, self.wbase = geo, nblk, depth, wbase
        npw = geo.np
        self.s = npw // nblk if nblk <= npw else 1  # steps per window
        self.w = nblk // npw if nblk > npw else 1  # windows per step
        self.period = (
            depth * self.s if self.w == 1 else depth // math.gcd(depth, self.w)
        )

    def window(self, u, b):
        if self.w == 1:
            return self.wbase + (u // self.s) % self.depth
        return self.wbase + (u * self.w + b // self.geo.np) % self.depth

    def pos(self, u, b):
        if self.w == 1:
            return (u % self.s) * self.nblk + b
        return b % self.geo.np

    def reg(self, u, b, r=0):
        return 8 * self.window(u, b) + r * self.geo.rs + self.pos(u, b) * self.geo.lmul

    def regs(self, u, b):
        g = self.geo
        return [self.reg(u, b, r) + x for r in range(g.kmax) for x in range(g.lmul)]


@dataclass(frozen=True)
class Load:
    stream: str  # pointer variable
    order: int  # issue order within the stream
    reg: int
    nregs: int
    adv: str  # C expression added to the stream's pointer after the load
    items: tuple  # (unit, block) operand groups it fills
    operand: str  # "A" or "B"


@dataclass(frozen=True)
class Plan:
    """One candidate K loop: block shape, load granularity, ring depths, multiply order."""

    geo: Geometry
    bm: int
    bn: int
    a_gran: str  # "window" | "block"
    b_gran: str  # "row" | "block"
    da: int
    db: int
    order: str  # "ij" (A block outer) | "ji"
    #: Multiplies a load waits after the last reader of its registers (1: no WAR stall at issue).
    lag: int = 0
    #: Loop bodies of the rings' period unrolled together.
    umul: int = 1

    @property
    def mb(self):
        return self.bm * self.geo.e

    @property
    def nb(self):
        return self.bn * self.geo.e

    @property
    def mps(self):
        return self.bm * self.bn

    def rings(self):
        return _Ring(self.geo, self.bm, self.da, 0), _Ring(
            self.geo, self.bn, self.db, self.da
        )

    @property
    def u(self):
        ra, rb = self.rings()
        u = ra.period * rb.period // math.gcd(ra.period, rb.period)
        while u * self.mps < 4:
            u *= 2
        return u * self.umul

    @property
    def kq(self):
        """K elements per loop body."""
        return self.geo.kmax * self.u

    @property
    def a_period(self):
        """``(K elements, bytes)`` of one repeat of the A panel's layout."""
        ra, _ = self.rings()
        g = self.geo
        if ra.w == 1:
            return g.kmax * ra.s, 8 * g.vlenb
        return g.kmax, 8 * g.vlenb * ra.w

    def a_offset(self, k, b):
        """Byte offset of K-row ``k``, A group ``b`` in the panel (the register image)."""
        g = self.geo
        ra, _ = self.rings()
        pk, wb = self.a_period
        kk = k % pk
        u, r = kk // g.kmax, kk % g.kmax
        if ra.w == 1:
            pos, win = (u % ra.s) * self.bm + b, 0
        else:
            pos, win = b % g.np, b // g.np
        return (k // pk) * wb + win * 8 * g.vlenb + (r * g.rs + pos * g.lmul) * g.vlenb

    def valid(self):
        g = self.geo
        if self.mps > g.ntiles:
            return False
        if self.a_gran == "block" and g.kmax != 1:
            return False
        ra, rb = self.rings()
        if self.da < ra.w or self.db < rb.w or self.da + self.db > WINDOWS:
            return False
        pk, _ = self.a_period
        if g.e % pk:
            return False
        # a body must divide the K extents that matter (powers of two from MAX_KQ up), else
        # those take the padded path
        kq = self.kq
        return kq & (kq - 1) == 0 and kq <= MAX_KQ

    # -- the streams of loads, in issue order per stream

    def loads(self, units):
        g = self.geo
        ra, rb = self.rings()
        out = []
        # A: the panel, sequential (the register image)
        n = 0
        if self.a_gran == "window":
            if ra.w == 1:
                for c in range(units // ra.s):
                    items = tuple(
                        (c * ra.s + s, b) for s in range(ra.s) for b in range(self.bm)
                    )
                    out.append(
                        Load(
                            "pa",
                            n,
                            8 * ra.window(c * ra.s, 0),
                            8,
                            str(8 * g.vlenb),
                            items,
                            "A",
                        )
                    )
                    n += 1
            else:
                for u in range(units):
                    for h in range(ra.w):
                        items = tuple((u, b) for b in range(h * g.np, (h + 1) * g.np))
                        out.append(
                            Load(
                                "pa",
                                n,
                                8 * ra.window(u, h * g.np),
                                8,
                                str(8 * g.vlenb),
                                items,
                                "A",
                            )
                        )
                        n += 1
        else:  # block (KMAX 1): one group per load
            for u in range(units):
                for b in range(self.bm):
                    out.append(
                        Load(
                            "pa",
                            n,
                            ra.reg(u, b),
                            g.lmul,
                            str(g.lmul * g.vlenb),
                            ((u, b),),
                            "A",
                        )
                    )
                    n += 1
        # B: in rows of the natural layout (one stream per K-row of a step)
        run = g.sw * g.ies  # bytes per group
        for r in range(g.kmax):
            n = 0
            stream = f"pb{r}"
            for u in range(units):
                if self.b_gran == "row" and rb.w == 1:
                    pieces = [
                        (tuple(range(self.bn)), rb.reg(u, 0, r), self.bn * g.lmul)
                    ]
                elif self.b_gran == "row":
                    pieces = [
                        (
                            tuple(range(h * g.np, (h + 1) * g.np)),
                            rb.reg(u, h * g.np, r),
                            g.rs,
                        )
                        for h in range(rb.w)
                    ]
                else:
                    pieces = [((b,), rb.reg(u, b, r), g.lmul) for b in range(self.bn)]
                for x, (blocks, reg, nregs) in enumerate(pieces):
                    last = x == len(pieces) - 1
                    used = sum(len(p[0]) for p in pieces[:-1]) * run
                    adv = (
                        "bstep"
                        if last and used == 0
                        else (f"bstep_{used}" if last else str(len(blocks) * run))
                    )
                    out.append(
                        Load(
                            stream,
                            n,
                            reg,
                            nregs,
                            adv,
                            tuple((u, b) for b in blocks),
                            "B",
                        )
                    )
                    n += 1
        return out

    def mmas(self, units):
        """``(unit, i, j)`` in issue order."""
        out = []
        for u in range(units):
            if self.order == "ij":
                out += [(u, i, j) for i in range(self.bm) for j in range(self.bn)]
            else:
                out += [(u, i, j) for j in range(self.bn) for i in range(self.bm)]
        return out


@dataclass(frozen=True)
class Schedule:
    """The K loop as emitted: prologue ops, then per body position the mma and the ops after it.

    An op is ``(kind, load, next_body)``: ``kind`` "load" (the whole-register load) or
    "adv" (its pointer advance, placed separately so it can fill a later issue slot);
    ``next_body`` marks the prefetches the last body drops.
    """

    plan: Plan
    cap: int  # non-mma instructions a gap between two multiplies takes without delaying the next
    lead: int  # multiplies a load keeps ahead of its first use when it slides to a free gap
    prologue: tuple  # ops for body 0, issued before the loop
    body: tuple  # per mma position: ((u, i, j), (op, ...))


def schedule(plan, cap=1, lead=0):
    """Place every load after the last reader of its registers' old contents (plus ``plan.lag``).

    Loads and pointer advances fill the gaps between multiplies ``cap`` at a time
    (the body's last gap keeps two slots for the loop's own increment and branch);
    a load slides to a later free gap only while it stays ``lead`` multiplies ahead of
    its first use (its latency), else it overfills its earliest gap. Each stream's
    order is kept. Returns None when the rings are too shallow.
    """
    u = plan.u
    periods = 6
    units = periods * u
    mmas = plan.mmas(units)
    loads = plan.loads(units)
    readers = {}
    for idx, (uu, i, j) in enumerate(mmas):
        readers.setdefault(("A", uu, i), []).append(idx)
        readers.setdefault(("B", uu, j), []).append(idx)
    info = []
    for ld in loads:
        rs = [x for it in ld.items for x in readers[(ld.operand, *it)]]
        info.append((ld, min(rs), max(rs)))
    # previous occupant of each register: the last earlier load (by first use) writing it
    by_first = sorted(
        range(len(info)),
        key=lambda x: (info[x][1], info[x][0].stream, info[x][0].order),
    )
    last_use_of_reg = {}
    earliest = {}
    for x in by_first:
        ld, first, last = info[x]
        prev = -1
        for r in range(ld.reg, ld.reg + ld.nregs):
            prev = max(prev, last_use_of_reg.get(r, -1))
        if prev >= first:
            return None  # the ring is too shallow: the old contents are still read after this use
        earliest[x] = min(prev + plan.lag, first - 1) if prev >= 0 else prev
        for r in range(ld.reg, ld.reg + ld.nregs):
            last_use_of_reg[r] = last
    pbody = u * plan.mps

    def capacity(g):
        return max(0, cap - 2) if g % pbody == pbody - 1 else cap

    fill = {}

    def take(lo, hi):
        """The first gap in [lo, hi] with a free slot, else lo (over capacity: it costs a cycle there)."""
        g = lo
        while g <= hi and fill.get(g, 0) >= capacity(g):
            g += 1
        g = g if g <= hi else lo
        fill[g] = fill.get(g, 0) + 1
        return g

    gap, adv_gap = {}, {}
    stream_last = {}
    for x in by_first:
        ld, first, _ = info[x]
        lo = max(earliest[x], stream_last.get(ld.stream, -1))
        if lo > first - 1:
            return None
        gap[x] = take(lo, max(lo, first - 1 - lead))
        adv_gap[x] = take(
            gap[x], min(gap[x] + 2, first - 1) if first - 1 > gap[x] else gap[x] + 2
        )
        stream_last[ld.stream] = adv_gap[x]

    # the steady state: the first period whose placement the next one repeats, shifted by a body
    def period_of(x):
        return info[x][0].items[0][0] // u

    per = {}
    for x in gap:
        ld = info[x][0]
        per.setdefault(period_of(x), {})[
            (ld.stream, ld.order % _orders(loads, ld.stream, u, units))
        ] = (
            gap[x],
            adv_gap[x],
        )
    p = next(
        (
            p
            for p in range(1, periods - 2)
            if per.get(p)
            and per.get(p).keys() == per.get(p + 1, {}).keys()
            and all(
                per[p + 1][key] == (a + pbody, b + pbody)
                for key, (a, b) in per[p].items()
            )
        ),
        None,
    )
    if p is None:
        return None
    base = p * pbody
    body = []
    for q in range(pbody):
        uu, i, j = mmas[base + q]
        body.append(((uu - p * u, i, j), []))
    placed = []
    for x in gap:
        ld, _, _ = info[x]
        if period_of(x) != p:
            continue
        ld = Load(
            ld.stream,
            ld.order,
            ld.reg,
            ld.nregs,
            ld.adv,
            tuple((it[0] - p * u, it[1]) for it in ld.items),
            ld.operand,
        )
        for kind, g in (("load", gap[x]), ("adv", adv_gap[x])):
            rel = g - base
            if not -pbody <= rel < pbody:
                return None
            nxt = rel < 0
            # within a position, this body's ops go before the next body's (they come first in each stream)
            placed.append(
                (
                    rel + pbody if nxt else rel,
                    nxt,
                    ld.stream,
                    ld.order,
                    kind == "adv",
                    (kind, ld, nxt),
                )
            )
    placed.sort(key=lambda t: t[:5])
    for t in placed:
        body[t[0]][1].append(t[5])
    prologue = [
        t[5] for t in sorted(placed, key=lambda t: (t[0], t[2], t[3], t[4])) if t[1]
    ]
    return Schedule(
        plan, cap, lead, tuple(prologue), tuple((m, tuple(ops)) for m, ops in body)
    )


def _orders(loads, stream, u, units):
    """Loads of ``stream`` per period of ``u`` units (the stream's order modulo it repeats)."""
    return sum(1 for ld in loads if ld.stream == stream) * u // units


def _simulate(sched, costs, steps, init_ops, store_ops):
    """Modelled ``(cycles, instructions)`` of one block: prologue, ``steps / U`` bodies, init and store."""
    plan = sched.plan
    g = plan.geo
    ra, rb = plan.rings()
    nb = steps // plan.u
    if not costs.timed:
        n = len(sched.prologue) + init_ops + store_ops + 4
        per = sum(1 + len(ops) for _, ops in sched.body) + 2
        last = sum(1 + sum(1 for op in ops if not op[2]) for _, ops in sched.body)
        n += (nb - 1) * per + last
        return n, n
    cyc = 0
    instr = 0
    ready = [0] * 32
    read_until = [0] * 32
    lsu = 0
    mma_free = 0
    tile_ready = {}

    def alu(k=1):
        nonlocal cyc, instr
        cyc += k
        instr += k

    def op(o):
        nonlocal cyc, lsu, instr
        kind, ld, _ = o
        if kind == "adv":
            alu()
            return
        regs = range(ld.reg, ld.reg + ld.nregs)
        t = max([cyc, lsu] + [read_until[r] for r in regs])
        dt = costs.load_time(ld.nregs * g.vlenb)
        lsu = t + dt
        for r in regs:
            ready[r] = t + costs.load_latency + dt
        cyc = t + 1
        instr += 1

    def mma(item):
        nonlocal cyc, mma_free, instr
        uu, i, j = item
        regs = ra.regs(uu, i) + rb.regs(uu, j)
        tile = (i, j)
        t = max([cyc, mma_free] + [ready[r] for r in regs])
        if not costs.forwarding:
            t = max(t, tile_ready.get(tile, 0))
        mma_free = t + costs.mma_occ
        tile_ready[tile] = mma_free + costs.mma_drain
        for r in regs:
            read_until[r] = t + costs.mma_read
        cyc = t + 1
        instr += 1

    for o in sched.prologue:
        op(o)
    alu(init_ops + 4)
    for b in range(nb):
        for item, ops in sched.body:
            # registers are reused every U steps: the unit numbers only matter modulo the rings
            mma((item[0] + b * plan.u, item[1], item[2]))
            for o in ops:
                if not (o[2] and b == nb - 1):
                    op(o)
        if b < nb - 1:
            alu(2)
    cyc = max(cyc, mma_free + costs.mma_drain)
    alu(store_ops)
    return cyc, instr


def _ref_cost(sched, costs, ref):
    plan = sched.plan
    g = plan.geo
    steps = max(plan.u, (ref // g.kmax) // plan.u * plan.u)
    blocks = math.ceil(ref / plan.mb) * math.ceil(ref / plan.nb)
    init_ops = plan.mps + 6
    store_ops = 3 * plan.mps * g.e + 4 * plan.bm
    nb = steps // plan.u
    if nb <= 3:
        cyc, instr = _simulate(sched, costs, steps, init_ops, store_ops)
    else:  # the bodies are periodic: extrapolate from two and three of them
        c2, i2 = _simulate(sched, costs, 2 * plan.u, init_ops, store_ops)
        c3, i3 = _simulate(sched, costs, 3 * plan.u, init_ops, store_ops)
        cyc, instr = c2 + (nb - 2) * (c3 - c2), i2 + (nb - 2) * (i3 - i2)
    return blocks * cyc, blocks * instr


def candidates(geo, block=None):
    shapes = []
    p = [1, 2, 4, 8, 16]
    for bm in p:
        for bn in p:
            if bm * bn <= geo.ntiles and (block is None or (bm, bn) == tuple(block)):
                shapes.append((bm, bn))
    out = []
    for bm, bn in shapes:
        for ag in ("window", "block"):
            for bg in ("row", "block"):
                for da in range(1, WINDOWS + 1):
                    for db in range(1, WINDOWS + 1 - da):
                        for order in ("ij", "ji"):
                            for lag in (0, 1):
                                plan = Plan(geo, bm, bn, ag, bg, da, db, order, lag)
                                if plan.valid():
                                    out.append(plan)
    return out


@lru_cache(maxsize=256)
def choose(geo, costs, block=None, ref=REF_SIZE):
    """The best :class:`Schedule` for ``geo`` under ``costs`` (see the module docstring)."""

    def key(sched):
        plan = sched.plan
        cyc, instr = _ref_cost(sched, costs, ref)
        return (
            cyc,
            instr,
            -plan.mps,
            abs(plan.bm - plan.bn),
            plan.umul,
            plan.da + plan.db,
            plan.lag,
            plan.order,
            plan.a_gran,
            plan.b_gran,
        )

    cap = max(1, costs.mma_occ - 1) if costs.timed else 1
    lead = 0
    if costs.timed:  # multiplies that cover a whole-window load's latency
        lead = math.ceil(
            (costs.load_latency + costs.load_time(8 * geo.vlenb)) / costs.mma_occ
        )
    leads = sorted({0, lead})
    scheds = (schedule(p, cap, ld) for p in candidates(geo, block) for ld in leads)
    ranked = sorted(((key(s), s) for s in scheds if s is not None), key=lambda t: t[0])
    best = ranked[0] if ranked else None
    # then the best few with longer bodies (less loop overhead, more room to place loads)
    for _, sched in ranked[:8]:
        for umul in (2, 4):
            plan = sched.plan
            plan = Plan(
                geo,
                plan.bm,
                plan.bn,
                plan.a_gran,
                plan.b_gran,
                plan.da,
                plan.db,
                plan.order,
                plan.lag,
                umul,
            )
            longer = schedule(plan, cap, sched.lead) if plan.valid() else None
            if longer is not None and key(longer) < best[0]:
                best = (key(longer), longer)
    if best is None:
        raise ValueError(
            f"{geo.in_fmt} -> {geo.acc_fmt}: no VME blocking fits"
            + (f" (block {block[0]}x{block[1]})" if block else "")
        )
    return best[1]


# -- C emission ---------------------------------------------------------------------------------


def _lcm(a, b):
    return a * b // math.gcd(a, b)


class _PairEmitter:
    def __init__(self, sched, narrow=(), costs=None, dual=False):
        self.main, self.narrow = sched, tuple(narrow)
        #: dual-issue loops (``timing.scalar.issue_width: 2``) and the costs their schedules assume
        self.dual, self.costs = dual, costs or Costs()
        self.sc, self.pl, self.sfx = sched, sched.plan, ""
        self.g = sched.plan.geo
        g = self.g
        self.p = f"naiad_gemm_{g.in_fmt}_{g.acc_fmt}"
        self.it = C_TYPES[g.in_fmt][0]
        self.at = C_TYPES[g.acc_fmt][0]
        self.kl = _lcm(g.e, self.pl.kq)  # K granularity of a padded chunk
        self.kc = max(self.kl, KC_TARGET // self.kl * self.kl)

    def tile(self, i, j):
        return (i * self.pl.bn + j) * self.g.acc_step

    def _op_stmt(self, op):
        """A load or a pointer advance, each its own volatile asm, so both stay where the schedule put them."""
        g = self.g
        kind, ld, _ = op
        if kind == "load":
            return f"NAIAD_GEMM_VLRE({ld.nregs}, {g.sew}, {ld.reg}, {ld.stream});"
        if ld.adv.lstrip("-").isdigit() and -2048 <= int(ld.adv) < 2048:
            return f"NAIAD_GEMM_ADDI({ld.stream}, {ld.adv});"
        return f"NAIAD_GEMM_ADD({ld.stream}, {self._adv_var(ld.adv)});"

    def _adv_var(self, adv):
        if adv.lstrip("-").isdigit():
            return f"adv_{adv.replace('-', 'm')}"
        return adv

    def _adv_decls(self):
        """``const long`` declarations for the register-held pointer advances the K loop uses."""
        out = []
        seen = set()
        lds = [op[1] for op in self.sc.prologue] + [
            op[1] for _, ops in self.sc.body for op in ops
        ]
        for ld in lds:
            if ld.adv in seen:
                continue
            seen.add(ld.adv)
            if ld.adv.startswith("bstep_"):
                out.append(f"const long {ld.adv} = bstep - {ld.adv[6:]};")
            elif ld.adv.lstrip("-").isdigit() and not -2048 <= int(ld.adv) < 2048:
                out.append(f"const long {self._adv_var(ld.adv)} = {ld.adv};")
        return out

    def _mma_stmt(self, item):
        ra, rb = self.pl.rings()
        u, i, j = item
        return f"{self.g.mma}({self.tile(i, j)}, {ra.reg(u, i)}, {rb.reg(u, j)});"

    def emit(self):
        g, pl, p = self.g, self.pl, self.p
        it, at = self.it, self.at
        ra, rb = pl.rings()
        pk, wb = pl.a_period
        L = []
        w = L.append
        w(f"/* ---- {g.in_fmt} in, {g.acc_fmt} accumulate ---- */")
        w(f"#if defined(NAIAD_VME_EXT_{_ext_macro(g)})")
        w(f"#define NAIAD_GEMM_HAS_{g.in_fmt.upper()}_{g.acc_fmt.upper()} 1")
        w(f"typedef {it} {p}_in_t;")
        w(f"typedef {at} {p}_acc_t;")
        w(
            f"/* SEW {g.sew}, TWIDEN {g.tw} (TEW {g.tew}), KMAX {g.kmax}, LMUL {g.lmul}, tile edge E = {g.e}; "
            f"block {pl.bm} x {pl.bn} tiles = {pl.mb} x {pl.nb} of C;"
        )
        w(
            f"   A: {'one load per window' if pl.a_gran == 'window' else 'one load per group'}, "
            f"{ra.s if ra.w == 1 else 1} step(s) a window, {pl.da} window(s); "
            f"B: {'one load per K-row' if pl.b_gran == 'row' else 'one load per group'}, {pl.db} window(s); "
            f"multiplies {'A-block' if pl.order == 'ij' else 'B-block'} outer; {pl.u} K step(s) a loop body. */"
        )
        w(f"#define {p.upper()}_MB {pl.mb}")
        w(f"#define {p.upper()}_NB {pl.nb}")
        P = p.upper()
        kl, grp = self.kl, pl.bn * g.sw
        w(
            "/* K elements per chunk (C is reloaded between chunks); NAIAD_GEMM_MAX_K / _MAX_N, when the"
        )
        w(
            "   includer defines them, bound the problem and shrink the static panels to fit. */"
        )
        w("#ifdef NAIAD_GEMM_MAX_K")
        w(
            f"#define {P}_KC ((NAIAD_GEMM_MAX_K + {kl - 1}) / {kl} * {kl} < {self.kc} ? "
            f"(NAIAD_GEMM_MAX_K + {kl - 1}) / {kl} * {kl} : {self.kc})"
        )
        w("#else")
        w(f"#define {P}_KC {self.kc}")
        w("#endif")
        w("#ifdef NAIAD_GEMM_MAX_N")
        w(f"#define {P}_NBK ((NAIAD_GEMM_MAX_N + {pl.nb - 1}) / {pl.nb})")
        w("#else")
        w(
            f"#define {P}_NBK ({max(1, BPAD_BYTES // g.ies // (self.kc * grp))} * {self.kc} / {P}_KC)"
        )
        w("#endif")
        w(f"static char {p}_apanel[{P}_KC / {pk} * {wb}] NAIAD_GEMM_STAGE;")
        w(f"static {it} {p}_bpad[{P}_KC * {P}_NBK * {grp}] NAIAD_GEMM_STAGE;")
        bpad = f"({P}_KC * {P}_NBK * {grp})"
        w("")
        self._emit_aoff(w)
        self._emit_pack_a(w)
        self._emit_pack_b(w, bpad)
        self._emit_init_store(w)
        self._emit_block(w)
        for v in self.narrow:  # the N edge: the same A panel, fewer B groups
            self.sc, self.pl, self.sfx = v, v.plan, f"_n{v.plan.bn}"
            self._emit_init_store(w)
            self._emit_block(w)
        self.sc, self.pl, self.sfx = self.main, self.main.plan, ""
        self._emit_run(w, bpad)
        w(f"#endif /* {g.in_fmt} x {g.acc_fmt} */")
        w("")
        return L

    # A panel offsets ---------------------------------------------------------------------------

    def _emit_aoff(self, w):
        g, pl, p = self.g, self.pl, self.p
        ra, _ = pl.rings()
        pk, wb = pl.a_period
        w(
            f"/* Byte offset of K-row k, A group b in the panel (the order vl8re{g.sew} fills registers). */"
        )
        w(f"NAIAD_GEMM_INLINE long {p}_aoff(long k, long b) {{")
        w(f"    const long kk = k % {pk}, u = kk / {g.kmax}, r = kk % {g.kmax};")
        if ra.w == 1:
            w(f"    const long pos = (u % {ra.s}) * {pl.bm} + b, win = 0;")
        else:
            w(f"    const long pos = b % {g.np}, win = b / {g.np};")
        w(
            f"    return (k / {pk}) * {wb} + win * {8 * g.vlenb} + (r * {g.rs} + pos * {g.lmul}) * {g.vlenb};"
        )
        w("}")
        w("")

    # transpose -----------------------------------------------------------------------------------

    def _rows_cols(self, src, dst, tl, ts, rows, cols, offset=2):
        """Unrolled row loads (``rows``) interleaved with column stores (``cols``), stores ``offset`` behind."""
        g, pl = self.g, self.pl
        e = g.e
        deltas = [pl.a_offset(c + 1, 0) - pl.a_offset(c, 0) for c in range(e)]
        if self.dual and all(-2048 <= d < 2048 for d in deltas):
            return self._rows_cols_dual(src, dst, tl, ts, rows, cols, offset, deltas)
        out = []
        r_seq = (
            [
                f"vme_vtle({g.sew}, s_, tl_ + {r}); s_ += la; NAIAD_GEMM_OPAQUE(s_);"
                for r in range(e)
            ]
            if rows
            else []
        )
        c_seq = (
            [
                f"vme_vtse({g.sew}, d_, ts_ + {c}); d_ += {deltas[c]}; NAIAD_GEMM_OPAQUE(d_);"
                for c in range(e)
            ]
            if cols
            else []
        )
        if rows and cols:
            out += r_seq[:offset]
            for x in range(e):
                if x + offset < e:
                    out.append(r_seq[x + offset])
                out.append(c_seq[x])
        else:
            out = r_seq + c_seq
        head = ["{"]
        if rows:
            head.append(f"    const char *s_ = {src};")
            head.append(f"    unsigned long tl_ = {tl};")
            head.append("    NAIAD_GEMM_OPAQUE(tl_);")
        if cols:
            head.append(f"    char *d_ = {dst};")
            head.append(f"    unsigned long ts_ = {ts};")
            head.append("    NAIAD_GEMM_OPAQUE(ts_);")
        return head + [f"    {s}" for s in out] + ["}"]

    def _tile_gap(self, nbytes):
        """Cycles the load / store unit holds between two tile ops of ``nbytes`` (1 untimed)."""
        return self.costs.load_time(nbytes)

    def _rows_cols_dual(self, src, dst, tl, ts, rows, cols, offset, deltas):
        """:meth:`_rows_cols` for the dual-issue front end: the same ops in the same interleave, as
        explicit ops (tile op, pointer advance, index advance) ordered by :func:`dual_sched.schedule`.

        A row stride of zero (the ones-tile reductions read one A row for every tile row) is a run-time
        test here, and the rows then need no pointer advance at all: the compiler folded it away in the
        single-issue text when the stride was a constant, and the explicit advance would not fold.
        """
        g = self.g
        e = g.e
        gap = self._tile_gap(e * g.ies)

        def body(zero_la):
            r_seq, c_seq = [], []
            for r in range(e):
                seq = [
                    dual_sched.Op(
                        f"vme_vtle({g.sew}, s_, ti_);", ("s_", "ti_"), (), True
                    )
                ]
                if r < e - 1:
                    if not zero_la:
                        seq.append(
                            dual_sched.Op(
                                "NAIAD_GEMM_ADD(s_, la);", ("s_", "la"), ("s_",)
                            )
                        )
                    seq.append(
                        dual_sched.Op("NAIAD_GEMM_ADDI(ti_, 1);", ("ti_",), ("ti_",))
                    )
                r_seq.append(seq)
            for c in range(e):
                seq = [
                    dual_sched.Op(
                        f"vme_vtse({g.sew}, d_, ts_);", ("d_", "ts_"), (), True
                    )
                ]
                if c < e - 1:
                    seq += [
                        dual_sched.Op(
                            f"NAIAD_GEMM_ADDI(d_, {deltas[c]});", ("d_",), ("d_",)
                        ),
                        dual_sched.Op("NAIAD_GEMM_ADDI(ts_, 1);", ("ts_",), ("ts_",)),
                    ]
                c_seq.append(seq)
            ops = []
            if rows and cols:
                for r in range(min(offset, e)):
                    ops += r_seq[r]
                for x in range(e):
                    if x + offset < e:
                        ops += r_seq[x + offset]
                    ops += c_seq[x]
            else:
                for seq in (r_seq if rows else []) + (c_seq if cols else []):
                    ops += seq
            return [o.text for o in dual_sched.schedule(ops, vec_gap=gap)[0]]

        head = ["{"]
        if rows:
            head += [
                f"    const char *s_ = {src};",
                f"    unsigned long ti_ = {tl};",
                "    NAIAD_GEMM_OPAQUE(ti_);",
            ]
        if cols:
            head += [
                f"    char *d_ = {dst};",
                f"    unsigned long ts_ = {ts};",
                "    NAIAD_GEMM_OPAQUE(ts_);",
            ]
        if not rows:
            return head + [f"    {t}" for t in body(False)] + ["}"]
        return (
            head
            + ["    if (la == 0) {"]
            + [f"        {t}" for t in body(True)]
            + ["    } else {"]
            + [f"        {t}" for t in body(False)]
            + ["    }", "}"]
        )

    def _emit_pack_a(self, w):
        g, pl, p = self.g, self.pl, self.p
        e, bm = g.e, pl.bm
        pk, wb = pl.a_period
        it = self.it
        t1 = g.t_step
        w(
            "/* A[MB x kq*E] (full) -> the panel: E x E sub-blocks through tiles mt0 / mt1 at TEW = SEW,"
        )
        w(
            "   rows in (vtle), columns out (vtse); one sub-block's columns go out under the next one's rows. */"
        )
        w(
            f"static __attribute__((noinline, unused)) void {p}_pack_a_full(const {it} *A, long lda, long kq) {{"
        )
        w(f"    const long la = lda * {g.ies};")
        w(f"    const long nq = kq * {bm};")
        w(f"    vme_msetmtypei({g.sew}, 0, 1);")
        w(f"    vme_msettm({e});")
        w(f"    vme_msettn({e});")
        src = f"(const char *)(A + (q % {bm}) * {e} * lda + (q / {bm}) * {e})"
        dst = f"{p}_apanel + (q / {bm}) * {e // pk * wb} + {p}_aoff(0, q % {bm})"
        srcm = src.replace("q", "(q - 1)")
        dstm = dst.replace("q", "(q - 1)")
        tl = f"(q & 1) ? {t1}ul << 27 : 0ul"
        ts = f"((q - 1) & 1) ? ({t1}ul << 27) | (1ul << 24) : (1ul << 24)"
        w("    {")
        w("        const long q = 0;")
        for line in self._rows_cols(src, None, tl, None, True, False):
            w(f"        {line}")
        w("    }")
        w("    for (long q = 1; q < nq; q++) {")
        for line in self._rows_cols(src, dstm, tl, ts, True, True):
            w(f"        {line}")
        w("    }")
        w("    {")
        w("        const long q = nq;")
        for line in self._rows_cols(srcm, dstm, None, ts, False, True):
            w(f"        {line}")
        w("    }")
        w("}")
        w("")
        w(
            "/* The general case: mv valid rows, kc valid K of kpad (a multiple of E); padded K is zeroed. */"
        )
        w(
            f"static __attribute__((noinline, unused)) void {p}_pack_a_edge(const {it} *A, long lda, long mv, "
            "long kc, long kpad) {"
        )
        w(f"    vme_msetmtypei({g.sew}, 0, 1);")
        w(f"    vme_msettm({e});")
        w(f"    for (long k0 = 0; k0 < kpad; k0 += {e}) {{")
        w(f"        for (long b = 0; b < {bm}; b++) {{")
        w(f"            long rows = mv - b * {e}, kv = kc - k0;")
        w("            if (rows <= 0)")
        w("                continue; /* only C rows past m would read it */")
        w(f"            rows = rows > {e} ? {e} : rows;")
        w(f"            kv = kv < 0 ? 0 : kv > {e} ? {e} : kv;")
        w(f"            if (kv < {e}) {{")
        w(f"                vme_msettn({e});")
        w("                vme_vtzero(0);")
        w("            }")
        w("            if (kv > 0) {")
        w("                vme_msettn(kv);")
        w("                for (long r = 0; r < rows; r++)")
        w(
            f"                    vme_vtle({g.sew}, &A[(b * {e} + r) * lda + k0], VME_TSS(0, 0, r));"
        )
        w("            }")
        w(f"            vme_msettn({e});")
        w(f"            for (long c = 0; c < {e}; c++)")
        w(
            f"                vme_vtse({g.sew}, {p}_apanel + {p}_aoff(k0 + c, b), VME_TSS(0, 1, c));"
        )
        w("        }")
        w("    }")
        w("}")
        w("")

    def _emit_pack_b(self, w, bpad):
        g, pl, p = self.g, self.pl, self.p
        it = self.it
        w(
            "/* B[kc x nc] -> a zero-padded copy: kpad rows of nbk blocks of BN groups of SW elements"
        )
        w(
            "   (E valid), the layout the K loop reads B in. RVV copies (outside the matrix region:"
        )
        w(
            "   the vsetvl they run clears mtype, which every matrix phase sets again). */"
        )
        w(
            f"static __attribute__((noinline, unused)) void {p}_pack_b(const {it} *B, long ldb, long kc, long kpad, "
            "long nc, long nbk) {"
        )
        u = f"u{g.sew}"
        vt = f"vuint{g.sew}m8_t"
        w(f"    const long ncp = nbk * {pl.bn * g.sw};")
        w(
            f"    const {vt} z = __riscv_vmv_v_x_{u}m8(0, __riscv_vsetvlmax_e{g.sew}m8());"
        )
        w("    for (long k = 0; k < kpad; k++) {")
        w(f"        uint{g.sew}_t *d = (uint{g.sew}_t *)({p}_bpad + k * ncp);")
        w(f"        const uint{g.sew}_t *src = (const uint{g.sew}_t *)(B + k * ldb);")
        if g.sw == g.e:
            w("        long x = 0;")
            w("        if (k < kc) {")
            w("            for (size_t vl; x < nc; x += vl) {")
            w(f"                vl = __riscv_vsetvl_e{g.sew}m8(nc - x);")
            w(
                f"                __riscv_vse{g.sew}_v_{u}m8(d + x, __riscv_vle{g.sew}_v_{u}m8(src + x, vl), vl);"
            )
            w("            }")
            w("        }")
            w("        for (size_t vl; x < ncp; x += vl) {")
            w(f"            vl = __riscv_vsetvl_e{g.sew}m8(ncp - x);")
            w(f"            __riscv_vse{g.sew}_v_{u}m8(d + x, z, vl);")
            w("        }")
        else:
            w(f"        for (long gi = 0; gi < nbk * {pl.bn}; gi++) {{")
            w(f"            long cnt = k < kc ? nc - gi * {g.e} : 0, x = 0;")
            w(f"            cnt = cnt < 0 ? 0 : cnt > {g.e} ? {g.e} : cnt;")
            w(f"            uint{g.sew}_t *dg = d + gi * {g.sw};")
            w("            for (size_t vl; x < cnt; x += vl) {")
            w(f"                vl = __riscv_vsetvl_e{g.sew}m8(cnt - x);")
            w(
                f"                __riscv_vse{g.sew}_v_{u}m8(dg + x, __riscv_vle{g.sew}_v_{u}m8(src + gi * {g.e} + x, vl), vl);"
            )
            w("            }")
            w(f"            for (size_t vl; x < {g.sw}; x += vl) {{")
            w(f"                vl = __riscv_vsetvl_e{g.sew}m8({g.sw} - x);")
            w(f"                __riscv_vse{g.sew}_v_{u}m8(dg + x, z, vl);")
            w("            }")
            w("        }")
        w("    }")
        w("}")
        w("")

    # init / store -----------------------------------------------------------------------------

    def _emit_init_store(self, w):
        g, pl, p = self.g, self.pl, self.p
        at = self.at
        e = g.e
        w(
            "/* The accumulators <- zero, or C[mv x nv] (row stride ldc) when accumulating. */"
        )
        w(
            f"NAIAD_GEMM_INLINE void {p}_init{self.sfx}(const {at} *C, long ldc, int acc, long mv, long nv) {{"
        )
        w(f"    vme_msetmtypei({g.tew}, 0, 1);")
        w(f"    vme_msettm({e});")
        w(f"    vme_msettn({e});")
        w("    if (!acc) {")
        for i in range(pl.bm):
            for j in range(pl.bn):
                w(f"        vme_vtzero({self.tile(i, j)});")
        w("        return;")
        w("    }")
        w(f"    for (long i = 0; i < {pl.bm}; i++) {{")
        w(f"        long rows = mv - i * {e};")
        w(f"        rows = rows > {e} ? {e} : rows;")
        w(f"        for (long j = 0; j < {pl.bn}; j++) {{")
        w(f"            long cols = nv - j * {e};")
        w("            if (rows <= 0 || cols <= 0)")
        w("                continue;")
        w(f"            vme_msettn(cols > {e} ? {e} : cols);")
        w(
            f"            const unsigned long t = (unsigned long)((i * {pl.bn} + j) * {g.acc_step}) << 27;"
        )
        w("            for (long r = 0; r < rows; r++)")
        w(
            f"                vme_vtle({g.tew}, &C[(i * {e} + r) * ldc + j * {e}], t | r);"
        )
        w("        }")
        w("    }")
        w("}")
        w("")
        w(
            "/* C[mv x nv] <- the accumulators, by tile rows (tm / tn clipped to the matrix). */"
        )
        w(
            f"NAIAD_GEMM_INLINE void {p}_store{self.sfx}({at} *C, long ldc, long mv, long nv) {{"
        )
        w(f"    if (mv == {pl.mb} && nv == {pl.nb}) {{")
        if self.dual and pl.bn <= MAX_DUAL_BN:
            self._emit_store_dual(w)
        else:
            w(f"        const long rstep = ldc * {g.aes} - {(pl.bn - 1) * e * g.aes};")
            w(f"        vme_msettn({e});")
            w(f"        for (long i = 0; i < {pl.bm}; i++) {{")
            w(f"            char *c_ = (char *)(C + i * {e} * ldc);")
            for j in range(pl.bn):
                w(
                    f"            unsigned long t{j}_ = (unsigned long)((i * {pl.bn} + {j}) * {g.acc_step}) << 27;"
                )
                w(f"            NAIAD_GEMM_OPAQUE(t{j}_);")
            for r in range(e):
                for j in range(pl.bn):
                    adv = "rstep" if j == pl.bn - 1 else str(e * g.aes)
                    w(
                        f"            vme_vtse({g.tew}, c_, t{j}_ + {r}); c_ += {adv}; NAIAD_GEMM_OPAQUE(c_);"
                    )
            w("        }")
        w("        return;")
        w("    }")
        w(f"    for (long i = 0; i < {pl.bm}; i++) {{")
        w(f"        long rows = mv - i * {e};")
        w(f"        rows = rows > {e} ? {e} : rows;")
        w(f"        for (long j = 0; j < {pl.bn}; j++) {{")
        w(f"            long cols = nv - j * {e};")
        w("            if (rows <= 0 || cols <= 0)")
        w("                continue;")
        w(f"            vme_msettn(cols > {e} ? {e} : cols);")
        w(
            f"            const unsigned long t = (unsigned long)((i * {pl.bn} + j) * {g.acc_step}) << 27;"
        )
        w("            for (long r = 0; r < rows; r++)")
        w(
            f"                vme_vtse({g.tew}, &C[(i * {e} + r) * ldc + j * {e}], t | r);"
        )
        w("        }")
        w("    }")
        w("}")
        w("")

    def _emit_store_dual(self, w):
        """The full-block C store for the dual-issue front end: one pointer and one index register per C
        column, so no advance waits on the one before it, ordered by :func:`dual_sched.schedule`."""
        g, pl = self.g, self.pl
        e, bn = g.e, pl.bn
        w(f"        const long cs_ = ldc * {g.aes};")
        w(f"        vme_msettn({e});")
        w(f"        for (long i = 0; i < {pl.bm}; i++) {{")
        for j in range(bn):
            w(f"            char *c{j}_ = (char *)(C + i * {e} * ldc + {j * e});")
            w(
                f"            unsigned long t{j}_ = (unsigned long)((i * {bn} + {j}) * {g.acc_step}) << 27;"
            )
            w(f"            NAIAD_GEMM_OPAQUE(t{j}_);")
        ops = []
        for r in range(e):
            for j in range(bn):
                ops.append(
                    dual_sched.Op(
                        f"vme_vtse({g.tew}, c{j}_, t{j}_);",
                        (f"c{j}_", f"t{j}_"),
                        (),
                        True,
                    )
                )
                if r < e - 1:
                    ops += [
                        dual_sched.Op(
                            f"NAIAD_GEMM_ADD(c{j}_, cs_);",
                            (f"c{j}_", "cs_"),
                            (f"c{j}_",),
                        ),
                        dual_sched.Op(
                            f"NAIAD_GEMM_ADDI(t{j}_, 1);", (f"t{j}_",), (f"t{j}_",)
                        ),
                    ]
        ordered, _ = dual_sched.schedule(ops, vec_gap=self._tile_gap(e * g.aes))
        for o in ordered:
            w(f"            {o.text}")
        w("        }")

    # the block: K loop -------------------------------------------------------------------------

    def _emit_block(self, w):
        g, pl, p, sc = self.g, self.pl, self.p, self.sc
        at = self.at
        w(
            f"/* One {pl.mb} x {pl.nb} block of C: (acc ? C : 0) + panel . B, nbody bodies of {pl.u} K step(s);"
        )
        w(
            "   pb is B's first row at this block's columns, ldbb its row stride in bytes. */"
        )
        w(
            f"static __attribute__((noinline, unused)) void {p}_block{self.sfx}(long nbody, const char *pb, long ldbb, "
            f"{at} *C, long ldc, int acc, long mv, long nv) {{"
        )
        w(f"    const char *pa = {p}_apanel;")
        for r in range(g.kmax):
            w(f"    const char *pb{r} = pb + {r} * ldbb;")
        w(f"    const long bstep = {g.kmax} * ldbb;")
        w("    (void)bstep;")
        for line in self._adv_decls():
            w(f"    {line}")
        for op in sc.prologue:
            w(f"    {self._op_stmt(op)}")
        w(f"    {p}_init{self.sfx}(C, ldc, acc, mv, nv);")
        w(f"    vme_msetmtypei({g.sew}, {g.alt}, {g.tw});")
        w(f"    vme_msettm({g.e});")
        w(f"    vme_msettk({g.kmax});")
        w(f"    vme_msettn({g.e});")
        w("    for (long it_ = nbody - 1; it_ > 0; it_--) {")
        for item, ops in sc.body:
            w(f"        {self._mma_stmt(item)}")
            for op in ops:
                w(f"        {self._op_stmt(op)}")
        w("    }")
        w("    /* the last body: no prefetch for a next one */")
        for item, ops in sc.body:
            w(f"    {self._mma_stmt(item)}")
            for op in ops:
                if not op[2]:
                    w(f"    {self._op_stmt(op)}")
        w(f"    {p}_store{self.sfx}(C, ldc, mv, nv);")
        w("}")
        w("")

    # driver -----------------------------------------------------------------------------------

    def _emit_run(self, w, bpad):
        g, pl, p = self.g, self.pl, self.p
        it, at = self.it, self.at
        e, mb, nb = g.e, pl.mb, pl.nb
        w(
            "/* C = (accumulate ? C : 0) + A[m x k] . B[k x n], any m, n, k (see the header). */"
        )
        w(
            f"static __attribute__((unused, noinline)) void {p}_run(long m, long n, long k, const {it} *A, long lda, "
            f"const {it} *B, long ldb, {at} *C, long ldc, int accumulate) {{"
        )
        w("    if (m <= 0 || n <= 0)")
        w("        return;")
        w("    if (k <= 0) {")
        w("        if (!accumulate)")
        w("            for (long i = 0; i < m; i++)")
        w("                for (long j = 0; j < n; j++)")
        w("                    C[i * ldc + j] = 0;")
        w("        return;")
        w("    }")
        P = p.upper()
        w(f"    for (long k0 = 0; k0 < k; k0 += {P}_KC) {{")
        w(f"        const long kc = k - k0 < {P}_KC ? k - k0 : {P}_KC;")
        w("        const int acc = accumulate || k0 > 0;")
        kq_all = max([pl.kq] + [v.plan.kq for v in self.narrow])
        step = e if self.narrow else nb
        w(
            f"        const int inplace = {int(g.inplace_ok)} && n % {step} == 0 && kc % {kq_all} == 0;"
        )
        w(f"        const long kpad_a = (kc + {e - 1}) / {e} * {e};")
        w(
            f"        const long kpad = inplace ? kc : (kc + {self.kl - 1}) / {self.kl} * {self.kl};"
        )
        w(f"        const long nbody = kpad / {pl.kq};")
        w(
            "        const long kp = kpad > kpad_a ? kpad : kpad_a; /* panel K rows (zero past kc) */"
        )
        w(
            f"        long nbk = inplace ? (n + {nb - 1}) / {nb} : {bpad} / (kpad * {pl.bn * g.sw});"
        )
        w("        nbk = nbk < 1 ? 1 : nbk;")
        w(f"        const long ncs = nbk * {nb};")
        w("        for (long j00 = 0; j00 < n; j00 += ncs) {")
        w("            const long nc = n - j00 < ncs ? n - j00 : ncs;")
        w("            const char *bsrc;")
        w("            long ldbb;")
        w("            if (inplace) {")
        w("                bsrc = (const char *)(B + k0 * ldb + j00);")
        w(f"                ldbb = ldb * {g.ies};")
        w("            } else {")
        w(f"                const long nbkc = (nc + {nb - 1}) / {nb};")
        w(f"                {p}_pack_b(B + k0 * ldb + j00, ldb, kc, kpad, nc, nbkc);")
        w(f"                bsrc = (const char *){p}_bpad;")
        w(f"                ldbb = nbkc * {pl.bn * g.sw * g.ies};")
        w("            }")
        w(f"            for (long i0 = 0; i0 < m; i0 += {mb}) {{")
        w(f"                const long mv = m - i0 < {mb} ? m - i0 : {mb};")
        w(f"                if (mv == {mb} && kc == kp)")
        w(f"                    {p}_pack_a_full(A + i0 * lda + k0, lda, kp / {e});")
        w("                else")
        w(f"                    {p}_pack_a_edge(A + i0 * lda + k0, lda, mv, kc, kp);")
        w("                for (long j0 = 0; j0 < nc;) {")
        w(f"                    const long nv = nc - j0 < {nb} ? nc - j0 : {nb};")
        w(
            f"                    const long boff = inplace ? j0 * {g.ies} : (j0 / {nb}) * {pl.bn * g.sw * g.ies};"
        )
        w(f"                    {at} *c = C + i0 * ldc + j00 + j0;")
        w("                    if (!inplace || nv == " + str(nb) + ") {")
        w(
            f"                        {p}_block(nbody, bsrc + boff, ldbb, c, ldc, acc, mv, nv);"
        )
        w(f"                        j0 += {nb};")
        for v in self.narrow:
            vnb = v.plan.nb
            w(
                f"                    }} else if (nv >= {vnb}) {{ /* in place, a multiple of E: the narrower blocks */"
            )
            w(
                f"                        {p}_block_n{v.plan.bn}(kpad / {v.plan.kq}, bsrc + boff, ldbb, c, ldc, acc, mv, {vnb});"
            )
            w(f"                        j0 += {vnb};")
        w("                    }")
        w("                }")
        w("            }")
        w("        }")
        w("    }")
        w("    vme_vtdiscard();")
        w("}")
        w("")
        w("/* C = A . B */")
        w(
            f"static inline __attribute__((unused)) void {p}(long m, long n, long k, const {it} *A, long lda, "
            f"const {it} *B, long ldb, {at} *C, long ldc) {{"
        )
        w(f"    {p}_run(m, n, k, A, lda, B, ldb, C, ldc, 0);")
        w("}")
        w("")
        w("/* C += A . B */")
        w(
            f"static inline __attribute__((unused)) void {p}_acc(long m, long n, long k, const {it} *A, long lda, "
            f"const {it} *B, long ldb, {at} *C, long ldc) {{"
        )
        w(f"    {p}_run(m, n, k, A, lda, B, ldb, C, ldc, 1);")
        w("}")


def _ext_macro(geo):
    for name, macro in _MMA:
        if macro == geo.mma:
            row = vme.arith_row(name, geo.sew, geo.tw, geo.alt)
            return row.ext.upper()
    raise AssertionError(geo)


def plan_for(design, in_fmt, acc_fmt, block=None, ref=REF_SIZE):
    """The chosen :class:`Schedule` for one pair of ``design`` (a VME ``Design``)."""
    spec = design.matrix
    geo = Geometry.for_pair(spec, in_fmt, acc_fmt)
    if isinstance(block, str):
        block = tuple(int(v) for v in block.lower().split("x"))
    return choose(geo, costs_for(design, geo), tuple(block) if block else None, ref)


def emit_vme_gemm_header(design, *, block=None, pairs=None, ref=REF_SIZE):
    """The ``naiad_gemm.h`` text for a VME design."""
    spec = design.matrix
    available = format_pairs(spec)
    if pairs is None:
        pairs = available
    else:
        pairs = [tuple(p.split(":")) if isinstance(p, str) else tuple(p) for p in pairs]
        missing = [p for p in pairs if p not in available]
        if missing:
            raise ValueError(
                f"format pair(s) {missing} not implemented by this unit; have {available}"
            )
    scheds = []
    for in_fmt, acc_fmt in pairs:
        try:
            main = plan_for(design, in_fmt, acc_fmt, block, ref)
            narrow = []
            bn = main.plan.bn // 2
            while bn >= 1:
                narrow.append(
                    plan_for(design, in_fmt, acc_fmt, (main.plan.bm, bn), ref)
                )
                bn //= 2
            scheds.append((main, narrow))
        except ValueError:
            if pairs is not available:
                raise
            # a default pair this emitter cannot block (or not at a forced shape): leave it out
    if not scheds:
        raise ValueError(f"no VME GEMM could be emitted for pairs {pairs}")
    L = [
        "/*",
        f" * naiad_gemm.h: GEMM driver for design '{design.name}'",
        f" * (vme, Zvt v0.4: TE {spec.te}, VLEN {spec.vlen}).",
        " * Generated by python -m framework.naiad.emit.gemm (framework/naiad/emit/gemm_vme.py).",
        " * DO NOT EDIT: regenerate from the design file instead.",
        " */",
        "#ifndef NAIAD_GEMM_H",
        "#define NAIAD_GEMM_H",
        "",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <riscv_vector.h>",
        '#include "dwarf_util.h"',
        "",
        "#ifndef NAIAD_MATRIX_VME",
        '#error "naiad_gemm.h was emitted for a VME unit, but naiad.h describes another mode: re-emit"',
        "#endif",
        '#include "vme.h"',
        f"#if NAIAD_VME_TE != {spec.te} || NAIAD_VLEN != {spec.vlen}",
        '#error "naiad_gemm.h does not match this design\'s naiad.h (TE or VLEN): re-emit"',
        "#endif",
        "",
        "#define NAIAD_GEMM_INLINE static inline __attribute__((always_inline, unused))",
        "#define NAIAD_GEMM_STAGE __attribute__((aligned(64), unused))",
        "/* Hide a value from the optimiser, so x + c stays one addi rather than a fresh constant. */",
        '#define NAIAD_GEMM_OPAQUE(x) __asm__("" : "+r"(x))',
        "/* The K loop's ops: vl<nf>re<eew>.v into literal register vd; a pointer advance (immediate or register). */",
        "#define NAIAD_GEMM_VLRE(nf, eew, vd, p) \\",
        '    __asm__ volatile("vl" VME_STR(nf) "re" VME_STR(eew) ".v v" VME_STR(vd) ", (%0)" ::"r"(p) : "memory")',
        '#define NAIAD_GEMM_ADDI(p, imm) __asm__ volatile("addi %0, %0, %1" : "+r"(p) : "i"(imm))',
        '#define NAIAD_GEMM_ADD(p, r) __asm__ volatile("add %0, %0, %1" : "+r"(p) : "r"(r))',
        "",
        "#define NAIAD_GEMM_FN_(in, acc) naiad_gemm_##in##_##acc",
        "#define NAIAD_GEMM_FN(in, acc) NAIAD_GEMM_FN_(in, acc)",
        "",
    ]
    timing = getattr(design, "timing", None)
    dual = timing is not None and timing.scalar.issue_width >= 2
    for main, narrow in scheds:
        costs = costs_for(design, main.plan.geo)
        L += _PairEmitter(main, narrow, costs, dual).emit()
    L += [
        "/* The unsuffixed names, for the pair named by NAIAD_GEMM_IN / NAIAD_GEMM_ACC. */",
        "#if defined(NAIAD_GEMM_IN) && defined(NAIAD_GEMM_ACC)",
        "#define NAIAD_GEMM_SEL_(in, acc, f) naiad_gemm_##in##_##acc##f",
        "#define NAIAD_GEMM_SEL(in, acc, f) NAIAD_GEMM_SEL_(in, acc, f)",
        "#define naiad_gemm NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, )",
        "#define naiad_gemm_acc NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _acc)",
        "typedef NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _in_t) naiad_gemm_in_t;",
        "typedef NAIAD_GEMM_SEL(NAIAD_GEMM_IN, NAIAD_GEMM_ACC, _acc_t) naiad_gemm_acc_t;",
        "#endif",
        "",
        "#endif /* NAIAD_GEMM_H */",
        "",
    ]
    return "\n".join(L)


__all__ = [
    "C_TYPES",
    "Costs",
    "Geometry",
    "Plan",
    "Schedule",
    "candidates",
    "choose",
    "costs_for",
    "emit_vme_gemm_header",
    "format_pairs",
    "plan_for",
    "schedule",
]
