"""IME's cycle model: Zvvm instructions on the vector unit's register file and ports.

Timing only (the functional side is :mod:`framework.matrix.ime_isa`).
:class:`IMETiming` extends the RVV model (:class:`framework.pe.rv.timing.vector.VectorTiming`)
because IME has no state of its own: its operands are vector register groups,
so it shares the RVV scoreboard (one ready cycle per vector register), the
vector load/store unit, and the VRF's ports. A design in ``mode: ime`` gives
its ``timing.matrix`` section this module's :class:`IMETimingSpec`, and the
cost state builds an :class:`IMETiming` in place of the plain vector model.

**The mma unit.** One matrix datapath beside the RVV arithmetic datapath (the
two run concurrently). A multiply-accumulate of ``M x N x K_eff`` MACs in
input format ``f`` occupies it for

    occ = max(ceil(M*N*K_eff * passes[f] / macs_per_cycle),      # compute
              ceil(R / vrf_read_bytes_per_cycle),                 # operand reads
              ceil(Wr / vrf_write_bytes_per_cycle))               # C write-back

(or ``mma_ii`` in place of the compute term), where ``R`` is the A, B and C
bytes it reads from the VRF (``M*lambda*LMUL*SEW/8`` for A, the same with ``N``
for B, ``M*N*SEW/8`` for C) and ``Wr`` the C bytes it writes. C is ready
``pipeline_depth - 1`` cycles after the unit frees; with ``acc_forwarding`` a
following mma into the same C group may start as soon as the unit is free
(accumulation in place), while any other reader waits for the full latency.
With ``c_bypass`` such a chained mma does not re-read C through the ports
(the array keeps it), which isolates how much of IME's port traffic is C.

**VRF ports.** IME adds no ports: by default it has the RVV datapath's, three
reads and one write of ``datapath_bits`` a cycle. Port windows are booked
from issue, so the model bounds throughput rather than placing each byte:
an mma holds the read port for ``ceil(R / read bw)`` cycles and the write port
for ``ceil(Wr / write bw)``, and an RVV instruction (arithmetic, load or
store, and the tile loads/stores, which go through the vector LSU) books its
own source / destination bytes likewise (``VL``-scaled group bytes). The two
*classes* exclude each other: an RVV instruction waits (``vector_busy``) while
an mma holds a port it needs, and an mma waits (``matrix_busy``) while RVV
does. RVV against RVV is not re-checked, so an RVV-only program times exactly
as under the plain vector model.

**Hazards.** An mma reads its A and B groups and C, and writes C, through the
shared scoreboard (``matrix_raw`` when it waits); an RVV instruction reading
an mma's C waits for it like any other producer (``vector_raw``), and one
writing a group an mma is still reading or writing waits for that too.

**Tile loads/stores** (``vmtl``/``vmttl``/``vmts``/``vmtts``) use the vector LSU
and its ``mem_bytes_per_cycle`` and ``mem_startup``, with one extra limit: a
transfer is ``segments`` contiguous runs in memory, and the LSU starts
``tile_segments_per_cycle`` of them a cycle. ``vmtl`` with ``LD = lambda*LMUL``
(and ``vmttl`` with ``LD = VL / (lambda*LMUL)``) is one run; otherwise
``vmtl`` is ``VL / (lambda*LMUL)`` runs and ``vmttl`` ``lambda*LMUL`` runs.
Counted at full ``VL`` (masked-off elements still take their lane), as the
RVV model counts.
A tile store registers its rows (``LD``-strided) with the stores-in-flight
table (:mod:`framework.pe.rv.timing.store_hazards`) until the LSU frees; a load
overlapping it waits (``mem_raw``).
"""

import math
from dataclasses import dataclass, field

from framework.matrix import ime
from framework.pe.rv.timing.store_hazards import footprint
from framework.pe.rv.timing.vector import (
    CFG,
    MEM,
    MEM_RAW,
    VECTOR_BUSY,
    VECTOR_RAW,
    VectorTiming,
)

MATRIX_BUSY = "matrix_busy"
MATRIX_RAW = "matrix_raw"


@dataclass(frozen=True)
class IMETimingSpec:
    """``timing.matrix`` for ``mode: ime``. Bandwidths are bytes per cycle; ``None`` takes the default shown."""

    #: Peak multiply-accumulates per cycle for a one-pass format.
    macs_per_cycle: int = 512
    #: Fixed mma compute interval, overriding the one from ``macs_per_cycle``.
    mma_ii: int | None = None
    #: Input-format name -> passes per mma (as AME's ``format_passes``); default 1.
    format_passes: dict = field(default_factory=dict, hash=False)
    #: Cycles from the unit freeing to C being readable, plus one.
    pipeline_depth: int = 4
    #: A following mma into the same C group may start as soon as the unit is free.
    acc_forwarding: bool = True
    #: A chained mma on the same C group does not re-read C through the VRF ports.
    c_bypass: bool = False
    #: VRF read / write port bandwidth; None = the RVV datapath's (3 reads, 1 write of datapath_bits).
    vrf_read_bytes_per_cycle: float | None = None
    vrf_write_bytes_per_cycle: float | None = None
    #: Contiguous memory runs the vector LSU starts per cycle for a tile load/store.
    tile_segments_per_cycle: float = 1

    def passes(self, in_format):
        return self.format_passes.get(in_format, 1)

    def compute_interval(self, macs, in_format):
        if self.mma_ii is not None:
            return self.mma_ii
        return max(1, math.ceil(macs * self.passes(in_format) / self.macs_per_cycle))

    def ports(self, datapath_bits):
        """``(read, write)`` VRF port bandwidth in bytes per cycle."""
        rd = self.vrf_read_bytes_per_cycle
        wr = self.vrf_write_bytes_per_cycle
        return (
            3 * datapath_bits / 8 if rd is None else rd,
            datapath_bits / 8 if wr is None else wr,
        )

    def mma_occupancy(self, geo, in_format, datapath_bits, c_read=True):
        """``(occupancy, compute, read cycles, write cycles)`` of one mma of geometry ``geo``."""
        rbw, wbw = self.ports(datapath_bits)
        reads = geo.a_bytes() + geo.b_bytes() + (geo.c_bytes() if c_read else 0)
        compute = self.compute_interval(geo.macs, in_format)
        t_read = max(1, math.ceil(reads / rbw))
        t_write = max(1, math.ceil(geo.c_bytes() / wbw))
        return max(compute, t_read, t_write), compute, t_read, t_write


def parse_ime_timing(raw, design, parse_fields, number, flag, error):
    """``timing.matrix`` for an IME design (the caller's schema helpers do the field checks)."""
    from framework.matrix.formats import get_format

    rules = {
        "mma_ii": lambda v, w: number(v, w, minimum=1, allow_none=True),
        "acc_forwarding": flag,
        "c_bypass": flag,
        "format_passes": lambda v, w: v,
        "vrf_read_bytes_per_cycle": lambda v, w: number(
            v, w, minimum=1, allow_none=True, allow_float=True
        ),
        "vrf_write_bytes_per_cycle": lambda v, w: number(
            v, w, minimum=1, allow_none=True, allow_float=True
        ),
        "tile_segments_per_cycle": lambda v, w: number(
            v, w, minimum=1, allow_float=True
        ),
    }
    spec = parse_fields(raw, IMETimingSpec, "timing.matrix", rules)
    passes = spec.format_passes
    if not isinstance(passes, dict):
        raise error(
            "timing.matrix.format_passes: expected a mapping of in-format -> passes"
        )
    formats = {f for pair in design.matrix.format_pairs for f in pair[:1]}
    canonical = {}
    for name, n in passes.items():
        try:
            fmt = get_format(str(name).lower()).name
        except ValueError:
            fmt = str(name).lower()
        if fmt not in formats:
            raise error(
                f"timing.matrix.format_passes: {fmt} is not an input format of the design's extensions {sorted(formats)}"
            )
        canonical[fmt] = number(n, f"timing.matrix.format_passes.{fmt}", minimum=1)
    return IMETimingSpec(
        **{
            **{k: getattr(spec, k) for k in spec.__dataclass_fields__},
            "format_passes": canonical,
        }
    )


@dataclass(frozen=True)
class IMEOp:
    """A decoded Zvvm word, for the cost state (``ssrc``/``sdst`` are the scalar scoreboard's)."""

    d: ime.Decoded
    ssrc: tuple
    sdst: int = -1


def _rvv_class_bytes(timing, op):
    """``(read, write)`` VRF bytes of an RVV op: its group bytes, scaled by VL / VLMAX."""
    vs = timing.vs
    frac = min(1.0, vs.vl / vs.vlmax) if vs.vlmax else 1.0
    rd = sum(len(timing._regs(b, r)) for b, r in op.vsrc) * vs.vlenb * frac
    wr = 0 if op.vdst is None else len(timing._regs(*op.vdst)) * vs.vlenb * frac
    return rd, wr


class IMETiming(VectorTiming):
    """The vector timing state plus Zvvm: the mma unit, the VRF port classes, IME counters."""

    def __init__(self, spec, vector_state, memory, ime_spec, unit):
        self.ime_spec = ime_spec
        self.unit = unit
        super().__init__(spec, vector_state, memory)
        self.peak_macs_per_cycle = ime_spec.macs_per_cycle

    def reset(self):
        super().reset()
        self.mma_free = 0
        self.mma_ready = [
            0
        ] * 32  # forwarding: when a chained mma may read a C register
        self.read_until = [0] * 32  # last cycle an in-flight mma reads each register
        self.mma_writes = [False] * 32  # the register's last writer was an mma
        self.last_c = None  # (vd, emul_c) the last mma wrote, while untouched since
        self.ime_read_free = self.ime_write_free = 0
        self.rvv_read_free = self.rvv_write_free = 0
        self.mmas = self.macs = 0
        self.mma_busy = self.vrf_read_busy = self.vrf_write_busy = 0

    def counters(self):
        out = super().counters()
        out.update(
            {
                "matrix_busy_cycles": self.mma_busy,
                "matrix_mma": self.mmas,
                "matrix_macs": self.macs,
                "matrix_peak_macs_per_cycle": self.peak_macs_per_cycle,
                "ime_vrf_read_busy_cycles": self.vrf_read_busy,
                "ime_vrf_write_busy_cycles": self.vrf_write_busy,
            }
        )
        return out

    # -- decode ---------------------------------------------------------------------

    def decode(self, instr):
        op = self._ops.get(instr, 0)
        if op == 0:
            d = ime.decode(instr) if ime.in_space(instr) else None
            if d is None:
                op = super().decode(instr)
            else:
                ssrc = (d.rs1, d.rs2) if d.insn.kind in ("load", "store") else ()
                op = IMEOp(d, tuple(s for s in ssrc if s))
            self._ops[instr] = op
        return op

    # -- issue ----------------------------------------------------------------------

    def issue(self, op, cycle, register_file):
        if isinstance(op, IMEOp):
            if op.d.insn.kind in ("int", "fp"):
                return self._issue_mma(op.d, cycle)
            return self._issue_tile(op.d, cycle, register_file)
        if op.kind == CFG:
            return super().issue(op, cycle, register_file)
        rd, wr = _rvv_class_bytes(self, op)
        if (rd and self.ime_read_free > cycle) or (wr and self.ime_write_free > cycle):
            return self._stall(
                max(self.ime_read_free, self.ime_write_free), VECTOR_BUSY
            )
        if op.vdst is not None:
            for r in self._regs(*op.vdst):
                at = max(self.read_until[r], self.ready[r] if self.mma_writes[r] else 0)
                if at > cycle:
                    return self._stall(at, VECTOR_RAW)
        done = super().issue(op, cycle, register_file)
        if done < 0:
            return done
        self._book_rvv(cycle, rd, wr)
        if op.vdst is not None:
            self._touch(self._regs(*op.vdst))
        return done

    def _book_rvv(self, cycle, rd, wr):
        rbw, wbw = self.ime_spec.ports(self.spec.datapath_bits)
        if rd:
            self.rvv_read_free = max(self.rvv_read_free, cycle + math.ceil(rd / rbw))
        if wr:
            self.rvv_write_free = max(self.rvv_write_free, cycle + math.ceil(wr / wbw))

    def _touch(self, regs):
        for r in regs:
            self.mma_writes[r] = False
        if self.last_c is not None:
            vd, n = self.last_c
            if any(vd <= r < vd + n for r in regs):
                self.last_c = None

    def _geometry(self, d):
        """``(geo, in_format)`` for an mma under the current vtype, or None if it would trap."""
        vs, unit = self.vs, self.unit
        if vs.vill or vs.lmul_log2 < 0 or vs.vstart:
            return None
        lam = ime.lambda_of(vs.vtype)
        if lam is None:
            return None
        geo = ime.gemm_geometry(
            vs.vlen, vs.sew, 1 << vs.lmul_log2, lam, vs.vl, d.insn.w
        )
        if isinstance(geo, str):
            return None
        vt = vs.vtype
        row = ime.encoding_row(
            d.insn.kind,
            d.insn.w,
            vs.sew,
            (vt >> ime.ALTFMT_A_SHIFT) & 1,
            (vt >> ime.ALTFMT_B_SHIFT) & 1,
            (vt >> ime.ALTFMT_SHIFT) & 1,
        )
        if (
            row is None
            or unit is None
            or any(e not in unit.extensions for e in row.requires)
        ):
            return None
        return geo, row.fmt_a

    def _issue_mma(self, d, cycle):
        found = self._geometry(d)
        if found is None:
            return cycle + 1  # it traps on execute; no cost is owed
        geo, fmt = found
        spec = self.ime_spec
        a = range(d.vs1, d.vs1 + geo.lmul)
        b = range(d.vs2, d.vs2 + geo.lmul)
        c = range(d.vd, min(32, d.vd + geo.emul_c))
        ready = self.ready
        for r in (*a, *b):
            if ready[r] > cycle:
                return self._stall(ready[r], MATRIX_RAW)
        chained = self.last_c == (d.vd, geo.emul_c)
        for r in c:
            at = (
                self.mma_ready[r]
                if spec.acc_forwarding and chained and self.mma_writes[r]
                else ready[r]
            )
            if at > cycle:
                return self._stall(at, MATRIX_RAW)
        if self.mma_free > cycle:
            return self._stall(self.mma_free, MATRIX_BUSY)
        if self.rvv_read_free > cycle or self.rvv_write_free > cycle:
            return self._stall(
                max(self.rvv_read_free, self.rvv_write_free), MATRIX_BUSY
            )
        # -- commit
        c_read = not (spec.c_bypass and chained)
        occ, _, t_read, t_write = spec.mma_occupancy(
            geo, fmt, self.spec.datapath_bits, c_read
        )
        self.mma_free = cycle + occ
        self.ime_read_free = cycle + t_read
        self.ime_write_free = cycle + t_write
        done = cycle + occ + spec.pipeline_depth - 1
        for r in (*a, *b):
            self.read_until[r] = max(self.read_until[r], cycle + t_read)
        for r in c:
            ready[r] = done
            self.mma_ready[r] = cycle + occ
            self.mma_writes[r] = True
            self.read_until[r] = max(self.read_until[r], cycle + t_read)
        self.last_c = (d.vd, geo.emul_c)
        self.mmas += 1
        self.macs += geo.macs
        self.mma_busy += occ
        self.vrf_read_busy += t_read
        self.vrf_write_busy += t_write
        return cycle + 1

    def _issue_tile(self, d, cycle, register_file):
        vs = self.vs
        if vs.vill or vs.lmul_log2 < 0:
            return cycle + 1
        lmul = 1 << vs.lmul_log2
        lam = ime.decode_lambda(d.lam_imm) if d.lam_imm else ime.lambda_of(vs.vtype)
        if lam is None or vs.vl % (lam * lmul):
            return cycle + 1
        ls = lam * lmul
        is_load = d.insn.kind == "load"
        regs = range(d.vd, min(32, d.vd + lmul))
        ready = self.ready
        if not is_load:
            for r in regs:
                if ready[r] > cycle:
                    return self._stall(ready[r], VECTOR_RAW)
        else:
            for r in regs:
                at = max(self.read_until[r], ready[r] if self.mma_writes[r] else 0)
                if at > cycle:
                    return self._stall(at, VECTOR_RAW)
        if not d.vm and ready[0] > cycle:
            return self._stall(ready[0], VECTOR_RAW)
        if self.lsu_free > cycle:
            return self._stall(self.lsu_free, VECTOR_BUSY)
        nbytes = vs.vl * vs.sew // 8
        if (is_load and self.ime_write_free > cycle) or (
            not is_load and self.ime_read_free > cycle
        ):
            return self._stall(
                self.ime_write_free if is_load else self.ime_read_free, VECTOR_BUSY
            )
        addr = register_file[d.rs1].read_uint()
        region = self.memory.lookup(addr)
        if region.port_free > cycle:
            return self._stall(region.port_free, MEM)
        ld = register_file[d.rs2].read_uint()
        nl = vs.vl // ls
        if ld == 0:
            ld = (vs.vlen // vs.sew // lam) if d.transposing else ls
        # Memory RAW: rows of the tile at ld elements (framework.pe.rv.timing.store_hazards).
        eb = vs.sew // 8
        rows, row_elems = (ls, nl) if d.transposing else (nl, ls)
        fp = footprint(addr, ld * eb, rows, row_elems * eb)
        inflight = self.memory.inflight
        if is_load and inflight.until > cycle:
            at = inflight.wait(fp, cycle)
            if at:
                return self._stall(at, MEM_RAW)
        # -- commit
        dense = ld == (nl if d.transposing else ls)
        segments = 1 if dense else (ls if d.transposing else nl)
        bw = self.spec.mem_bytes_per_cycle
        if region.bandwidth is not None:
            bw = min(bw, region.bandwidth)
        t = max(
            1,
            math.ceil(nbytes / bw),
            math.ceil(segments / self.ime_spec.tile_segments_per_cycle),
        )
        latency = region.access(addr, nbytes, not is_load, cycle, t)
        self.lsu_free = cycle + t
        self.mem_busy_cycles += t
        if is_load:
            done = cycle + self.spec.mem_startup + latency + t
            for r in regs:
                ready[r] = done
            self._touch(regs)
            self._book_rvv(cycle, 0, nbytes)
        else:
            self._book_rvv(cycle, nbytes, 0)
            if nbytes:
                inflight.add(fp, cycle + t, cycle)
        return cycle + 1


__all__ = ["IMEOp", "IMETiming", "IMETimingSpec", "parse_ime_timing"]
