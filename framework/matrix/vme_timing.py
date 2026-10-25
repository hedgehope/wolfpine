"""VME's cycle model: Zvt instructions beside the vector unit, on its register file and LSU.

Timing only (the functional side is :mod:`framework.matrix.vme_isa`). VME's
operands come from vector register groups and its results go to its own
accumulator tiles, so :class:`VMETiming` extends the IME model
(:class:`framework.matrix.ime_timing.IMETiming`, itself the RVV model plus a
matrix datapath): it shares the RVV scoreboard, the vector LSU and the VRF's
ports, and IME's rule that matrix and RVV instructions book the ports as two
mutually exclusive classes (RVV against RVV is not re-checked, so an RVV-only
program times exactly as under the plain vector model). A design in ``mode:
vme`` gives ``timing.matrix`` this module's :class:`VMETimingSpec`.

**The multiply unit.** A ``vtfmm``/``vtmm*`` of ``tm x tn x tk`` MACs occupies it for

    occ = max(ceil(tm*tn*tk * passes[f] / macs_per_cycle),       # compute
              ceil((A + B bytes) / vrf_read_bytes_per_cycle))     # operand reads

(or ``mma_ii`` for the compute term), A being ``tk*tm*SEW/8`` bytes and B
``tk*tn*SEW/8``. There is no C traffic through the VRF (C lives in a tile):
this is the coupling difference from IME, whose C is re-read and written
back through the ports on every mma. The tile is ready ``pipeline_depth - 1``
cycles after the unit frees; with ``acc_forwarding`` a following multiply into
the same tile may start as soon as the unit is free.

**Tile traffic.** The moves (``vtmv.v.t`` tile -> VRF, ``vtmv.t.v`` VRF ->
tile), loads/stores (``vtle``/``vtse``, through the vector LSU at
``mem_bytes_per_cycle``, one contiguous run) and ``vtzero`` use the tile port,
``tile_bytes_per_cycle`` wide (one at a time); a move also holds the VRF write
(``vtmv.v.t``) or read (``vtmv.t.v``) port for its bytes. ``vtzero`` takes one
cycle. Hazards are per TEW-8 storage tile (``vme.physical_tiles``), so punned
views of the same storage order correctly; tile results of moves and loads
are ready ``1`` (moves) or ``mem_startup + latency + transfer`` (loads) cycles
after issue. Configuration instructions (``msetmtype[i]``, ``msett*``) and
``vtdiscard`` cost one cycle and write rd like an ALU op.
"""

import math
from dataclasses import dataclass, field

from framework.matrix import vme
from framework.matrix.ime_timing import MATRIX_BUSY, MATRIX_RAW, IMETiming
from framework.naiad.vector_timing import MEM, VECTOR_BUSY, VECTOR_RAW, VectorTiming

TILE_RAW = MATRIX_RAW


@dataclass(frozen=True)
class VMETimingSpec:
    """``timing.matrix`` for ``mode: vme``. Bandwidths are bytes per cycle; ``None`` takes the default shown."""

    #: Peak multiply-accumulates per cycle for a one-pass format.
    macs_per_cycle: int = 512
    #: Fixed multiply compute interval, overriding the one from ``macs_per_cycle``.
    mma_ii: int | None = None
    #: Input-format name -> passes per multiply (as AME's ``format_passes``); default 1.
    format_passes: dict = field(default_factory=dict, hash=False)
    #: Cycles from the unit freeing to the tile being readable, plus one.
    pipeline_depth: int = 4
    #: A following multiply into the same tile may start as soon as the unit is free.
    acc_forwarding: bool = True
    #: VRF read / write port bandwidth; None = the RVV datapath's (3 reads, 1 write of datapath_bits).
    vrf_read_bytes_per_cycle: float | None = None
    vrf_write_bytes_per_cycle: float | None = None
    #: The tile state's move/load/store port, bytes per cycle.
    tile_bytes_per_cycle: float = 64

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

    def mma_occupancy(self, macs, operand_bytes, in_format, datapath_bits):
        """``(occupancy, compute, read cycles)`` of one multiply."""
        rbw, _ = self.ports(datapath_bits)
        compute = self.compute_interval(macs, in_format)
        t_read = max(1, math.ceil(operand_bytes / rbw))
        return max(compute, t_read), compute, t_read


def parse_vme_timing(raw, design, parse_fields, number, flag, error):
    """``timing.matrix`` for a VME design (the caller's schema helpers do the field checks)."""
    from framework.matrix.formats import get_format

    rules = {
        "mma_ii": lambda v, w: number(v, w, minimum=1, allow_none=True),
        "acc_forwarding": flag,
        "format_passes": lambda v, w: v,
        "vrf_read_bytes_per_cycle": lambda v, w: number(
            v, w, minimum=1, allow_none=True, allow_float=True
        ),
        "vrf_write_bytes_per_cycle": lambda v, w: number(
            v, w, minimum=1, allow_none=True, allow_float=True
        ),
        "tile_bytes_per_cycle": lambda v, w: number(v, w, minimum=1, allow_float=True),
    }
    spec = parse_fields(raw, VMETimingSpec, "timing.matrix", rules)
    passes = spec.format_passes
    if not isinstance(passes, dict):
        raise error(
            "timing.matrix.format_passes: expected a mapping of in-format -> passes"
        )
    formats = {p[0] for p in design.matrix.format_pairs}
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
    return VMETimingSpec(
        **{
            **{k: getattr(spec, k) for k in spec.__dataclass_fields__},
            "format_passes": canonical,
        }
    )


@dataclass(frozen=True)
class VMEOp:
    """A decoded Zvt word, for the cost state (``ssrc``/``sdst`` are the scalar scoreboard's)."""

    insn: vme.Instruction
    instr: int
    ssrc: tuple
    sdst: int = -1

    def field(self, name):
        return self.insn.field(self.instr, name)


_SCALAR_SRC = {
    "msetmtype": ("rs1", "rs2"),
    "msettn": ("rs1",),
    "msettm": ("rs1",),
    "msettk": ("rs1",),
    "vtle": ("rs1", "rs2"),
    "vtse": ("rs1", "rs2"),
    "vtmv.v.t": ("rs1",),
    "vtmv.t.v": ("rs1",),
}


class VMETiming(IMETiming):
    """The vector timing state plus Zvt: the multiply unit, the tile port, per-storage-tile hazards."""

    def __init__(self, spec, vector_state, memory, vme_spec, unit):
        super().__init__(spec, vector_state, memory, vme_spec, unit)

    def reset(self):
        super().reset()
        self.tile_ready = [0] * 16  # when each TEW-8 storage tile's last write lands
        self.tile_mma_ready = [
            0
        ] * 16  # forwarding: when a chained multiply may accumulate
        self.tile_mma_writes = [
            False
        ] * 16  # the storage tile's last writer was a multiply
        self.tile_read_until = [0] * 16  # last cycle an in-flight transfer reads it
        self.tile_port_free = 0
        self.tile_port_busy = 0

    def counters(self):
        out = VectorTiming.counters(self)
        out.update(
            {
                "matrix_busy_cycles": self.mma_busy,
                "matrix_mma": self.mmas,
                "matrix_macs": self.macs,
                "matrix_peak_macs_per_cycle": self.peak_macs_per_cycle,
                "vme_vrf_read_busy_cycles": self.vrf_read_busy,
                "vme_vrf_write_busy_cycles": self.vrf_write_busy,
                "vme_tile_port_busy_cycles": self.tile_port_busy,
            }
        )
        return out

    # -- decode ---------------------------------------------------------------------

    def decode(self, instr):
        op = self._ops.get(instr, 0)
        if op == 0:
            insn = vme.decode(instr)
            if insn is None:
                op = VectorTiming.decode(self, instr)
            else:
                srcs = tuple(
                    insn.field(instr, s) for s in _SCALAR_SRC.get(insn.name, ())
                )
                rd = insn.field(instr, "rd") if "rd" in insn.operands else 0
                op = VMEOp(insn, instr, tuple(s for s in srcs if s), rd or -1)
            self._ops[instr] = op
        return op

    # -- issue ----------------------------------------------------------------------

    def issue(self, op, cycle, register_file):
        if not isinstance(op, VMEOp):
            return super().issue(op, cycle, register_file)
        name = op.insn.name
        unit = self.unit
        if name.startswith("mset") or name == "vtdiscard":
            return cycle + 1
        if self.vs.vill or unit is None or not unit.mtwiden:
            return cycle + 1  # it traps on execute; no cost is owed
        if name in ("vtle", "vtse"):
            return self._issue_tile_mem(op, cycle, register_file)
        if name.startswith("vtmv"):
            return self._issue_move(op, cycle, register_file)
        if name == "vtzero":
            return self._issue_zero(op, cycle)
        return self._issue_mma(op, cycle)

    def _ptiles(self, tile, tew):
        return vme.physical_tiles(tile - tile % vme.tile_step(tew), tew)

    def _tile_write_ok(self, pts, cycle):
        """A move/load/zero into a tile waits for in-flight readers (WAR) and an in-flight multiply (WAW).

        Not for an earlier move or load: those complete in issue order (one tile
        port), so a later row or column write cannot overtake them.
        """
        at = 0
        for p in pts:
            at = max(
                at,
                self.tile_read_until[p],
                self.tile_ready[p] if self.tile_mma_writes[p] else 0,
            )
        return at <= cycle, at

    def _issue_mma(self, op, cycle):
        vs, unit = self.vs, self.unit
        if vs.vstart:
            return cycle + 1
        sew, tw = vs.sew, unit.twiden
        row = vme.arith_row(op.insn.name, sew, tw, (vs.vtype >> vme.ALTFMT_SHIFT) & 1)
        if row is None or row.ext not in unit.extensions:
            return cycle + 1
        tew = sew * tw
        e = vme.ete(unit.te, tew)
        tm, tn, tk = min(unit.tm, e), min(vs.vl, e), unit.tk
        if not (tm and tn and tk):
            return cycle + 1
        kmax = vme.KMAX[(sew, tw)]
        stride, lmul = 8 // kmax, 1 << max(0, vs.lmul_log2)
        regs = [
            r
            for base in (op.field("vs2"), op.field("vs1"))
            for k in range(tk)
            for r in range(base + k * stride, min(32, base + k * stride + lmul))
        ]
        ready = self.ready
        for r in regs:
            if ready[r] > cycle:
                return self._stall(ready[r], MATRIX_RAW)
        pts = self._ptiles(op.field("mtd") << 1, tew)
        spec = self.ime_spec
        for p in pts:
            at = (
                self.tile_mma_ready[p]
                if spec.acc_forwarding and self.tile_mma_writes[p]
                else self.tile_ready[p]
            )
            at = max(at, self.tile_read_until[p])
            if at > cycle:
                return self._stall(at, TILE_RAW)
        if self.mma_free > cycle:
            return self._stall(self.mma_free, MATRIX_BUSY)
        if self.rvv_read_free > cycle:
            return self._stall(self.rvv_read_free, MATRIX_BUSY)
        # -- commit
        macs = tm * tn * tk
        nbytes = tk * (tm + tn) * sew // 8
        occ, _, t_read = spec.mma_occupancy(
            macs, nbytes, row.fmt_a, self.spec.datapath_bits
        )
        self.mma_free = cycle + occ
        self.ime_read_free = max(self.ime_read_free, cycle + t_read)
        for r in regs:
            self.read_until[r] = max(self.read_until[r], cycle + t_read)
        done = cycle + occ + spec.pipeline_depth - 1
        for p in pts:
            self.tile_ready[p] = done
            self.tile_mma_ready[p] = cycle + occ
            self.tile_mma_writes[p] = True
        self.mmas += 1
        self.macs += macs
        self.mma_busy += occ
        self.vrf_read_busy += t_read
        return cycle + 1

    def _tile_of(self, register_file, op, reg_field, tew):
        tile, _, _ = vme.decode_tss(
            register_file[op.field(reg_field)].read_uint(), tew, self.unit.te
        )
        return self._ptiles(tile, tew)

    def _issue_move(self, op, cycle, register_file):
        vs = self.vs
        tew = vs.sew
        to_vec = op.insn.name == "vtmv.v.t"
        n = max(0, min(vs.vl, vme.ete(self.unit.te, tew)) - vs.vstart)
        nbytes = n * tew // 8
        lmul = 1 << max(0, vs.lmul_log2)
        reg = op.field("vd" if to_vec else "vs2")
        regs = range(reg, min(32, reg + lmul))
        pts = self._tile_of(register_file, op, "rs1", tew)
        ready = self.ready
        if to_vec:
            for p in pts:
                if self.tile_ready[p] > cycle:
                    return self._stall(self.tile_ready[p], TILE_RAW)
            for r in regs:
                at = max(self.read_until[r], ready[r])
                if at > cycle:
                    return self._stall(at, VECTOR_RAW)
            if self.rvv_write_free > cycle:
                return self._stall(self.rvv_write_free, MATRIX_BUSY)
        else:
            for r in regs:
                if ready[r] > cycle:
                    return self._stall(ready[r], VECTOR_RAW)
            ok, at = self._tile_write_ok(pts, cycle)
            if not ok:
                return self._stall(at, TILE_RAW)
            if self.rvv_read_free > cycle:
                return self._stall(self.rvv_read_free, MATRIX_BUSY)
        if self.tile_port_free > cycle:
            return self._stall(self.tile_port_free, MATRIX_BUSY)
        # -- commit
        spec = self.ime_spec
        rbw, wbw = spec.ports(self.spec.datapath_bits)
        t_port = max(1, math.ceil(nbytes / spec.tile_bytes_per_cycle))
        t_vrf = max(1, math.ceil(nbytes / (wbw if to_vec else rbw)))
        occ = max(t_port, t_vrf)
        self.tile_port_free = cycle + occ
        self.tile_port_busy += occ
        if to_vec:
            self.ime_write_free = max(self.ime_write_free, cycle + t_vrf)
            self.vrf_write_busy += t_vrf
            for r in regs:
                ready[r] = cycle + occ + 1
            self._touch(regs)
            for p in pts:
                self.tile_read_until[p] = max(self.tile_read_until[p], cycle + occ)
        else:
            self.ime_read_free = max(self.ime_read_free, cycle + t_vrf)
            self.vrf_read_busy += t_vrf
            for r in regs:
                self.read_until[r] = max(self.read_until[r], cycle + t_vrf)
            for p in pts:
                self.tile_ready[p] = max(self.tile_ready[p], cycle + occ + 1)
                self.tile_mma_writes[p] = False
        return cycle + 1

    def _issue_tile_mem(self, op, cycle, register_file):
        eee = op.field("eee")
        if eee > 3:
            return cycle + 1
        tew = 8 << eee
        is_load = op.insn.name == "vtle"
        vs = self.vs
        n = max(0, min(vs.vl, vme.ete(self.unit.te, tew)) - vs.vstart)
        nbytes = n * tew // 8
        pts = self._tile_of(register_file, op, "rs2", tew)
        if is_load:
            ok, at = self._tile_write_ok(pts, cycle)
            if not ok:
                return self._stall(at, TILE_RAW)
        else:
            for p in pts:
                if self.tile_ready[p] > cycle:
                    return self._stall(self.tile_ready[p], TILE_RAW)
        if self.lsu_free > cycle:
            return self._stall(self.lsu_free, VECTOR_BUSY)
        if self.tile_port_free > cycle:
            return self._stall(self.tile_port_free, MATRIX_BUSY)
        addr = register_file[op.field("rs1")].read_uint()
        region = self.memory.lookup(addr)
        if region.port_free > cycle:
            return self._stall(region.port_free, MEM)
        # -- commit
        bw = self.spec.mem_bytes_per_cycle
        if region.bandwidth is not None:
            bw = min(bw, region.bandwidth)
        t = max(
            1,
            math.ceil(nbytes / bw),
            math.ceil(nbytes / self.ime_spec.tile_bytes_per_cycle),
        )
        latency = region.access(addr, nbytes, not is_load, cycle, t)
        self.lsu_free = cycle + t
        self.mem_busy_cycles += t
        self.tile_port_free = cycle + t
        self.tile_port_busy += t
        if is_load:
            done = cycle + self.spec.mem_startup + latency + t
            for p in pts:
                self.tile_ready[p] = max(self.tile_ready[p], done)
                self.tile_mma_writes[p] = False
        else:
            for p in pts:
                self.tile_read_until[p] = max(self.tile_read_until[p], cycle + t)
        return cycle + 1

    def _issue_zero(self, op, cycle):
        tew = self.vs.sew
        pts = self._ptiles(op.field("mtd"), tew)
        ok, at = self._tile_write_ok(pts, cycle)
        if not ok:
            return self._stall(at, TILE_RAW)
        if self.tile_port_free > cycle:
            return self._stall(self.tile_port_free, MATRIX_BUSY)
        self.tile_port_free = cycle + 1
        self.tile_port_busy += 1
        for p in pts:
            self.tile_ready[p] = max(self.tile_ready[p], cycle + 1)
            self.tile_mma_writes[p] = False
        return cycle + 1


__all__ = ["VMEOp", "VMETiming", "VMETimingSpec", "parse_vme_timing"]
