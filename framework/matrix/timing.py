"""The matrix unit's cycle model (AME), and the equal-resource normaliser.

Timing only. Nothing here changes what an instruction computes: the functional
unit (:mod:`framework.matrix.unit`, :mod:`framework.matrix.ame_isa`) executes
every matrix instruction in the tick it issues, exactly as it does untimed, and
this module decides only *when* the core may issue it and when its results may
be read. It is driven by the Naiad core's cost state
(:class:`framework.naiad.timing.NaiadCostState`), which hands it each matrix
word before the word executes.

**The model.** An attached unit fed by an in-order, single-issue core with no
instruction queue between them, so every hazard is resolved at issue:

* Three independent pipes: the **mma pipe** (mma, and the ascal/aadds
  epilogue), the **load path** (tload.a/.b, aload) and the **store path**
  (tstore.a/.b, astore). An instruction waits for its pipe to be free (stall
  reason ``matrix_busy``), then holds it for its initiation interval.
* A **scoreboard** on every tile and accumulator: the cycle its value becomes
  readable (RAW / WAW) and the cycle its last in-flight reader is done with it
  (WAR). A dependent access waits (``matrix_raw``); an independent scalar
  instruction does not, so scalar code runs under an in-flight mma. With
  ``overlap: false`` the core instead waits for each matrix instruction to
  complete before issuing anything (``matrix_busy``), the no-overlap baseline.
* ``mma`` on in-format ``f``: initiation interval
  ``ii = ceil(M * N * K * passes[f] / macs_per_cycle)`` (or ``mma_ii`` given
  outright), result readable at ``issue + ii + pipeline_depth - 1``. With
  ``acc_forwarding`` an mma into the accumulator the previous mma wrote may
  start as soon as the pipe is free (accumulation in place, as a systolic array
  or the Tensix Dst does), while an ``astore`` of it still waits for the full
  latency.
* A tile/accumulator transfer of ``B`` bytes to or from a memory region takes
  ``T = ceil(B / min(path bandwidth, region bandwidth))`` cycles on its path
  (and on the region's port, which scalar loads share), and is readable at
  ``issue + path latency + region latency + T - 1``. A region with a latency
  provider (a cache) charges the provider's answer in place of its latency.
* ``azero``, ``tshift`` and ``mcfg`` are fixed latencies on no pipe.

Conventions shared with the scalar model: "ready at ``r``" means an
instruction reading the value may issue at cycle ``r``; a latency of 1 is
back-to-back, i.e. untimed. Instruction fetch is free.

**Parameters** are :class:`AMETimingSpec`, the design's ``timing.matrix``
section; every field has a documented default. :func:`normalise` derives an
equal-resource unit and its timing from a peak-FLOPs / storage / bandwidth
budget, so coupling modes are compared at equal hardware cost.

**Calibration.** ``naiad-ame-timed.yaml`` takes its AME numbers from the Tensix
FPU as this repo models it (the Tensix instruction-cost and unit-cost
tables); the design file cites each entry it took, and nothing here reads
those tables, so the TT cost model's consumer set is unchanged. The
p150 hardware step is user-run and goes:

1. On the card, time a ``matmul_tiles`` inner loop at LoFi, HiFi2 and HiFi4
   (``perfbench/tensixbench`` phase B) and difference the fidelities: the
   per-phase step divided by the 16 MVMULs per tile phase is the cycles per
   MVMUL, hence ``macs_per_cycle = 2048 / that`` and ``format_passes``.
2. Time an unpack-only loop over N tiles at each throttle mode and fit
   cycles = a + b * bytes: ``b`` gives ``tile_load_bandwidth``, ``a`` the
   ``tile_load_latency``. The same on a pack-only loop gives the store path.
3. Time a dependent chain of matmuls into one Dst tile against independent
   ones: the difference per step is ``pipeline_depth - 1`` (zero confirms
   ``acc_forwarding``).
4. Put the fitted numbers in a design and rerun the ``ubench`` tests' formulas
   against Naiad, and ``kernels/naiad/gemm_ame`` against tt-metal's matmul at the
   same shape, recording the error per kernel.
"""

import math
from dataclasses import dataclass, field

#: Stall reasons this model reports (a subset of the core's).
MATRIX_BUSY = "matrix_busy"
MATRIX_RAW = "matrix_raw"
MEM = "mem"

#: ``exec`` names (``encodings.yaml``) -> the pipe they occupy.
_PIPE = {
    "tile_load": "load",
    "acc_load": "load",
    "tile_store": "store",
    "acc_store": "store",
    "mma": "mma",
    "acc_scalar": "mma",
}


@dataclass(frozen=True)
class AMETimingSpec:
    """``timing.matrix``: the AME unit's cycle parameters, with their defaults.

    Bandwidths are bytes per cycle (``None`` is unlimited), latencies and
    intervals cycles. The defaults are an illustrative mid-size unit, not a
    calibrated one: a 16x16x16 fp32 mma every 8 cycles, 64 B/cycle tile paths.
    """

    #: Peak multiply-accumulates per cycle for a one-pass format.
    macs_per_cycle: int = 512
    #: Fixed mma initiation interval, overriding the one derived from ``macs_per_cycle``.
    mma_ii: int | None = None
    #: In-format name -> passes through the array per mma (the Tensix fidelity phases); default 1.
    format_passes: dict = field(default_factory=dict, hash=False)
    #: Cycles from an mma's last pass to its accumulator being readable, plus one.
    pipeline_depth: int = 4
    #: An mma may accumulate into the accumulator the previous mma is still writing.
    acc_forwarding: bool = True
    #: Scalar instructions issue while matrix instructions are in flight.
    overlap: bool = True
    tile_load_bandwidth: float | None = 64
    tile_load_latency: int = 1
    tile_store_bandwidth: float | None = 64
    tile_store_latency: int = 1
    azero: int = 1
    mcfg: int = 1
    tshift: int = 1
    #: ascal / aadds latency; they hold the mma pipe for this long.
    epilogue: int = 1

    def mma_interval(self, m, n, k, in_format):
        if self.mma_ii is not None:
            return self.mma_ii
        passes = self.format_passes.get(in_format, 1)
        return max(1, math.ceil(m * n * k * passes / self.macs_per_cycle))


def transfer_cycles(nbytes, *bandwidths):
    """Cycles to move ``nbytes`` at the narrowest of ``bandwidths`` (``None`` = unlimited), at least 1."""
    limits = [b for b in bandwidths if b is not None]
    if not limits:
        return 1
    return max(1, math.ceil(nbytes / min(limits)))


class _Op:
    """One decoded matrix word, memoised by the cost state (operands are literals)."""

    __slots__ = (
        "exec",
        "srcs",
        "dst",
        "tiles_read",
        "tiles_written",
        "accs_read",
        "accs_written",
        "base",
        "nbytes",
        "ii",
        "macs",
        "pipe",
    )


class AMETiming:
    """One hart's matrix-unit timing state: pipes, tile/acc scoreboard, counters.

    ``unit`` is the functional :class:`~framework.matrix.unit.MatrixUnit` (read
    for its shape, formats and decoder, never written); ``memory`` is the
    core's :class:`~framework.naiad.timing.MemoryTiming`, for the region a
    transfer touches.
    """

    def __init__(self, spec, unit, memory):
        self.spec = spec
        self.unit = unit
        self.memory = memory
        self.decoder = unit.decoder
        self.peak_macs_per_cycle = spec.macs_per_cycle
        self.reset()

    def reset(self):
        unit = self.unit
        self.tile_ready = [0] * unit.num_tiles
        self.tile_read_free = [0] * unit.num_tiles
        self.acc_ready = [0] * unit.num_accs
        self.acc_mma_ready = [0] * unit.num_accs
        self.acc_read_free = [0] * unit.num_accs
        self.pipe_free = {"mma": 0, "load": 0, "store": 0}
        self.busy = {"mma": 0, "load": 0, "store": 0}
        self.mmas = 0
        self.macs = 0
        self.stall_until = 0
        self.stall_reason = MATRIX_BUSY

    def counters(self):
        return {
            "matrix_busy_cycles": self.busy["mma"],
            "matrix_load_busy_cycles": self.busy["load"],
            "matrix_store_busy_cycles": self.busy["store"],
            "matrix_mma": self.mmas,
            "matrix_macs": self.macs,
            "matrix_peak_macs_per_cycle": self.peak_macs_per_cycle,
        }

    # -- decode --------------------------------------------------------------

    def decode(self, instr):
        """``_Op`` for a matrix word, or None if the unit would reject it (it then traps on execute)."""
        decoded = self.decoder.decode(instr)
        if decoded is None:
            return None
        insn, ops = decoded
        unit, spec = self.unit, self.spec
        op = _Op()
        op.exec = insn.exec
        op.pipe = _PIPE.get(insn.exec)
        op.srcs, op.dst = [], -1
        op.tiles_read, op.tiles_written, op.accs_read, op.accs_written = (), (), (), ()
        op.base, op.nbytes, op.ii, op.macs = None, 0, 0, 0
        for o in insn.operands:
            if o.kind == "gpr":
                if o.field == "rd":
                    op.dst = ops[o.name]
                elif ops[o.name]:
                    op.srcs.append(ops[o.name])
            elif o.kind == "tile" and ops[o.name] >= unit.num_tiles:
                return None
            elif o.kind == "acc" and ops[o.name] >= unit.num_accs:
                return None
        op.srcs = tuple(op.srcs)
        if "base" in ops:
            op.base = ops["base"]

        def fmt(name, table):
            f = table.get(ops[name])
            return None if f is None else f

        kind = insn.exec
        if kind == "tile_load":
            f = fmt("fmt", unit.in_formats)
            if f is None:
                return None
            rows, cols = unit.shape(insn.params["role"])
            op.nbytes = rows * cols * f.dtype.itemsize
            op.tiles_written = (ops["td"],)
        elif kind == "tile_store":
            f = fmt("fmt", unit.in_formats)
            if f is None:
                return None
            rows, cols = unit.shape(insn.params["role"])
            op.nbytes = rows * cols * f.dtype.itemsize
            op.tiles_read = (ops["ts"],)
        elif kind in ("acc_load", "acc_store"):
            f = fmt("fmt", unit.acc_formats)
            if f is None:
                return None
            op.nbytes = unit.m * unit.n * f.dtype.itemsize
            if kind == "acc_load":
                op.accs_written = (ops["ad"],)
            else:
                op.accs_read = (ops["as"],)
        elif kind == "acc_zero":
            op.accs_written = (ops["ad"],)
        elif kind == "tile_shift":
            op.tiles_read, op.tiles_written = (ops["ts"],), (ops["td"],)
        elif kind == "mma":
            f = fmt("in", unit.in_formats)
            if f is None or fmt("acc", unit.acc_formats) is None:
                return None
            op.ii = spec.mma_interval(unit.m, unit.n, unit.k, f.name)
            op.macs = unit.m * unit.n * unit.k
            op.tiles_read = (ops["ta"], ops["tb"])
            op.accs_read = op.accs_written = (ops["ad"],)
        elif kind == "acc_scalar":
            op.ii = spec.epilogue
            op.accs_read = op.accs_written = (ops["ad"],)
        return op

    # -- issue ---------------------------------------------------------------

    def _stall(self, until, reason):
        self.stall_until, self.stall_reason = until, reason
        return -1

    def issue(self, op, cycle, register_file):
        """Issue ``op`` at ``cycle`` if nothing blocks it.

        Returns the cycle the instruction completes (its last result readable),
        or -1 with :attr:`stall_until` / :attr:`stall_reason` set. Like the
        core's ``can_issue``, nothing is mutated before the last point that can
        return -1.
        """
        spec = self.spec
        kind = op.exec
        # Hazards on the unit's registers.
        for t in op.tiles_read:
            if self.tile_ready[t] > cycle:
                return self._stall(self.tile_ready[t], MATRIX_RAW)
        for t in op.tiles_written:
            at = max(self.tile_ready[t], self.tile_read_free[t])
            if at > cycle:
                return self._stall(at, MATRIX_RAW)
        for a in op.accs_read:
            at = (
                self.acc_mma_ready[a]
                if kind == "mma" and spec.acc_forwarding
                else self.acc_ready[a]
            )
            if at > cycle:
                return self._stall(at, MATRIX_RAW)
        for a in op.accs_written:
            at = self.acc_read_free[a]
            if kind != "mma":
                at = max(at, self.acc_ready[a])
            if at > cycle:
                return self._stall(at, MATRIX_RAW)
        pipe = op.pipe
        if pipe is not None and self.pipe_free[pipe] > cycle:
            return self._stall(self.pipe_free[pipe], MATRIX_BUSY)

        # The memory side: region port, then the transfer's length and latency.
        region = None
        if op.base is not None:
            base = register_file[op.base].read_uint() if op.base else 0
            region = self.memory.lookup(base)
            if region.port_free > cycle:
                return self._stall(region.port_free, MEM)

        # -- commit: nothing below can stall --------------------------------
        if kind == "mma" or kind == "acc_scalar":
            ii = op.ii
            self.pipe_free["mma"] = cycle + ii
            self.busy["mma"] += ii
            done = cycle + ii + (spec.pipeline_depth - 1 if kind == "mma" else 0)
            a = op.accs_written[0]
            self.acc_ready[a] = done
            self.acc_mma_ready[a] = cycle + ii if kind == "mma" else done
            for t in op.tiles_read:
                self.tile_read_free[t] = max(self.tile_read_free[t], cycle + ii)
            if kind == "mma":
                self.mmas += 1
                self.macs += op.macs
            return done
        if region is not None:
            is_write = pipe == "store"
            if is_write:
                bandwidth, latency = spec.tile_store_bandwidth, spec.tile_store_latency
            else:
                bandwidth, latency = spec.tile_load_bandwidth, spec.tile_load_latency
            t_cycles = transfer_cycles(op.nbytes, bandwidth, region.bandwidth)
            latency += region.access(base, op.nbytes, is_write, cycle, t_cycles)
            self.pipe_free[pipe] = cycle + t_cycles
            self.busy[pipe] += t_cycles
            done = cycle + latency + t_cycles - 1
            for t in op.tiles_written:
                self.tile_ready[t] = done
            for a in op.accs_written:
                self.acc_ready[a] = self.acc_mma_ready[a] = done
            for t in op.tiles_read:
                self.tile_read_free[t] = max(self.tile_read_free[t], cycle + t_cycles)
            for a in op.accs_read:
                self.acc_read_free[a] = max(self.acc_read_free[a], cycle + t_cycles)
            return done
        if kind == "acc_zero":
            done = cycle + spec.azero
            a = op.accs_written[0]
            self.acc_ready[a] = self.acc_mma_ready[a] = done
            return done
        if kind == "tile_shift":
            done = cycle + spec.tshift
            self.tile_ready[op.tiles_written[0]] = done
            ts = op.tiles_read[0]
            self.tile_read_free[ts] = max(self.tile_read_free[ts], done)
            return done
        # mcfg (config_read): its x-register result is the core's to scoreboard.
        return cycle + spec.mcfg


# -- the equal-resource normaliser ---------------------------------------------


@dataclass(frozen=True)
class NormalisedConfig:
    """An equal-resource unit for one coupling mode.

    ``matrix`` is a design's ``matrix:`` section and ``timing`` its
    ``timing.matrix`` section, ready to merge into a raw design mapping;
    ``derived`` records the arithmetic (shape, storage used, interval, achieved peak).
    """

    mode: str
    matrix: dict
    timing: dict
    derived: dict


def normalise(
    peak_flops_per_cycle,
    operand_bytes,
    mem_bytes_per_cycle,
    *,
    mode="ame",
    in_format="fp32",
    acc_format="fp32",
    tiles=8,
    accs=4,
    power_of_two=True,
):
    """Derive a unit and its timing from a hardware budget, so modes compare at equal cost.

    The three budgets are what any coupling mode has to pay for: ``peak_flops_per_cycle``
    (2 FLOPs per MAC), ``operand_bytes`` of operand storage next to the
    multipliers, and ``mem_bytes_per_cycle`` between that storage and memory.

    **AME** (implemented). Square tiles of side ``d``: ``tiles`` input tiles of
    ``d*d`` ``in_format`` elements and ``accs`` accumulators of ``d*d``
    ``acc_format`` elements must fit the storage,

        d = floor(sqrt(operand_bytes / (tiles * in_bytes + accs * acc_bytes)))

    (rounded down to a power of two when ``power_of_two``), and the array runs at
    ``macs_per_cycle = peak_flops_per_cycle / 2``, so an mma's interval is
    ``ceil(d**3 / macs_per_cycle)``. Both tile paths get ``mem_bytes_per_cycle``.
    """
    from framework.matrix.formats import get_format

    if mode in ("ime", "vme"):
        return _NORMALISERS[mode](
            peak_flops_per_cycle, operand_bytes, mem_bytes_per_cycle
        )
    if mode != "ame":
        raise ValueError(f"unknown coupling mode {mode!r}; expected ame, ime or vme")
    if peak_flops_per_cycle < 2 or peak_flops_per_cycle % 2:
        raise ValueError(
            f"peak_flops_per_cycle must be an even number >= 2, got {peak_flops_per_cycle}"
        )
    if mem_bytes_per_cycle is not None and mem_bytes_per_cycle <= 0:
        raise ValueError(f"mem_bytes_per_cycle must be > 0, got {mem_bytes_per_cycle}")
    in_fmt, acc_fmt = get_format(in_format), get_format(acc_format)
    in_bytes, acc_bytes = in_fmt.dtype.itemsize, acc_fmt.dtype.itemsize
    per_d2 = tiles * in_bytes + accs * acc_bytes
    d = math.isqrt(operand_bytes // per_d2)
    if power_of_two and d:
        d = 1 << (d.bit_length() - 1)
    d = min(d, 256)
    if d < 1:
        raise ValueError(
            f"{operand_bytes} bytes cannot hold {tiles} {in_fmt.name} tiles and {accs} {acc_fmt.name} accumulators of even 1x1"
        )
    macs = peak_flops_per_cycle // 2
    ii = max(1, math.ceil(d**3 / macs))
    matrix = {
        "mode": "ame",
        "m": d,
        "n": d,
        "k": d,
        "tiles": tiles,
        "accs": accs,
        "formats": {"in": [in_fmt.name], "acc": [acc_fmt.name]},
    }
    timing = {
        "macs_per_cycle": macs,
        "tile_load_bandwidth": mem_bytes_per_cycle,
        "tile_store_bandwidth": mem_bytes_per_cycle,
    }
    derived = {
        "d": d,
        "storage_bytes": d * d * per_d2,
        "mma_ii": ii,
        "achieved_flops_per_cycle": 2 * d**3 / ii,
        "tile_load_cycles": transfer_cycles(d * d * in_bytes, mem_bytes_per_cycle),
    }
    return NormalisedConfig("ame", matrix, timing, derived)


def _normalise_ime(peak_flops_per_cycle, operand_bytes, mem_bytes_per_cycle):
    """IME (Phase C): mma reads and writes vector register groups; no new state.

    Sketch: the operand storage *is* the VRF, so ``VLEN = operand_bytes * 8 / 32``.
    With element width ``SEW`` and a register group of ``LMUL`` registers holding
    an ``M x K`` tile, ``M * K * SEW = LMUL * VLEN``; A, B and C tiles each take a
    group, so the largest square tile is ``d = sqrt(LMUL * VLEN / SEW)`` with
    ``3 * LMUL <= 32`` (minus the groups the kernel needs for itself). Peak MACs
    as AME (``peak / 2``), but each mma also reads ``2 * d * d * SEW / 8`` and
    writes ``d * d * acc_bytes`` through the VRF ports, so the timing adds a
    VRF-port bandwidth term: ``ii = max(ceil(d**3 / macs), ceil(vrf_bytes / vrf_port_bytes_per_cycle))``.
    Memory bandwidth goes to vector loads/stores instead of tile loads.
    """
    raise NotImplementedError("IME normalisation needs the RVV register file (Phase C)")


def _normalise_vme(peak_flops_per_cycle, operand_bytes, mem_bytes_per_cycle):
    """VME (Phase D): vector-length-agnostic matrix registers.

    Sketch: matrix registers of ``R = VLEN / SEW`` rows by ``VLEN / SEW`` columns
    (the proposal's ``MLEN = VLEN * VLEN / SEW`` bits each), with the
    storage budget fixing ``VLEN`` from ``n_mregs * MLEN / 8 = operand_bytes``.
    The outer-product form issues ``K`` rank-1 updates per tile, each
    ``R * R`` MACs, so ``ii = ceil(R * R / macs)`` per update and the tile takes
    ``K`` of them; loads move one ``VLEN``-bit row per cycle at most, so the
    tile-load term is ``R * max(1, VLEN / 8 / mem_bytes_per_cycle)``. Pin the
    definition to a dated task-group draft before implementing.
    """
    raise NotImplementedError(
        "VME normalisation is not defined yet (Phase D; pin the proposal first)"
    )


_NORMALISERS = {"ime": _normalise_ime, "vme": _normalise_vme}
