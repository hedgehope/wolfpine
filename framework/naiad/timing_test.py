"""Naiad's cycle model: the ``timing:`` schema, and microbenchmarks with analytic cycle counts (gate G4).

Each microbenchmark is a hand-encoded RV64 program bracketed by two
``csrrs t, cycle, x0`` reads; the test states the formula for the difference
and asserts it exactly. With the first read issuing at cycle ``c1``, an
instruction that issues at ``c1 + 1 + x`` and is followed directly by the
second read makes the difference ``x + 2``. So a body of ``N`` instructions with
nothing to wait for reads ``N + 1``.
"""

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from framework.matrix.encodings import DECODER, TABLE
from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.run import main
from framework.naiad.timing import NaiadCostState
from framework.pe.rv.isa.isa64_testing import b_type, csr, i_type, r_type

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS = REPO_ROOT / "kernels" / "naiad"
TIMED_DESIGN = DESIGNS_DIR / "naiad-ame-timed.yaml"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")

PROG = 0x80000000
SP_BASE = 0x10000000
DRAM_DATA = 0x80008000
T0, T1, S0, A0 = 5, 6, 8, 10
SPIN = 0x0000006F  # j .
FP32 = TABLE.formats["fp32"]


def _rdcycle(rd):
    return csr(2, 0xC00, rd=rd, rs1=0)


def _addi(rd, rs1, imm):
    return i_type(0x13, 0, imm, rd=rd, rs1=rs1)


def _ld(rd, rs1, imm=0):
    return i_type(0x03, 3, imm, rd=rd, rs1=rs1)


def _mul(rd, rs1, rs2):
    return r_type(0x33, 0, 1, rd=rd, rs1=rs1, rs2=rs2)


def _div(rd, rs1, rs2):
    return r_type(0x33, 4, 1, rd=rd, rs1=rs1, rs2=rs2)


def _bne(rs1, rs2, offset):
    return b_type(1, offset, rs1=rs1, rs2=rs2)


def _mma(ad, ta, tb):
    return DECODER.encode("mma", ad=ad, ta=ta, tb=tb, **{"in": FP32, "acc": FP32})


def _tload_a(td, base, stride):
    return DECODER.encode("tload.a", td=td, base=base, stride=stride, fmt=FP32)


def _astore(acc, base, stride):
    return DECODER.encode("astore", base=base, stride=stride, fmt=FP32, **{"as": acc})


def _raw(timing=None, matrix=True):
    raw = {
        "name": "ubench",
        "cores": {"xlen": 64, "isa": ["i", "m", "zicsr", "zicntr"], "reset_pc": PROG},
        "blocks": [
            {"name": "sp", "type": "scratchpad", "base": SP_BASE, "window": 0x10000},
            {"name": "dram", "type": "dram", "base": PROG, "window": "1MiB"},
        ],
    }
    if matrix:
        raw["matrix"] = {"m": 16, "n": 16, "k": 16, "tiles": 8, "accs": 4}
    if timing is not None:
        raw["timing"] = copy.deepcopy(timing)
    return raw


#: Every latency 1 and nothing limited: the model reproduces untimed execution.
UNIT_TIMING = {
    "scalar": {
        "alu": 1,
        "mul": 1,
        "div": 1,
        "load": 1,
        "store": 1,
        "branch_taken": 0,
        "csr": 1,
    },
    "memory": {"default": {"latency": 0}, "dram": {"latency": 0, "bandwidth": None}},
    "matrix": {
        "mma_ii": 1,
        "pipeline_depth": 1,
        "tile_load_bandwidth": None,
        "tile_store_bandwidth": None,
    },
}


def _timing(scalar=None, memory=None, matrix=None):
    """A timing section: unit timing, with the given overrides."""
    out = copy.deepcopy(UNIT_TIMING)
    out["scalar"].update(scalar or {})
    out["memory"].update(memory or {})
    out["matrix"].update(matrix or {})
    return out


def _run(raw, body, regs=None, mem=None, max_cycles=20000):
    """Run ``rdcycle t0; body; rdcycle t1; j .`` and return ``(t1 - t0, soc)``."""
    soc = build(parse_design(raw))
    soc.reset()
    words = [_rdcycle(T0), *body, _rdcycle(T1), SPIN]
    for i, w in enumerate(words):
        soc.memory.write(PROG + 4 * i, w.to_bytes(4, "little"))
    for addr, data in (mem or {}).items():
        soc.memory.write(addr, data)
    core = soc.cores[0]
    for idx, value in (regs or {}).items():
        core.register_file[idx].write(value.to_bytes(8, "little"))
    spin = PROG + 4 * (len(words) - 1)
    for cycle in range(max_cycles):
        soc.clock.clock_tick(cycle)
        if core.pc_register.read_uint() == spin:
            break
    else:
        raise AssertionError("the program did not reach its spin loop")
    rf = core.register_file
    return rf[T1].read_uint() - rf[T0].read_uint(), soc


# -- timing off ----------------------------------------------------------------


def test_no_timing_section_leaves_the_cores_untimed():
    soc = build(load_design(DESIGNS_DIR / "naiad-ame.yaml"))
    assert soc.design.timing is None
    assert all(core.rv_cost is None for core in soc.cores)
    assert not any(".timing." in name for name in soc.counters())


def test_unit_timing_reproduces_untimed_cycles_exactly():
    """All latencies 1, no penalties, no limits: the cost state never stalls."""
    body = [
        _addi(A0, 0, 7),
        _mul(11, A0, A0),
        _div(12, 11, A0),
        _ld(13, S0),
        _addi(14, 13, 1),
        _addi(15, 0, 3),
        _addi(15, 15, -1),
        _bne(15, 0, -4),
        _tload_a(0, S0, 16),
        _mma(0, 0, 1),
        _astore(0, S0, 16),
    ]
    regs = {S0: SP_BASE, 16: 64}
    untimed, _ = _run(_raw(), body, regs)
    timed, soc = _run(_raw(_timing()), body, regs)
    assert (
        timed == untimed == len(body) + 5
    )  # 15 executed (the loop runs three times), + 1
    assert soc.counters()["hart0.timing.stall_cycles"] == 0


# -- scalar microbenchmarks ------------------------------------------------------

N = 16


@pytest.mark.parametrize("alu", [1, 3])
def test_dependent_alu_chain(alu):
    """``N`` x ``addi a0, a0, 1``: each waits for the last, so ``2 + (N - 1) * alu``."""
    cycles, soc = _run(_raw(_timing({"alu": alu})), [_addi(A0, A0, 1)] * N)
    assert cycles == 2 + (N - 1) * alu
    counters = soc.counters()
    assert counters["hart0.timing.stall_raw"] == (N - 1) * (alu - 1)
    assert counters["hart0.timing.stall_cycles"] == (N - 1) * (alu - 1)


def test_independent_alu_stream_hides_the_latency():
    """Eight registers round robin, latency 3 <= 8: never waits, so ``N + 1``."""
    body = [_addi(10 + i % 8, 10 + i % 8, 1) for i in range(N)]
    cycles, _ = _run(_raw(_timing({"alu": 3})), body)
    assert cycles == N + 1


def test_dependent_multiply_chain_and_unpipelined_divide():
    """``mul`` pipelines (a dependent chain pays its latency); back-to-back independent divides
    queue on the one divider, so each starts ``div`` cycles after the last."""
    cycles, _ = _run(_raw(_timing({"mul": 3})), [_mul(A0, A0, A0)] * N, {A0: 1})
    assert cycles == 2 + (N - 1) * 3
    body = [_div(10 + i % 8, 20, 21) for i in range(N)]
    cycles, soc = _run(_raw(_timing({"div": 5})), body, {20: 100, 21: 7})
    assert cycles == 2 + (N - 1) * 5
    assert soc.counters()["hart0.timing.stall_structural"] == (N - 1) * 4
    cycles, _ = _run(
        _raw(_timing({"div": 5, "div_pipelined": True})), body, {20: 100, 21: 7}
    )
    assert cycles == N + 1


@pytest.mark.parametrize(
    "where, latency",
    [(SP_BASE + 0x100, 0), (DRAM_DATA, 30)],
    ids=["scratchpad", "dram"],
)
def test_load_use_chain(where, latency):
    """Pointer chase (each word holds its own address): ``2 + (N - 1) * (load + region latency)``."""
    timing = _timing(
        {"load": 2}, {"sp": {"latency": 0}, "dram": {"latency": 30, "bandwidth": 16}}
    )
    cycles, soc = _run(
        _raw(timing),
        [_ld(A0, A0)] * N,
        {A0: where},
        {where: where.to_bytes(8, "little")},
    )
    assert cycles == 2 + (N - 1) * (2 + latency)
    counters = soc.counters()
    assert counters["hart0.timing.stall_mem"] == (N - 1) * (1 + latency)
    region = "sp" if where < PROG else "dram"
    assert counters[f"hart0.timing.mem.{region}.loads"] == N


def test_region_bandwidth_limits_an_independent_load_stream():
    """8-byte loads through a 4 B/cycle port hold it 2 cycles each: ``2 + (N - 1) * 2``."""
    body = [_ld(10 + i % 8, S0) for i in range(N)]
    timing = _timing({"load": 2}, {"sp": {"latency": 0, "bandwidth": 4}})
    cycles, soc = _run(_raw(timing), body, {S0: SP_BASE})
    assert cycles == 2 + (N - 1) * 2
    assert soc.counters()["hart0.timing.stall_mem"] == N - 1


@pytest.mark.parametrize("penalty", [0, 2])
def test_taken_branch_loop(penalty):
    """``addi a0, a0, -1; bne a0, x0, loop`` for ``N`` iterations, the last not taken:
    ``3 + (N - 1) * (2 + penalty)``."""
    cycles, soc = _run(
        _raw(_timing({"branch_taken": penalty})),
        [_addi(A0, A0, -1), _bne(A0, 0, -4)],
        {A0: N},
    )
    assert cycles == 3 + (N - 1) * (2 + penalty)
    assert soc.counters()["hart0.timing.stall_branch"] == (N - 1) * penalty


class _StubCache:
    """A latency provider: 10 cycles the first time a 64-byte line is touched, 1 after."""

    def __init__(self):
        self.calls = []
        self.lines = set()

    def access(self, addr, size, is_write):
        self.calls.append((addr, size, is_write))
        line = addr >> 6
        if line in self.lines:
            return 1
        self.lines.add(line)
        return 10


def test_a_latency_provider_replaces_the_region_latency():
    """The chain's first load misses (2 + 10), the rest hit (2 + 1): ``2 + 12 + (N - 2) * 3``."""
    raw = _raw(_timing({"load": 2}, {"dram": {"latency": 30}}))
    soc = build(parse_design(raw))
    stub = _StubCache()
    soc.cores[0].rv_cost.memory.attach_provider("dram", stub)
    # Re-run through the same SoC: _run builds its own, so drive this one by hand.
    soc.reset()
    words = [_rdcycle(T0), *[_ld(A0, A0)] * N, _rdcycle(T1), SPIN]
    for i, w in enumerate(words):
        soc.memory.write(PROG + 4 * i, w.to_bytes(4, "little"))
    soc.memory.write(DRAM_DATA, DRAM_DATA.to_bytes(8, "little"))
    core = soc.cores[0]
    core.register_file[A0].write(DRAM_DATA.to_bytes(8, "little"))
    for cycle in range(500):
        soc.clock.clock_tick(cycle)
    rf = core.register_file
    assert rf[T1].read_uint() - rf[T0].read_uint() == 2 + 12 + (N - 2) * 3
    assert stub.calls == [(DRAM_DATA, 8, False)] * N


def test_the_builder_puts_a_cache_block_in_front_of_what_it_fronts():
    """A ``cache`` with ``miss_latency: backing`` fills at the fronted region's latency (30).

    The chain's first load misses: hit 1 + fill 30, so 2 + 31; the rest hit at
    2 + 1: ``2 + 33 + (N - 2) * 3``.
    """
    raw = _raw(_timing({"load": 2}, {"dram": {"latency": 30}}))
    raw["blocks"].append(
        {
            "name": "l1d",
            "type": "cache",
            "params": {
                "fronts": ["dram"],
                "size": 1024,
                "assoc": 1,
                "line": 64,
                "miss_latency": "backing",
            },
        }
    )
    regs, mem = {A0: DRAM_DATA}, {DRAM_DATA: DRAM_DATA.to_bytes(8, "little")}
    cycles, soc = _run(raw, [_ld(A0, A0)] * N, regs, mem)
    cache = soc.blocks["l1d"]
    assert soc.cores[0].rv_cost.memory.by_name["dram"].provider is cache
    assert cycles == 2 + 33 + (N - 2) * 3
    assert (cache.stats["misses"], cache.stats["hits"]) == (1, N - 1)


# -- matrix microbenchmarks ------------------------------------------------------

# A 16x16x16 fp32 mma at 1024 MACs/cycle: ii = 4096 / 1024 = 4; depth 6 makes
# the accumulator readable ii + depth - 1 = 9 cycles after issue.
II, DEPTH = 4, 6
LAT = II + DEPTH - 1
MMA_TIMING = {"macs_per_cycle": 1024, "mma_ii": None, "pipeline_depth": DEPTH}


def _mma_raw(**matrix):
    return _raw(_timing(matrix={**MMA_TIMING, **matrix}))


@pytest.mark.parametrize(
    "forwarding, per_step",
    [(False, LAT), (True, II)],
    ids=["no-forwarding", "acc-forwarding"],
)
def test_dependent_mma_chain(forwarding, per_step):
    """``N`` mma into one accumulator: ``2 + (N - 1) * per_step``, where a step is the full
    latency without forwarding and the interval with it."""
    cycles, soc = _run(_mma_raw(acc_forwarding=forwarding), [_mma(0, 0, 1)] * N)
    assert cycles == 2 + (N - 1) * per_step
    counters = soc.counters()
    assert counters["hart0.timing.matrix_busy_cycles"] == N * II
    assert counters["hart0.timing.matrix_mma"] == N
    assert counters["hart0.timing.matrix_macs"] == N * 16**3


def test_independent_mma_stream_runs_at_the_interval():
    """Four accumulators round robin (4 * ii >= latency): ``2 + (N - 1) * ii`` even without forwarding."""
    cycles, soc = _run(
        _mma_raw(acc_forwarding=False), [_mma(i % 4, 0, 1) for i in range(N)]
    )
    assert cycles == 2 + (N - 1) * II
    counters = soc.counters()
    assert counters["hart0.timing.stall_matrix_busy"] == (N - 1) * (II - 1)
    assert counters["hart0.timing.stall_matrix_raw"] == 0


def test_without_overlap_the_core_waits_for_every_mma():
    """``overlap: false``: each mma holds the core to its completion, so ``1 + N * latency``."""
    cycles, _ = _run(_mma_raw(overlap=False, acc_forwarding=False), [_mma(0, 0, 1)] * N)
    assert cycles == 1 + N * LAT


K_SCALAR = 5
TILE_BYTES = 16 * 16 * 4


@pytest.mark.parametrize("overlap", [True, False])
def test_scalar_work_overlaps_an_inflight_mma(overlap):
    """``mma; K x addi; astore`` with ``K < latency - 1``.

    Overlapped, the addis run under the mma and only the astore waits for the
    accumulator: ``latency + 2``. Not overlapped, everything serialises and the
    astore (1024 bytes at 64 B/cycle, latency 1) is waited for too:
    ``1 + latency + K + 1024 / 64``.
    """
    body = [
        _mma(0, 0, 1),
        *[_addi(10 + i, 0, i) for i in range(K_SCALAR)],
        _astore(0, S0, 16),
    ]
    cycles, soc = _run(
        _mma_raw(overlap=overlap, tile_store_bandwidth=64, tile_store_latency=1),
        body,
        {S0: SP_BASE, 16: 64},
    )
    if overlap:
        assert cycles == LAT + 2
        assert soc.counters()["hart0.timing.stall_matrix_raw"] == LAT - 1 - K_SCALAR
    else:
        assert cycles == 1 + LAT + K_SCALAR + TILE_BYTES // 64


@pytest.mark.parametrize("bandwidth", [16, 32, 64, 128, 256])
def test_tile_load_bandwidth_sweep(bandwidth):
    """``N`` tile loads of 1 KiB each hold the load path ``T = 1024 / bandwidth`` cycles:
    ``2 + (N - 1) * T``."""
    body = [_tload_a(i % 8, S0, 16) for i in range(N)]
    timing = _timing(matrix={"tile_load_bandwidth": bandwidth})
    cycles, soc = _run(_raw(timing), body, {S0: SP_BASE, 16: 64})
    t = TILE_BYTES // bandwidth
    assert cycles == 2 + (N - 1) * t
    assert soc.counters()["hart0.timing.matrix_load_busy_cycles"] == N * t


def test_a_narrow_region_caps_the_tile_load_rate():
    """Path 128 B/cycle, region port 32 B/cycle: the region wins, ``T = 32``."""
    body = [_tload_a(i % 8, S0, 16) for i in range(4)]
    timing = _timing(
        memory={"sp": {"bandwidth": 32}}, matrix={"tile_load_bandwidth": 128}
    )
    cycles, _ = _run(_raw(timing), body, {S0: SP_BASE, 16: 64})
    assert cycles == 2 + 3 * 32


@pytest.mark.parametrize(
    "base, region_latency", [(SP_BASE, 0), (DRAM_DATA, 30)], ids=["scratchpad", "dram"]
)
def test_an_mma_waits_for_its_tile_load(base, region_latency):
    """``tload.a t0; mma`` : the mma issues when the tile is ready,
    ``1 + tile_load_latency + region latency + T`` (T = 1024 / 64)."""
    timing = _timing(
        memory={"dram": {"latency": 30, "bandwidth": None}},
        matrix={"tile_load_bandwidth": 64, "tile_load_latency": 3},
    )
    cycles, soc = _run(
        _raw(timing), [_tload_a(0, S0, 16), _mma(0, 0, 1)], {S0: base, 16: 64}
    )
    assert cycles == 1 + 3 + region_latency + TILE_BYTES // 64
    assert (
        soc.counters()["hart0.timing.stall_matrix_raw"]
        == 3 + region_latency + TILE_BYTES // 64 - 2
    )


# -- schema ----------------------------------------------------------------------


def test_defaults_are_filled_in():
    design = parse_design(_raw({}))
    timing = design.timing
    assert timing.scalar.issue_width == 1
    assert (timing.scalar.load, timing.scalar.div, timing.scalar.div_pipelined) == (
        2,
        20,
        False,
    )
    assert timing.region("dram", "dram").latency == 30
    assert timing.region("sp", "scratchpad").bandwidth is None
    assert timing.matrix.macs_per_cycle == 512
    assert parse_design(_raw({}, matrix=False)).timing.matrix is None


@pytest.mark.parametrize(
    "timing, message",
    [
        ({"scalar": {"issue_width": 3}}, r"only single \(1\) and dual \(2\)"),
        ({"scalar": {"alu": 0}}, "must be >= 1"),
        ({"scalar": {"bogus": 1}}, "unknown key"),
        ({"scalar": {"div_pipelined": 1}}, "true or false"),
        ({"memory": {"nowhere": {"latency": 1}}}, "neither a mapped block"),
        ({"memory": {"dram": {"bandwidth": 0}}}, "must be >= 1"),
        ({"memory": {"dram": {"speed": 3}}}, "unknown key"),
        (
            {"matrix": {"format_passes": {"bf16": 2}}},
            "not one of the unit's in formats",
        ),
        ({"matrix": {"format_passes": {"nope": 2}}}, "format_passes"),
        ({"vector": {"vlen": 128}}, "no vector unit to time"),
        ({"cache": {}}, "unknown key"),
    ],
)
def test_bad_timing_sections_are_rejected(timing, message):
    with pytest.raises(DesignError, match=message):
        parse_design(_raw(timing))


def test_matrix_timing_needs_a_matrix_unit():
    with pytest.raises(DesignError, match="no matrix: section"):
        parse_design(_raw({"matrix": {}}, matrix=False))


def test_memory_entries_resolve_by_name_then_type_then_default():
    timing = parse_design(
        _raw(
            {
                "memory": {
                    "scratchpad": {"latency": 4},
                    "dram": {"latency": 9},
                    "default": {"latency": 1},
                }
            }
        )
    ).timing
    assert timing.region("sp", "scratchpad").latency == 4
    assert timing.region("dram", "dram").latency == 9
    assert timing.region("perf", "perfcounters").latency == 1
    named = parse_design(
        _raw({"memory": {"sp": {"latency": 2}, "scratchpad": {"latency": 4}}})
    ).timing
    assert named.region("sp", "scratchpad").latency == 2


def test_the_cost_state_is_a_counter_source():
    soc = build(parse_design(_raw({})))
    assert isinstance(soc.cores[0].rv_cost, NaiadCostState)
    names = soc.counters()
    for reason in ("raw", "mem", "structural", "branch", "matrix_busy", "matrix_raw"):
        assert f"hart0.timing.stall_{reason}" in names
    assert "hart0.timing.matrix_busy_cycles" in names


# -- the calibrated design, end to end ---------------------------------------------

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


def _make(build_dir, **variables):
    args = [
        "make",
        "-C",
        str(KERNELS),
        f"BUILD={build_dir}",
        f"PYTHON={sys.executable}",
        f"NAIAD_CC={NAIAD_CC}",
    ]
    args += [f"{k}={v}" for k, v in variables.items()]
    subprocess.run(args, check=True, capture_output=True, text=True)


def test_the_timed_design_parses():
    design = load_design(TIMED_DESIGN)
    m = design.timing.matrix
    assert (m.macs_per_cycle, m.format_passes, m.pipeline_depth) == (
        2048,
        {"fp32": 4, "int8": 1},
        5,
    )
    assert m.mma_interval(16, 16, 16, "fp32") == 8
    assert m.mma_interval(16, 16, 16, "int8") == 2
    assert design.program_memory.name == "l1"


@needs_clang
def test_gemm_ame_under_the_timed_design(tmp_path, capsys):
    _make(tmp_path, PROG="gemm_ame", DESIGN=TIMED_DESIGN)
    assert main([str(TIMED_DESIGN), str(tmp_path / "gemm_ame.elf")]) == 0
    out, err = capsys.readouterr()
    assert out.endswith("PASS\n")
    # 64 fp32 mma at 8 cycles + 64 int8 mma at 2; 64 + 64 tile loads of 1 KiB / 256 B
    # at 64 B/cycle; 16 + 16 accumulator stores of 1 KiB.
    assert "hart0.timing.matrix_busy_cycles = 640" in err
    assert "hart0.timing.matrix_load_busy_cycles = 1280" in err
    assert "hart0.timing.matrix_store_busy_cycles = 512" in err


@needs_clang
def test_gemm_scalar_under_the_timed_design(tmp_path, capsys):
    _make(tmp_path, PROG="gemm_scalar", DESIGN=TIMED_DESIGN, CFLAGS="-O2 -DGEMM_M=32")
    assert main([str(TIMED_DESIGN), str(tmp_path / "gemm_scalar.elf")]) == 0
    out, err = capsys.readouterr()
    assert out.endswith("PASS\n")
    assert "hart0.timing.stall_mem = 0" not in err
