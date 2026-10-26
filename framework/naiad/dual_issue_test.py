"""The dual-issue front end (``timing.scalar.issue_width: 2``): pairing rules with analytic counts.

Same method as ``timing_test.py``: a hand-encoded program bracketed by two
``rdcycle`` reads. ``rdcycle`` is a CSR access, so it issues alone: with the
first read at cycle ``c1``, a body whose last instruction issues at ``c1 + x``
reads ``x + 1``. ``N`` independent instructions that all pair read ``N/2 + 1``
(single issue: ``N + 1``).
"""

import copy

import pytest
import yaml

from framework.naiad import timing_test as tt
from framework.naiad.builder import NaiadClock, build
from framework.naiad.design import parse_design
from framework.pe.rv.isa.isa64_testing import b_type, s_type
from framework.pe.rv.timing import vector_test as vt
from framework.pe.rv.timing.core import DualIssueCostState, NaiadCostState

N = 16
A0, A1 = 10, 11


def _dual(timing):
    timing = dict(timing)
    timing["scalar"] = {**timing.get("scalar", {}), "issue_width": 2}
    return timing


def _scalar(scalar=None, width=2):
    timing = tt._timing(scalar)
    return tt._raw(_dual(timing) if width == 2 else timing)


def _vector(width=2):
    raw = vt._raw()
    if width == 2:
        raw["timing"] = _dual(raw["timing"])
    return raw


def _sd(rs2, rs1, imm=0):
    return s_type(3, imm, rs1=rs1, rs2=rs2)


def _beq(rs1, rs2, offset):
    return b_type(0, offset, rs1=rs1, rs2=rs2)


# -- wiring ----------------------------------------------------------------------


def test_single_issue_stays_the_default():
    soc = build(parse_design(tt._raw(tt._timing())))
    assert type(soc.cores[0].rv_cost) is NaiadCostState
    assert soc.clock.issue_width == 1
    assert "hart0.timing.paired" not in soc.counters()


def test_issue_width_2_installs_the_dual_issue_state_and_clock():
    soc = build(parse_design(_scalar()))
    assert soc.design.timing.scalar.issue_width == 2
    assert isinstance(soc.cores[0].rv_cost, DualIssueCostState)
    assert soc.clock.issue_width == 2
    assert soc.counters()["hart0.timing.paired"] == 0


class _Core:
    def __init__(self, log):
        self.log, self.csrs = log, type("C", (), {"cycles": 0})()

    def clock_tick(self, cycle):
        self.csrs.cycles += 1
        self.log.append(("core", self, cycle))


class _Block:
    def __init__(self, log):
        self.log = log

    def clock_tick(self, cycle):
        self.log.append(("block", self, cycle))


def test_clock_ticks_cores_once_per_slot_and_blocks_once_per_cycle():
    log = []
    a, b, dma = _Core(log), _Core(log), _Block(log)
    clock = NaiadClock([a, b, dma])
    clock.set_issue_width(2, [a, b])
    for cycle in range(3):
        clock.clock_tick(cycle)
    assert log[:5] == [
        ("core", a, 0),
        ("core", b, 0),
        ("core", a, 0),
        ("core", b, 0),
        ("block", dma, 0),
    ]
    assert len(log) == 15
    assert a.csrs.cycles == b.csrs.cycles == 3  # cycles, not slots
    assert clock.current_cycle == 2


# -- scalar pairing --------------------------------------------------------------


def test_independent_alu_stream_pairs():
    body = [tt._addi(10 + i % 8, 10 + i % 8, 1) for i in range(N)]
    single, _ = tt._run(_scalar(width=1), body)
    dual, soc = tt._run(_scalar(), body)
    assert (single, dual) == (N + 1, N // 2 + 1)
    c = soc.counters()
    assert c["hart0.timing.paired"] == N // 2
    assert c["hart0.timing.stall_cycles"] == 0


def test_dependent_chain_does_not_pair():
    """A RAW on slot 0's result refuses slot 1, and the refusal is not a stall."""
    cycles, soc = tt._run(_scalar(), [tt._addi(A0, A0, 1)] * N)
    assert cycles == N + 1
    c = soc.counters()
    assert c["hart0.timing.paired"] == 0
    assert c["hart0.timing.stall_cycles"] == 0


def test_latency_still_waits_in_slot_0():
    """``mul a0; addi a1, a0`` x N/2 at mul latency 3: the addi waits 2 cycles in slot 0, as single issue."""
    body = [tt._mul(A0, 20, 20), tt._addi(A1, A0, 1)] * (N // 2)
    single, _ = tt._run(_scalar({"mul": 3}, width=1), body, {20: 3})
    dual, soc = tt._run(_scalar({"mul": 3}), body, {20: 3})
    # Each addi issues 3 after its mul; the next mul pairs with the addi.
    assert single == 1 + (N // 2) * 4 - 1 + 1
    assert dual == 3 * (N // 2) + 2
    assert soc.counters()["hart0.timing.stall_raw"] == 2 * (N // 2)
    assert soc.counters()["hart0.timing.paired"] == N // 2 - 1


@pytest.mark.parametrize(
    "first, second, pairs",
    [
        (
            lambda i: tt._ld(10 + i % 4, tt.S0),
            lambda i: tt._addi(20 + i % 4, 0, 1),
            True,
        ),
        (lambda i: tt._ld(10 + i % 4, tt.S0), lambda i: _sd(0, tt.S0, 8), False),
        (
            lambda i: tt._ld(10 + i % 4, tt.S0),
            lambda i: tt._ld(20 + i % 4, tt.S0),
            False,
        ),
        (
            lambda i: tt._mul(10 + i % 4, 20, 20),
            lambda i: tt._addi(24 + i % 4, 0, 1),
            True,
        ),
        (
            lambda i: tt._mul(10 + i % 4, 20, 20),
            lambda i: tt._mul(24 + i % 4, 20, 20),
            False,
        ),
        (
            lambda i: tt._mul(10 + i % 4, 20, 20),
            lambda i: tt._ld(24 + i % 4, tt.S0),
            True,
        ),
    ],
    ids=["ld+alu", "ld+sd", "ld+ld", "mul+alu", "mul+mul", "mul+ld"],
)
def test_single_pipes(first, second, pairs):
    """One memory op and one multiply/divide per cycle; ALU work pairs with either."""
    body = [w for i in range(N // 2) for w in (first(i), second(i))]
    cycles, soc = tt._run(_scalar(), body, {tt.S0: tt.SP_BASE, 20: 3})
    assert cycles == (N // 2 + 1 if pairs else N + 1)
    assert soc.counters()["hart0.timing.stall_cycles"] == 0


def test_a_branch_ends_the_issue_group():
    """``addi a1; addi a0, -1; bne a0, loop``: the two addis pair, nothing pairs after the bne."""
    body = [tt._addi(A1, A1, 1), tt._addi(A0, A0, -1), tt._bne(A0, 0, -8)]
    single, _ = tt._run(_scalar(width=1), body, {A0: N})
    dual, soc = tt._run(_scalar(), body, {A0: N})
    assert (single, dual) == (3 * N + 1, 2 * N + 1)
    assert soc.counters()["hart0.timing.paired"] == N


def test_a_not_taken_branch_also_ends_the_group():
    body = [_beq(0, 21, 8), tt._addi(A0, 0, 1), tt._addi(A1, 0, 1)]
    cycles, _ = tt._run(_scalar(), body, {21: 1})
    assert cycles == 3  # beq | addi + addi | rdcycle


def test_mcycle_counts_cycles_not_slots():
    soc = build(parse_design(_scalar()))
    soc.reset()
    soc.memory.write(tt.PROG, tt.SPIN.to_bytes(4, "little"))  # j .
    for cycle in range(200):
        soc.clock.clock_tick(cycle)
    assert soc.cores[0].csrs.cycles == 200
    assert soc.counters()["hart0.cycle"] == 200


# -- vector and matrix pairing ---------------------------------------------------


def test_vector_op_pairs_with_scalar_alu_work():
    """vadd (one datapath cycle at VL 8) + addi: one cycle per pair."""
    body = [
        w
        for i in range(N // 2)
        for w in (vt._vadd(8 + i % 4, 1, 2), tt._addi(10 + i % 4, 0, 1))
    ]
    single, _ = vt._run(_vector(width=1), body)
    dual, soc = vt._run(_vector(), body)
    assert (single, dual) == (N + 1, N // 2 + 1)
    assert soc.counters()["hart0.timing.paired"] == N // 2


def test_vector_ops_do_not_pair_with_each_other():
    """vle32 (one LSU cycle) + vadd use different vector resources but one dispatch port."""
    body = [
        w
        for i in range(N // 2)
        for w in (vt.vmem(True, 32, 8 + i % 4, vt.S0), vt._vadd(16 + i % 4, 1, 2))
    ]
    cycles, _ = vt._run(_vector(), body)
    assert cycles == N + 1


@pytest.mark.parametrize("scalar, pairs", [("ld", False), ("addi", True)])
def test_vector_memory_op_takes_the_memory_pipe(scalar, pairs):
    second = {
        "ld": lambda i: tt._ld(10 + i % 4, vt.S0),
        "addi": lambda i: tt._addi(10 + i % 4, 0, 1),
    }[scalar]
    body = [
        w
        for i in range(N // 2)
        for w in (vt.vmem(True, 32, 8 + i % 4, vt.S0), second(i))
    ]
    cycles, _ = vt._run(_vector(), body)
    assert cycles == (N // 2 + 1 if pairs else N + 1)


def test_matrix_op_pairs_with_scalar_alu_work():
    """AME mma (ii 1) round robin over the four accumulators, each with an addi beside it."""
    body = [
        w
        for i in range(N // 2)
        for w in (tt._mma(i % 4, 0, 1), tt._addi(10 + i % 4, 0, 1))
    ]
    single, _ = tt._run(_scalar(width=1), body)
    dual, soc = tt._run(_scalar(), body)
    assert (single, dual) == (N + 1, N // 2 + 1)
    assert soc.counters()["hart0.timing.paired"] == N // 2


def test_matrix_ops_do_not_pair_with_each_other():
    body = [tt._mma(i % 4, 0, 1) for i in range(N)]
    cycles, _ = tt._run(_scalar(), body)
    assert cycles == N + 1


# -- end to end ------------------------------------------------------------------


@tt.needs_clang
def test_dual_issue_runs_a_compiled_kernel_identically_in_fewer_cycles(tmp_path):
    """gemm_ame under naiad-ame-timed: the same exit code and console, fewer cycles, some pairs."""
    tt._make(tmp_path, PROG="gemm_ame", DESIGN=tt.TIMED_DESIGN)
    raw = yaml.safe_load(tt.TIMED_DESIGN.read_text())
    results = []
    for width in (1, 2):
        raw.setdefault("timing", {}).setdefault("scalar", {})["issue_width"] = width
        soc = build(parse_design(copy.deepcopy(raw), tt.TIMED_DESIGN))
        soc.load(tmp_path / "gemm_ame.elf")
        results.append((soc.run(10_000_000), soc))
    (one, _), (two, soc) = results
    assert one.exit_code == two.exit_code == 0
    # The console differs only in the cycle counts the kernel prints.
    assert one.console.endswith(b"PASS\n")
    assert two.console.endswith(b"PASS\n")
    assert two.cycles < one.cycles
    c = soc.counters()
    assert c["hart0.timing.paired"] > 0
    assert c["hart0.instret"] == one.counters["hart0.instret"]
