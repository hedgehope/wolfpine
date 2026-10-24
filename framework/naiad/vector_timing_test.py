"""The vector cycle model (``timing.vector``): schema checks and microbenchmarks with analytic counts.

Same method as ``timing_test.py``: a hand-encoded program bracketed by two
``rdcycle`` reads, whose difference the test derives from the model's rules.
An instruction issuing at ``c1 + 1 + x`` followed directly by the second read
gives ``x + 2``. The ``vsetvli`` runs before the first read, so it is outside
every bracket.
"""

import copy

import pytest

from framework.naiad.builder import build
from framework.naiad.design import DesignError, parse_design
from framework.naiad.vector_timing import classify
from framework.pe.rv.isa.isa64_testing import csr, i_type, r_type
from framework.pe.rv.isa.v.vtesting import (
    OPIVV,
    OPIVX,
    OPMVV,
    opv,
    vmem,
    vsetvli,
    vtype,
)

PROG = 0x80000000
SP = 0x10000000
T0, T1, A0, A1, A2, S0, S1 = 5, 6, 10, 11, 12, 8, 9
SPIN = 0x0000006F
DP, STARTUP, DIV_II, MEM_STARTUP, LSU, DRAM_LAT = 256, 4, 8, 2, 32, 10


def _raw(vector=None, memory=None, scalar=None):
    timing = {
        "scalar": {
            "alu": 1,
            "mul": 3,
            "div": 1,
            "load": 1,
            "store": 1,
            "branch_taken": 0,
            "csr": 1,
        },
        "memory": {
            "sp": {"latency": 0, "bandwidth": None},
            "dram": {"latency": DRAM_LAT, "bandwidth": None},
        },
        "vector": {
            "datapath_bits": DP,
            "startup": STARTUP,
            "div_ii": DIV_II,
            "mem_startup": MEM_STARTUP,
            "mem_bytes_per_cycle": LSU,
            "indexed_elements_per_cycle": 1,
        },
    }
    timing["vector"].update(vector or {})
    timing["memory"].update(memory or {})
    timing["scalar"].update(scalar or {})
    return {
        "name": "vbench",
        "cores": {
            "xlen": 64,
            "isa": ["i", "m", "f", "d", "v", "zicsr", "zicntr"],
            "vlen": 256,
            "elen": 64,
            "reset_pc": PROG,
        },
        "blocks": [
            {"name": "sp", "type": "scratchpad", "base": SP, "window": 0x10000},
            {"name": "dram", "type": "dram", "base": PROG, "window": "1MiB"},
        ],
        "timing": copy.deepcopy(timing),
    }


def _rdcycle(rd):
    return csr(2, 0xC00, rd=rd, rs1=0)


def _run(raw, body, sew=32, lmul=1, regs=None, vl=None):
    """``vsetvli; rdcycle t0; body; rdcycle t1; j .`` -> ``(t1 - t0, soc)``."""
    soc = build(parse_design(raw))
    soc.reset()
    core = soc.cores[0]
    regs = dict(regs or {})
    regs.setdefault(S0, SP)
    regs.setdefault(S1, PROG + 0x8000)
    head = (
        [vsetvli(0, A2, vtype(sew, lmul))]
        if vl is not None
        else [vsetvli(A2, 0, vtype(sew, lmul))]
    )
    if vl is not None:
        regs[A2] = vl
    words = [*head, _rdcycle(T0), *body, _rdcycle(T1), SPIN]
    for i, w in enumerate(words):
        soc.memory.write(PROG + 4 * i, w.to_bytes(4, "little"))
    for idx, value in regs.items():
        core.register_file[idx].write(value.to_bytes(8, "little"))
    spin = PROG + 4 * (len(words) - 1)
    for cycle in range(20000):
        soc.clock.clock_tick(cycle)
        if core.pc_register.read_uint() == spin:
            break
    else:
        raise AssertionError("the program did not reach its spin loop")
    rf = core.register_file
    return rf[T1].read_uint() - rf[T0].read_uint(), soc


def _vadd(vd, vs2, vs1):
    return opv(OPIVV, 0x00, vd, vs2, vs1)


N = 6


@pytest.mark.parametrize("lmul,occupancy", [(1, 1), (2, 2), (8, 8)])
def test_independent_arithmetic_streams_at_the_datapath_rate(lmul, occupancy):
    """``N`` independent ``vadd.vv`` at e32, VL = VLMAX: each holds the datapath ceil(VL*32/256) = LMUL cycles."""
    body = [_vadd(16 if i % 2 else 24, 8, 8) for i in range(N)]
    cycles, soc = _run(_raw(), body, lmul=lmul)
    assert cycles == 2 + (N - 1) * occupancy
    c = soc.counters()
    assert c["hart0.timing.stall_vector_busy"] == (N - 1) * (occupancy - 1)
    assert c["hart0.timing.vector_busy_cycles"] == N * occupancy


def test_dependent_chain_waits_for_the_whole_result():
    """``vadd v1, v1, v2`` x N, no chaining: each issues ``1 + startup`` after the last."""
    cycles, soc = _run(_raw(), [_vadd(1, 1, 2)] * N)
    assert cycles == 2 + (N - 1) * (1 + STARTUP)
    assert soc.counters()["hart0.timing.stall_vector_raw"] == (N - 1) * STARTUP


def test_short_vl_occupies_one_cycle():
    """VL = 3 at LMUL 8 is one element group: occupancy 1, whatever the group size."""
    body = [_vadd(16 if i % 2 else 24, 8, 8) for i in range(N)]
    cycles, _ = _run(_raw(), body, lmul=8, vl=3)
    assert cycles == N + 1


def test_divide_occupies_div_ii_per_group():
    vdivu = opv(OPMVV, 0x20, 16, 8, 8)
    vdivu2 = opv(OPMVV, 0x20, 24, 8, 8)
    cycles, _ = _run(_raw(), [vdivu, vdivu2] * 2, lmul=2)
    assert cycles == 2 + 3 * 2 * DIV_II


def test_widening_counts_the_wide_elements():
    """vwadd.vv at e16, LMUL 1, VL 16: 16 x 32-bit results = 2 datapath cycles each."""
    vwadd = [opv(OPMVV, 0x31, 8 + 2 * (i % 2), 2, 3) for i in range(N)]
    cycles, _ = _run(_raw(), vwadd, sew=16)
    assert cycles == 2 + (N - 1) * 2


def test_reduction_then_scalar_move():
    """vredsum (+ log2(8 lanes) = 3 tree cycles), then vmv.x.s, then a dependent addi."""
    body = [
        opv(OPMVV, 0x00, 4, 8, 2),  # vredsum.vs v4, v8, v2
        opv(OPMVV, 0x10, A0, 4, 0),  # vmv.x.s a0, v4
        i_type(0x13, 0, 1, rd=A1, rs1=A0),
    ]
    cycles, _ = _run(_raw(), body)
    red_done = 1 + 1 + STARTUP + 3
    mv_done = red_done + 1 + STARTUP
    assert cycles == mv_done + 1


def test_scalar_operand_waits_on_the_scalar_scoreboard():
    """vadd.vx reads a0 straight after a 3-cycle mul writes it."""
    mul = r_type(0x33, 0, 1, rd=A0, rs1=A1, rs2=A1)
    vaddvx = opv(OPIVX, 0x00, 4, 8, A0)
    cycles, _ = _run(_raw(), [mul, vaddvx])
    assert cycles == 1 + 3 + 1


@pytest.mark.parametrize("lmul", [1, 8])
def test_unit_stride_load_then_use(lmul):
    """vle32 of VLMAX x 4 bytes from DRAM: ready mem_startup + latency + bytes/32 after issue."""
    nbytes = 8 * lmul * 4
    t = nbytes // LSU
    body = [vmem(True, 32, 8, S1), _vadd(16, 8, 8)]
    cycles, soc = _run(_raw(), body, lmul=lmul)
    assert cycles == 1 + MEM_STARTUP + DRAM_LAT + t + 1
    c = soc.counters()
    assert c["hart0.timing.mem.dram.loads"] == 1
    assert c["hart0.timing.mem.dram.bytes"] == nbytes
    assert c["hart0.timing.vector_mem_busy_cycles"] == t


def test_loads_back_to_back_share_the_load_store_unit():
    body = [vmem(True, 32, 8 + 8 * (i % 2), S0) for i in range(N)]
    cycles, _ = _run(_raw(), body, lmul=8)
    assert cycles == 2 + (N - 1) * (256 // LSU)


def test_region_bandwidth_caps_the_transfer():
    body = [vmem(True, 32, 8, S1), vmem(True, 32, 16, S1)]
    cycles, _ = _run(
        _raw(memory={"dram": {"latency": DRAM_LAT, "bandwidth": 16}}), body, lmul=8
    )
    assert cycles == 2 + 256 // 16


def test_strided_load_pays_one_address_per_element():
    """vlse32 of 8 elements: max(32 B / 32, 8 elements / 1) = 8 cycles on the LSU."""
    body = [vmem(True, 32, 8, S0, mop=2, rs2=A1), vmem(True, 32, 9, S0, mop=2, rs2=A1)]
    cycles, _ = _run(_raw(), body, regs={A1: 64})
    assert cycles == 2 + 8


def test_store_waits_for_its_data_then_is_posted():
    body = [_vadd(8, 2, 3), vmem(False, 32, 8, S0), i_type(0x13, 0, 1, rd=A1, rs1=0)]
    cycles, _ = _run(_raw(), body)
    assert cycles == 1 + (1 + STARTUP) + 1 + 1


def test_zero_startup_and_wide_datapath_reproduce_untimed_issue():
    raw = _raw(vector={"startup": 0, "datapath_bits": 4096})
    cycles, _ = _run(raw, [_vadd(1, 1, 2)] * N)
    assert cycles == N + 1


def test_every_decodable_vector_word_classifies():
    from framework.pe.rv.isa.v import isa as visa

    for funct3 in range(7):
        for funct6 in range(64):
            for vm in (0, 1):
                for src in (0, 1, 2, 3, 5, 0x10, 0x11, 0x14, 0x15, 0x1D):
                    w = opv(funct3, funct6, 8, 16, src, vm)
                    if visa.decode(w) is not None:
                        assert classify(w) is not None


# -- schema ------------------------------------------------------------------------------


def test_vector_defaults_when_the_section_is_absent():
    raw = _raw()
    del raw["timing"]["vector"]
    design = parse_design(raw)
    assert design.timing.vector.datapath_bits == 256
    assert build(design).cores[0].rv_cost.vector is not None


def test_timing_vector_without_a_vector_unit_is_refused():
    raw = _raw()
    raw["cores"]["isa"] = ["i", "m", "zicsr"]
    del raw["cores"]["vlen"], raw["cores"]["elen"]
    with pytest.raises(DesignError, match="no vector unit"):
        parse_design(raw)


def test_chaining_is_not_modelled():
    with pytest.raises(DesignError, match="chaining"):
        parse_design(_raw(vector={"chaining": True}))


def test_unknown_vector_timing_key():
    with pytest.raises(DesignError, match="unknown key"):
        parse_design(_raw(vector={"lanes": 4}))
