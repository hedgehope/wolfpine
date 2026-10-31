"""Opt-in memory-ordering hazards: ``timing.memory.{scalar_stores, dma_coherence, war, waw}``.

Core-side microbenchmarks in the ``store_hazards_test`` bracket style (``rdcycle;
body; rdcycle``: ``cycles`` is t1 - t0), each run with the option off and on:
off must reproduce the unmodelled issue exactly, on must add exactly the wait.
The DMA side is in :mod:`framework.ip.dma_ordering_test`.
"""

import pytest

from framework.naiad import timing_test as tt
from framework.naiad.design import DesignError, parse_design
from framework.pe.rv.isa.isa64_testing import s_type
from framework.pe.rv.isa.v.vtesting import vmem
from framework.pe.rv.timing import vector_test as vt

T_VEC = 64 * 4 // vt.LSU  # e32, LMUL 8, vl 64: 256 bytes at the LSU rate
VL = 64


def _sd(rs1, rs2, imm=0):
    return s_type(3, imm, rs1=rs1, rs2=rs2)


def _vec(body, memory=None, scalar=None, regs=None):
    raw = vt._raw(memory=memory, scalar=scalar)
    return vt._run(raw, body, sew=32, lmul=8, vl=VL, regs=regs)


def _delta(body, flag, memory=None, **kw):
    """``(off cycles, on cycles, on soc)`` for a vector-design body with ``timing.memory.<flag>`` toggled."""
    off, _ = _vec(body, memory=memory, **kw)
    on, soc = _vec(body, memory={**(memory or {}), flag: True}, **kw)
    return off, on, soc


# -- the option ----------------------------------------------------------------


def _spec(memory):
    raw = vt._raw()
    raw["timing"]["memory"].update(memory)
    return parse_design(raw).timing.ordering


def test_every_hazard_is_off_by_default():
    assert not _spec({}).any


def test_hazards_turns_on_all_four_and_a_named_flag_overrides_it():
    spec = _spec({"hazards": True})
    assert spec.scalar_stores
    assert spec.dma_coherence
    assert spec.war
    assert spec.waw
    spec = _spec({"hazards": True, "war": False})
    assert spec.scalar_stores
    assert spec.waw
    assert not spec.war
    assert _spec({"war": True}).war
    assert not _spec({"war": True}).waw


def test_a_non_boolean_flag_is_refused():
    with pytest.raises(DesignError, match="timing.memory.war"):
        _spec({"war": 1})


def test_region_entries_still_parse_beside_the_flags():
    design = parse_design(vt._raw(memory={"war": True}))
    assert design.timing.memory["dram"].latency == vt.DRAM_LAT
    assert "war" not in design.timing.memory


# -- scalar stores in flight -----------------------------------------------------


def test_a_slow_scalar_store_is_in_flight_for_a_vector_load():
    """``sd`` occupies the scalar LSU 6 cycles; the vector LSU is a different path,
    so the load could issue at once. With ``scalar_stores`` it waits for the store."""
    body = [_sd(vt.S0, vt.A0), vmem(True, 32, 16, vt.S0)]
    off, on, soc = _delta(body, "scalar_stores", scalar={"store": 6})
    assert (off, on) == (3, 8)
    assert soc.counters()["hart0.timing.stall_mem_raw"] == 5


def test_a_slow_scalar_store_does_not_hold_a_disjoint_vector_load():
    body = [_sd(vt.S0, vt.A0), vmem(True, 32, 16, vt.A1)]
    off, on, _ = _delta(
        body, "scalar_stores", scalar={"store": 6}, regs={vt.A1: vt.SP + 1024}
    )
    assert off == on == 3


def test_a_one_cycle_scalar_store_is_never_registered():
    """Its bytes are visible next cycle, so there is nothing to wait for (and no
    dependence on hart tick order)."""
    body = [_sd(vt.S0, vt.A0), vmem(True, 32, 16, vt.S0)]
    off, on, soc = _delta(body, "scalar_stores")
    assert off == on == 3
    assert soc.cores[0].rv_cost.inflight.until == 0


# -- WAW ------------------------------------------------------------------------


def test_a_scalar_store_waits_for_an_earlier_vector_store_to_the_same_bytes():
    body = [vmem(False, 32, 8, vt.S0), _sd(vt.S0, vt.A0, 4)]
    off, on, soc = _delta(body, "waw")
    assert (off, on) == (3, T_VEC + 2)
    assert soc.counters()["hart0.timing.stall_mem_waw"] == T_VEC - 1


def test_waw_leaves_a_disjoint_store_alone():
    body = [vmem(False, 32, 8, vt.S0), _sd(vt.A1, vt.A0)]
    off, on, soc = _delta(body, "waw", regs={vt.A1: vt.SP + 1024})
    assert off == on == 3
    assert soc.counters()["hart0.timing.stall_mem_waw"] == 0


def test_waw_does_not_apply_to_a_load():
    """A load after a store is RAW, which is always on: ``waw`` changes nothing there."""
    body = [vmem(False, 32, 8, vt.S0), tt._ld(vt.A0, vt.S0)]
    off, on, soc = _delta(body, "waw")
    assert off == on == T_VEC + 2
    assert soc.counters()["hart0.timing.stall_mem_waw"] == 0


def _ame(body, matrix, memory):
    timing = tt._timing(matrix=matrix, memory=memory)
    return tt._run(tt._raw(timing), body, {tt.S0: tt.SP_BASE, 16: 64})


def test_a_scalar_store_waits_for_an_earlier_tile_store():
    t_store = tt.TILE_BYTES // 64
    matrix = {"tile_store_bandwidth": 64, "tile_store_latency": 1}
    body = [tt._astore(0, tt.S0, 16), _sd(tt.S0, tt.A0)]
    off, _ = _ame(body, matrix, {})
    on, soc = _ame(body, matrix, {"waw": True})
    assert (off, on) == (3, t_store + 2)
    assert soc.counters()["hart0.timing.stall_mem_waw"] == t_store - 1


# -- WAR ------------------------------------------------------------------------


def test_a_scalar_store_waits_for_an_earlier_vector_load_to_finish_reading():
    body = [vmem(True, 32, 16, vt.S0), _sd(vt.S0, vt.A0, 4)]
    off, on, soc = _delta(body, "war")
    assert (off, on) == (3, T_VEC + 2)
    assert soc.counters()["hart0.timing.stall_mem_war"] == T_VEC - 1


def test_war_leaves_a_store_to_other_bytes_alone():
    body = [vmem(True, 32, 16, vt.S0), _sd(vt.A1, vt.A0)]
    off, on, _ = _delta(body, "war", regs={vt.A1: vt.SP + 1024})
    assert off == on == 3


def test_a_vector_store_after_a_vector_load_is_already_ordered_by_the_lsu():
    """Same path, so the LSU already serialises them: ``war`` adds nothing."""
    body = [vmem(True, 32, 16, vt.S0), vmem(False, 32, 8, vt.S0)]
    off, on, soc = _delta(body, "war")
    assert off == on
    assert soc.counters()["hart0.timing.stall_mem_war"] == 0


def test_a_scalar_store_waits_for_an_earlier_tile_load():
    t_load = tt.TILE_BYTES // 64
    matrix = {"tile_load_bandwidth": 64}
    body = [tt._tload_a(0, tt.S0, 16), _sd(tt.S0, tt.A0)]
    off, _ = _ame(body, matrix, {})
    on, soc = _ame(body, matrix, {"war": True})
    assert (off, on) == (3, t_load + 2)
    assert soc.counters()["hart0.timing.stall_mem_war"] == t_load - 1


def test_the_options_change_timing_never_the_bytes():
    body = [vmem(False, 32, 8, vt.S0), _sd(vt.S0, vt.A0, 4), vmem(True, 32, 16, vt.S0)]
    data = []
    for memory in ({}, {"hazards": True}):
        _, soc = _vec(body, memory=memory, regs={vt.A0: 0x1234})
        data.append(bytes(soc.memory.read(vt.SP, 256)))
    assert data[0] == data[1]
