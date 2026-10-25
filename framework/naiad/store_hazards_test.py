"""Memory RAW between an asynchronous store and a later load (:mod:`framework.naiad.store_hazards`).

The footprint arithmetic, then microbenchmarks in the ``timing_test.py``
bracket style (``rdcycle; body; rdcycle``: an instruction issuing at
``c1 + 1 + x`` followed directly by the second read gives ``x + 2``).
"""

import pytest

from framework.naiad import timing_test as tt
from framework.naiad import vector_timing_test as vt
from framework.naiad.builder import build
from framework.naiad.design import parse_design
from framework.naiad.store_hazards import StoresInFlight, footprint, overlaps
from framework.pe.rv.isa.v.vtesting import vmem

# -- footprints ---------------------------------------------------------------------


def _span(lo, n):
    return footprint(lo, 0, 1, n)


def test_contiguous_spans_overlap_only_when_they_share_a_byte():
    assert overlaps(_span(0, 8), _span(7, 1))
    assert not overlaps(_span(0, 8), _span(8, 8))
    assert not overlaps(_span(8, 8), _span(0, 8))


def test_touching_rows_collapse_to_one_span():
    fp = footprint(100, 16, 4, 16)
    assert fp == (100, 164, 100, 0, 1, 64)
    assert footprint(100, -16, 4, 16) == (52, 116, 52, 0, 1, 64)


def test_interleaved_column_blocks_do_not_overlap():
    """Two 16 x 64-byte blocks side by side in a 128-byte-row matrix: the
    bounding intervals interleave, the bytes do not."""
    left, right = footprint(0, 128, 16, 64), footprint(64, 128, 16, 64)
    assert not overlaps(left, right)
    assert not overlaps(right, left)
    assert overlaps(left, footprint(60, 128, 16, 8))  # straddles a row end
    assert overlaps(left, _span(15 * 128 + 63, 1))  # the last byte
    assert not overlaps(left, _span(15 * 128 + 64, 1000))
    assert not overlaps(left, _span(64, 64))  # one gap, exactly


def test_negative_stride_is_the_same_rows():
    assert footprint(15 * 128, -128, 16, 64) == footprint(0, 128, 16, 64)


def test_empty_access_overlaps_nothing():
    assert not overlaps(footprint(64, 128, 0, 64), _span(0, 1 << 20))
    assert not overlaps(_span(0, 1 << 20), footprint(64, 0, 1, 0))


def test_the_table_reports_the_latest_overlapping_completion_and_prunes():
    t = StoresInFlight(capacity=2)
    assert t.wait(_span(0, 8), 0) == 0
    t.add(_span(0, 8), 10, 0)
    t.add(_span(4, 8), 12, 1)
    assert t.wait(_span(0, 4), 2) == 10
    assert t.wait(_span(4, 1), 2) == 12
    assert t.wait(_span(100, 1), 2) == 0
    t.add(_span(100, 4), 14, 11)  # full: the completed entry is pruned, nothing dropped
    assert t.dropped == 0
    assert len(t.entries) == 2
    t.add(_span(200, 4), 15, 11)  # still full of live entries: the earliest is dropped
    assert t.dropped == 1
    assert t.wait(_span(4, 1), 11) == 0
    assert t.wait(_span(200, 1), 11) == 15
    assert t.wait(_span(200, 1), 15) == 0  # complete
    t.add(_span(0, 4), 5, 5)  # done == issue: nothing in flight
    assert t.until == 15


# -- AME --------------------------------------------------------------------------

T_STORE = tt.TILE_BYTES // 64  # one fp32 accumulator at 64 B/cycle
ROW = 64  # bytes per accumulator row (16 fp32)


def _ame(body, regs):
    timing = tt._timing(
        matrix={
            "tile_store_bandwidth": 64,
            "tile_store_latency": 1,
            "tile_load_bandwidth": 64,
        }
    )
    return tt._run(tt._raw(timing), body, {tt.S0: tt.SP_BASE, **regs})


def test_a_tile_load_of_the_bytes_an_astore_writes_waits_for_it():
    """``astore a0; tload.a t0`` of the same tile: the load issues when the
    store completes, ``1 + tile_store_latency + T - 1`` after it, so ``T + 2``."""
    body = [tt._astore(0, tt.S0, 16), tt._tload_a(0, tt.S0, 16)]
    cycles, soc = _ame(body, {16: ROW})
    assert cycles == T_STORE + 2
    assert soc.counters()["hart0.timing.stall_mem_raw"] == T_STORE - 1


def test_a_disjoint_tile_load_does_not_wait():
    body = [tt._astore(0, tt.S0, 16), tt._tload_a(0, 17, 16)]
    cycles, soc = _ame(body, {16: ROW, 17: tt.SP_BASE + tt.TILE_BYTES})
    assert cycles == 3
    assert soc.counters()["hart0.timing.stall_mem_raw"] == 0


def test_the_neighbouring_column_block_does_not_wait():
    """Store the left 16 x 16 block of a 32-wide fp32 matrix, load the right one."""
    body = [tt._astore(0, tt.S0, 16), tt._tload_a(0, 17, 16)]
    cycles, soc = _ame(body, {16: 2 * ROW, 17: tt.SP_BASE + ROW})
    assert cycles == 3
    assert soc.counters()["hart0.timing.stall_mem_raw"] == 0


@pytest.mark.parametrize("offset", [0, tt.TILE_BYTES - 8], ids=["first", "last"])
def test_a_scalar_load_of_an_astored_row_waits(offset):
    body = [tt._astore(0, tt.S0, 16), tt._ld(tt.A0, 17)]
    cycles, soc = _ame(body, {16: ROW, 17: tt.SP_BASE + offset})
    assert cycles == T_STORE + 2
    assert soc.counters()["hart0.timing.stall_mem_raw"] == T_STORE - 1


# -- RVV ----------------------------------------------------------------------------

VL = 64  # e32, LMUL 8: 256 bytes, T = 256 / LSU
T_VEC = VL * 4 // vt.LSU


def _vec(body, regs=None):
    return vt._run(vt._raw(), body, sew=32, lmul=8, vl=VL, regs=regs)


def test_a_scalar_load_after_a_vector_store_waits_for_the_lsu():
    body = [vmem(False, 32, 8, vt.S0), tt._ld(vt.A0, vt.S0)]
    cycles, soc = _vec(body)
    assert cycles == T_VEC + 2
    assert soc.counters()["hart0.timing.stall_mem_raw"] == T_VEC - 1


def test_a_disjoint_scalar_load_does_not_wait():
    body = [vmem(False, 32, 8, vt.S0), tt._ld(vt.A0, vt.A1)]
    cycles, soc = _vec(body, {vt.A1: vt.SP + VL * 4})
    assert cycles == 3
    assert soc.counters()["hart0.timing.stall_mem_raw"] == 0


def test_vector_store_then_vector_load_is_unchanged():
    """The vector load already waits for the LSU (``vector_busy``), which frees
    exactly when the store completes: no ``mem_raw`` on top."""
    body = [vmem(False, 32, 8, vt.S0), vmem(True, 32, 16, vt.S0)]
    cycles, soc = _vec(body)
    counters = soc.counters()
    assert cycles == T_VEC + 2
    assert counters["hart0.timing.stall_mem_raw"] == 0
    assert counters["hart0.timing.stall_vector_busy"] == T_VEC - 1


def test_a_strided_vector_store_leaves_its_gaps_free():
    """``vsse32`` at an 8-byte stride writes every other word (``VL`` cycles, one
    element a cycle): a load of its last word waits, a load of a gap does not."""
    store = vmem(False, 32, 8, vt.S0, mop=2, rs2=vt.A1)
    for addr, want in ((vt.SP + 8 * (VL - 1), VL + 2), (vt.SP + 4, 3)):
        cycles, _ = _vec([store, _lw(vt.A0, 13)], {vt.A1: 8, 13: addr})
        assert cycles == want, hex(addr)


def _lw(rd, rs1):
    return tt.i_type(0x03, 2, 0, rd=rd, rs1=rs1)


# -- multi-hart --------------------------------------------------------------------


def test_another_harts_store_in_flight_stalls_this_harts_load():
    """One table per SoC: a store hart 1 has in flight delays hart 0's load of those bytes."""
    raw = tt._raw(tt._timing())
    raw["cores"]["count"] = 2
    raw["cores"]["isa"].append("a")
    raw["blocks"].append(
        {"name": "clint", "type": "clint", "base": 0x02000000, "window": 0x10000}
    )
    soc = build(parse_design(raw))
    soc.reset()
    h0, h1 = (core.rv_cost for core in soc.cores)
    assert h0.inflight is h1.inflight is h1.matrix.memory.inflight
    h1.inflight.add(_span(tt.SP_BASE, 64), 50, 0)
    rf = soc.cores[0].register_file
    rf[tt.S0].write(tt.SP_BASE.to_bytes(8, "little"))
    load = tt._ld(tt.A0, tt.S0, 8)
    assert not h0.can_issue(load, 10, rf)
    assert h0.take_pending_stall() == (1, "mem_raw")
    assert h0.can_issue(load, 50, rf)
