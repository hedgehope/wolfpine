"""Cycle costs of the Ztt-tracking AME ops: microbenchmarks with analytic counts.

The harness and conventions are ``timing_test``'s: a hand-encoded body between
two ``rdcycle`` reads, where an instruction issuing ``x`` cycles after the first
read's successor and followed directly by the second read gives ``x + 2``, so
``N`` back-to-back instructions give ``N + 1``.

The unit is 16 x 16 x 16 fp32 at ``timing_test``'s mma timing (ii 4, depth 6).
Each register-to-register op holds the mma pipe for its interval ``ii`` and
is readable ``latency - 1`` after it, so a dependent reader waits
``ii + latency - 1`` and the body ``op; reader`` reads ``ii + latency + 1``.
"""

import math

import pytest

from framework.matrix.encodings import DECODER, TABLE
from framework.naiad.design import DesignError, parse_design
from framework.naiad.timing_test import (
    DEPTH,
    II,
    LAT,
    MMA_TIMING,
    S0,
    SP_BASE,
    TILE_BYTES,
    _mma,
    _raw,
    _run,
    _timing,
)

FP32 = TABLE.formats["fp32"]
FEATURES = ["shift", "acc_move", "convert", "mma_transpose", "reduce", "shift2"]
ROWS = 16


def _design(acc_mem=True, **matrix_timing):
    raw = _raw(_timing(matrix={**MMA_TIMING, **matrix_timing}))
    raw["matrix"].update({"features": FEATURES, "acc_mem": acc_mem})
    return raw


def _w(name, **ops):
    return DECODER.encode(name, **ops)


def _amov_ta(td, acc):
    return _w("amov.t.a", td=td, fmt=FP32, b=0, **{"as": acc})


def _amov_at(ad, ts):
    return _w("amov.a.t", ad=ad, ts=ts, fmt=FP32, b=0)


def _tshift(td, ts):
    # amount register x0 = 0: a plain dependent read of ts
    return _w("tshift", td=td, ts=ts, amount=0, fmt=FP32, cols=0, b=0)


def _tconv(td, ts):
    return _w("tconv", td=td, ts=ts, to=FP32, b=0, **{"from": FP32})


def _treduce(td, ts, prefix=False, cols=0):
    return _w(
        "tprefix" if prefix else "treduce", td=td, ts=ts, op=0, fmt=FP32, cols=cols, b=0
    )


def _tshift2(td, ta, tb):
    return _w("tshift2", td=td, ta=ta, tb=tb, esz=2, cols=0, amount=1)


def _tstore(ts, base, stride):
    return _w("tstore.a", ts=ts, base=base, stride=stride, fmt=FP32)


# -- acc <-> tile moves ------------------------------------------------------------


@pytest.mark.parametrize("rows_per_cycle, latency", [(4, 2), (8, 1), (1, 3)])
def test_amov_t_a_then_a_dependent_tile_read(rows_per_cycle, latency):
    """``amov.t.a t0; tshift t1 <- t0``: ``ceil(16 / rows) + latency + 1``."""
    raw = _design(
        acc_to_tile_rows_per_cycle=rows_per_cycle, acc_to_tile_latency=latency
    )
    cycles, soc = _run(raw, [_amov_ta(0, 0), _tshift(1, 0)])
    ii = math.ceil(ROWS / rows_per_cycle)
    assert cycles == ii + latency + 1
    assert soc.counters()["hart0.timing.matrix_busy_cycles"] == ii


def test_amov_t_a_stream_holds_the_mma_pipe():
    """``N`` independent amov.t.a: ``2 + (N - 1) * ii`` with the TIC default ii = 16 / 4."""
    n = 6
    cycles, _ = _run(_design(), [_amov_ta(i % 8, i % 4) for i in range(n)])
    assert cycles == 2 + (n - 1) * 4


def test_amov_waits_for_the_accumulator_it_moves():
    """``mma acc0; amov.t.a t2 <- acc0``: the move issues at the mma's full latency."""
    cycles, _ = _run(_design(), [_mma(0, 0, 1), _amov_ta(2, 0)])
    assert cycles == LAT + 2


def test_amov_a_t_then_mma_into_that_accumulator():
    """``amov.a.t acc0 <- t0; mma acc0``: ``16 / 8 + 4 + 1`` (TIC MOVA2D defaults)."""
    cycles, _ = _run(_design(), [_amov_at(0, 0), _mma(0, 0, 1)])
    assert cycles == 2 + 4 + 1


def test_the_ztt_store_is_a_move_then_a_tile_store():
    """No acc_mem: ``mma; amov.t.a t7 <- acc0; tstore t7`` reads
    ``LAT + 4 + 2 + 1``: the tstore issues when the move is readable."""
    cycles, _ = _run(
        _design(acc_mem=False),
        [_mma(0, 0, 1), _amov_ta(7, 0), _tstore(7, S0, 16)],
        {S0: SP_BASE, 16: 64},
    )
    assert cycles == LAT + (4 + 2 - 1) + 2
    # the store itself then occupies the store path TILE_BYTES / 64 cycles (not in the bracket)
    assert TILE_BYTES // 64 == 16


def test_acc_mem_false_rejects_aload_in_the_design_too():
    raw = _design(acc_mem=False)
    raw["matrix"]["features"] = FEATURES + ["acc_mem"]
    with pytest.raises(DesignError, match="acc_mem is not listed"):
        parse_design(raw)


# -- conversion ------------------------------------------------------------------


@pytest.mark.parametrize("tconv, epilogue, want_ii", [(None, 3, 3), (5, 3, 5)])
def test_tconv_interval_defaults_to_the_epilogue(tconv, epilogue, want_ii):
    """``tconv; mma``: the mma waits for the pipe, ``ii + 2``; a dependent read also ``ii + 2``."""
    raw = _design(tconv=tconv, epilogue=epilogue)
    cycles, _ = _run(raw, [_tconv(2, 2), _mma(0, 0, 1)])
    assert cycles == want_ii + 2
    cycles, _ = _run(raw, [_tconv(2, 2), _tshift(3, 2)])
    assert cycles == want_ii + 2


# -- transposed mma ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["mma.at", "mma.bt"])
def test_a_transposed_mma_costs_an_mma(name):
    """A dependent chain of ``N`` with forwarding: ``2 + (N - 1) * ii``, as plain mma."""
    n = 5
    word = _w(name, ad=0, ta=0, tb=1, **{"in": FP32, "acc": FP32})
    cycles, soc = _run(_design(), [word] * n)
    assert cycles == 2 + (n - 1) * II
    counters = soc.counters()
    assert counters["hart0.timing.matrix_mma"] == n
    assert counters["hart0.timing.matrix_macs"] == n * 16**3


# -- reductions and prefixes -----------------------------------------------------------


@pytest.mark.parametrize("rows_per_cycle, latency", [(8, 5), (16, 3), (4, 1)])
def test_treduce_then_a_dependent_read(rows_per_cycle, latency):
    """``treduce; tshift``: ``ceil(16 / rows) + latency + 1``."""
    raw = _design(treduce_rows_per_cycle=rows_per_cycle, treduce_latency=latency)
    cycles, _ = _run(raw, [_treduce(2, 1), _tshift(3, 2)])
    assert cycles == math.ceil(ROWS / rows_per_cycle) + latency + 1


@pytest.mark.parametrize("cols", [0, 1])
def test_tprefix_takes_log2_length_passes(cols):
    """A 16-long scan: ``log2(16) = 4`` passes of ``16 / 8``: ``8 + 5 + 1``."""
    cycles, _ = _run(_design(), [_treduce(2, 1, prefix=True, cols=cols), _tshift(3, 2)])
    assert cycles == 4 * 2 + 5 + 1


# -- tshift2 -----------------------------------------------------------------------------


@pytest.mark.parametrize("tshift, tshift2, want", [(2, None, 2), (2, 3, 3)])
def test_tshift2_is_a_fixed_latency_off_the_mma_pipe(tshift, tshift2, want):
    """``tshift2; tshift <- it`` reads ``want + 2``; an independent mma after it is not held."""
    raw = _design(tshift=tshift, tshift2=tshift2)
    cycles, _ = _run(raw, [_tshift2(2, 0, 1), _tshift(3, 2)])
    assert cycles == want + 2
    cycles, _ = _run(raw, [_tshift2(2, 0, 1), _mma(0, 0, 1)])
    assert cycles == 3


def test_the_defaults_are_the_tensix_analogues():
    spec = parse_design(_design()).timing.matrix
    assert (spec.acc_to_tile_rows_per_cycle, spec.acc_to_tile_latency) == (
        4,
        2,
    )  # MOVD2A
    assert (spec.tile_to_acc_rows_per_cycle, spec.tile_to_acc_latency) == (
        8,
        4,
    )  # MOVA2D
    assert (spec.treduce_rows_per_cycle, spec.treduce_latency) == (
        8,
        5,
    )  # GAPOOL/GMPOOL
    assert spec.tconv is None
    assert spec.tshift2 is None
    assert DEPTH == 6


# -- tloadh (the halo tile load: NOT in Ztt) ----------------------------------------------


def _tloadh(td, h, base=S0, stride=16):
    return _w("tloadh", td=td, base=base, stride=stride, fmt=FP32, h=h)


def _halo_design(**matrix_timing):
    raw = _design(**matrix_timing)
    raw["matrix"]["features"] = [*FEATURES, "halo"]
    return raw


@pytest.mark.parametrize("h", [1, 2, 4, 7])
def test_tloadh_moves_only_its_2h_rows(h):
    """``N`` halo loads hold the load path ``2h * 64 B / 64`` cycles each: ``2 + (N - 1) * 2h``."""
    n = 5
    raw = _halo_design(tile_load_bandwidth=64)
    cycles, soc = _run(
        raw,
        [_tloadh(i % 8, h) for i in range(n)],
        {S0: SP_BASE + 8 * 64, 16: 64},
    )
    assert cycles == 2 + (n - 1) * 2 * h
    assert soc.counters()["hart0.timing.matrix_load_busy_cycles"] == n * 2 * h


def test_an_mma_waits_for_its_halo_tile():
    """``tloadh t0; mma``: ``1 + tile_load_latency + 2h`` (h = 2, 64 B rows at 64 B/cycle)."""
    raw = _halo_design(tile_load_bandwidth=64, tile_load_latency=3)
    cycles, _ = _run(
        raw, [_tloadh(0, 2), _mma(0, 0, 1)], {S0: SP_BASE + 8 * 64, 16: 64}
    )
    assert cycles == 1 + 3 + 4


def test_tloadh_needs_the_halo_feature():
    cycles, _ = _run(_halo_design(), [_tloadh(0, 1)], {S0: SP_BASE + 8 * 64, 16: 64})
    assert cycles > 0
    with pytest.raises(Exception, match="halo"):
        _run(_design(), [_tloadh(0, 1)], {S0: SP_BASE + 8 * 64, 16: 64})
