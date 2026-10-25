"""Cycle costs of the element-wise accumulator ops (custom-3): microbenchmarks with analytic counts.

The harness and conventions are ``timing_test``'s (see ``ame_ztt_timing_test``):
``N`` back-to-back instructions read ``N + 1``, and an op of interval ``ii`` and
latency ``lat`` followed by a dependent reader reads ``ii + lat + 1``.

The unit is 16 x 16 x 16 fp32, so an accumulator is 1 KiB: at the default
64 B/cycle an element-wise op holds its pipe for 16 cycles.
"""

import pytest

from framework.matrix.encodings import DECODER, TABLE
from framework.naiad.design import DesignError, parse_design
from framework.naiad.timing_test import (
    II,
    LAT,
    MMA_TIMING,
    S0,
    SP_BASE,
    _mma,
    _raw,
    _run,
    _timing,
)

FP32 = TABLE.formats["fp32"]
EW_II = 16 * 16 * 4 // 64  # 16
EW_LAT, TX_LAT, TX_FACTOR = 4, 12, 4


def _design(**matrix_timing):
    raw = _raw(_timing(matrix={**MMA_TIMING, **matrix_timing}))
    raw["matrix"]["features"] = ["ewise", "ewise_tx"]
    return raw


def _mul(d, a, b):
    return DECODER.encode("aew.aa", ad=d, as1=a, as2=b, fmt=FP32, op=2)


def _mul_t(d, a, t):
    return DECODER.encode("aew.at", ad=d, as1=a, ts=t, fmt=FP32, op=2)


def _fma(d, a, b):
    return DECODER.encode("amacc.aa", ad=d, as1=a, as2=b, fmt=FP32, neg=0)


def _rsqrt(d, a):
    return DECODER.encode("aewu", ad=d, as1=a, fmt=FP32, op=1)


def test_dependent_pair_waits_interval_plus_latency():
    cycles, soc = _run(_design(), [_mul(0, 1, 2), _mul(3, 0, 0)])
    assert cycles == EW_II + EW_LAT + 1
    c = soc.counters()
    assert c["hart0.timing.matrix_ewise_ops"] == 2
    assert c["hart0.timing.matrix_ewise_busy_cycles"] == 2 * EW_II
    assert c["hart0.timing.matrix_busy_cycles"] == 0  # not the mma pipe


def test_independent_stream_is_one_per_interval():
    n = 5
    cycles, _ = _run(_design(), [_mul(i % 2, 2, 3) for i in range(n)])
    assert cycles == 2 + (n - 1) * EW_II


@pytest.mark.parametrize("bpc, ii", [(128, 8), (32, 32), (None, 1)])
def test_interval_follows_ewise_bytes_per_cycle(bpc, ii):
    cycles, _ = _run(_design(ewise_bytes_per_cycle=bpc), [_mul(0, 1, 2), _mul(3, 0, 0)])
    assert cycles == ii + EW_LAT + 1


def test_rsqrt_is_a_quarter_rate_with_its_own_latency():
    cycles, _ = _run(_design(), [_rsqrt(0, 1), _mul(3, 0, 0)])
    assert cycles == TX_FACTOR * EW_II + TX_LAT + 1


def test_fma_reads_its_destination():
    """``mul acc0; amacc acc0 += acc1 * acc2``: the fma waits for acc0."""
    cycles, _ = _run(_design(), [_mul(0, 1, 2), _fma(0, 1, 2)])
    assert cycles == EW_II + EW_LAT + 1


@pytest.mark.parametrize("pipe, expect", [("ewise", 1 + 2), ("mma", II + 2)])
def test_ewise_overlaps_mma_unless_it_shares_the_pipe(pipe, expect):
    """``mma acc0; aew acc1 = acc2 * acc3``: independent, so only the pipe can delay it."""
    cycles, _ = _run(_design(ewise_pipe=pipe), [_mma(0, 0, 1), _mul(1, 2, 3)])
    assert cycles == expect


def test_ewise_waits_for_the_mma_result_and_mma_for_the_ewise_result():
    cycles, _ = _run(_design(), [_mma(0, 0, 1), _mul(0, 0, 0)])
    assert cycles == LAT + 2
    cycles, _ = _run(_design(), [_mul(0, 1, 2), _mma(0, 0, 1)])
    assert cycles == EW_II + EW_LAT + 1


def test_tile_operand_waits_for_its_load():
    tload = DECODER.encode("tload.a", td=0, base=S0, stride=16, fmt=FP32)
    raw = _design(tile_load_bandwidth=64, tile_load_latency=3)
    cycles, _ = _run(raw, [tload, _mul_t(1, 2, 0)], {S0: SP_BASE, 16: 64})
    # the 1 KiB load: 16 cycles + latency 3 - 1 -> readable at 18, then the aew issues
    assert cycles == 16 + 3 - 1 + 2


def test_bad_ewise_pipe_is_a_design_error():
    with pytest.raises(DesignError, match="ewise_pipe"):
        parse_design(_design(ewise_pipe="vector"))
