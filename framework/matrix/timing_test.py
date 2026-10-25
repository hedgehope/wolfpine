"""The AME timing parameters' arithmetic and the equal-resource normaliser."""

import copy

import pytest

from framework.matrix.timing import AMETimingSpec, normalise, transfer_cycles
from framework.naiad.builder import build
from framework.naiad.design import parse_design


def test_mma_interval_from_peak_macs_and_format_passes():
    spec = AMETimingSpec(macs_per_cycle=2048, format_passes={"fp32": 4})
    assert spec.mma_interval(16, 16, 16, "fp32") == 8  # 4096 MACs x 4 passes / 2048
    assert spec.mma_interval(16, 16, 16, "int8") == 2  # one pass by default
    assert spec.mma_interval(2, 2, 2, "int8") == 1  # never below one cycle
    assert AMETimingSpec(mma_ii=3).mma_interval(16, 16, 16, "fp32") == 3


def test_transfer_cycles_takes_the_narrowest_bandwidth():
    assert transfer_cycles(1024, 64, None) == 16
    assert transfer_cycles(1024, 128, 32) == 32
    assert transfer_cycles(1024, None, None) == 1
    assert transfer_cycles(100, 64) == 2
    assert transfer_cycles(0, 64) == 1


def test_normalise_ame_square_power_of_two():
    """4096 FLOPs/cycle (the Tensix FPU at LoFi), 64 KiB of operands, 64 B/cycle, fp32.

    8 tiles x 4 B + 4 accs x 4 B = 48 B per d^2; 65536 / 48 = 1365, sqrt -> 36,
    down to a power of two -> 32. 2048 MACs/cycle: ii = 32^3 / 2048 = 16.
    """
    cfg = normalise(4096, 64 * 1024, 64)
    assert cfg.mode == "ame"
    assert (cfg.matrix["m"], cfg.matrix["n"], cfg.matrix["k"]) == (32, 32, 32)
    assert cfg.timing == {
        "macs_per_cycle": 2048,
        "tile_load_bandwidth": 64,
        "tile_store_bandwidth": 64,
    }
    assert cfg.derived["storage_bytes"] == 32 * 32 * 48
    assert cfg.derived["mma_ii"] == 16
    assert cfg.derived["achieved_flops_per_cycle"] == 4096
    assert cfg.derived["tile_load_cycles"] == 32 * 32 * 4 // 64


def test_normalise_without_power_of_two_rounding():
    cfg = normalise(4096, 64 * 1024, 64, power_of_two=False)
    assert cfg.derived["d"] == 36
    assert cfg.derived["mma_ii"] == 23  # ceil(36^3 / 2048)
    assert cfg.derived["achieved_flops_per_cycle"] < 4096


def test_narrower_formats_get_bigger_tiles_at_equal_storage():
    fp32 = normalise(4096, 64 * 1024, 64, in_format="fp32", acc_format="fp32")
    bf16 = normalise(
        4096, 64 * 1024, 64, in_format="bf16", acc_format="fp32", power_of_two=False
    )
    assert bf16.derived["d"] > fp32.derived["d"]
    assert bf16.derived["storage_bytes"] <= 64 * 1024


@pytest.mark.parametrize(
    "args, message",
    [
        ((3, 1024, 64), "even"),
        ((4096, 16, 64), "cannot hold"),
        ((4096, 1024, 0), "must be > 0"),
    ],
)
def test_normalise_rejects_impossible_budgets(args, message):
    with pytest.raises(ValueError, match=message):
        normalise(*args)


def test_vme_is_implemented():  # IME's / VME's normalisers: ime_timing_test.py, vme_timing_test.py
    assert normalise(4096, 64 * 1024, 64, mode="vme").mode == "vme"


def test_a_normalised_config_builds_as_a_design():
    cfg = normalise(1024, 16 * 1024, 32)
    raw = {
        "name": "normalised",
        "cores": {"xlen": 64, "isa": ["i", "m"], "reset_pc": 0x80000000},
        "blocks": [
            {"name": "dram", "type": "dram", "base": 0x80000000, "window": "1MiB"}
        ],
        "matrix": copy.deepcopy(cfg.matrix),
        "timing": {"matrix": copy.deepcopy(cfg.timing)},
    }
    soc = build(parse_design(raw))
    timing = soc.cores[0].rv_cost.matrix
    unit = soc.cores[0].matrix
    assert unit.m == cfg.derived["d"]
    assert (
        timing.spec.mma_interval(unit.m, unit.n, unit.k, "fp32")
        == cfg.derived["mma_ii"]
    )
