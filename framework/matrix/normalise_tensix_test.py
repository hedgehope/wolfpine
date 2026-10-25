"""normalise(mode="tensix"): a fixed Tensix tile matched by its clock and the DMA's bandwidth."""

from fractions import Fraction

import pytest

from framework.matrix.timing import (
    TENSIX_DST_BYTES,
    TENSIX_MACS_PER_CYCLE,
    TENSIX_SRC_BYTES,
    normalise,
)


def test_naiad_dwarfs_budget_is_the_tile_at_its_own_clock():
    cfg = normalise(4096, 12288, 64, mode="tensix", in_format="bf16")
    assert cfg.mode == "tensix"
    assert cfg.matrix == {}
    assert cfg.blocks == {"tensix": {"clock_ratio": 1}, "dma": {"bytes_per_cycle": 16}}
    # The (unused) AME unit of a tensix design gets the same per-pass peak, for the ratio column.
    assert cfg.timing == {"macs_per_cycle": 2048}
    d = cfg.derived
    assert d["achieved_flops_per_cycle"] == 4096
    assert d["dma_channels"] == 4
    assert d["storage_bytes"] == TENSIX_SRC_BYTES + TENSIX_DST_BYTES
    assert d["storage_ratio"] == pytest.approx((9728 + 32768) / 12288)
    assert d["tile_unpack_bytes_per_naiad_cycle"] == 64


@pytest.mark.parametrize(
    "peak, ratio", [(1024, "1/4"), (2048, "1/2"), (8192, 2), (6144, "3/2")]
)
def test_lower_or_higher_peak_moves_the_tile_clock(peak, ratio):
    cfg = normalise(peak, 12288, 64, mode="tensix")
    assert cfg.blocks["tensix"]["clock_ratio"] == ratio
    assert Fraction(str(ratio)) * TENSIX_MACS_PER_CYCLE * 2 == peak
    assert cfg.derived["tile_unpack_bytes_per_naiad_cycle"] == float(
        64 * Fraction(str(ratio))
    )


def test_bandwidth_splits_over_the_channels():
    assert normalise(4096, 12288, 128, mode="tensix").blocks["dma"] == {
        "bytes_per_cycle": 32
    }


@pytest.mark.parametrize(
    "args, message",
    [
        ((4095, 12288, 64), "even number"),
        ((4096, 12288, None), "mem_bytes_per_cycle > 0"),
        ((4096, 12288, 0), "mem_bytes_per_cycle > 0"),
        ((4096, 12288, 66), "does not split over 4"),
    ],
)
def test_rejects_impossible_budgets(args, message):
    with pytest.raises(ValueError, match=message):
        normalise(*args, mode="tensix")


def test_other_modes_have_no_block_params():
    for mode, kw in (("ame", {}), ("ime", {"vlen": 256}), ("vme", {"vlen": 256})):
        assert normalise(4096, 12288, 64, mode=mode, **kw).blocks == {}


def test_unknown_mode_names_tensix():
    with pytest.raises(ValueError, match="tensix"):
        normalise(4096, 12288, 64, mode="nope")
