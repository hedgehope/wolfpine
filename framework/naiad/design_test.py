import copy

import pytest

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import DesignError, load_design, parse_design, parse_int

BASE = {
    "name": "t",
    "cores": {"xlen": 32, "isa": ["i", "m"], "reset_pc": 0x1000},
    "blocks": [
        {"name": "bootrom", "type": "bootrom", "base": 0x1000, "window": 0x1000},
        {"name": "dram", "type": "dram", "base": 0x80000000, "window": "1MiB"},
        {"name": "htif", "type": "htif"},
    ],
}


def _design(**changes):
    raw = copy.deepcopy(BASE)
    raw.update(changes)
    return raw


def _with_block(block):
    raw = copy.deepcopy(BASE)
    raw["blocks"].append(block)
    return raw


def test_shipped_designs_validate():
    base = load_design(DESIGNS_DIR / "naiad-base.yaml")
    assert base.cores.xlen == 32
    assert base.cores.march == "rv32im_zicsr"
    assert base.block("dram").size == 1 << 30
    assert base.program_memory.name == "dram"
    rv64 = load_design(DESIGNS_DIR / "naiad-base-rv64.yaml")
    assert rv64.cores.xlen == 64
    assert rv64.cores.mabi == "lp64"


def test_rv64_design_builds_an_rv64_core():
    soc = build(load_design(DESIGNS_DIR / "naiad-base-rv64.yaml"))
    assert soc.cores[0].XLEN_BYTES == 8
    assert soc.cores[0].csrs.time_source is not None


def test_overlapping_windows_are_rejected():
    raw = _with_block(
        {"name": "sp", "type": "scratchpad", "base": 0x1800, "window": 0x1000}
    )
    with pytest.raises(DesignError, match="'sp' window .* overlaps block 'bootrom'"):
        parse_design(raw)


def test_adjacent_windows_are_fine():
    parse_design(
        _with_block(
            {"name": "sp", "type": "scratchpad", "base": 0x2000, "window": 0x1000}
        )
    )


def test_size_larger_than_window_is_rejected():
    raw = _with_block(
        {
            "name": "sp",
            "type": "scratchpad",
            "base": 0x10000,
            "window": 0x1000,
            "params": {"size": 0x2000},
        }
    )
    with pytest.raises(DesignError, match="size 0x2000 does not fit its window 0x1000"):
        parse_design(raw)


def test_unknown_type_is_rejected():
    with pytest.raises(DesignError, match="unknown type 'gpu'"):
        parse_design(
            _with_block({"name": "g", "type": "gpu", "base": 0x10000, "window": 0x1000})
        )


def test_unknown_param_and_isa_are_rejected():
    raw = _with_block(
        {
            "name": "c",
            "type": "clint",
            "base": 0x2000000,
            "window": 0x10000,
            "params": {"harts": 2},
        }
    )
    with pytest.raises(DesignError, match="unknown param"):
        parse_design(raw)
    with pytest.raises(DesignError, match="unknown rv32 extension"):
        parse_design(_design(cores={"isa": ["i", "q"]}))
    # Zalrsc exists only on the RV64 core, so the check follows xlen.
    with pytest.raises(DesignError, match="unknown rv32 extension"):
        parse_design(_design(cores={"isa": ["i", "zalrsc"]}))
    assert (
        parse_design(
            _design(cores={"xlen": 64, "isa": ["i", "a", "zalrsc"]})
        ).cores.xlen
        == 64
    )


def test_unmapped_block_takes_no_window():
    raw = copy.deepcopy(BASE)
    raw["blocks"][2] = {"name": "htif", "type": "htif", "base": 0x3000, "window": 0x10}
    with pytest.raises(DesignError, match="not memory-mapped"):
        parse_design(raw)


def test_reset_pc_must_be_mapped():
    with pytest.raises(DesignError, match="reset_pc 0x2000 is not inside"):
        parse_design(_design(cores={"reset_pc": 0x2000}))


def test_duplicate_names_are_rejected():
    raw = _with_block(
        {"name": "dram", "type": "scratchpad", "base": 0x10000, "window": 0x1000}
    )
    with pytest.raises(DesignError, match="duplicate block name"):
        parse_design(raw)


def test_dma_block_maps_one_register_bank_per_channel():
    raw = _with_block(
        {
            "name": "dma0",
            "type": "dma",
            "base": 0x3000000,
            "window": 0x1000,
            "params": {"channels": 2},
        }
    )
    soc = build(parse_design(raw))
    assert soc.blocks["dma0"].size == 0x80
    assert soc.memory.read(0x3000000 + 0x40 + 0x38, 8) == bytes(8)


def test_parse_int():
    assert parse_int("64KiB", "x") == 65536
    assert parse_int("0x8000_0000", "x") == 0x80000000
    assert parse_int(12, "x") == 12
    with pytest.raises(DesignError):
        parse_int("lots", "x")
