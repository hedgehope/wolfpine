import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import DesignError, parse_design
from framework.naiad.ip import dma as D
from framework.naiad.run import main

KERNELS = Path(__file__).resolve().parents[2] / "kernels" / "naiad"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")

DMA_YAML = DESIGNS_DIR / "naiad-dma.yaml"
DMA = 0x10010000
SP = 0x20000000
DRAM = 0x80100000
PERF = 0x02010000


def _raw(**params):
    raw = yaml.safe_load(DMA_YAML.read_text())
    for block in raw["blocks"]:
        if block["type"] == "dma":
            block["params"] = {**block["params"], **params}
    return raw


def _soc(**params):
    soc = build(parse_design(_raw(**params)))
    soc.reset()
    return soc


def _wr(soc, reg, value, ch=0):
    soc.memory.write(DMA + ch * D.CH_STRIDE + reg, value.to_bytes(8, "little"))


def _rd(soc, reg, ch=0):
    return int.from_bytes(soc.memory.read(DMA + ch * D.CH_STRIDE + reg, 8), "little")


def _start(soc, src, dst, length, rows=None, src_stride=0, dst_stride=0, ch=0):
    _wr(soc, D.SRC, src, ch)
    _wr(soc, D.DST, dst, ch)
    _wr(soc, D.LEN, length, ch)
    ctrl = D.CTRL_START
    if rows is not None:
        _wr(soc, D.ROWS, rows, ch)
        _wr(soc, D.SRC_STRIDE, src_stride, ch)
        _wr(soc, D.DST_STRIDE, dst_stride, ch)
        ctrl |= D.CTRL_2D
    _wr(soc, D.CTRL, ctrl, ch)


def _tick_until_idle(soc, ch=0, limit=100_000):
    """Tick from the next cycle until STATUS drops BUSY; returns the cycle it finished in."""
    clock = soc.clock
    for _ in range(limit):
        cycle = clock.current_cycle + 1
        # Only the DMA: the core would run off into unloaded memory.
        clock.current_cycle = cycle
        soc.blocks["dma"].clock_tick(cycle)
        if not _rd(soc, D.STATUS, ch) & D.STATUS_BUSY:
            return cycle
    raise AssertionError("DMA never finished")


def _pattern(n, seed=1):
    return bytes((i * 37 + seed) & 0xFF for i in range(n))


# -- registers ---------------------------------------------------------------


def test_registers_read_back_and_start_is_self_clearing():
    soc = _soc()
    for reg, value in (
        (D.SRC, DRAM),
        (D.DST, SP),
        (D.LEN, 64),
        (D.ROWS, 3),
        (D.SRC_STRIDE, 128),
        (D.DST_STRIDE, 64),
    ):
        _wr(soc, reg, value)
        assert _rd(soc, reg) == value
    _wr(soc, D.CTRL, D.CTRL_2D)
    assert _rd(soc, D.CTRL) == D.CTRL_2D
    _wr(soc, D.CTRL, D.CTRL_2D | D.CTRL_START)
    assert _rd(soc, D.CTRL) == D.CTRL_2D
    assert _rd(soc, D.STATUS) == D.STATUS_BUSY


def test_rv32_style_half_register_access_and_status_is_write_one_to_clear():
    soc = _soc()
    soc.memory.write(DMA + D.SRC, DRAM.to_bytes(4, "little"))
    soc.memory.write(DMA + D.SRC + 4, (1).to_bytes(4, "little"))
    assert _rd(soc, D.SRC) == (1 << 32) | DRAM
    _start(soc, DRAM, SP, 8)
    _tick_until_idle(soc)
    assert _rd(soc, D.STATUS) == D.STATUS_DONE
    _wr(soc, D.STATUS, 0)  # writing zeros clears nothing
    assert _rd(soc, D.STATUS) == D.STATUS_DONE
    _wr(soc, D.STATUS, D.STATUS_DONE)
    assert _rd(soc, D.STATUS) == 0


def test_start_while_busy_is_dropped_with_an_error():
    soc = _soc()
    soc.memory.write(DRAM, _pattern(256))
    _start(soc, DRAM, SP, 256)
    _start(soc, DRAM, SP + 0x1000, 8)
    status = _rd(soc, D.STATUS)
    assert status & (D.STATUS_BUSY | D.STATUS_ERROR) == D.STATUS_BUSY | D.STATUS_ERROR
    assert status >> D.STATUS_ERR_SHIFT == D.ERR_BUSY
    _tick_until_idle(soc)
    assert soc.memory.read(SP, 256) == _pattern(
        256
    )  # the first transfer ran to completion
    assert soc.memory.read(SP + 0x1000, 8) == bytes(8)  # the second never started
    assert _rd(soc, D.STATUS) & D.STATUS_DONE


# -- timing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, bpc, setup", [(4096, 8, 16), (100, 8, 16), (1, 8, 0), (64, 64, 3), (0, 8, 5)]
)
def test_transfer_takes_setup_plus_ceil_n_over_bpc(n, bpc, setup):
    soc = _soc(bytes_per_cycle=bpc, setup_latency=setup)
    soc.memory.write(DRAM, _pattern(n))
    start = soc.clock.current_cycle
    _start(soc, DRAM, SP, n)
    expected = setup + -(-n // bpc)
    if expected == 0:
        assert _rd(soc, D.STATUS) == D.STATUS_DONE
    else:
        assert _tick_until_idle(soc) - start == expected
    assert soc.memory.read(SP, n) == _pattern(n)
    assert soc.counters()["dma.busy_cycles"] == expected


def test_2d_timing_counts_bytes_not_rows():
    soc = _soc(bytes_per_cycle=8, setup_latency=4)
    _start(soc, DRAM, SP, 12, rows=4, src_stride=64, dst_stride=16)
    assert (
        _tick_until_idle(soc) == 4 + 6
    )  # 48 bytes at 8 per cycle, rows split mid-cycle


def test_cycle_of_start_is_not_a_transfer_cycle_in_a_running_soc():
    soc = _soc(setup_latency=2)
    clock = soc.clock
    clock.clock_tick(0)
    _start(soc, DRAM, SP, 8)  # as if a core stored CTRL during cycle 0
    soc.blocks["dma"].clock_tick(0)  # the DMA ticking after that core, same cycle
    assert _rd(soc, D.STATUS) == D.STATUS_BUSY
    assert _tick_until_idle(soc) == 3


# -- data ----------------------------------------------------------------------


def test_1d_copy_dram_to_scratchpad_and_back():
    soc = _soc()
    data = _pattern(1000)
    soc.memory.write(DRAM, data)
    _start(soc, DRAM, SP + 8, len(data))
    _tick_until_idle(soc)
    _start(soc, SP + 8, DRAM + 0x10000, len(data))
    _tick_until_idle(soc)
    assert soc.memory.read(DRAM + 0x10000, 1000) == data


def test_2d_tile_gather_and_scatter():
    soc = _soc()
    rows, cols, stride = 8, 32, 128  # an 8 x 32-byte tile out of 128-byte rows
    matrix = _pattern(stride * 16)
    soc.memory.write(DRAM, matrix)
    _start(
        soc,
        DRAM + 2 * stride + 32,
        SP,
        cols,
        rows=rows,
        src_stride=stride,
        dst_stride=cols,
    )
    _tick_until_idle(soc)
    tile = b"".join(
        matrix[(2 + r) * stride + 32 : (2 + r) * stride + 64] for r in range(rows)
    )
    assert soc.memory.read(SP, rows * cols) == tile
    _start(soc, SP, DRAM + 0x8000, cols, rows=rows, src_stride=cols, dst_stride=stride)
    _tick_until_idle(soc)
    for r in range(rows):
        assert (
            soc.memory.read(DRAM + 0x8000 + r * stride, cols)
            == tile[r * cols : (r + 1) * cols]
        )
        assert soc.memory.read(DRAM + 0x8000 + r * stride + cols, 8) == bytes(
            8
        )  # gaps untouched


def test_reading_the_destination_before_done_races_the_copy():
    soc = _soc(bytes_per_cycle=8, setup_latency=4)
    data = _pattern(256)
    soc.memory.write(DRAM, data)
    soc.memory.write(SP, b"\xaa" * 256)
    _start(soc, DRAM, SP, 256)
    dma = soc.blocks["dma"]
    for cycle in range(1, 1 + 4 + 3):  # setup, then three 8-byte chunks
        soc.clock.current_cycle = cycle
        dma.clock_tick(cycle)
    # The race: mid-transfer the destination is part new, part old.
    assert _rd(soc, D.STATUS) == D.STATUS_BUSY
    assert soc.memory.read(SP, 24) == data[:24]
    assert soc.memory.read(SP + 24, 232) == b"\xaa" * 232
    # The pattern: poll STATUS to DONE, then read.
    _tick_until_idle(soc)
    assert _rd(soc, D.STATUS) == D.STATUS_DONE
    assert soc.memory.read(SP, 256) == data


def test_channels_run_independently():
    soc = _soc(channels=2, bytes_per_cycle=8, setup_latency=0)
    soc.memory.write(DRAM, _pattern(64))
    _start(soc, DRAM, SP, 64, ch=0)
    _start(soc, DRAM, SP + 0x100, 16, ch=1)
    assert _tick_until_idle(soc, ch=1) == 2
    assert _rd(soc, D.STATUS, ch=0) == D.STATUS_BUSY
    assert _tick_until_idle(soc, ch=0) == 8
    assert soc.memory.read(SP + 0x100, 16) == _pattern(16)


# -- errors ------------------------------------------------------------------


def _error_code(soc):
    status = _rd(soc, D.STATUS)
    assert status & 0xFF == D.STATUS_ERROR
    return status >> D.STATUS_ERR_SHIFT


def test_unmapped_or_overrunning_spans_are_refused_at_start():
    soc = _soc()
    _start(soc, 0x40000000, SP, 8)  # nothing mapped there
    assert _error_code(soc) == D.ERR_SRC
    _start(soc, DRAM, SP + 0x40000 - 4, 8)  # runs off the end of the 256 KiB scratchpad
    assert _error_code(soc) == D.ERR_DST
    _start(
        soc, DRAM, SP, 64, rows=8, src_stride=64, dst_stride=0x10000
    )  # last row past the end
    assert _error_code(soc) == D.ERR_DST
    assert soc.counters()["dma.errors"] == 3
    assert soc.counters()["dma.transfers"] == 0


def test_misaligned_2d_is_refused_but_1d_is_byte_granular():
    soc = _soc(stride_align=8)
    _start(soc, DRAM + 4, SP, 16, rows=2, src_stride=64, dst_stride=16)
    assert _error_code(soc) == D.ERR_ALIGN
    _start(soc, DRAM, SP, 16, rows=2, src_stride=60, dst_stride=16)
    assert _error_code(soc) == D.ERR_ALIGN
    _start(soc, DRAM + 3, SP + 5, 7)
    _tick_until_idle(soc)
    assert _rd(soc, D.STATUS) == D.STATUS_DONE


def test_an_mmio_access_the_target_rejects_stops_with_a_bus_error():
    soc = _soc(bytes_per_cycle=8)
    # The perf counters are 8-byte registers; an 8-byte read at offset 5 crosses one.
    _start(soc, PERF + 5, SP, 16)
    _tick_until_idle(soc)
    assert _error_code(soc) == D.ERR_BUS
    soc.memory.write(DMA + D.STATUS, (D.STATUS_ERROR).to_bytes(8, "little"))
    assert _rd(soc, D.STATUS) == 0


# -- counters, reset, design ---------------------------------------------------


def test_counters_split_bytes_by_path():
    soc = _soc()
    soc.memory.write(DRAM, _pattern(64))
    _start(soc, DRAM, SP, 64)
    _tick_until_idle(soc)
    _start(soc, SP, DRAM + 0x1000, 24)
    _tick_until_idle(soc)
    _start(soc, PERF, SP + 0x100, 8)  # any path with an MMIO end
    _tick_until_idle(soc)
    c = soc.counters()
    assert c["dma.bytes"] == 96
    assert c["dma.transfers"] == 3
    assert c["dma.dram_to_scratchpad_bytes"] == 64
    assert c["dma.scratchpad_to_dram_bytes"] == 24
    assert c["dma.dram_to_dram_bytes"] == 0
    assert c["dma.mmio_bytes"] == 8
    assert c["dma.busy_cycles"] == (16 + 8) + (16 + 3) + (16 + 1)


def test_reset_clears_registers_transfers_and_counters():
    soc = _soc()
    _start(soc, DRAM, SP, 64)
    soc.reset()
    assert _rd(soc, D.STATUS) == 0
    assert _rd(soc, D.SRC) == 0
    assert soc.blocks["dma"].active == []
    assert soc.counters()["dma.transfers"] == 0
    assert "dma.dram_to_scratchpad_bytes" in soc.counters()


@pytest.mark.parametrize(
    "params, match",
    [
        ({"channels": 0}, "channels must be >= 1"),
        ({"bytes_per_cycle": 0}, "bytes_per_cycle must be >= 1"),
        ({"setup_latency": -1}, "setup_latency must be >= 0"),
        ({"stride_align": 6}, "power of two"),
        ({"channels": 65}, "does not fit its window"),
        ({"burst": 4}, "unknown param"),
    ],
)
def test_bad_dma_params_are_rejected(params, match):
    with pytest.raises(DesignError, match=match):
        parse_design(_raw(**params))


def test_header_names_the_register_layout():
    from framework.naiad.gen import gen_header

    header = gen_header(parse_design(copy.deepcopy(_raw())))
    assert "#define NAIAD_HAS_DMA 1" in header
    assert "#define NAIAD_DMA_ADDR NAIAD_DMA_BASE" in header
    assert "#define NAIAD_DMA_STATUS 0x00000038UL" in header
    assert "#define NAIAD_DMA_BYTES_PER_CYCLE 0x00000008UL" in header


# -- end to end: the dma kernel ------------------------------------------------


@pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
def test_dma_kernel_builds_runs_and_passes(tmp_path, capsys):
    subprocess.run(
        [
            "make",
            "-C",
            str(KERNELS),
            "PROG=dma",
            f"DESIGN={DMA_YAML}",
            f"BUILD={tmp_path}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert main([str(DMA_YAML), str(tmp_path / "dma.elf")]) == 0
    out, err = capsys.readouterr()
    assert "race: last word before DONE: 0xaaaaaaaa (expect 0xaaaaaaaa) ok" in out
    assert "2D round trip mismatches: 0x0 (expect 0x0) ok" in out
    assert "MISMATCH" not in out
    assert out.endswith("PASS\n")
    assert "dma.dram_to_scratchpad_bytes = 5120" in err
    assert "dma.scratchpad_to_dram_bytes = 1024" in err
