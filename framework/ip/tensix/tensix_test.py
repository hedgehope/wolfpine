"""The Naiad Tensix accelerator: images from captured traces, the MMIO block, and the gemm offload.

The block tests drive ``framework.ip.tensix.block.TensixBlock`` from Python
the way ``kernels/naiad/include/tensix_accel.h`` does from C (boot, launch,
doorbell, STATUS polls), ticking its clock by hand, and check the result
against the readback the live tt-metal run recorded in the shipped trace, bit
for bit. The end-to-end tests build the gemm dwarf's ``tensix`` variant and
run it through the dwarf harness on ``naiad-tensix.yaml``.
"""

import importlib.util
import struct
from pathlib import Path

import pytest

from framework.bridge import protocol as proto
from framework.bridge.trace import parse_trace_line
from framework.ip.tensix import IMAGES_DIR
from framework.ip.tensix import block as tx
from framework.ip.tensix import image as tensix_image
from framework.naiad.design import BlockSpec, DesignError, load_design, parse_design

REPO = Path(__file__).resolve().parents[3]
IMAGES = IMAGES_DIR
TRACE_FP32 = IMAGES / "bh_gemm_bf16_fp32.trace"
TRACE_BF16 = IMAGES / "bh_gemm_bf16_bf16.trace"
INCLUDE = REPO / "kernels" / "naiad" / "include"
DESIGN = REPO / "framework" / "naiad" / "designs" / "naiad-tensix.yaml"
HAVE_ML_DTYPES = importlib.util.find_spec("ml_dtypes") is not None


# -- images ------------------------------------------------------------------


@pytest.mark.parametrize(
    "trace,c_addr,page", [(TRACE_FP32, 0x100000, 4096), (TRACE_BF16, 0x120000, 2048)]
)
def test_image_parses_the_launch_message(trace, c_addr, page):
    img = tensix_image.from_trace(trace)
    assert img.arch == "blackhole"
    assert img.core == (1, 2)
    assert img.go_addr == 0x4F0
    assert img.launch_addr == 0x70
    # Compute-only program: the three TRISCs, no data-movement kernel.
    assert img.enables == 0b11100
    assert set(img.rta) == {"trisc0", "trisc1", "trisc2"}
    assert set(img.cbs) == {0, 1, 2}
    a, b, c = img.cbs[0], img.cbs[1], img.cbs[2]
    assert (a.num_pages, a.page_size) == (64, 2048)
    assert (b.num_pages, b.page_size) == (64, 2048)
    assert (c.addr, c.num_pages, c.page_size) == (c_addr, 64, page)
    # The capture's operand writes landed in the FIFOs and are not part of the recipe.
    fifos = [(cb.addr, cb.addr + cb.size) for cb in img.cbs.values()]
    for addr, data in img.launch:
        assert not any(addr < hi and addr + len(data) > lo for lo, hi in fifos)
    # The firmware for all five cores, and nothing outside L1.
    assert img.boot_bytes > 10_000
    assert all(addr + len(d) <= 0x180000 for addr, d in img.boot + img.launch)


@pytest.mark.parametrize(
    "trace,name", [(TRACE_FP32, "gemm_fp32"), (TRACE_BF16, "gemm_bf16")]
)
def test_shipped_header_is_regenerated_from_its_trace(trace, name):
    header = INCLUDE / f"tensix_{name}.h"
    assert header.read_text() == tensix_image.c_header(
        tensix_image.from_trace(trace), name
    )


def test_segments_are_disjoint_and_keep_the_last_write():
    writes = [(0, b"\x01" * 8), (4, b"\x02" * 8), (2, b"\x03\x03"), (20, b"\x04")]
    segs = tensix_image._segments(writes)
    mem = bytearray(24)
    spans = []
    for addr, data in segs:
        spans.append((addr, addr + len(data)))
        mem[addr : addr + len(data)] = data
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert bytes(mem[:13]) == b"\x01\x01\x03\x03\x02\x02\x02\x02\x02\x02\x02\x02\x00"
    assert mem[20] == 4


def test_trace_without_a_launch_is_refused(tmp_path):
    bad = tmp_path / "bad.trace"
    bad.write_text("WRITE core=1,2 addr=0x0 size=4 data=6f30101e\n")
    with pytest.raises(tensix_image.TensixImageError):
        tensix_image.from_trace(bad, core=(1, 2))


# -- the block, driven from Python --------------------------------------------


def _block(**params):
    spec = BlockSpec("tensix", "tensix", 0x11000000, 16 << 20, params)
    return tx.TensixBlock(spec)


class _Host:
    """What tensix_accel.h does, through the block's MMIO, with the clock ticked by hand."""

    def __init__(self, block):
        self.block = block
        self.cycle = 0

    def tick(self, n=1):
        for _ in range(n):
            self.block.clock_tick(self.cycle)
            self.cycle += 1

    def reg(self, off):
        return int.from_bytes(self.block.read(tx.CTRL_OFFSET + off, 8), "little")

    def set_reg(self, off, value):
        self.block.write(tx.CTRL_OFFSET + off, value.to_bytes(8, "little"))

    def l1_write(self, addr, data):
        self.block.write(addr, data)

    def l1_read(self, addr, n):
        return self.block.read(addr, n)

    def poll(self, done, cap=200_000):
        start = self.cycle
        while not done():
            self.tick()
            assert self.cycle - start < cap, "the tile never answered"
        return self.cycle - start

    def signal(self):
        return (self.reg(tx.STATUS) >> tx.STATUS_SIGNAL_SHIFT) & 0xFF

    def boot(self, img):
        self.set_reg(tx.CTRL, tx.CTRL_RESET_ASSERT)
        for addr, data in img.boot:
            self.l1_write(addr, data)
        assert self.reg(tx.STATUS) & tx.STATUS_IN_RESET
        self.set_reg(tx.CTRL, tx.CTRL_RESET_RELEASE)
        assert not self.reg(tx.STATUS) & tx.STATUS_IN_RESET
        return self.poll(lambda: self.signal() == tx.RUN_MSG_DONE)

    def launch(self, img, args):
        for addr, data in img.launch:
            self.l1_write(addr, data)
        for addr in set(img.rta.values()):
            self.l1_write(addr, struct.pack(f"<{len(args)}I", *args))
        self.set_reg(tx.DOORBELL, tx.RUN_MSG_GO)
        assert self.reg(tx.STATUS) & tx.STATUS_BUSY
        return self.poll(lambda: self.reg(tx.STATUS) & tx.STATUS_DONE)


def _capture_io(trace, img):
    """The capture's operand writes (into the CB FIFOs) and its C readback replies."""
    fifos = {cb.index: (cb.addr, cb.addr + cb.size) for cb in img.cbs.values()}
    lo_c, hi_c = fifos[2]
    writes, reads = [], {}
    for line in trace.read_text().splitlines():
        p = parse_trace_line(line)
        if p is None:
            continue
        addr = p["address"]
        if p["cmd"] == proto.CMD_WRITE and any(
            lo <= addr < hi for lo, hi in fifos.values()
        ):
            writes.append((addr, p["data"]))
        elif p["cmd"] == proto.CMD_READ and lo_c <= addr < hi_c:
            reads[addr] = p["reply"]
    return writes, reads


@pytest.mark.parametrize("trace", [TRACE_FP32, TRACE_BF16], ids=["fp32", "bf16"])
def test_block_reproduces_the_live_run_bit_for_bit(trace):
    img = tensix_image.from_trace(trace)
    writes, reads = _capture_io(trace, img)
    host = _Host(_block())
    assert host.reg(tx.ID) == (0x54454E53 << 32) | 2
    assert host.reg(tx.L1_SIZE) == 1536 * 1024
    assert host.reg(tx.GO_ADDR) == 0x4F0
    host.boot(img)
    for addr, data in writes:
        host.l1_write(addr, data)
    waited = host.launch(img, (2, 2, 2))  # the capture's 64^3 problem
    c = img.cbs[2]
    got = host.l1_read(c.addr, 4 * c.page_size)
    want = b"".join(reads[a] for a in sorted(reads))[: len(got)]
    assert got == want
    stats = host.block.counters()
    assert stats["launches"] == 1
    assert 0 < stats["busy_cycles"] <= waited
    assert host.reg(tx.BUSY_CYCLES) == stats["busy_cycles"]
    assert host.reg(tx.TILE_CYCLES) == stats["tile_cycles"] == host.cycle
    assert stats["l1_read_bytes"] >= 4 * c.page_size


def test_relaunch_with_new_args_recomputes():
    img = tensix_image.from_trace(TRACE_FP32)
    writes, reads = _capture_io(TRACE_FP32, img)
    host = _Host(_block())
    host.boot(img)
    for addr, data in writes:
        host.l1_write(addr, data)
    host.launch(img, (2, 2, 2))
    c = img.cbs[2]
    first = host.l1_read(c.addr, 2 * c.page_size)
    host.l1_write(c.addr, bytes(4 * c.page_size))
    # Mt = 1: only C's first tile row, (0, 0) and (0, 1), the same sums as before.
    for addr in set(img.rta.values()):
        host.l1_write(addr, struct.pack("<3I", 1, 2, 2))
    host.set_reg(tx.DOORBELL, tx.RUN_MSG_GO)
    host.poll(lambda: host.reg(tx.STATUS) & tx.STATUS_DONE)
    assert host.l1_read(c.addr, 2 * c.page_size) == first
    assert host.l1_read(c.addr + 2 * c.page_size, 2 * c.page_size) == bytes(
        2 * c.page_size
    )
    assert host.block.counters()["launches"] == 2


def test_clock_ratio_and_lazy_construction():
    block = _block(clock_ratio="1/2", cost_model=False)
    host = _Host(block)
    host.tick(10)
    assert block.device is None
    assert block.counters()["tile_cycles"] == 0
    host.reg(tx.STATUS)  # first touch builds the tile, its clock starting now
    assert block.device is not None
    host.tick(10)
    assert block.counters()["tile_cycles"] == 10  # cycles 10..19 -> floor(20 / 2)
    block.reset()
    assert block.device is None
    assert block.counters()["tile_cycles"] == 0


def test_entry_pcs_program_the_blackhole_override():
    host = _Host(_block())
    host.set_reg(tx.ENTRY_TRISC1, 0x6FE0)
    host.set_reg(tx.ENTRY_NCRISC, 0x5BE0)
    assert host.reg(tx.ENTRY_TRISC1) == 0x6FE0
    host.set_reg(tx.CTRL, tx.CTRL_RESET_ASSERT)
    host.set_reg(tx.CTRL, tx.CTRL_RESET_RELEASE)
    host.set_reg(tx.TILE_ADDR, 0xFFB1222C)
    assert host.reg(tx.TILE_DATA) == 0x6FE0
    host.set_reg(tx.TILE_ADDR, 0xFFB12234)
    assert host.reg(tx.TILE_DATA) & 0b111 == 0b010
    host.set_reg(tx.TILE_ADDR, 0xFFB1223C)
    assert host.reg(tx.TILE_DATA) & 1


def test_bad_accesses_are_refused():
    block = _block()
    with pytest.raises(IndexError):
        block.read(0x180000 - 4, 8)  # past L1
    with pytest.raises(IndexError):
        block.write(tx.CTRL_OFFSET + tx.STATUS, (0).to_bytes(8, "little"))  # read-only
    with pytest.raises(IndexError):
        block.read(tx.CTRL_OFFSET + 4, 8)  # crosses a register


def _design(params, window="16MiB"):
    import yaml

    raw = yaml.safe_load(DESIGN.read_text())
    for block in raw["blocks"]:
        if block["name"] == "tensix":
            block["params"] = params
            block["window"] = window
    return parse_design(raw, DESIGN)


@pytest.mark.skipif(not HAVE_ML_DTYPES, reason="naiad-tensix's formats need ml_dtypes")
@pytest.mark.parametrize(
    "params,window",
    [
        ({"arch": "grayskull"}, "16MiB"),
        ({"clock_ratio": 0}, "16MiB"),
        ({"clock_ratio": "x"}, "16MiB"),
        ({"cost_model": "yes"}, "16MiB"),
        ({"frequency": 1}, "16MiB"),
        ({}, "1MiB"),
    ],
)
def test_design_validation(params, window):
    with pytest.raises(DesignError):
        _design(params, window)


@pytest.mark.skipif(not HAVE_ML_DTYPES, reason="naiad-tensix's formats need ml_dtypes")
def test_preset_header_defines():
    from framework.naiad.gen import gen_header

    text = gen_header(load_design(DESIGN))
    for name in (
        "NAIAD_TENSIX_PRESENT",
        "NAIAD_TENSIX_CTRL_OFFSET 0x00200000UL",
        "NAIAD_HAS_DMA",
    ):
        assert name in text


def test_wormhole_tile_builds_and_maps_its_l1():
    host = _Host(_block(arch="wormhole"))
    assert host.reg(tx.ID) == (0x54454E53 << 32) | 1
    assert host.reg(tx.L1_SIZE) == 1464 * 1024
    assert host.reg(tx.GO_ADDR) == 0x4A0
    assert host.reg(tx.STATUS) & tx.STATUS_IN_RESET
    host.l1_write(0x100, b"abcd")
    assert host.l1_read(0x100, 4) == b"abcd"
    host.tick(5)
    assert host.reg(tx.TILE_CYCLES) == 5
    with pytest.raises(IndexError):
        host.set_reg(tx.ENTRY_TRISC0, 0x1000)
