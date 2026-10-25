"""A Tensix tile on the Naiad bus: a loosely coupled accelerator driven by MMIO and DMA.

The block owns one Wolfpine Tensix tile, the same model the tt-metal replay
guards drive (``framework.device.blackhole`` by default, ``wormhole`` with
``arch: wormhole``): its L1, the five baby RISC-V cores (BRISC, NCRISC,
TRISC0-2) and the Tensix coprocessor, inside an otherwise empty device (the
DRAM tiles exist but nothing addresses them). Nothing about the tile is
changed: the Naiad side reaches it only through the tile's own L1 and soft
reset, exactly as a tt-metal host does over the wire.

Address map (offsets from the block's base)::

    0x0000_0000  L1 window: the tile's L1, byte-addressable (1.5 MiB Blackhole,
                 1464 KiB Wormhole); any size, any alignment, so both the cores
                 and the DMA engine read and write it like memory
    0x0020_0000  control registers (64-bit, at 8-byte strides)

==========  ===========  ======================================================
offset      name         meaning
==========  ===========  ======================================================
0x00        ID           RO: ``0x54454E53_000000aa`` ("TENS"; aa = 1 Wormhole, 2 Blackhole)
0x08        L1_SIZE      RO: bytes of L1 behind the window
0x10        CTRL         WO: bit 0 RESET_ASSERT holds all five cores in soft
                         reset; bit 1 RESET_RELEASE releases BRISC (its firmware
                         brings the other four up, as after a tt-metal DEASSERT)
0x18        DOORBELL     WO: the low byte is written to the go message's signal
                         byte in L1. ``0x80`` (RUN_MSG_GO) is a launch: it sets
                         BUSY and starts the busy counter
0x20        STATUS       RO: bit 0 IN_RESET (BRISC held), bit 1 BUSY (a launch
                         whose go signal has not yet read DONE), bit 2 DONE (the
                         last launch finished; cleared by the next doorbell),
                         bits 15:8 the go signal byte as the firmware left it
0x28        GO_ADDR      RO: L1 address of the go message (go_msg_t)
0x30        ENTRY_NCRISC RW: reset PC for NCRISC (0 = the core's default)
0x38        ENTRY_TRISC0 RW: likewise TRISC0, TRISC1, TRISC2 (0x40, 0x48).
                         Programs the tile's reset-PC override (Blackhole: the
                         RISCV_DEBUG_REG *_RESET_PC / _OVERRIDE registers);
                         BRISC always boots at L1 0. Takes effect at the next
                         release of that core
0x50        TILE_ADDR    RW: address inside the tile for TILE_DATA
0x58        TILE_DATA    RW: 32-bit access at TILE_ADDR (any tile register,
                         e.g. the Tensix config space or the mailboxes)
0x60        TILE_CYCLES  RO: Tensix cycles since reset
0x68        BUSY_CYCLES  RO: Tensix cycles spent BUSY
0x70        LAUNCHES     RO: doorbells with RUN_MSG_GO
==========  ===========  ======================================================

**Clock.** The tile runs at ``clock_ratio`` Tensix cycles per Naiad cycle (an
int, a float or ``"num/den"``; default 1). At the end of Naiad cycle ``t`` the
tile has run ``floor((t + 1) * clock_ratio)`` cycles, and the block ticks after
the cores, so an MMIO access in cycle ``t`` sees the tile as of the end of
cycle ``t - 1``. While BUSY the tile is stepped one cycle at a time so the
cycle the firmware writes DONE is counted exactly. ``cost_model`` (default
true) builds the tile with the Tensix cost model on (``TT_SIM_COST_MODEL``):
the occupancy floor the replay guards use for timing, not cycle-accurate.

**Running a kernel** without a tt-metal host is the job of a *Tensix image*
(``framework.naiad.tensix_image``): the firmware and kernel writes a captured
tt-metal launch made, replayed by the Naiad program through the L1 window
(``kernels/naiad/include/tensix_accel.h``). The block itself knows nothing
about tt-metal beyond where the go message is.

**Counters** (``<block>.<key>``): ``busy_cycles``, ``tile_cycles``,
``launches``, ``l1_read_bytes`` / ``l1_write_bytes`` (every access through the
L1 window, by a core or the DMA engine).

A fresh tile is built on every reset, so runs never see each other's L1.
"""

import os
from fractions import Fraction

from framework.naiad.ip.base import IPBlock
from framework.pe.rv.babyriscv import BabyRISCVCoreType

L1_OFFSET = 0x0
CTRL_OFFSET = 0x0020_0000
CTRL_SIZE = 0x1000

ID = 0x00
L1_SIZE = 0x08
CTRL = 0x10
DOORBELL = 0x18
STATUS = 0x20
GO_ADDR = 0x28
ENTRY_NCRISC = 0x30
ENTRY_TRISC0 = 0x38
ENTRY_TRISC1 = 0x40
ENTRY_TRISC2 = 0x48
TILE_ADDR = 0x50
TILE_DATA = 0x58
TILE_CYCLES = 0x60
BUSY_CYCLES = 0x68
LAUNCHES = 0x70

CTRL_RESET_ASSERT = 1 << 0
CTRL_RESET_RELEASE = 1 << 1
STATUS_IN_RESET = 1 << 0
STATUS_BUSY = 1 << 1
STATUS_DONE = 1 << 2
STATUS_SIGNAL_SHIFT = 8

RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
SOFT_RESET_ADDR = 0xFFB121B0
_BRISC_RESET_BIT = 1 << 11
_ID_MAGIC = 0x54454E53 << 32
_MASK64 = (1 << 64) - 1

#: Per arch: (ID code, tile coord, go_msg_t L1 address).
ARCHES = {
    "wormhole": (1, (18, 18), 0x4A0),
    "blackhole": (2, (1, 2), 0x4F0),
}
_ENTRY_CORES = {
    ENTRY_NCRISC: BabyRISCVCoreType.NCRISC,
    ENTRY_TRISC0: BabyRISCVCoreType.TRISC0,
    ENTRY_TRISC1: BabyRISCVCoreType.TRISC1,
    ENTRY_TRISC2: BabyRISCVCoreType.TRISC2,
}
# Blackhole reset-PC override registers (tt-metal blackhole/tensix.h, as
# framework.pe.rv.babyriscv reads them): per core (PC register, override
# register, override bit).
_BH_DEBUG = 0xFFB12000
_BH_ENTRY_REGS = {
    BabyRISCVCoreType.NCRISC: (0x238, 0x23C, 0),
    BabyRISCVCoreType.TRISC0: (0x228, 0x234, 0),
    BabyRISCVCoreType.TRISC1: (0x22C, 0x234, 1),
    BabyRISCVCoreType.TRISC2: (0x230, 0x234, 2),
}


def _ratio(value):
    try:
        ratio = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return ratio if ratio > 0 else None


def _make_device(arch, cost_model):
    """One-tile device of ``arch``; the Tensix cost model is read at construction."""
    if arch == "blackhole":
        from framework.device.blackhole import Blackhole as device_class
    else:
        from framework.device.wormhole import Wormhole as device_class
    coord = ARCHES[arch][1]
    saved = os.environ.get("TT_SIM_COST_MODEL")
    os.environ["TT_SIM_COST_MODEL"] = "1" if cost_model else "0"
    try:
        return device_class(tensix_coords=[coord])
    finally:
        if saved is None:
            del os.environ["TT_SIM_COST_MODEL"]
        else:
            os.environ["TT_SIM_COST_MODEL"] = saved


class TensixBlock(IPBlock):
    PARAMS = frozenset({"arch", "clock_ratio", "cost_model"})

    @classmethod
    def block_size(cls, params, window):
        return CTRL_OFFSET + CTRL_SIZE

    @classmethod
    def validate_params(cls, spec, error):
        arch = spec.params.get("arch", "blackhole")
        if arch not in ARCHES:
            raise error(
                f"block '{spec.name}': arch must be one of {sorted(ARCHES)}, got '{arch}'"
            )
        if _ratio(spec.params.get("clock_ratio", 1)) is None:
            raise error(
                f"block '{spec.name}': clock_ratio must be a positive number or 'num/den'"
            )
        if not isinstance(spec.params.get("cost_model", True), bool):
            raise error(f"block '{spec.name}': cost_model must be true or false")

    def __init__(self, spec):
        super().__init__(spec)
        params = spec.params
        self.arch = params.get("arch", "blackhole")
        self.arch_code, self.coord, self.go_addr = ARCHES[self.arch]
        self.clock_ratio = _ratio(params.get("clock_ratio", 1))
        self.cost_model = bool(params.get("cost_model", True))
        from framework.arch.blackhole import BLACKHOLE_PROFILE
        from framework.arch.wormhole import WORMHOLE_PROFILE

        profile = BLACKHOLE_PROFILE if self.arch == "blackhole" else WORMHOLE_PROFILE
        self.l1_size = profile.tensix_l1_size
        self.device = None
        self.reset()

    def reset(self):
        # Built lazily: a design that never touches the tile pays nothing.
        self.device = None
        self.busy = False
        self.done = False
        self.entry = dict.fromkeys(_ENTRY_CORES, 0)
        self.tile_addr = 0
        self.stats = {
            "busy_cycles": 0,
            "tile_cycles": 0,
            "launches": 0,
            "l1_read_bytes": 0,
            "l1_write_bytes": 0,
        }

    def _dev(self):
        if self.device is None:
            self.device = _make_device(self.arch, self.cost_model)
            # Until now every core was in reset and the tile had nothing to do:
            # start its clock at the Naiad time it is first touched.
            clock = getattr(self.soc, "clock", None)
            now = 0 if clock is None else clock.current_cycle
            self.stats["tile_cycles"] = int(now * self.clock_ratio)
        return self.device

    # -- clock ---------------------------------------------------------------

    def _signal(self):
        return self.device.read(self.coord, self.go_addr + 3, 1)[0]

    def clock_tick(self, cycle):
        if self.device is None:
            return
        target = int((cycle + 1) * self.clock_ratio)
        n = target - self.stats["tile_cycles"]
        if n <= 0:
            return
        if not self.busy:
            self.device.run(n)
            self.stats["tile_cycles"] = target
            return
        stats = self.stats
        for _ in range(n):
            self.device.run(1)
            stats["tile_cycles"] += 1
            stats["busy_cycles"] += 1
            if self._signal() == RUN_MSG_DONE:
                self.busy = False
                self.done = True
                self.device.run(target - stats["tile_cycles"])
                stats["tile_cycles"] = target
                return

    # -- MMIO ----------------------------------------------------------------

    def _l1_span(self, addr, size):
        if addr + size > self.l1_size:
            raise IndexError(
                f"{self.name}: L1 access of {size} bytes at 0x{addr:x} runs past "
                f"the {self.l1_size}-byte L1"
            )

    def read(self, addr, size):
        if addr < CTRL_OFFSET:
            self._l1_span(addr, size)
            self.stats["l1_read_bytes"] += size
            return bytes(self._dev().read(self.coord, addr, size))
        reg = addr - CTRL_OFFSET
        off = reg & 7
        if size <= 0 or off + size > 8 or reg + size > CTRL_SIZE:
            raise IndexError(
                f"{self.name}: access of {size} bytes at offset {hex(addr)} crosses a register"
            )
        value = self.read_reg(reg - off)
        return value.to_bytes(8, "little")[off : off + size]

    def write(self, addr, value, size=None):
        if size is None:
            size = len(value)
        if addr < CTRL_OFFSET:
            self._l1_span(addr, size)
            self.stats["l1_write_bytes"] += size
            self._dev().write(self.coord, addr, bytes(value[:size]))
            return
        reg = addr - CTRL_OFFSET
        off = reg & 7
        if size <= 0 or off + size > 8 or reg + size > CTRL_SIZE:
            raise IndexError(
                f"{self.name}: access of {size} bytes at offset {hex(addr)} crosses a register"
            )
        reg -= off
        if off or size != 8:
            current = bytearray(self.read_reg(reg).to_bytes(8, "little"))
            current[off : off + size] = value[:size]
            value = bytes(current)
        self.write_reg(reg, int.from_bytes(value[:8], "little"))

    def _soft_reset(self):
        return int.from_bytes(
            self._dev().read(self.coord, SOFT_RESET_ADDR, 4), "little"
        )

    def read_reg(self, reg):
        if reg == ID:
            return _ID_MAGIC | self.arch_code
        if reg == L1_SIZE:
            return self.l1_size
        if reg in (CTRL, DOORBELL):
            return 0
        if reg == STATUS:
            status = (self._signal_now() << STATUS_SIGNAL_SHIFT) | (
                STATUS_BUSY if self.busy else 0
            )
            if self.done:
                status |= STATUS_DONE
            if self._soft_reset() & _BRISC_RESET_BIT:
                status |= STATUS_IN_RESET
            return status
        if reg == GO_ADDR:
            return self.go_addr
        if reg in _ENTRY_CORES:
            return self.entry[reg]
        if reg == TILE_ADDR:
            return self.tile_addr
        if reg == TILE_DATA:
            return int.from_bytes(
                self._dev().read(self.coord, self.tile_addr, 4), "little"
            )
        if reg == TILE_CYCLES:
            return self.stats["tile_cycles"]
        if reg == BUSY_CYCLES:
            return self.stats["busy_cycles"]
        if reg == LAUNCHES:
            return self.stats["launches"]
        return 0

    def _signal_now(self):
        self._dev()
        return self._signal()

    def write_reg(self, reg, value):
        value &= _MASK64
        dev = self._dev()
        if reg == CTRL:
            if value & CTRL_RESET_ASSERT:
                dev.assert_soft_reset(self.coord)
                self.busy = False
            if value & CTRL_RESET_RELEASE:
                self._apply_entries()
                dev.deassert_soft_reset(self.coord, BabyRISCVCoreType.BRISC)
                # As the wire bridge does on DEASSERT: clear the tile's stale
                # core state (PCs) before the released cores run.
                dev.reset_tile(self.coord)
            return
        if reg == DOORBELL:
            signal = value & 0xFF
            dev.write(self.coord, self.go_addr + 3, bytes([signal]))
            self.done = False
            if signal == RUN_MSG_GO:
                self.busy = True
                self.stats["launches"] += 1
            return
        if reg in _ENTRY_CORES:
            if self.arch != "blackhole":
                raise IndexError(
                    f"{self.name}: ENTRY_* is modelled on Blackhole only; on {self.arch} "
                    "program the Tensix config's reset-PC override through TILE_ADDR/TILE_DATA"
                )
            self.entry[reg] = value & 0xFFFFFFFF
            return
        if reg == TILE_ADDR:
            self.tile_addr = value & 0xFFFFFFFF
            return
        if reg == TILE_DATA:
            dev.write(
                self.coord, self.tile_addr, (value & 0xFFFFFFFF).to_bytes(4, "little")
            )
            return
        raise IndexError(f"{self.name}: register at offset {hex(reg)} is read-only")

    def _apply_entries(self):
        if self.arch != "blackhole":
            return
        dev = self.device
        for reg, core in _ENTRY_CORES.items():
            pc_off, ovr_off, bit = _BH_ENTRY_REGS[core]
            ovr = int.from_bytes(dev.read(self.coord, _BH_DEBUG + ovr_off, 4), "little")
            pc = self.entry[reg]
            if pc:
                dev.write(self.coord, _BH_DEBUG + pc_off, pc.to_bytes(4, "little"))
                ovr |= 1 << bit
            else:
                ovr &= ~(1 << bit)
            dev.write(self.coord, _BH_DEBUG + ovr_off, ovr.to_bytes(4, "little"))

    # -- reporting -----------------------------------------------------------

    def counters(self):
        """``busy_cycles`` / ``tile_cycles`` (Tensix cycles), ``launches``, and the bytes read
        and written through the L1 window by cores and the DMA engine."""
        return dict(self.stats)

    def header_defines(self):
        return [
            ("PRESENT", 1),
            ("L1_OFFSET", L1_OFFSET),
            ("L1_SIZE_BYTES", self.l1_size),
            ("CTRL_OFFSET", CTRL_OFFSET),
            ("GO_MSG_ADDR", self.go_addr),
            ("ARCH_CODE", self.arch_code),
            ("CLOCK_RATIO_NUM", self.clock_ratio.numerator),
            ("CLOCK_RATIO_DEN", self.clock_ratio.denominator),
            ("REG_ID", ID),
            ("REG_L1_SIZE", L1_SIZE),
            ("REG_CTRL", CTRL),
            ("REG_DOORBELL", DOORBELL),
            ("REG_STATUS", STATUS),
            ("REG_GO_ADDR", GO_ADDR),
            ("REG_ENTRY_NCRISC", ENTRY_NCRISC),
            ("REG_ENTRY_TRISC0", ENTRY_TRISC0),
            ("REG_ENTRY_TRISC1", ENTRY_TRISC1),
            ("REG_ENTRY_TRISC2", ENTRY_TRISC2),
            ("REG_TILE_ADDR", TILE_ADDR),
            ("REG_TILE_DATA", TILE_DATA),
            ("REG_TILE_CYCLES", TILE_CYCLES),
            ("REG_BUSY_CYCLES", BUSY_CYCLES),
            ("REG_LAUNCHES", LAUNCHES),
            ("CTRL_RESET_ASSERT", CTRL_RESET_ASSERT),
            ("CTRL_RESET_RELEASE", CTRL_RESET_RELEASE),
            ("STATUS_IN_RESET", STATUS_IN_RESET),
            ("STATUS_BUSY", STATUS_BUSY),
            ("STATUS_DONE", STATUS_DONE),
            ("STATUS_SIGNAL_SHIFT", STATUS_SIGNAL_SHIFT),
            ("RUN_MSG_GO", RUN_MSG_GO),
            ("RUN_MSG_DONE", RUN_MSG_DONE),
        ]
