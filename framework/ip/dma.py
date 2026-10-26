"""DMA engine: clocked copies over the SoC bus, programmed through MMIO registers.

Each channel is eight 64-bit registers at ``CH_STRIDE * channel``:

====  ============  ==========================================================
0x00  SRC           source address (byte)
0x08  DST           destination address (byte)
0x10  LEN           bytes per row (the whole transfer in 1D mode)
0x18  SRC_STRIDE    2D: bytes from one source row's start to the next
0x20  DST_STRIDE    2D: bytes from one destination row's start to the next
0x28  ROWS          2D: number of rows
0x30  CTRL          bit 0 START (write 1; reads 0), bit 1 MODE_2D
0x38  STATUS        bit 0 BUSY, bit 1 DONE, bit 2 ERROR, bits 15:8 error code
====  ============  ==========================================================

Writing CTRL with START latches SRC..ROWS, clears DONE/ERROR and sets BUSY, so
the registers can be reprogrammed for the next transfer while one runs. Writing
STATUS clears whichever of DONE/ERROR are set in the written value (W1C). A
START while the channel is busy is dropped and sets ERROR with ``ERR_BUSY``;
the running transfer carries on.

Timing: a transfer started during cycle ``t`` spends ``setup_latency`` cycles
in setup, then moves ``bytes_per_cycle`` bytes on each following cycle, so an
``N``-byte transfer shows DONE at the end of cycle ``t + setup_latency +
ceil(N / bytes_per_cycle)`` and a poll in the cycle after sees it. Channels do
not contend with each other or with the cores for the bus: each moves its own
``bytes_per_cycle``. A chunk is one bus read and one bus write, through
``soc.memory``, so any mapped region (DRAM, scratchpad, another block's MMIO)
is a valid end, and a chunk crossing a 2D row boundary is split at the row.

Ordering: the copy is visible progressively. A core that reads the destination
before STATUS shows DONE sees whichever chunks have landed and the old bytes
everywhere else, exactly the race a real engine has; software polls STATUS and
reads the destination only after DONE. There are no interrupts (Naiad cores
take none). Overlapping source and destination copy forward chunk by chunk,
which is no promise: like hardware, an overlapping transfer is undefined.

Errors are reported, not raised: a transfer whose source or destination span
is not inside one mapped block (``ERR_SRC`` / ``ERR_DST``), or a 2D transfer
whose SRC, DST or strides are not multiples of ``stride_align``
(``ERR_ALIGN``), is refused at START with ERROR set and nothing moved. A bus
access refused mid-transfer (an MMIO block rejecting the access size) stops
the transfer with ``ERR_BUS``.
"""

from framework.ip.base import RegisterBlock

SRC = 0x00
DST = 0x08
LEN = 0x10
SRC_STRIDE = 0x18
DST_STRIDE = 0x20
ROWS = 0x28
CTRL = 0x30
STATUS = 0x38
CH_STRIDE = 0x40

CTRL_START = 1 << 0
CTRL_2D = 1 << 1
STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1
STATUS_ERROR = 1 << 2
STATUS_ERR_SHIFT = 8

ERR_NONE = 0
ERR_SRC = 1
ERR_DST = 2
ERR_ALIGN = 3
ERR_BUS = 4
ERR_BUSY = 5

#: Block types whose bytes are counted per path; any other end counts as MMIO.
#: A ``tensix`` block's end is its L1 window (``framework.ip.tensix.block``).
MEMORY_TYPES = ("dram", "scratchpad", "tensix")
_MASK64 = (1 << 64) - 1
_DEFAULTS = {
    "channels": 1,
    "bytes_per_cycle": 8,
    "setup_latency": 16,
    "stride_align": 8,
}


class _Channel:
    def __init__(self):
        self.regs = [0] * (CH_STRIDE // 8)
        self.status = 0
        # The latched transfer, valid while BUSY.
        self.setup_left = 0
        self.start_cycle = 0
        self.src_row = self.dst_row = 0
        self.col = 0
        self.row_len = 0
        self.rows_left = 0
        self.src_stride = self.dst_stride = 0
        self.path = None


class DMAEngine(RegisterBlock):
    PARAMS = frozenset(_DEFAULTS)

    @classmethod
    def _param(cls, params, name):
        return int(params.get(name, _DEFAULTS[name]))

    @classmethod
    def block_size(cls, params, window):
        return CH_STRIDE * cls._param(params, "channels")

    @classmethod
    def validate_params(cls, spec, error):
        for name, low in (
            ("channels", 1),
            ("bytes_per_cycle", 1),
            ("setup_latency", 0),
            ("stride_align", 1),
        ):
            if cls._param(spec.params, name) < low:
                raise error(f"block '{spec.name}': {name} must be >= {low}")
        align = cls._param(spec.params, "stride_align")
        if align & (align - 1):
            raise error(
                f"block '{spec.name}': stride_align must be a power of two, got {align}"
            )

    def __init__(self, spec):
        super().__init__(spec)
        params = spec.params
        self.channels = self._param(params, "channels")
        self.bytes_per_cycle = self._param(params, "bytes_per_cycle")
        self.setup_latency = self._param(params, "setup_latency")
        self.stride_align = self._param(params, "stride_align")
        self.path_keys = ()
        self.reset()

    def attach(self, soc):
        super().attach(soc)
        # The counter set is fixed at build time (the counter hub discovers it
        # then): one key per memory-to-memory path, and one for any transfer
        # with an MMIO end.
        memories = sorted(
            {b.spec.type for b in soc.blocks.values() if b.spec.type in MEMORY_TYPES}
        )
        self.path_keys = tuple(
            f"{s}_to_{d}_bytes" for s in memories for d in memories
        ) + ("mmio_bytes",)
        self.reset()

    def reset(self):
        self.chans = [_Channel() for _ in range(self.channels)]
        self.active = []
        self.stats = {"bytes": 0, "transfers": 0, "busy_cycles": 0, "errors": 0}
        self.stats.update(dict.fromkeys(self.path_keys, 0))

    # -- registers ---------------------------------------------------------

    def read_reg(self, offset):
        ch = self.chans[offset // CH_STRIDE]
        reg = offset % CH_STRIDE
        if reg == STATUS:
            return ch.status
        if reg == CTRL:
            return ch.regs[reg // 8] & ~CTRL_START
        return ch.regs[reg // 8]

    def write_reg(self, offset, value):
        ch = self.chans[offset // CH_STRIDE]
        reg = offset % CH_STRIDE
        value &= _MASK64
        if reg == STATUS:
            ch.status &= ~(value & (STATUS_DONE | STATUS_ERROR)) | STATUS_BUSY
            if not ch.status & STATUS_ERROR:
                ch.status &= ~(0xFF << STATUS_ERR_SHIFT)
            return
        ch.regs[reg // 8] = value
        if reg == CTRL and value & CTRL_START:
            self._start(ch)

    # -- transfers ---------------------------------------------------------

    def _fail(self, ch, code):
        ch.status = (
            (ch.status & STATUS_BUSY) | STATUS_ERROR | (code << STATUS_ERR_SHIFT)
        )
        self.stats["errors"] += 1

    def _span_type(self, low, length):
        """The type of the one mapped block holding ``[low, low + length)``, else None."""
        addr_range, block = self.soc.memory.memory_map.locate(low)
        if addr_range is None or low + max(length, 1) - 1 > addr_range.high:
            return None
        return block.spec.type if block.spec.type in MEMORY_TYPES else "mmio"

    def _start(self, ch):
        if ch.status & STATUS_BUSY:
            self._fail(ch, ERR_BUSY)
            return
        ch.status = 0
        src, dst, length = ch.regs[SRC // 8], ch.regs[DST // 8], ch.regs[LEN // 8]
        if ch.regs[CTRL // 8] & CTRL_2D:
            rows, src_stride, dst_stride = (
                ch.regs[ROWS // 8],
                ch.regs[SRC_STRIDE // 8],
                ch.regs[DST_STRIDE // 8],
            )
            if (src | dst | src_stride | dst_stride) & (self.stride_align - 1):
                self._fail(ch, ERR_ALIGN)
                return
        else:
            rows, src_stride, dst_stride = 1, 0, 0
        total = length * rows
        if total:
            src_type = self._span_type(src, (rows - 1) * src_stride + length)
            if src_type is None:
                self._fail(ch, ERR_SRC)
                return
            dst_type = self._span_type(dst, (rows - 1) * dst_stride + length)
            if dst_type is None:
                self._fail(ch, ERR_DST)
                return
            mmio = "mmio" in (src_type, dst_type)
            ch.path = "mmio_bytes" if mmio else f"{src_type}_to_{dst_type}_bytes"
        ch.src_row, ch.dst_row, ch.col = src, dst, 0
        ch.row_len, ch.rows_left = length, rows if total else 0
        ch.src_stride, ch.dst_stride = src_stride, dst_stride
        ch.setup_left = self.setup_latency
        ch.start_cycle = self.soc.clock.current_cycle
        self.stats["transfers"] += 1
        if not total and not ch.setup_left:
            ch.status = STATUS_DONE
            return
        ch.status = STATUS_BUSY
        self.active.append(ch)

    def _finish(self, ch, status):
        ch.status = (ch.status & ~STATUS_BUSY) | status
        self.active.remove(ch)

    def _move(self, ch):
        """One cycle's worth of bytes: a few whole-chunk bus copies, split at row ends."""
        memory = self.soc.memory
        budget = self.bytes_per_cycle
        moved = 0
        try:
            while budget and ch.rows_left:
                take = min(budget, ch.row_len - ch.col)
                memory.write(
                    ch.dst_row + ch.col, memory.read(ch.src_row + ch.col, take)
                )
                moved += take
                budget -= take
                ch.col += take
                if ch.col == ch.row_len:
                    ch.col = 0
                    ch.rows_left -= 1
                    ch.src_row += ch.src_stride
                    ch.dst_row += ch.dst_stride
        except IndexError:
            self._fail(ch, ERR_BUS)
            self._finish(ch, 0)
        stats = self.stats
        stats["bytes"] += moved
        stats[ch.path] += moved

    def clock_tick(self, cycle):
        if not self.active:
            return
        for ch in tuple(self.active):
            # The cycle a START landed in is not one of the transfer's cycles,
            # whether this block ticks before or after the core that wrote it.
            if cycle <= ch.start_cycle:
                continue
            self.stats["busy_cycles"] += 1
            if ch.setup_left:
                ch.setup_left -= 1
                if ch.setup_left or ch.rows_left:
                    continue
            else:
                self._move(ch)
                if not ch.status & STATUS_BUSY:
                    continue
            if not ch.rows_left:
                self._finish(ch, STATUS_DONE)

    # -- reporting ---------------------------------------------------------

    def counters(self):
        """``bytes`` and ``transfers`` (started, not refused), ``errors``, ``busy_cycles``
        (channel-cycles, so two busy channels count two a cycle), and bytes per path:
        ``<src type>_to_<dst type>_bytes`` between memories, ``mmio_bytes`` otherwise."""
        return dict(self.stats)

    def header_defines(self):
        regs = [
            ("SRC", SRC),
            ("DST", DST),
            ("LEN", LEN),
            ("SRC_STRIDE", SRC_STRIDE),
            ("DST_STRIDE", DST_STRIDE),
            ("ROWS", ROWS),
            ("CTRL", CTRL),
            ("STATUS", STATUS),
            ("CH_STRIDE", CH_STRIDE),
            ("CTRL_START", CTRL_START),
            ("CTRL_2D", CTRL_2D),
            ("STATUS_BUSY", STATUS_BUSY),
            ("STATUS_DONE", STATUS_DONE),
            ("STATUS_ERROR", STATUS_ERROR),
            ("STATUS_ERR_SHIFT", STATUS_ERR_SHIFT),
            ("ERR_SRC", ERR_SRC),
            ("ERR_DST", ERR_DST),
            ("ERR_ALIGN", ERR_ALIGN),
            ("ERR_BUS", ERR_BUS),
            ("ERR_BUSY", ERR_BUSY),
        ]
        return regs + [
            ("CHANNELS", self.channels),
            ("BYTES_PER_CYCLE", self.bytes_per_cycle),
            ("SETUP_LATENCY", self.setup_latency),
            ("STRIDE_ALIGN", self.stride_align),
        ]
