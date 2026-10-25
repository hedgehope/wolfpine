"""Tensix images: what a Naiad host program loads into a Tensix tile to run a kernel.

A tt-metal host brings a worker up and launches a program by writing its L1
over the wire: firmware for the five baby cores and the init handshake while
the worker is held in reset, then (after the firmware has answered the
handshake) the compiled kernels, the CB config, the runtime args and the launch
message, then the go signal. A wire trace of one such run
(``TT_SIM_RECORD``, :mod:`framework.bridge.trace`) is therefore a complete
recipe, and nothing in it depends on the host after capture. This module turns
one into a :class:`TensixImage`:

``boot``
    the L1 writes the host made before releasing the worker from reset, in
    order (firmware text and data, the zeroed mailboxes, ``go = INIT``). Load
    them into a worker held in reset, release BRISC (the firmware brings up the
    other four cores itself) and wait for the go message to read ``DONE``.
``launch``
    the L1 writes between that handshake and the go signal (kernel binaries, CB
    config, runtime args, launch message). Load them, write ``GO`` to the go
    message's signal byte and wait for ``DONE`` again. With slow dispatch the
    firmware always reads launch slot 0, so relaunching is ``launch`` + ``GO``
    again, with the runtime args rewritten in between.
``cbs`` / ``rta``
    parsed from the launch message (``kernel_config_msg_t``): every local
    circular buffer's L1 FIFO (address, size, pages, page size), and each
    enabled processor's runtime-argument base. Host writes that land inside a
    CB's FIFO are the program's operand data, not part of the recipe, so they
    are dropped from ``launch`` (the Naiad side DMAs its own operands there).

Writes outside L1 (the soft-reset register the host pokes before the reset
message) are dropped too: the Naiad Tensix block owns reset (``framework.naiad.ip.tensix``).

CLI (used by ``kernels/naiad/tensix/capture/capture.sh``)::

    python -m framework.naiad.tensix_image filter <trace> --core X,Y -o <out.trace> [--note TEXT]
    python -m framework.naiad.tensix_image header <trace> --name NAME -o <out.h>
    python -m framework.naiad.tensix_image info <trace>

``filter`` keeps only one worker's messages (minus the go-message polls), and
trims trailing zeros off large writes, so a capture taken on a full-grid device
shrinks to a small, self-contained trace of that worker; the readbacks keep
their replies, so the filtered trace still records the live run's result.
"""

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

from framework.bridge import protocol as proto
from framework.bridge.trace import parse_trace_line

#: Per arch: (L1 address of launch slot 0, L1 address of go_msg_t, L1 size).
#: Blackhole: mailboxes at 0x60 (``blackhole/dev_mem_map.h``), launch[0] at 0x70,
#: go_messages[0] at 0x4F0. Wormhole: launch[0] at 0x20, go_messages[0] at 0x4A0.
ARCH_LAYOUT = {
    "blackhole": (0x70, 0x4F0, 0x180000),
    "wormhole": (0x20, 0x4A0, 0x16E000),
}

RUN_MSG_INIT = 0x40
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
#: Processor index -> name, the order of ``kernel_config_msg_t.enables`` and ``rta_offset``.
PROCESSORS = ("brisc", "ncrisc", "trisc0", "trisc1", "trisc2")

# kernel_config_msg_t (hostdev/dev_msgs.h, packed; tt-metal 0.74, ProgrammableCoreType::COUNT
# = 4, five processors): kernel_config_base[4] u32, sem_offset[4] u16, local_cb_offset u16
# (+0x18), remote_cb_offset u16, rta_offset[5] of {u16 rta, u16 crta} (+0x1C), mode, pad,
# kernel_text_offset[5], kernel_text_size[5], pad, local_cb_mask u64 (+0x68), ...,
# enables u32 (+0x78, which is what driver/blackhole's LAUNCH_ENABLES_OFFSET 0xE8 points at).
_KCFG_BASE = 0x00
_KCFG_LOCAL_CB_OFFSET = 0x18
_KCFG_RTA_OFFSET = 0x1C
_KCFG_LOCAL_CB_MASK = 0x68
_KCFG_ENABLES = 0x78
_CB_CONFIG_WORDS = 4  # fifo_addr, fifo_size, fifo_num_pages, fifo_page_size (bytes)


class TensixImageError(Exception):
    pass


@dataclass(frozen=True)
class CircularBuffer:
    index: int
    addr: int
    size: int
    num_pages: int
    page_size: int


@dataclass
class TensixImage:
    arch: str
    core: tuple
    boot: list = field(default_factory=list)  # [(addr, bytes)]
    launch: list = field(default_factory=list)  # [(addr, bytes)]
    go_addr: int = 0
    launch_addr: int = 0
    cbs: dict = field(default_factory=dict)  # index -> CircularBuffer
    rta: dict = field(default_factory=dict)  # processor name -> L1 address
    enables: int = 0
    note: str = ""

    @property
    def boot_bytes(self):
        return sum(len(d) for _, d in self.boot)

    @property
    def launch_bytes(self):
        return sum(len(d) for _, d in self.launch)

    def summary(self):
        lines = [
            f"{self.arch} worker {self.core}: boot {len(self.boot)} writes / {self.boot_bytes} B, "
            f"launch {len(self.launch)} writes / {self.launch_bytes} B, go 0x{self.go_addr:x}",
            "enables: "
            + ", ".join(p for i, p in enumerate(PROCESSORS) if self.enables >> i & 1),
        ]
        for cb in self.cbs.values():
            lines.append(
                f"  cb{cb.index}: 0x{cb.addr:x} size 0x{cb.size:x} ({cb.num_pages} x {cb.page_size} B)"
            )
        for proc, addr in self.rta.items():
            lines.append(f"  rta {proc}: 0x{addr:x}")
        if self.note:
            lines.append(f"note: {self.note}")
        return "\n".join(lines)


def _messages(path):
    """Parsed trace lines, plus each ``# note:`` comment."""
    notes = []
    msgs = []
    with open(path) as f:
        for line in f:
            if line.startswith("# note:"):
                notes.append(line[len("# note:") :].strip())
                continue
            p = parse_trace_line(line)
            if p is not None:
                msgs.append(p)
    return msgs, notes


def _pick_core(msgs, core, go_addr):
    if core is not None:
        return tuple(core)
    launched = {
        p["core"]
        for p in msgs
        if p["cmd"] == proto.CMD_WRITE
        and p["address"] == go_addr
        and len(p["data"]) == 4
        and p["data"][3] == RUN_MSG_GO
    }
    if len(launched) != 1:
        raise TensixImageError(
            f"expected exactly one launched worker, found {sorted(launched)}; pass core="
        )
    return launched.pop()


def _parse_launch(data, image, l1_size):
    if len(data) < _KCFG_ENABLES + 4:
        raise TensixImageError(f"launch message is {len(data)} bytes, too short")
    base = struct.unpack_from("<I", data, _KCFG_BASE)[0]
    local_cb = struct.unpack_from("<H", data, _KCFG_LOCAL_CB_OFFSET)[0]
    mask = struct.unpack_from("<Q", data, _KCFG_LOCAL_CB_MASK)[0]
    image.enables = struct.unpack_from("<I", data, _KCFG_ENABLES)[0]
    for i, proc in enumerate(PROCESSORS):
        if image.enables >> i & 1:
            rta = struct.unpack_from("<H", data, _KCFG_RTA_OFFSET + 4 * i)[0]
            image.rta[proc] = base + rta
    if not 0 < base < l1_size:
        raise TensixImageError(f"kernel_config_base 0x{base:x} is outside L1")
    return base + local_cb, mask


def _l1_bytes(image, addr, n):
    """The bytes the image's writes leave at ``[addr, addr + n)`` (zeros elsewhere)."""
    out = bytearray(n)
    for a, d in [*image.boot, *image.launch]:
        lo, hi = max(a, addr), min(a + len(d), addr + n)
        if lo < hi:
            out[lo - addr : hi - addr] = d[lo - a : hi - a]
    return bytes(out)


def from_trace(path, arch="blackhole", core=None):
    """Build a :class:`TensixImage` from a wire trace of one tt-metal launch.

    The worker is the one ``core`` names, else the one worker the trace sends
    ``GO``. Only the first launch is taken.
    """
    if arch not in ARCH_LAYOUT:
        raise TensixImageError(f"unknown arch '{arch}'; known: {sorted(ARCH_LAYOUT)}")
    launch_addr, go_addr, l1_size = ARCH_LAYOUT[arch]
    msgs, notes = _messages(path)
    core = _pick_core(msgs, core, go_addr)
    image = TensixImage(
        arch=arch,
        core=core,
        go_addr=go_addr,
        launch_addr=launch_addr,
        note="; ".join(notes),
    )
    phase = "boot"
    deasserted = False
    launch_writes = []
    for p in msgs:
        if p["core"] != core:
            continue
        cmd, addr, data = p["cmd"], p["address"], p["data"]
        if cmd == proto.CMD_RESET_DEASSERT:
            deasserted = True
            phase = "launch"
            continue
        if cmd != proto.CMD_WRITE or addr >= l1_size:
            continue
        if addr == go_addr and len(data) == 4 and data[3] == RUN_MSG_GO:
            break
        if phase == "boot":
            image.boot.append((addr, bytes(data)))
        else:
            launch_writes.append((addr, bytes(data)))
    else:
        raise TensixImageError(f"{path}: no GO for worker {core}")
    if not deasserted:
        raise TensixImageError(f"{path}: worker {core} is never released from reset")
    launch_msg = [d for a, d in launch_writes if a == launch_addr]
    if not launch_msg:
        raise TensixImageError(f"{path}: no launch message at 0x{launch_addr:x}")
    cb_cfg_addr, mask = _parse_launch(launch_msg[-1], image, l1_size)
    image.launch = launch_writes
    # The CB config, as the launch phase leaves it.
    n_cbs = mask.bit_length()
    raw = _l1_bytes(image, cb_cfg_addr, 16 * n_cbs)
    for i in range(n_cbs):
        if mask >> i & 1:
            addr, size, pages, page = struct.unpack_from("<4I", raw, 16 * i)
            image.cbs[i] = CircularBuffer(i, addr, size, pages, page)
    # Operand data the capture host wrote into a FIFO is not part of the recipe.
    fifos = [(cb.addr, cb.addr + cb.size) for cb in image.cbs.values()]
    image.launch = [
        (a, d)
        for a, d in launch_writes
        if not any(a < hi and a + len(d) > lo for lo, hi in fifos)
    ]
    return image


# -- filter ----------------------------------------------------------------


def filter_trace(src, dst, core, note=""):
    """Write ``src``'s messages for ``core`` (minus go-message polls) to ``dst``.

    Writes over 4 KiB lose their trailing zero bytes (a fresh L1 is zero), so a
    capture's full-buffer operand writes shrink to the tiles it used.
    """
    core = tuple(core)
    out = [f"# note: {note}\n"] if note else []
    go_addrs = {go for _, go, _ in ARCH_LAYOUT.values()}
    with open(src) as f:
        for line in f:
            p = parse_trace_line(line)
            if p is None or p["core"] != core:
                continue
            if p["cmd"] == proto.CMD_READ and p["address"] in go_addrs:
                continue
            if p["cmd"] == proto.CMD_WRITE and len(p["data"]) > 4096:
                data = p["data"]
                keep = (len(data.rstrip(b"\0")) + 3) & ~3
                parts = line.split()
                parts[3] = f"size={keep}"
                parts[4] = f"data={data[:keep].hex() or '-'}"
                line = " ".join(parts) + "\n"
            out.append(line)
    Path(dst).write_text("".join(out))


# -- C header --------------------------------------------------------------


def _segments(writes):
    """``writes`` as disjoint ``(addr, bytes)`` segments, in first-write order.

    A later write wins where it overlaps an earlier one: the earlier segment
    loses those bytes (it may split in two). The result leaves L1 exactly as the
    writes did, whatever order the segments land in, which matters because the
    Naiad driver spreads them over the DMA engine's channels. Nothing is padded:
    bytes no write set are never touched (after boot the firmware owns them).
    """
    segs = []
    for addr, data in writes:
        end = addr + len(data)
        kept = []
        for a, d in segs:
            e = a + len(d)
            if e <= addr or a >= end:
                kept.append((a, d))
                continue
            if a < addr:
                kept.append((a, d[: addr - a]))
            if e > end:
                kept.append((end, d[end - a :]))
        kept.append((addr, bytes(data)))
        segs = kept
    return segs


def c_header(image, name):
    """A C header carrying ``image`` for ``kernels/naiad/include/tensix_accel.h``.

    Defines ``tensix_image_t <name>`` (``TENSIX_IMAGE_<NAME>``) plus each CB's
    FIFO and each runtime-arg base as ``TENSIX_<NAME>_*`` macros. Boot and launch
    are separate segment lists, because the firmware must have answered the
    init handshake before the launch is written.
    """
    ident = name.lower()
    macro = name.upper()
    boot = _segments(image.boot)
    launch = _segments(image.launch)
    blob = b"".join(d for _, d in boot) + b"".join(d for _, d in launch)
    lines = ["/* Generated by framework.naiad.tensix_image -- do not edit. */"]
    lines += [f"/* {line} */" for line in image.summary().splitlines()]
    lines += [
        f"#ifndef TENSIX_IMAGE_{macro}_H",
        f"#define TENSIX_IMAGE_{macro}_H",
        "",
        '#include "tensix_accel.h"',
        "",
        f'#define TENSIX_{macro}_ARCH "{image.arch}"',
        f"#define TENSIX_{macro}_GO_ADDR 0x{image.go_addr:x}",
    ]
    for cb in image.cbs.values():
        p = f"TENSIX_{macro}_CB{cb.index}"
        lines += [
            f"#define {p}_ADDR 0x{cb.addr:x}",
            f"#define {p}_SIZE 0x{cb.size:x}",
            f"#define {p}_PAGES {cb.num_pages}",
            f"#define {p}_PAGE_SIZE {cb.page_size}",
        ]
    rta = sorted(set(image.rta.values()))
    for proc, addr in image.rta.items():
        lines.append(f"#define TENSIX_{macro}_RTA_{proc.upper()} 0x{addr:x}")
    lines += [
        "",
        f"static const uint8_t tensix_{ident}_blob[{len(blob)}] __attribute__((aligned(64))) = {{",
    ]
    for i in range(0, len(blob), 16):
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in blob[i : i + 16]) + ",")
    lines.append("};")

    def seg_table(tag, segs, offset):
        rows = []
        for addr, data in segs:
            rows.append(f"    {{0x{addr:x}, {offset}, {len(data)}}},")
            offset += len(data)
        lines.append(
            f"static const tensix_segment_t tensix_{ident}_{tag}[{len(segs)}] = {{"
        )
        lines.extend(rows)
        lines.append("};")
        return offset

    offset = seg_table("boot", boot, 0)
    seg_table("launch", launch, offset)
    lines.append(f"static const uint32_t tensix_{ident}_rta[{len(rta)}] = {{")
    lines.extend(f"    0x{a:x}," for a in rta)
    lines += [
        "};",
        f"static const tensix_image_t tensix_{ident} = {{",
        f"    tensix_{ident}_blob,",
        f"    tensix_{ident}_boot, {len(boot)},",
        f"    tensix_{ident}_launch, {len(launch)},",
        f"    tensix_{ident}_rta, {len(rta)},",
        f"    0x{image.go_addr:x},",
        "};",
        "",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m framework.naiad.tensix_image")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter", help="keep one worker's messages")
    f.add_argument("trace")
    f.add_argument("--core", required=True)
    f.add_argument("-o", "--output", required=True)
    f.add_argument("--note", default="")
    h = sub.add_parser("header", help="write the C image header")
    h.add_argument("trace")
    h.add_argument("--name", required=True)
    h.add_argument("--arch", default="blackhole")
    h.add_argument("-o", "--output", required=True)
    i = sub.add_parser("info", help="summarise an image")
    i.add_argument("trace")
    i.add_argument("--arch", default="blackhole")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "filter":
            core = tuple(int(v) for v in args.core.split(","))
            filter_trace(args.trace, args.output, core, args.note)
        elif args.cmd == "header":
            image = from_trace(args.trace, args.arch)
            Path(args.output).write_text(c_header(image, args.name))
        else:
            print(from_trace(args.trace, args.arch).summary())
    except TensixImageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
