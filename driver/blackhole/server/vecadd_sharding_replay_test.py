"""Socket-free replay of upstream ``vecadd_sharding`` on Blackhole.

tt-metal's ``programming_examples/vecadd_sharding`` is the only workload in
reach that drives **L1-sharded buffers with no data-movement kernel at all**
(``docs/upstream-examples-status.md``): the host allocates three sharded L1
``MeshBuffer``s, binds circular buffers ``c_0`` / ``c_1`` / ``c_2`` directly to
them with ``set_globally_allocated_address``, writes the two input shards
straight into each worker's L1 with ``EnqueueWriteMeshBuffer``, and runs a
**compute-only** program (``add_tiles``) over CBs that were never filled by a
reader. Nothing else in the tree has that shape — every other compute example
reaches L1 through a BRISC/NCRISC dataflow kernel.

Height-sharded, the default: 4x4 tiles over four workers in the x direction,
one row of 4 tiles each. Logical ``(0..3, 0)`` -> Blackhole physical
``(1..4, 2)``. Each worker therefore holds 4 bfloat16 tiles = 4096 elements per
buffer, at fixed L1 addresses:

===========  ==========  ===================================================
buffer       L1 address  role
===========  ==========  ===================================================
``a``        0x17e000    first input shard, written by the host
``b``        0x17c000    second input shard, written by the host
``c``        0x17a000    output shard, read back by the host
===========  ==========  ===================================================

**The golden is computed, not frozen.** The inputs are in the trace (the host
writes them over the wire), so the guard reads ``a`` and ``b`` back out of each
worker's L1 and requires ``c`` to be their sum — bit for bit, all 16384
elements over the four workers — as the FPU's ELWADD forms it into a 16-bit Dst
(``fp32_dest_acc_en`` is off): the smaller operand's 11-bit mantissa is aligned
to the larger's exponent with a round-half-up, the two are added, and the sum is
rounded half-up into bfloat16. That is *two* roundings, so it is neither a
truncation nor a single round of the exact sum (0.1533203125 + 8.125 gives
8.3125, not 8.25). :func:`_elwadd_bf16` restates the rule independently of
``framework/pe/tensix/backends/matrix.py``; ``optests/diff.sh elwmul`` pins the
same op bit-exact against ttsim. A rounding change would show up here and should
be re-validated against ttsim rather than papered over. (The live host checks the
same thing to a slack 0.2 absolute tolerance and prints "All results match
expected values within tolerance".)

All four workers are launched, so pump whichever tile's go-message is still GO
until it reports DONE.

Run:  python3 -m driver.blackhole.server.vecadd_sharding_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from framework.bridge import DramCore, Fabric, TensixCore, Transport
from framework.bridge import protocol as proto
from framework.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "vecadd_sharding.trace"
# Upstream logical (0..3, 0) -> Blackhole physical (1..4, 2).
TENSIX_POOL = [(1, 2), (2, 2), (3, 2), (4, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

A_ADDR = 0x17E000
B_ADDR = 0x17C000
C_ADDR = 0x17A000
TILES_PER_CORE = 4
SHARD_ELEMS = TILES_PER_CORE * 32 * 32
SHARD_BYTES = 2 * SHARD_ELEMS


def _build_fabric():
    device = make_device()
    fabric = Fabric()
    for translated, tile in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, tile))
    for physical in TENSIX_POOL:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
    return device, fabric


def _go_signal(device, core):
    return device.tt_device.read(TENSIX_COORD_MAP[core], GO_MSG_ADDR, 4)[3]


def _bf16(raw):
    """Widen a little-endian bfloat16 buffer to float32."""
    widened = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return widened.view(np.float32)


def _elwadd_bf16(a, b):
    """``a + b`` as ELWADD forms it into a 16-bit Dst (see the module docstring).

    Exact in float64 throughout: the operands are bfloat16 and the aligned sum
    has 12 significant bits. Assumes normal, non-zero operands and a non-zero
    sum, which the host's U(0, 10) inputs guarantee (``main`` checks the zeros).
    """
    a, b = a.astype(np.float64), b.astype(np.float64)
    swap = np.abs(a) < np.abs(b)
    big, small = np.where(swap, b, a), np.where(swap, a, b)
    ulp11 = np.exp2(np.floor(np.log2(np.abs(big))) - 10)
    small = np.sign(small) * np.floor(np.abs(small) / ulp11 + 0.5) * ulp11
    total = big + small
    ulp_bf16 = np.exp2(np.floor(np.log2(np.abs(total))) - 7)
    rounded = np.sign(total) * np.floor(np.abs(total) / ulp_bf16 + 0.5) * ulp_bf16
    return rounded.astype(np.float32)


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric()
    transport = Transport(addr=None)

    n_msgs = 0
    with TRACE.open() as f:
        for line in f:
            parsed = parse_trace_line(line)
            if parsed is None:
                continue
            req = SimpleNamespace(
                cmd=parsed["cmd"],
                core=parsed["core"],
                address=parsed["address"],
                size=parsed["size"],
                data=parsed["data"],
            )
            transport._handle(fabric, req)
            n_msgs += 1
            if (
                parsed["cmd"] == proto.CMD_READ
                and parsed["core"] in TENSIX_POOL
                and parsed["address"] == GO_MSG_ADDR
                and _go_signal(device, parsed["core"]) == RUN_MSG_GO
            ):
                pumped = 0
                while (
                    _go_signal(device, parsed["core"]) != RUN_MSG_DONE
                    and pumped < PUMP_CAP
                ):
                    device.tt_device.run(PUMP_CHUNK)
                    pumped += PUMP_CHUNK

    shards = {}
    for core in TENSIX_POOL:
        tile = TENSIX_COORD_MAP[core]
        shards[core] = tuple(
            bytes(device.tt_device.read(tile, addr, SHARD_BYTES))
            for addr in (A_ADDR, B_ADDR, C_ADDR)
        )
    device.tt_device.shutdown()

    for core, (a_raw, b_raw, c_raw) in shards.items():
        a, b, c = _bf16(a_raw), _bf16(b_raw), _bf16(c_raw)
        # A shard of zeros would make the comparison below vacuous: the host
        # seeds from a fixed 0x1234567 and draws from U(0, 10), so every input
        # element is non-zero.
        if not (a.all() and b.all()):
            raise AssertionError(
                f"replaying {TRACE.name}: worker {core}'s input shard is not "
                f"fully populated ({int((a == 0).sum())} zeros in a, "
                f"{int((b == 0).sum())} in b) — the host's L1 writes did not land"
            )
        expected = _elwadd_bf16(a, b)
        wrong = np.flatnonzero(c != expected)
        if wrong.size:
            i = int(wrong[0])
            raise AssertionError(
                f"{wrong.size}/{SHARD_ELEMS} elements wrong replaying "
                f"{TRACE.name} on worker {core} (L1-sharded add_tiles); "
                f"first: c[{i}]={float(c[i])} expected {float(expected[i])} "
                f"(= {float(a[i])} + {float(b[i])})"
            )

    print(
        f"blackhole vecadd_sharding_replay test OK ({n_msgs} messages; all "
        f"{len(TENSIX_POOL) * SHARD_ELEMS} bf16 elements across "
        f"{len(TENSIX_POOL)} workers == a + b, from L1-sharded buffers driven "
        "by a compute-only kernel)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
