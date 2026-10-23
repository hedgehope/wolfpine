"""RV32M division unit tests: rounding towards zero, the remainder's sign, and
the spec's divide-by-zero / overflow results (no trap)."""

import pytest

from framework.device.device import DeviceMemory
from framework.memory.memory import DRAM
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.pe.rv.rv32 import RV32IM

M32 = 0xFFFFFFFF
MIN32 = 0x80000000
PC = 0x100
RD, RS1, RS2 = 1, 15, 20


def _run(funct3, a, b):
    mem_map = MemoryMap()
    mem_map[AddressRange(0, 0x1000)] = DRAM(0x1000)
    memory = DeviceMemory(mem_map)
    instr = (0x01 << 25) | (RS2 << 20) | (RS1 << 15) | (funct3 << 12) | (RD << 7) | 0x33
    memory.write(PC, instr.to_bytes(4, "little"))
    core = RV32IM(PC, [memory], unknown_instr_is_error=True)
    core.start()
    core.register_file[RS1].write((a & M32).to_bytes(4, "little"))
    core.register_file[RS2].write((b & M32).to_bytes(4, "little"))
    core.clock_tick(0)
    return core.register_file[RD].read_uint()


@pytest.mark.parametrize(
    "funct3,a,b,expected",
    [
        # div: rounds towards zero
        (4, -7, 2, -3),
        (4, 7, -2, -3),
        (4, -7, -2, 3),
        (4, 0x7FFFFFFF, 1, 0x7FFFFFFF),
        (4, 5, 0, -1),  # by zero -> -1
        (4, MIN32, -1, MIN32),  # overflow -> dividend
        # divu
        (5, M32, 2, 0x7FFFFFFF),
        (5, M32, M32, 1),
        (5, 5, 0, M32),  # by zero -> 2**32 - 1
        # rem: sign of the dividend
        (6, -1000, 7, -6),
        (6, 1000, -7, 6),
        (6, -1000, -7, -6),
        (6, -7, 0, -7),  # by zero -> dividend
        (6, MIN32, -1, 0),  # overflow -> 0
        # remu
        (7, M32, 10, M32 % 10),
        (7, 1234, 0, 1234),
    ],
)
def test_div_rem(funct3, a, b, expected):
    assert _run(funct3, a, b) == expected & M32
