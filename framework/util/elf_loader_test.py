import pytest

from framework.device.device import DeviceMemory
from framework.memory.memory import DRAM, SparseDRAM
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.util.elf_loader import ELFLoadError, load_elf
from framework.util.elf_testing import write_elf


def _memory():
    mem_map = MemoryMap()
    mem_map[AddressRange(0x1000, 0x1000)] = DRAM(0x1000)
    mem_map[AddressRange(0x80000000, 1 << 20)] = SparseDRAM(1 << 20)
    return DeviceMemory(mem_map)


@pytest.mark.parametrize("elfclass", [32, 64])
def test_segments_entry_and_symbols(tmp_path, elfclass):
    path = write_elf(
        tmp_path / "prog.elf",
        [(0x80000000, b"\x13\x00\x00\x00" * 4, 16), (0x1000, b"rom!", 4)],
        entry=0x80000004,
        symbols={"tohost": 0x80000100, "fromhost": 0x80000140},
        elfclass=elfclass,
    )
    memory = _memory()
    image = load_elf(path, memory)
    assert image.xlen == elfclass
    assert image.entry == 0x80000004
    assert image.symbol("tohost") == 0x80000100
    assert image.symbols["fromhost"] == 0x80000140
    assert [(s.paddr, s.filesz, s.memsz) for s in image.segments] == [
        (0x80000000, 16, 16),
        (0x1000, 4, 4),
    ]
    assert memory.read(0x80000000, 16) == b"\x13\x00\x00\x00" * 4
    assert memory.read(0x1000, 4) == b"rom!"
    with pytest.raises(KeyError, match="no symbol 'missing'"):
        image.symbol("missing")


def test_bss_is_zeroed(tmp_path):
    memory = _memory()
    memory.write(0x80000000, b"\xff" * 64)  # stale contents where .bss will land
    path = write_elf(
        tmp_path / "bss.elf", [(0x80000000, b"\xab" * 8, 64)], entry=0x80000000
    )
    load_elf(path, memory)
    assert memory.read(0x80000000, 8) == b"\xab" * 8
    assert memory.read(0x80000008, 56) == bytes(56)


def test_segment_outside_map_is_rejected_before_writing(tmp_path):
    memory = _memory()
    path = write_elf(
        tmp_path / "bad.elf",
        [(0x80000000, b"\x01" * 4, 4), (0x40000000, b"\x02" * 4, 4)],
        entry=0x80000000,
    )
    with pytest.raises(ELFLoadError, match=r"0x40000000"):
        load_elf(path, memory)
    assert memory.read(0x80000000, 4) == bytes(4)  # nothing was written


def test_segment_overrunning_its_range_is_rejected(tmp_path):
    # Starts inside the 4 KiB range at 0x1000 but its .bss runs past the end.
    path = write_elf(
        tmp_path / "overrun.elf", [(0x1F00, b"\x01" * 4, 0x200)], entry=0x1F00
    )
    with pytest.raises(ELFLoadError, match="not wholly inside"):
        load_elf(path, _memory())


def test_non_riscv_is_rejected(tmp_path):
    path = write_elf(
        tmp_path / "x86.elf", [(0x80000000, b"\x90", 1)], entry=0x80000000, machine=62
    )
    with pytest.raises(ELFLoadError, match="not a RISC-V ELF"):
        load_elf(path, _memory())
