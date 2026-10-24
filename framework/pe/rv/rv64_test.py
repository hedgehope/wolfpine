"""RV64 core tests: construction, the extension registry, the RV32 core's widths
being unchanged, and an end-to-end rv64im C program built with clang."""

import math
import subprocess
from pathlib import Path

import pytest

from framework.device.clock import Clock
from framework.device.device import Device, DeviceMemory
from framework.device.reset import Reset
from framework.memory.memory import DRAM
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.pe.rv.isa.a64_isa import RV64_ZAAMO_ISA, RV64_ZALRSC_ISA
from framework.pe.rv.isa.b64_isa import RV64_ZBA_ISA, RV64_ZBB_ISA
from framework.pe.rv.isa.i64_isa import RV64_I_ISA
from framework.pe.rv.isa.m64_isa import RV64_M_ISA
from framework.pe.rv.isa.zicsr64_isa import RV64_ZICSR_ISA
from framework.pe.rv.rv32 import RV32IM
from framework.pe.rv.rv64 import RV64_ISA_REGISTRY, RV64I, RV64IM, rv64_extensions

LLVM = Path("/opt/llvm/20.1.7/bin")
CLANG = LLVM / "clang"
OBJCOPY = LLVM / "llvm-objcopy"


def _memory(size=0x10000):
    mem_map = MemoryMap()
    mem_map[AddressRange(0, size)] = DRAM(size)
    return DeviceMemory(mem_map)


def test_rv64_extensions_resolves_names_in_order_without_duplicates():
    assert rv64_extensions(["i", "m", "A", "zba", "zbb", "zicsr", "zicntr"]) == [
        RV64_M_ISA,
        RV64_ZAAMO_ISA,
        RV64_ZALRSC_ISA,
        RV64_ZBA_ISA,
        RV64_ZBB_ISA,
        RV64_ZICSR_ISA,
    ]
    assert rv64_extensions(["zalrsc", "a"]) == [RV64_ZALRSC_ISA, RV64_ZAAMO_ISA]
    with pytest.raises(KeyError, match="unknown RV64 extension 'q'"):
        rv64_extensions(["q"])
    assert set(RV64_ISA_REGISTRY) >= {
        "m",
        "a",
        "f",
        "d",
        "zfbfmin",
        "zba",
        "zbb",
        "zicsr",
        "zicntr",
    }


def test_rv64i_construction():
    core = RV64I(0x80000000, [_memory()], rv64_extensions(["m", "zicsr"]), core_id=2)
    assert core.isas == [RV64_I_ISA, RV64_M_ISA, RV64_ZICSR_ISA]
    assert core.csrs is not None
    assert core.register_file.csrs is core.csrs
    assert core.csrs.hart_id == 2
    assert len(core.register_file["pc"].read()) == 8
    assert len(core.register_file.registers) == 34
    core.start()
    assert core.register_file["pc"].read_uint() == 0x80000000

    bare = RV64IM(0, [_memory()])
    assert bare.isas == [RV64_I_ISA, RV64_M_ISA]
    assert bare.csrs is None


def test_rv64_fp_registers_are_flen_64():
    core = RV64I(0, [_memory()], fp_registers=True)
    assert len(core.register_file["f31"].read()) == 8
    assert len(core.register_file["fcsr"].read()) == 4


def test_rv32_core_is_unchanged():
    core = RV32IM(0, [_memory()], fp_registers=True)
    assert all(len(r.read()) == 4 for r in core.register_file.registers)
    assert core.csrs is None


def test_pc_above_4g():
    mem_map = MemoryMap()
    mem_map[AddressRange(0x1_0000_0000, 0x1000)] = DRAM(0x1000)
    memory = DeviceMemory(mem_map)
    memory.write(0x1_0000_0000, (0x00100093).to_bytes(4, "little"))  # addi ra, x0, 1
    core = RV64I(0x1_0000_0000, [memory], unknown_instr_is_error=True)
    core.start()
    core.clock_tick(0)
    assert core.register_file[1].read_uint() == 1
    assert core.register_file["pc"].read_uint() == 0x1_0000_0004


# -- end to end ------------------------------------------------------------

_PROGRAM = r"""
typedef unsigned long u64;
typedef long i64;
typedef unsigned int u32;
typedef int i32;

#define OUT ((volatile u64 *)0x8000)
#define DONE 0xC0FFEE

__attribute__((naked, section(".text.start"))) void _start(void) {
    __asm__ volatile("li sp, 0xF000\n call main\n 1: j 1b\n");
}

volatile i64 in[6] = {-7, 3, 0x123456789abcdefL, -1, 0x7fffffff, 0};
volatile i32 arr[8] = {5, -3, 100, 0, -2147483647, 42, 7, -1};

__attribute__((noinline)) u64 isqrt(u64 n) {
    u64 r = 0, bit = 1UL << 62;
    while (bit > n) bit >>= 2;
    while (bit) {
        if (n >= r + bit) { n -= r + bit; r = (r >> 1) + bit; }
        else r >>= 1;
        bit >>= 2;
    }
    return r;
}

__attribute__((noinline)) u64 fib(u32 n) {
    u64 a = 0, b = 1;
    for (u32 i = 0; i < n; i++) { u64 t = a + b; a = b; b = t; }
    return a;
}

int main(void) {
    i64 a = in[0], b = in[1], c = in[2], m1 = in[3];
    i32 w = (i32)in[4];
    OUT[0] = a / b;
    OUT[1] = a % b;
    OUT[2] = (u64)c * (u64)c;
    OUT[3] = (u64)(((unsigned __int128)(u64)c * (u64)c) >> 64);
    OUT[4] = (i64)(w + 1);
    OUT[5] = (u64)(u32)(w + 1);
    OUT[6] = (u64)(-c >> 13);
    OUT[7] = (u64)c >> 52;
    OUT[8] = fib(90);
    OUT[9] = (u64)m1 / 3;
    OUT[10] = (i64)((i32)a / (i32)b);
    OUT[11] = isqrt((u64)c);
    i32 s[8];
    for (int i = 0; i < 8; i++) s[i] = arr[i];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j + 1 < 8 - i; j++)
            if (s[j] > s[j + 1]) { i32 t = s[j]; s[j] = s[j + 1]; s[j + 1] = t; }
    i64 acc = 0;
    for (int i = 0; i < 8; i++) acc = acc * 31 + s[i];
    OUT[12] = (u64)acc;
    u64 cyc, ret;
    __asm__ volatile("csrr %0, cycle" : "=r"(cyc));
    __asm__ volatile("csrr %0, instret" : "=r"(ret));
    OUT[13] = cyc;
    OUT[14] = ret;
    OUT[15] = DONE;
    return 0;
}
"""


# _start first, at 0, so the raw image's first word is the entry point.
_LINKER_SCRIPT = """
ENTRY(_start)
SECTIONS {
    . = 0;
    .text : { *(.text.start) *(.text*) }
    .rodata : { *(.rodata*) *(.srodata*) }
    .data : { *(.data*) *(.sdata*) }
    .bss : { *(.bss*) *(.sbss*) }
}
"""


def _expected():
    m64 = (1 << 64) - 1
    c = 0x123456789ABCDEF

    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, (a + b) & m64
        return a

    acc = 0
    for v in sorted([5, -3, 100, 0, -2147483647, 42, 7, -1]):
        acc = acc * 31 + v
    return {
        0: -2 & m64,
        1: -1 & m64,
        2: (c * c) & m64,
        3: (c * c) >> 64,
        4: -(1 << 31) & m64,
        5: 1 << 31,
        6: (-c >> 13) & m64,
        7: c >> 52,
        8: fib(90),
        9: m64 // 3,
        10: -2 & m64,
        11: math.isqrt(c),
        12: acc & m64,
        15: 0xC0FFEE,
    }


@pytest.mark.skipif(
    not (CLANG.exists() and OBJCOPY.exists()), reason="clang 20.1.7 not installed"
)
def test_end_to_end_rv64im_program(tmp_path):
    src = tmp_path / "prog.c"
    src.write_text(_PROGRAM)
    script = tmp_path / "link.ld"
    script.write_text(_LINKER_SCRIPT)
    elf = tmp_path / "prog.elf"
    binary = tmp_path / "prog.bin"
    subprocess.run(
        [
            str(CLANG),
            "--target=riscv64-unknown-elf",
            "-march=rv64im_zicsr",
            "-mabi=lp64",
            "-mno-relax",
            "-O2",
            "-ffreestanding",
            "-nostdlib",
            "-fuse-ld=lld",
            f"-Wl,-T,{script}",
            "-o",
            str(elf),
            str(src),
        ],
        check=True,
    )
    subprocess.run([str(OBJCOPY), "-O", "binary", str(elf), str(binary)], check=True)
    image = binary.read_bytes()
    assert len(image) < 0x8000

    memory = _memory()
    memory.write(0, image)
    core = RV64IM(0, [memory], rv64_extensions(["zicsr"]), unknown_instr_is_error=True)
    device = Device(memory, [Clock([core])], [Reset([core])])
    device.reset()

    def out(i):
        return int.from_bytes(memory.read(0x8000 + 8 * i, 8), "little")

    for _ in range(100):
        device.run(1000)
        if out(15) == 0xC0FFEE:
            break
    assert out(15) == 0xC0FFEE, "program did not finish"
    for i, value in _expected().items():
        assert out(i) == value, f"OUT[{i}] = {hex(out(i))}, expected {hex(value)}"
    # One instruction retires per tick, and ``cycle`` is read one instruction
    # before ``instret``: tick N reads cycle = N, tick N+1 reads instret = N.
    assert out(14) > 100
    assert out(13) == out(14)
    assert core.unknown_instructions == 0
