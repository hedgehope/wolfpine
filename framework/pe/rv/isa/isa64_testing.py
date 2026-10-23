"""Shared harness for the RV64 ISA unit tests: encoders and a one-shot core.

Not a test module itself. Each test builds a real ``RV64I`` over a small DRAM,
places the instruction words at :data:`PC`, seeds registers, and clocks the
core once per instruction, so decode, execute and the PC update are all
exercised together.
"""

from framework.device.device import DeviceMemory
from framework.memory.memory import DRAM
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.pe.rv.isa.a64_isa import RV64_ZAAMO_ISA, RV64_ZALRSC_ISA
from framework.pe.rv.isa.b64_isa import RV64_ZBA_ISA, RV64_ZBB_ISA
from framework.pe.rv.isa.m64_isa import RV64_M_ISA
from framework.pe.rv.isa.zicsr64_isa import RV64_ZICSR_ISA
from framework.pe.rv.rv64 import RV64I

M64 = 0xFFFFFFFFFFFFFFFF
PC = 0x1000
DATA = 0x2000
MEM_SIZE = 0x4000
RD, RS1, RS2 = 1, 15, 20

ALL = [
    RV64_M_ISA,
    RV64_ZBA_ISA,
    RV64_ZBB_ISA,
    RV64_ZAAMO_ISA,
    RV64_ZALRSC_ISA,
    RV64_ZICSR_ISA,
]


def r_type(opcode, funct3, funct7, rd=RD, rs1=RS1, rs2=RS2):
    return (
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    )


def i_type(opcode, funct3, imm, rd=RD, rs1=RS1):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def s_type(funct3, imm, rs1=RS1, rs2=RS2):
    imm &= 0xFFF
    return (
        ((imm >> 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm & 0x1F) << 7)
        | 0x23
    )


def b_type(funct3, imm, rs1=RS1, rs2=RS2):
    return (
        (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 1) << 7)
        | 0x63
    )


def u_type(opcode, imm20, rd=RD):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def j_type(imm, rd=RD):
    return (
        (((imm >> 20) & 1) << 31)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 1) << 20)
        | (((imm >> 12) & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
    )


def amo(funct5, funct3, rd=RD, rs1=RS1, rs2=RS2):
    return r_type(0x2F, funct3, funct5 << 2, rd, rs1, rs2)


def csr(funct3, addr, rd=RD, rs1=RS1):
    return i_type(0x73, funct3, addr, rd, rs1)


def make_core(extensions=None, **kwargs):
    """An ``RV64I`` over one DRAM at 0, started at :data:`PC`."""
    mem_map = MemoryMap()
    dram = DRAM(MEM_SIZE)
    mem_map[AddressRange(0, MEM_SIZE)] = dram
    memory = DeviceMemory(mem_map)
    core = RV64I(
        PC,
        [memory],
        list(ALL if extensions is None else extensions),
        unknown_instr_is_error=True,
        **kwargs,
    )
    core.start()
    return core, memory


def set_reg(core, idx, value):
    core.register_file[idx].write((value & M64).to_bytes(8, "little"))


def reg(core, idx):
    return core.register_file[idx].read_uint()


def run(instrs, regs=None, extensions=None, mem=None, core=None, **kwargs):
    """Execute ``instrs`` (a word or a list) at the PC; return the core.

    A fresh core starts at :data:`PC`; passing ``core`` continues one. ``regs``
    maps register index -> value; ``mem`` maps address -> bytes.
    """
    if isinstance(instrs, int):
        instrs = [instrs]
    if core is None:
        core, _ = make_core(extensions, **kwargs)
    memory = core.visible_memory
    base = reg(core, "pc")
    for i, word in enumerate(instrs):
        memory.write(base + 4 * i, word.to_bytes(4, "little"))
    for addr, data in (mem or {}).items():
        memory.write(addr, data)
    for idx, value in (regs or {}).items():
        set_reg(core, idx, value)
    for cycle in range(len(instrs)):
        core.clock_tick(cycle)
    return core


def run1(instr, **regs_by_name):
    """Run one instruction with rs1/rs2 given by keyword; return rd's value."""
    regs = {}
    if "a" in regs_by_name:
        regs[RS1] = regs_by_name["a"]
    if "b" in regs_by_name:
        regs[RS2] = regs_by_name["b"]
    return reg(run(instr, regs), RD)
