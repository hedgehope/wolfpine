"""Shared harness for the RVV unit tests: encoders and a vector core over a small DRAM.

Not a test module. :class:`VCore` builds a real ``RV64I`` with F, D, Zicsr and
the vector unit, and :meth:`VCore.exec` places one word at the PC and clocks the
core, so decode, dispatch and execution are exercised together; an unclaimed
or illegal word raises (the core is built with ``unknown_instr_is_error``).
"""

import numpy as np

from framework.device.device import DeviceMemory
from framework.memory.memory import DRAM
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.pe.rv.isa.fd_isa import RV64_D_ISA, RV64_F_ISA, read_f, write_f
from framework.pe.rv.isa.v import RV64_V_ISA, attach
from framework.pe.rv.isa.v.state import FLOAT, SINT, UINT
from framework.pe.rv.isa.zicsr64_isa import RV64_ZICSR_ISA
from framework.pe.rv.rv32 import FCSR_INDEX
from framework.pe.rv.rv64 import RV64I

M64 = (1 << 64) - 1
PC = 0x1000
DATA = 0x10000
MEM_SIZE = 0x40000

# funct3 of OP-V
OPIVV, OPFVV, OPMVV, OPIVI, OPIVX, OPFVF, OPMVX, OPCFG = range(8)
_LMUL_CODE = {1: 0, 2: 1, 4: 2, 8: 3, 0.125: 5, 0.25: 6, 0.5: 7}
_SEW_CODE = {8: 0, 16: 1, 32: 2, 64: 3}
_WIDTH = {8: 0, 16: 5, 32: 6, 64: 7}


def vtype(sew, lmul=1, ta=0, ma=0):
    return (ma << 7) | (ta << 6) | (_SEW_CODE[sew] << 3) | _LMUL_CODE[lmul]


def opv(funct3, funct6, vd, vs2, src, vm=1):
    return (
        (funct6 << 26)
        | (vm << 25)
        | (vs2 << 20)
        | (src << 15)
        | (funct3 << 12)
        | (vd << 7)
        | 0x57
    )


def vsetvli(rd, rs1, vt):
    return (vt << 20) | (rs1 << 15) | (OPCFG << 12) | (rd << 7) | 0x57


def vsetivli(rd, uimm, vt):
    return (0x3 << 30) | (vt << 20) | (uimm << 15) | (OPCFG << 12) | (rd << 7) | 0x57


def vsetvl(rd, rs1, rs2):
    return (0x40 << 25) | (rs2 << 20) | (rs1 << 15) | (OPCFG << 12) | (rd << 7) | 0x57


def vmem(load, eew, vd, rs1, mop=0, rs2=0, vm=1, nf=1, mew=0):
    """A vector load/store: mop 0 unit (rs2 = lumop/sumop), 1 indexed-unordered, 2 strided, 3 indexed-ordered."""
    return (
        ((nf - 1) << 29)
        | (mew << 28)
        | (mop << 26)
        | (vm << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (_WIDTH[eew] << 12)
        | (vd << 7)
        | (0x07 if load else 0x27)
    )


class VCore:
    """An RV64 F/D/V core with helpers to configure vtype and poke registers and memory."""

    def __init__(
        self, vlen=256, elen=64, extensions=("v", "zvfh", "zvfbfmin", "zvfbfwma")
    ):
        mem_map = MemoryMap()
        mem_map[AddressRange(0, MEM_SIZE)] = DRAM(MEM_SIZE)
        self.memory = DeviceMemory(mem_map)
        self.core = RV64I(
            PC,
            [self.memory],
            [RV64_F_ISA, RV64_D_ISA, RV64_ZICSR_ISA, RV64_V_ISA],
            unknown_instr_is_error=True,
        )
        self.core.start()
        self.vs = attach(self.core, vlen=vlen, elen=elen, extensions=extensions)
        self.rf = self.core.register_file
        self.cycle = 0

    # -- execution ------------------------------------------------------------------

    def exec(self, word):
        self.rf["pc"].write(PC.to_bytes(8, "little"))
        self.memory.write(PC, word.to_bytes(4, "little"))
        self.core.clock_tick(self.cycle)
        self.cycle += 1

    def legal(self, word):
        """Execute ``word``; True if it was claimed, False if the core rejected it."""
        try:
            self.exec(word)
        except Exception as e:
            if str(e) == "Unknown instruction":
                return False
            raise
        return True

    def vset(self, sew, lmul=1, vl=None, ta=0, ma=0):
        """vsetvli with AVL = vl (VLMAX when None); returns the granted vl."""
        if vl is None:
            self.exec(vsetvli(5, 0, vtype(sew, lmul, ta, ma)))
        else:
            self.x(6, vl)
            self.exec(vsetvli(5, 6, vtype(sew, lmul, ta, ma)))
        return self.x(5)

    # -- registers ----------------------------------------------------------------------

    def x(self, idx, value=None):
        if value is None:
            return self.rf[idx].read_uint()
        self.rf[idx].write((value & M64).to_bytes(8, "little"))

    def f(self, idx, value=None):
        if value is None:
            return read_f(self.rf, idx)
        write_f(self.rf, idx, value)

    def fflags(self):
        return self.rf[FCSR_INDEX].read_uint() & 0x1F

    def clear_fflags(self):
        reg = self.rf[FCSR_INDEX]
        reg.write((reg.read_uint() & ~0x1F).to_bytes(4, "little"))

    def set_frm(self, rm):
        reg = self.rf[FCSR_INDEX]
        reg.write(((reg.read_uint() & 0x1F) | (rm << 5)).to_bytes(4, "little"))

    def v(self, base, dtype, nregs=1):
        """A view of ``nregs`` registers from ``base`` as ``dtype``."""
        return self.vs.vreg_bytes(base, nregs).view(dtype)

    def vset_data(self, base, values, dtype, nregs=1):
        view = self.v(base, dtype, nregs)
        view[: len(values)] = values

    def set_mask(self, bits, reg=0):
        self.vs.vrf[reg] = 0
        self.vs.set_mask(reg, np.asarray(bits, dtype=bool))

    def mem_write(self, addr, data):
        self.memory.write(addr, bytes(np.ascontiguousarray(data).view(np.uint8)))

    def mem_read(self, addr, n, dtype=np.uint8):
        return np.frombuffer(
            bytes(self.memory.read(addr, n * np.dtype(dtype).itemsize)), dtype=dtype
        )


__all__ = [
    "DATA",
    "FLOAT",
    "M64",
    "OPCFG",
    "OPFVF",
    "OPFVV",
    "OPIVI",
    "OPIVV",
    "OPIVX",
    "OPMVV",
    "OPMVX",
    "SINT",
    "UINT",
    "VCore",
    "opv",
    "vmem",
    "vsetivli",
    "vsetvl",
    "vsetvli",
    "vtype",
]
