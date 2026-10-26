"""The boot ROM: a three-instruction trampoline to the loaded ELF's entry point.

Spike's reset vector, minus the DTB and ``mhartid`` set-up (Naiad has neither a
device tree nor, yet, more than one hart)::

    0x00  auipc t0, 0
    0x04  lw/ld t0, 16(t0)     # lw on RV32, ld on RV64
    0x08  jr    t0
    0x0c  nop
    0x10  .word/.dword entry   # patched in by on_load

Read-only to the cores: a store into the ROM is a program bug and raises.
"""

from framework.ip.base import IPBlock

_AUIPC_T0 = 0x00000297
_LW_T0_16_T0 = 0x0102A283
_LD_T0_16_T0 = 0x0102B283
_JR_T0 = 0x00028067
_NOP = 0x00000013
_ENTRY_OFFSET = 16
_IMAGE_SIZE = 24


class BootROM(IPBlock):
    PARAMS = frozenset()

    @classmethod
    def block_size(cls, params, window):
        return _IMAGE_SIZE

    def __init__(self, spec):
        super().__init__(spec)
        self.xlen = 32
        self.entry = 0
        self.image = self._assemble()

    def attach(self, soc):
        super().attach(soc)
        self.xlen = soc.design.cores.xlen
        self.image = self._assemble()

    def on_load(self, image):
        self.entry = image.entry
        self.image = self._assemble()

    def _assemble(self):
        load = _LD_T0_16_T0 if self.xlen == 64 else _LW_T0_16_T0
        words = [_AUIPC_T0, load, _JR_T0, _NOP]
        code = b"".join(w.to_bytes(4, "little") for w in words)
        return code + self.entry.to_bytes(8, "little")

    def read(self, addr, size):
        if addr + size > len(self.image):
            raise IndexError(
                f"{self.name}: read of {size} bytes at offset {hex(addr)} is past the ROM"
            )
        return self.image[addr : addr + size]

    def write(self, addr, value, size=None):
        raise IndexError(
            f"{self.name}: the boot ROM is read-only (write at offset {hex(addr)})"
        )

    def header_defines(self):
        return [("ENTRY_OFFSET", _ENTRY_OFFSET)]
