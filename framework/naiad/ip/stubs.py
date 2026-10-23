"""Blocks the schema accepts but nothing implements yet (``dma``, ``cache``).

A design may place them so its memory map is final; any access raises, rather
than reading back zeros that a program could mistake for a working device.
"""

from framework.naiad.ip.base import IPBlock


class _StubBlock(IPBlock):
    PARAMS = frozenset({"size", "channels", "line_bytes", "ways", "sets"})

    def _refuse(self, addr):
        raise NotImplementedError(
            f"{self.name}: the '{self.spec.type}' block is a stub (access at offset {hex(addr)})"
        )

    def read(self, addr, size):
        self._refuse(addr)

    def write(self, addr, value, size=None):
        self._refuse(addr)


class DMAStub(_StubBlock):
    pass


class CacheStub(_StubBlock):
    pass
