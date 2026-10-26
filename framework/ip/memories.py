"""Plain memories: ``dram`` (sparse, for large windows) and ``scratchpad`` (dense).

Both subclass the Wolfpine memory leaves directly rather than wrapping them, so
the RV core's instruction-fetch fast path (``resolve_plain_ram_span``) still
recognises them as plain RAM.
"""

from framework.ip.base import IPBlock
from framework.memory.memory import AddressableMemory, SparseAddressableMemory


class DRAMBlock(IPBlock, SparseAddressableMemory):
    """Main memory. Backed by chunks allocated on first write, so a 1 GiB window costs nothing up front."""

    def __init__(self, spec):
        IPBlock.__init__(self, spec)
        SparseAddressableMemory.__init__(self, spec.size)


class ScratchpadBlock(IPBlock, AddressableMemory):
    """A small, dense on-chip memory."""

    def __init__(self, spec):
        IPBlock.__init__(self, spec)
        AddressableMemory.__init__(self, spec.size)
