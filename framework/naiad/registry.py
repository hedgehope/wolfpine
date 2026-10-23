"""Name -> class registries for Naiad designs.

``IP_REGISTRY`` is the only place a block type is bound to code: a design file
names a type, and the builder instantiates whatever is registered under it.

``NAIAD_ISA_REGISTRY`` is Naiad's own extension list, separate from the
Tenstorrent baby cores' ``ISA_EXTENSION_REGISTRY`` so neither can disturb the
other. ``"i"`` is the base ISA every core already has, so it maps to ``None``.
The classes here are the RV32 executors; an RV64 core brings its own (see
``builder.make_core``).
"""

from framework.naiad.ip.bootrom import BootROM
from framework.naiad.ip.cache import Cache
from framework.naiad.ip.clint import CLINT
from framework.naiad.ip.dma import DMAEngine
from framework.naiad.ip.htif import HTIF
from framework.naiad.ip.memories import DRAMBlock, ScratchpadBlock
from framework.naiad.ip.perfcounters import PerfCounters
from framework.pe.rv.isa.a_isa import RV_ZAAMO_ISA
from framework.pe.rv.isa.b_isa import RV_ZBA_ISA, RV_ZBB_ISA
from framework.pe.rv.isa.m_isa import RV_M_ISA
from framework.pe.rv.isa.zicsr_isa import RV_ZICSR_ISA

IP_REGISTRY = {
    "bootrom": BootROM,
    "dram": DRAMBlock,
    "scratchpad": ScratchpadBlock,
    "clint": CLINT,
    "htif": HTIF,
    "perfcounters": PerfCounters,
    "dma": DMAEngine,
    "cache": Cache,
}

NAIAD_ISA_REGISTRY = {
    "i": None,
    "m": RV_M_ISA,
    "zicsr": RV_ZICSR_ISA,
    "zba": RV_ZBA_ISA,
    "zbb": RV_ZBB_ISA,
    "zaamo": RV_ZAAMO_ISA,
}
