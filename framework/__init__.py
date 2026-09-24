"""Wolfpine — a framework for simulating AI accelerators.

The core here is architecture-agnostic: the memory map, the interconnect, the
RISC-V cores, the clock and reset fabric, and the cost and trace machinery are
all parameterised by an :mod:`framework.arch` profile. A target is a profile
plus a thin device subclass, not a fork.

The backends that ship today are the Tenstorrent architectures, Wormhole and
Blackhole, in :mod:`framework.device`.
"""
