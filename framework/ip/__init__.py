"""Reusable memory-mapped IP blocks: memories, boot ROM, CLINT, HTIF, DMA, caches, perf counters, a Tensix tile.

Each block is an :class:`~framework.ip.base.IPBlock`; a SoC target (the Naiad
reference SoC, :mod:`framework.naiad`) registers them by type and maps them
from its design file.
"""
