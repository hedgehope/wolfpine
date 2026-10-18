"""Simulator devices for tt-metal's UMD simulation backend, one directory per arch.

Each ``framework/sim/<arch>/`` is a complete ``TT_METAL_SIMULATOR`` target
(``run.sh`` plus the SoC and cluster descriptors), shipped inside the package
so a ``pip install`` has one to point at. ``simulator_dir`` locates it.
"""

import pathlib

ARCHES = ("wormhole", "blackhole")


def simulator_dir(arch):
    """The directory to export as ``TT_METAL_SIMULATOR`` for ``arch``."""
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {', '.join(ARCHES)}")
    return pathlib.Path(__file__).resolve().parent / arch
