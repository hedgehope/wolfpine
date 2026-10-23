"""Naiad: a standalone RISC-V reference SoC assembled from a YAML design file.

``design`` parses and validates a design, ``builder`` turns it into a Wolfpine
``Device``, ``gen`` writes the matching ``naiad.h`` / ``linker.ld`` / ``crt0.S``,
and ``run`` (``python -m framework.naiad``) loads an ELF and runs it to its HTIF
exit. Shipped designs live in ``designs/``.
"""

from pathlib import Path

DESIGNS_DIR = Path(__file__).parent / "designs"
PLATFORM_DIR = Path(__file__).parent / "platform"
