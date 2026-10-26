"""A Tenstorrent Tensix tile as a memory-mapped accelerator block.

``block`` is the MMIO block (:class:`~framework.ip.tensix.block.TensixBlock`);
``image`` turns a captured tt-metal wire trace into the launch image the block
replays; ``images/`` holds the captured traces.
"""

from pathlib import Path

IMAGES_DIR = Path(__file__).parent / "images"
