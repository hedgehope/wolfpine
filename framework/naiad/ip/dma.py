"""Moved to :mod:`framework.ip.dma`; ``framework.naiad.ip.dma`` is an alias of it, kept for older imports."""

import sys

from framework.ip import dma as _moved

sys.modules[__name__] = _moved
