"""Moved to :mod:`framework.ip.memories`; ``framework.naiad.ip.memories`` is an alias of it, kept for older imports."""

import sys

from framework.ip import memories as _moved

sys.modules[__name__] = _moved
