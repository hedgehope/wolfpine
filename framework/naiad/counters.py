"""Moved to :mod:`framework.ip.counters`; ``framework.naiad.counters`` is an alias of it, kept for older imports."""

import sys

from framework.ip import counters as _moved

sys.modules[__name__] = _moved
