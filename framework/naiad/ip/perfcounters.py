"""Moved to :mod:`framework.ip.perfcounters`; ``framework.naiad.ip.perfcounters`` is an alias of it, kept for older imports."""

import sys

from framework.ip import perfcounters as _moved

sys.modules[__name__] = _moved
