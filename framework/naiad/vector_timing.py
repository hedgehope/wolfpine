"""Moved to :mod:`framework.pe.rv.timing.vector`; ``framework.naiad.vector_timing`` is an alias of it, kept for older imports."""

import sys

from framework.pe.rv.timing import vector as _moved

sys.modules[__name__] = _moved
