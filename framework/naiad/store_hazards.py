"""Moved to :mod:`framework.pe.rv.timing.store_hazards`; this old path is an alias of it, kept for older imports."""

import sys

from framework.pe.rv.timing import store_hazards as _moved

sys.modules[__name__] = _moved
