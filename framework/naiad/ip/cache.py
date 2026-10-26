"""Moved to :mod:`framework.ip.cache`; ``framework.naiad.ip.cache`` is an alias of it, kept for older imports."""

import sys

from framework.ip import cache as _moved

sys.modules[__name__] = _moved
