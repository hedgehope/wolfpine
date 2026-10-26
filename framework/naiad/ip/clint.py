"""Moved to :mod:`framework.ip.clint`; ``framework.naiad.ip.clint`` is an alias of it, kept for older imports."""

import sys

from framework.ip import clint as _moved

sys.modules[__name__] = _moved
