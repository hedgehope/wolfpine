"""Moved to :mod:`framework.ip.htif`; ``framework.naiad.ip.htif`` is an alias of it, kept for older imports."""

import sys

from framework.ip import htif as _moved

sys.modules[__name__] = _moved
