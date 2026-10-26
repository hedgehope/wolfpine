"""Moved to :mod:`framework.ip.base`; ``framework.naiad.ip.base`` is an alias of it, kept for older imports."""

import sys

from framework.ip import base as _moved

sys.modules[__name__] = _moved
