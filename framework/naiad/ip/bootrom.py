"""Moved to :mod:`framework.ip.bootrom`; ``framework.naiad.ip.bootrom`` is an alias of it, kept for older imports."""

import sys

from framework.ip import bootrom as _moved

sys.modules[__name__] = _moved
