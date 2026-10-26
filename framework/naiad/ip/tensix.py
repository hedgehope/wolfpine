"""Moved to :mod:`framework.ip.tensix.block`; ``framework.naiad.ip.tensix`` is an alias of it, kept for older imports."""

import sys

from framework.ip.tensix import block as _moved

sys.modules[__name__] = _moved
