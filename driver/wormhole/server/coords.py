"""Moved to :mod:`framework.sim.wormhole.coords`; this alias keeps the old import path."""

import sys

from framework.sim.wormhole import coords

sys.modules[__name__] = coords
