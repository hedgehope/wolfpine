"""Moved to :mod:`framework.sim.blackhole.coords`; this alias keeps the old import path."""

import sys

from framework.sim.blackhole import coords

sys.modules[__name__] = coords
