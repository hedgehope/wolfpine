"""Moved to :mod:`framework.sim.wormhole.wh_device`; this alias keeps the old import path."""

import sys

from framework.sim.wormhole import wh_device

sys.modules[__name__] = wh_device
