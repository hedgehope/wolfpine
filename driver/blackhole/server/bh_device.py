"""Moved to :mod:`framework.sim.blackhole.bh_device`; this alias keeps the old import path."""

import sys

from framework.sim.blackhole import bh_device

sys.modules[__name__] = bh_device
