"""Architecture profiles for Wolfpine.

An :class:`~framework.arch.profile.ArchProfile` carries every hardware constant
that differs between Tenstorrent architectures (Wormhole, Blackhole, ...). The
device/tile classes read these values from a profile rather than hardcoding
them, so a new architecture is (largely) a new profile plus per-arch strategies
rather than a fork. See ``docs/plans/blackhole-support.md``.
"""

from framework.arch.blackhole import BLACKHOLE_PROFILE
from framework.arch.profile import ArchProfile
from framework.arch.wormhole import WORMHOLE_PROFILE

__all__ = ["ArchProfile", "BLACKHOLE_PROFILE", "WORMHOLE_PROFILE"]
