"""``python -m driver.wormhole.server`` — moved to ``python -m framework.sim.wormhole``.

Kept so existing scripts and ``driver/sim_procs.sh`` keep working.
"""

import sys

from framework.sim.wormhole.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
