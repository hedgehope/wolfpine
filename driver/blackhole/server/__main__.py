"""``python -m driver.blackhole.server`` — moved to ``python -m framework.sim.blackhole``.

Kept so existing scripts and ``driver/sim_procs.sh`` keep working.
"""

import sys

from framework.sim.blackhole.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
