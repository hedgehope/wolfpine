"""Moved to :mod:`framework.ip.tensix.image`; ``framework.naiad.tensix_image`` is an alias of it, kept for older imports."""

import sys

from framework.ip.tensix import image as _moved

if __name__ == "__main__":  # python -m framework.naiad.tensix_image still runs the CLI
    raise SystemExit(_moved.main())
else:
    sys.modules[__name__] = _moved
