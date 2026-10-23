"""``python -m experiments.naiad.all [--quick] [--jobs N] [sweep ...]``: run the talk's sweeps and redraw every figure.

With no sweep names it runs ``sweeps/talk_tier1.yaml``. Runs resume, so a
repeated call only simulates points that have no stored result and then
regenerates the CSVs and all figures.
"""

import argparse
import os
import sys
from pathlib import Path

from experiments.naiad.plot import plot_sweep
from experiments.naiad.run import resolve_sweep, run_sweep
from experiments.naiad.sweep import OUT

TALK_SWEEPS = ["talk_tier1"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m experiments.naiad.all", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument(
        "sweeps",
        nargs="*",
        default=TALK_SWEEPS,
        help=f"sweep names or files (default {TALK_SWEEPS})",
    )
    ap.add_argument(
        "--quick", action="store_true", help="the CI profile: smallest sizes"
    )
    ap.add_argument(
        "--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    ap.add_argument("--out", help="output root (default experiments/naiad/out)")
    ap.add_argument(
        "--force", action="store_true", help="rerun points that already have a result"
    )
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="rerun points whose stored result is not ok",
    )
    args = ap.parse_args(argv)
    for name in args.sweeps:
        sweep, _, _, _ = run_sweep(
            resolve_sweep(name),
            jobs=args.jobs,
            quick=args.quick,
            force=args.force,
            retry_failed=args.retry_failed,
            out_root=args.out,
        )
        plot_sweep(Path(args.out or OUT) / sweep.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
