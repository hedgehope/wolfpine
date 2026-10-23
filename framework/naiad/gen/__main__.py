"""``python -m framework.naiad.gen <design.yaml> -o <dir>``: write a design's generated files."""

import argparse
import sys

from framework.naiad.design import DesignError, load_design
from framework.naiad.gen import generate


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m framework.naiad.gen", description=__doc__
    )
    parser.add_argument("design", help="Naiad design YAML")
    parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="directory to write naiad.h, linker.ld, crt0.S, naiad.mk into",
    )
    args = parser.parse_args(argv)
    try:
        design = load_design(args.design)
    except DesignError as e:
        print(f"naiad: {args.design}: {e}", file=sys.stderr)
        return 2
    for path in generate(design, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
