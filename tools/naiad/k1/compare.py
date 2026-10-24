"""Diff two K1-pack outputs (Naiad, Spike, QEMU or the K1 board).

    python3 compare.py REFERENCE.out OTHER.out [--rtol 1e-5] [--times]

Lines (written by ``kernels/include/k1common.h``):

* ``R <case> <i> <hex>``: must match bit for bit;
* ``A <case> <i> <hex> <f32|f64>``: an implementation-defined summation order
  (``vfredusum``), compared as floats with relative tolerance ``--rtol``
  (NaNs must agree on being NaN);
* ``T <case> <ticks> <unit> <reps>``: timings, never compared; ``--times``
  prints them side by side (per call);
* ``# ...``: comments; ``# FAILED ...`` lines are reported.

Per case it prints OK, or the number of differing elements and the first one.
A case present on one side only is reported too. This script uses only the
standard library, so it also runs on the board.

Exit status: 0 if everything agrees, 1 otherwise.
"""

import argparse
import math
import struct
import sys


def parse(path):
    exact, approx, times, failed = {}, {}, {}, []
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "#":
                if len(parts) > 1 and parts[1] == "FAILED":
                    failed.append(line.strip())
                continue
            if parts[0] == "R" and len(parts) == 4:
                exact.setdefault(parts[1], {})[int(parts[2])] = parts[3].lower()
            elif parts[0] == "A" and len(parts) == 5:
                approx.setdefault(parts[1], {})[int(parts[2])] = (
                    parts[3].lower(),
                    parts[4],
                )
            elif parts[0] == "T" and len(parts) == 5:
                times[parts[1]] = (int(parts[2]), parts[3], int(parts[4]))
    return exact, approx, times, failed


def as_float(hexbits, kind):
    if kind == "f32":
        return struct.unpack("<f", struct.pack("<I", int(hexbits, 16)))[0]
    return struct.unpack("<d", struct.pack("<Q", int(hexbits, 16)))[0]


def close(a, b, rtol):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-30)


def compare(ref, other, rtol=1e-5):
    """``(lines, n_bad)``: the report for two parsed outputs."""
    (re_, ra, _, rf), (oe, oa, _, of) = ref, other
    lines, bad = [], 0
    for msg in rf + of:
        lines.append(f"FAILED  {msg}")
        bad += 1
    for case in sorted(set(re_) | set(oe)):
        if case not in re_ or case not in oe:
            lines.append(
                f"MISSING {case}: only in {'reference' if case in re_ else 'other'}"
            )
            bad += 1
            continue
        a, b = re_[case], oe[case]
        diff = [i for i in sorted(set(a) | set(b)) if a.get(i) != b.get(i)]
        if diff:
            i = diff[0]
            lines.append(
                f"DIFF    {case}: {len(diff)}/{max(len(a), len(b))} differ; first [{i}] ref {a.get(i)} other {b.get(i)}"
            )
            bad += 1
        else:
            lines.append(f"OK      {case} ({len(a)} exact)")
    for case in sorted(set(ra) | set(oa)):
        if case not in ra or case not in oa:
            lines.append(
                f"MISSING {case}: only in {'reference' if case in ra else 'other'}"
            )
            bad += 1
            continue
        a, b = ra[case], oa[case]
        diff = []
        for i in sorted(set(a) | set(b)):
            if (
                i not in a
                or i not in b
                or not close(as_float(*a[i]), as_float(*b[i]), rtol)
            ):
                diff.append(i)
        exact = sum(1 for i in a if a[i] == b.get(i))
        if diff:
            i = diff[0]
            lines.append(
                f"DIFF    {case}: {len(diff)}/{len(a)} beyond rtol {rtol}; first [{i}] ref {a.get(i)} other {b.get(i)}"
            )
            bad += 1
        else:
            lines.append(
                f"OK      {case} ({len(a)} within rtol {rtol}, {exact} bit-exact)"
            )
    return lines, bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference")
    parser.add_argument("other")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--times", action="store_true", help="print the timings side by side"
    )
    args = parser.parse_args(argv)
    ref, other = parse(args.reference), parse(args.other)
    lines, bad = compare(ref, other, args.rtol)
    print("\n".join(lines))
    if args.times:
        print(f"\n{'case':28} {'reference / call':>22} {'other / call':>22}")
        for case in sorted(set(ref[2]) | set(other[2])):
            cells = []
            for side in (ref[2], other[2]):
                if case in side:
                    ticks, unit, reps = side[case]
                    cells.append(f"{ticks / reps:.1f} {unit}")
                else:
                    cells.append("-")
            print(f"{case:28} {cells[0]:>22} {cells[1]:>22}")
    n_cases = len(set(ref[0]) | set(other[0]) | set(ref[1]) | set(other[1]))
    print(f"\n{n_cases} cases: {'all agree' if not bad else f'{bad} problems'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
