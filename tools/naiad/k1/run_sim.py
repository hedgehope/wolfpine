"""Run K1-pack kernel ELFs on Naiad or Spike and print their combined console output.

    python3 tools/naiad/k1/run_sim.py --sim naiad|spike [--design D] ELF... > naiad.out

The output has the same format as ``run_k1.sh`` on the board (see
``kernels/include/k1common.h``), so ``compare.py`` can diff any two.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DESIGN = REPO_ROOT / "framework" / "naiad" / "designs" / "naiad-rvv.yaml"


def run(sim, design, elf, max_cycles):
    from framework.naiad.difftest import run_naiad, run_spike, spike_isa

    if sim == "naiad":
        try:
            outcome = run_naiad(design, elf, max_cycles=max_cycles).outcome
        except Exception as e:  # noqa: BLE001 - report and go on to the next kernel
            return (
                f"# FAILED {Path(elf).stem}: naiad: {' '.join(str(e).split())[:300]}\n",
                1,
            )
    else:
        from framework.naiad.design import load_design

        outcome = run_spike(elf, spike_isa(load_design(design).cores.march))
    text = outcome.console.decode(errors="replace")
    if outcome.exit_code != 0:
        text += f"# FAILED {Path(elf).stem} exit {outcome.exit_code} ({outcome.detail.splitlines()[0][:200]})\n"
        return text, 1
    return text, 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", choices=("naiad", "spike"), required=True)
    parser.add_argument("--design", default=str(DEFAULT_DESIGN))
    parser.add_argument("--max-cycles", type=lambda s: int(s, 0), default=200_000_000)
    parser.add_argument("elfs", nargs="+")
    args = parser.parse_args(argv)
    status = 0
    print(f"# k1-pack run on {args.sim} design {Path(args.design).name}")
    for elf in args.elfs:
        text, bad = run(args.sim, args.design, elf, args.max_cycles)
        sys.stdout.write(text)
        sys.stdout.flush()
        status |= bad
    return status


if __name__ == "__main__":
    sys.exit(main())
