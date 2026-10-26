"""ex5 host side: feed the AME GEMM its inputs from numpy and check C on the host.

The C program (ex5.c) reads its problem size and operands from NAIAD_IO globals.
``framework.naiad.kernel_io.run_kernel`` builds the design, loads the ELF, writes
each input over its ELF symbol, runs to the HTIF exit and reads the outputs back,
so the data is generated and checked here in numpy, while the program still
checks itself against its own scalar loop.

    make -C driver/naiad ex5
    python3 driver/naiad/ex5/ex5.py driver/naiad/build/ex5/ex5.elf [--seed S] [--shape MxNxK]

The program's console streams as it runs; then a host PASS / FAIL; exits non-zero on any mismatch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from framework.naiad import DESIGNS_DIR
from framework.naiad.kernel_io import run_kernel

DESIGN = DESIGNS_DIR / "naiad-ame-timed.yaml"
MAX = 64  # ex5.c's MAX: the arrays are MAX * MAX, filled row-major at the problem's own strides


def run(elf, shape=(48, 33, 60), seed=1):
    m, n, k = shape
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
    result = run_kernel(
        DESIGN,
        elf,
        inputs={"m": m, "n": n, "k": k, "A": a, "B": b},
        outputs={"C": (np.int32, (m, n))},
    )
    want = a.astype(np.int32) @ b.astype(np.int32)
    host_ok = np.array_equal(result.outputs["C"], want)
    return result, host_ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("elf", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shape", default="48x33x60", help="MxNxK, each at most 64")
    args = parser.parse_args(argv)
    shape = tuple(int(x) for x in args.shape.lower().split("x"))
    if len(shape) != 3 or not all(0 < x <= MAX for x in shape):
        parser.error(f"--shape must be MxNxK with each in 1..{MAX}")
    result, host_ok = run(args.elf, shape, args.seed)
    print(
        f"host: exit code {result.exit_code} after {result.cycles} cycles; C vs numpy {'ok' if host_ok else 'MISMATCH'}"
    )
    ok = result.ok and host_ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
