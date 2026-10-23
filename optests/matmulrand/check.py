# ruff: noqa: INP001  (a standalone script beside the optest, run by path)
"""Check a ``matmulrand`` run's output against ``tile_mac``'s Tensix numerics (Naiad B6).

``python3 optests/matmulrand/check.py <log> [--arch blackhole|wormhole]`` reads the
operands (``OPDIFF_INPUT``), the compute config (``MATMULRAND_CONFIG``) and the
packed output tile (``OPDIFF_RESULT``) from a host log -- ttsim's or Wolfpine's,
as ``optests/diff.sh matmulrand`` leaves them in ``/tmp`` -- recomputes the
product with ``tile_mac(numerics="tensix_bh"/"tensix_wh")`` and compares every
element bit for bit. With a 32-bit Dst the packer narrows to the bf16 output CB
(round-half-up, NaN to Inf, denormals flushed), modelled by the in-tree
``DataFormatConversions.FP32InDstToBF16``; with a 16-bit Dst the tile is Dst.
"""

import argparse
import re
import sys

import numpy as np

from framework.matrix.tensix_numerics import tensix_mac_bits


def detilize(words):
    """A 32x32 tile in tt-metal's four-16x16-face layout -> row-major."""
    faces = np.asarray(words).reshape(2, 2, 16, 16)  # (face row, face col, r, c)
    return faces.transpose(0, 2, 1, 3).reshape(32, 32)


def _hex16(text):
    return np.array(
        [int(text[i : i + 4], 16) for i in range(0, len(text), 4)], np.int64
    )


def check(log, arch):
    text = open(log).read()
    config = re.search(r"MATMULRAND_CONFIG: fidelity=(\w+) fp32_dest=(\d)", text)
    inputs = re.search(r"OPDIFF_INPUT:([0-9a-f]+)", text)
    result = re.search(r"OPDIFF_RESULT:([0-9a-f]+)", text)
    if not (config and inputs and result):
        raise SystemExit(f"{log}: no matmulrand config/input/result lines")
    fidelity, fp32_dest = config.group(1), config.group(2) == "1"
    words = _hex16(inputs.group(1)).reshape(4, 1024)
    A = (
        np.hstack([detilize(words[0]), detilize(words[1])]) << 16
    )  # in0: K tiles side by side
    B = (
        np.vstack([detilize(words[2]), detilize(words[3])]) << 16
    )  # in1: K tiles stacked
    got = detilize(_hex16(result.group(1)))

    dst = tensix_mac_bits(
        np.zeros((32, 32), np.int64),
        A,
        B,
        dst32=fp32_dest,
        fidelity=fidelity,
        wormhole=arch == "wormhole",
    )
    if fp32_dest:
        from framework.pe.tensix.util import DataFormatConversions as DFC

        expected = np.vectorize(
            lambda v: DFC.FP32InDstToBF16(DFC.FP32ToDstFormatFP32(int(v)))
        )(dst)
    else:
        expected = dst >> 16
    bad = int(np.count_nonzero(expected != got))
    print(
        f"{'PASS' if not bad else 'FAIL'}  {log}: {arch} {fidelity} "
        f"{'fp32' if fp32_dest else 'bf16'} Dst, {1024 - bad}/1024 bf16 elements match tile_mac"
    )
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--arch", default="blackhole", choices=["blackhole", "wormhole"])
    args = ap.parse_args(argv)
    return 0 if all([check(log, args.arch) for log in args.logs]) else 1


if __name__ == "__main__":
    sys.exit(main())
