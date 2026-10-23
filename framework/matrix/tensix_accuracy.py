"""IEEE vs Tensix-fidelity accuracy for bf16-in / fp32-acc GEMMs (Naiad B6).

``python -m framework.matrix.tensix_accuracy [--size 64] [--seed 0]`` prints one
row per (K, numerics): the :func:`~framework.matrix.accuracy.compare` metrics
of a ``size x K @ K x size`` product of standard-normal operands, against two
fp64 goldens:

- ``end-to-end``: the product of the original fp64 operands, so it includes
  rounding the inputs to bf16 (the same for every mode);
- ``datapath``: the exact product of the bf16-rounded operands, so it isolates
  what the accumulation itself loses.

``normwise`` is ``compare``'s; ``cw_max`` is the largest ``|r - g|`` over
``(|A| @ |B|)``, the componentwise scale a GEMM error bound is stated in.
``compare``'s ``max_rel`` and ``ulp_max`` are kept in the returned rows but not
printed: on random data they are set by whichever output happens to cancel to
almost zero, and say nothing about the datapath. Wormhole's numbers are Blackhole's
bar the -1 renormalisation quirk, which random data essentially never reaches.
"""

import argparse

import numpy as np

from framework.matrix.accuracy import compare
from framework.matrix.mac import tile_mac
from framework.matrix.tensix_numerics import from_fp32_bits, to_fp32_bits

MODES = (
    ("ieee", {}),
    ("tensix LoFi", {"numerics": "tensix_bh", "fidelity": "LoFi"}),
    ("tensix HiFi2", {"numerics": "tensix_bh", "fidelity": "HiFi2"}),
    ("tensix HiFi4", {"numerics": "tensix_bh", "fidelity": "HiFi4"}),
)


def accuracy_table(ks=(64, 256, 1024), size=64, seed=0):
    """``[(K, mode, golden, metrics)]`` for bf16-in / fp32-acc products.

    ``metrics`` is :func:`compare`'s dict plus ``cw_max``.
    """
    rows = []
    for K in ks:
        rng = np.random.default_rng([seed, K])
        A = rng.standard_normal((size, K))
        B = rng.standard_normal((K, size))
        absAB = np.abs(A) @ np.abs(B)
        goldens = {
            "end-to-end": A @ B,
            "datapath": from_fp32_bits(to_fp32_bits(A, "bf16"), "fp32").astype(
                np.float64
            )
            @ from_fp32_bits(to_fp32_bits(B, "bf16"), "fp32").astype(np.float64),
        }
        for name, kw in MODES:
            result = tile_mac(None, A, B, "bf16", "fp32", **kw)
            for golden_name, golden in goldens.items():
                metrics = compare(result, golden, "fp32")
                metrics["cw_max"] = float(
                    np.max(np.abs(result.astype(np.float64) - golden) / absAB)
                )
                rows.append((K, name, golden_name, metrics))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=64, help="M = N (default 64)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    print(f"bf16 in, fp32 acc, {args.size}x{args.size} outputs, N(0,1) operands")
    print(f"{'K':>5}  {'numerics':<13} {'golden':<11} {'normwise':>10} {'cw_max':>10}")
    for K, mode, golden, m in accuracy_table(size=args.size, seed=args.seed):
        print(
            f"{K:>5}  {mode:<13} {golden:<11} {m['normwise']:>10.3e} {m['cw_max']:>10.3e}"
        )


if __name__ == "__main__":
    main()
