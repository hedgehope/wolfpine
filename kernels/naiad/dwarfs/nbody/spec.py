# ruff: noqa: INP001  (loaded by path by framework.naiad.dwarfs, not imported as a package)
"""nbody: N-body. All-pairs squared distances as a GEMM (the expansion), forces on the scalar side.

Variants:

* ``ame``: ``D = A B`` with augmented, K-padded coordinates
  (``A[i] = [-2x_i, |x_i|^2, 1]``, ``B[:, j] = [x_j, 1, |x_j|^2]``), so
  ``D[i][j] = |x_i|^2 + |x_j|^2 - 2 x_i.x_j`` comes out of the matrix unit in
  ``acc_fmt`` with every operand, norms included, in ``in_fmt``. Then the
  scalar side clamps ``D`` at 0 and accumulates ``w(r^2) (x_j - x_i)`` per
  particle. See ``ame/nbody_ame.c``.
* ``ame_fgemm``: the same distances, and the force sum as a second GEMM,
  ``F = W X - x * (W 1)``, so the scalar side only evaluates ``w``.
* ``scalar``: the direct pairwise loop, ``r^2`` from coordinate differences.

``size`` is ``n``, ``(n, dist)`` or ``(n, dist, law)``: ``dist`` is
``"uniform"`` (the unit cube) or ``"clustered"`` (four tight Gaussian blobs,
sigma 0.01, centred in the unit cube: close pairs far from the origin, where
the expansion cancels catastrophically at reduced precision); ``law`` is
``"soft"`` (softened gravity, eps = 0.01) or ``"lj"`` (Lennard-Jones, cut
off at 2.5 sigma). Unit masses.

Outputs: ``force`` always; ``d2`` (the squared distances the kernel used)
when ``n <= D2_MAX``, so the harness scores the distance error and the force
error separately, uniform against clustered.

Numpy only (bf16 by bit manipulation; no ``ml_dtypes``).
"""

import numpy as np

NAME = "nbody"
VARIANTS = {"ame": "ame", "ame_fgemm": "ame_fgemm", "scalar": "scalar"}
SIZES = (
    (64, "uniform"),
    (64, "clustered"),
    (256, "uniform"),
    (256, "clustered"),
    (384, "uniform"),
    (384, "clustered"),
)
FORMATS = [("fp32", "fp32"), ("fp64", "fp64"), ("bf16", "fp32"), ("fp16", "fp32")]
D2_MAX = 256
EPS = 0.01
CLUSTERS = 4
CLUSTER_SIGMA = 0.01

_MFMT = {
    "fp64": "NAIAD_MFMT_FP64",
    "fp32": "NAIAD_MFMT_FP32",
    "fp16": "NAIAD_MFMT_FP16",
    "bf16": "NAIAD_MFMT_BF16",
}


def _shape(size):
    if isinstance(size, (tuple, list)):
        n, *rest = size
        dist = rest[0] if rest else "uniform"
        law = rest[1] if len(rest) > 1 else "soft"
    else:
        n, dist, law = size, "uniform", "soft"
    if dist not in ("uniform", "clustered") or law not in ("soft", "lj"):
        raise ValueError(f"nbody: bad size {size!r}")
    return int(n), dist, law


def cflags(size, in_fmt, acc_fmt):
    n, _, law = _shape(size)
    return [
        f"-DDW_IN_FMT={_MFMT[in_fmt]}",
        f"-DDW_ACC_FMT={_MFMT[acc_fmt]}",
        f"-DNB_N={n}",
        f"-DNB_SAVE_D2={int(n <= D2_MAX)}",
        f"-DNB_LJ={int(law == 'lj')}",
    ]


def make_inputs(size, rng):
    n, dist, law = _shape(size)
    if dist == "uniform":
        pos = rng.random((n, 3))
    else:
        centres = rng.random((CLUSTERS, 3))
        pos = centres[np.arange(n) % CLUSTERS] + CLUSTER_SIGMA * rng.standard_normal(
            (n, 3)
        )
    sigma = 0.5 * n ** (-1 / 3)
    return {
        "pos": pos,
        "lj": law == "lj",
        "eps2": EPS**2,
        "sig2": sigma**2,
        "eps_lj": 1.0,
        "rc2": (2.5 * sigma) ** 2,
        # power-of-two weight scale for ame_fgemm, ~1/w_max, so fp16 weights do not overflow
        "wscale": 2.0 ** np.round(np.log2(EPS**3)),
    }


# -- formats (numpy only; a copy of the fft dwarf's helpers) -------------------


def _round_to_odd_f32(x):
    x = np.asarray(x, np.float64)
    f = x.astype(np.float32)
    bits = f.view(np.uint32).copy()
    fix = np.isfinite(x) & (f.astype(np.float64) != x) & ((bits & 1) == 0)
    away = np.abs(f.astype(np.float64)) > np.abs(x)
    bits[fix & away] -= 1
    bits[fix & ~away] += 1
    return bits.view(np.float32)


def to_storage(x, fmt):
    """fp64 values -> the kernel's storage array for ``fmt`` (RNE, one rounding)."""
    x = np.asarray(x, np.float64)
    if fmt == "fp64":
        return x.copy()
    if fmt == "fp32":
        return x.astype(np.float32)
    if fmt == "fp16":
        return x.astype(np.float16).view(np.uint16)
    if fmt == "bf16":
        u = _round_to_odd_f32(x).view(np.uint32).astype(np.uint64)
        nan = (u & 0x7FFFFFFF) > 0x7F800000
        out = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        out[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
        return out
    raise ValueError(f"nbody: unsupported format {fmt!r}")


def from_storage(a, fmt):
    a = np.asarray(a)
    if fmt in ("fp64", "fp32"):
        return a.astype(np.float64)
    if fmt == "fp16":
        return a.view(np.float16).astype(np.float64)
    if fmt == "bf16":
        return (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    raise ValueError(f"nbody: unsupported format {fmt!r}")


def round_to(x, fmt):
    return from_storage(to_storage(x, fmt), fmt)


def encode(inputs, in_fmt, acc_fmt):
    params = [
        inputs["eps2"],
        inputs["sig2"],
        24.0 * inputs["eps_lj"],
        inputs["rc2"],
        inputs["wscale"],
    ]
    return {
        "pos": to_storage(np.asarray(inputs["pos"]).reshape(-1), in_fmt),
        "params": to_storage(np.array(params), acc_fmt),
    }


def _acc_dtype(acc_fmt):
    return {"fp64": np.float64, "fp32": np.float32}[acc_fmt]


def outputs(size, acc_fmt):
    n, _, _ = _shape(size)
    dt = _acc_dtype(acc_fmt)
    out = {"force": (dt, (n, 3))}
    if n <= D2_MAX:
        out["d2"] = (dt, (n, n))
    return out


def decode(outputs, acc_fmt):
    return {name: np.asarray(a, np.float64) for name, a in outputs.items()}


def forces(pos, inputs, r2=None):
    """fp64 forces for ``pos``; ``r2`` overrides the squared distances (e.g. a kernel's D)."""
    pos = np.asarray(pos, np.float64)
    n = len(pos)
    diff = pos[None, :, :] - pos[:, None, :]  # x_j - x_i
    if r2 is None:
        r2 = np.einsum("ijc,ijc->ij", diff, diff)
    r2 = np.maximum(r2, 0.0)
    off = ~np.eye(n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if inputs["lj"]:
            inv = 1.0 / r2
            s6 = (inputs["sig2"] * inv) ** 3
            w = -24.0 * inputs["eps_lj"] * inv * s6 * (2 * s6 - 1)
            w = np.where(off & (r2 < inputs["rc2"]), w, 0.0)
        else:
            s = r2 + inputs["eps2"]
            w = np.where(off, 1.0 / (s * np.sqrt(s)), 0.0)
    return np.einsum("ij,ijc->ic", w, diff)


def expansion_model(inputs, in_fmt):
    """What the ``ame`` variant computes, in fp64 arithmetic: ``(d2, force)``.

    Positions and norms are rounded to ``in_fmt`` exactly as the kernel's
    operands are (the norms from the rounded positions, in fp32 first), and
    the expansion is summed exactly. Scoring the kernel against this isolates
    its accumulation error; scoring this against :func:`golden` gives the
    error of the expansion itself at ``in_fmt``, input rounding included.
    """
    pr = round_to(inputs["pos"], in_fmt)
    acc = np.float64 if in_fmt == "fp64" else np.float32
    n2 = round_to(np.sum(pr**2, axis=1).astype(acc), in_fmt)
    d2 = n2[:, None] + n2[None, :] - 2 * pr @ pr.T
    return d2, forces(pr, inputs, d2)


def golden(inputs):
    pos = np.asarray(inputs["pos"], np.float64)
    out = {"force": forces(pos, inputs)}
    if len(pos) <= D2_MAX:
        diff = pos[None, :, :] - pos[:, None, :]
        out["d2"] = np.einsum("ijc,ijc->ij", diff, diff)
    return out


#: FLOPs per ordered pair (i != j) of the direct method. Distance: 3 sub + 3 mul + 2 add.
#: Softened force: + eps^2, sqrt, mul, div, then 3 multiply-adds (6). LJ force: div,
#: sigma^2 * inv, two muls for s^6, 2 s^6 - 1 (2), two muls and the constant (3), then
#: 3 multiply-adds (6). sqrt and div count as one FLOP each.
FLOPS_DISTANCE = 8
FLOPS_FORCE = {"soft": 10, "lj": 15}


def useful_flops(size):
    n, _, law = _shape(size)
    return n * (n - 1) * (FLOPS_DISTANCE + FLOPS_FORCE[law])


def tolerance(in_fmt, acc_fmt):
    """Normwise threshold (worst output) of a numerically acceptable result, uniform particles.

    Set from the direct method: the ``ame`` variants miss it at bf16/fp16 (the
    expansion cancels), which is the finding, not a kernel bug; see
    :func:`expansion_model` for the check that they compute the expansion right.
    """
    return {
        ("fp64", "fp64"): 1e-10,
        ("fp32", "fp32"): 1e-3,
        ("fp16", "fp32"): 2e-2,
        ("bf16", "fp32"): 1e-1,
    }.get((in_fmt, acc_fmt), 1e-1)
