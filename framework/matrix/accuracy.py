"""The fp64-golden accuracy harness (Naiad WP5).

:func:`compare` scores a result against a golden; :func:`evaluate_formats` runs
a computation at a chosen format set and again with every format at fp64, and
scores the one against the other.

Zero convention. Relative error is undefined where the golden is zero, so
``max_rel`` is taken only over the elements whose golden is non-zero (and is 0
if there are none and the result matches, inf if it does not). An error at a
zero golden is still counted by ``max_abs``, by ``normwise``, and by
``ulp_max`` -- whose ulp there is the format's subnormal spacing, so any
non-zero result at a zero golden scores as the many ulps it really is.
``normwise`` is ``||r - g||_F / ||g||_F``, with the same 0-or-inf rule for an
all-zero golden. Elements that are equal (infinities included) contribute no
error; a NaN anywhere it is not also in the golden makes every metric NaN.
"""

import numpy as np

from framework.matrix.formats import format_of_dtype, get_format


def _zero_or_inf(err):
    return 0.0 if err == 0 else np.inf


def compare(result, golden, fmt=None):
    """Return ``{max_abs, max_rel, normwise, ulp_max}`` for ``result`` against ``golden``.

    ``ulp_max`` is in ulps of ``fmt``; by default the format of ``result``'s
    dtype (fp32 for float32), else fp64.
    """
    if fmt is None:
        fmt = format_of_dtype(np.asarray(result).dtype) or "fp64"
    fmt = get_format(fmt)
    r = np.asarray(result).astype(np.float64)
    g = np.asarray(golden).astype(np.float64)
    if r.shape != g.shape:
        raise ValueError(f"result {r.shape} and golden {g.shape} differ in shape")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        diff = np.where((r == g) | (np.isnan(r) & np.isnan(g)), 0.0, np.abs(r - g))
        if diff.size == 0:
            return {"max_abs": 0.0, "max_rel": 0.0, "normwise": 0.0, "ulp_max": 0.0}
        max_abs = float(np.max(diff))
        nz = g != 0
        max_rel = (
            float(np.max(diff[nz] / np.abs(g[nz])))
            if nz.any()
            else _zero_or_inf(max_abs)
        )
        gnorm = np.linalg.norm(g[np.isfinite(g)])
        dnorm = float(np.sqrt(np.sum(diff**2)))
        normwise = dnorm / gnorm if gnorm else _zero_or_inf(dnorm)
        ulp_max = float(np.max(diff / fmt.ulp(g)))
    if np.isnan(diff).any():
        return {
            "max_abs": np.nan,
            "max_rel": np.nan,
            "normwise": np.nan,
            "ulp_max": np.nan,
        }
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "normwise": normwise,
        "ulp_max": ulp_max,
    }


def evaluate_formats(fn, *args, formats, golden=None, ulp_fmt=None, **kwargs):
    """Score ``fn(*args, **kwargs, **formats)`` against an fp64 golden.

    ``formats`` maps ``fn``'s format keyword arguments to the formats under test,
    e.g. ``{"in_fmt": "bf16", "acc_fmt": "fp32"}``. The golden is ``golden`` if
    given, else ``fn`` re-run with every one of those keywords set to fp64.
    Returns :func:`compare`'s dict plus ``result`` and ``golden``.
    """
    result = fn(*args, **kwargs, **formats)
    if golden is None:
        golden = fn(*args, **kwargs, **{key: "fp64" for key in formats})
    return {**compare(result, golden, ulp_fmt), "result": result, "golden": golden}
