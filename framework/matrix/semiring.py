"""Tropical tile products: (min, +) and (max, +) over the tile_mac shapes.

``C ⊕= A ⊗ B`` with ⊗ = ``+`` and ⊕ = ``min`` (or ``max``), so element (i, j) is
``min(C[i, j], min_k A[i, k] + B[k, j])``. :func:`framework.matrix.mac.tile_mac`
selects these through ``semiring=`` and does the format rounding around them;
this module only works on arrays already in the compute dtype.

Min and max are exact, and rounding is monotonic, so unlike (+, ×) the result
does not depend on how K is chunked: only the A + B sums round, once each.
"""

import numpy as np

SEMIRINGS = ("plus_times", "min_plus", "max_plus")

# Cap on the (M, K-block, N) broadcast temporary, in elements
_BLOCK_ELEMS = 1 << 22


def _reducer(semiring):
    if semiring == "min_plus":
        return np.minimum
    if semiring == "max_plus":
        return np.maximum
    raise ValueError(
        f"unknown tropical semiring {semiring!r}; expected 'min_plus' or 'max_plus'"
    )


def identity(semiring, dtype):
    """⊕'s identity in ``dtype``: +inf for min, -inf for max (the int extremes for ints)."""
    dtype = np.dtype(dtype)
    lo, hi = (
        (-np.inf, np.inf)
        if dtype.kind == "f"
        else (np.iinfo(dtype).min, np.iinfo(dtype).max)
    )
    return dtype.type(hi if semiring == "min_plus" else lo)


def tropical_product(A, B, semiring):
    """``⊕_k A[:, k] + B[k, :]`` for (M, K) ``A`` and (K, N) ``B``, in their common dtype.

    K is walked in blocks so the broadcast sum stays bounded; an empty K gives
    the identity everywhere.
    """
    reduce = _reducer(semiring)
    (M, K), N = A.shape, B.shape[1]
    out = np.full((M, N), identity(semiring, A.dtype), dtype=A.dtype)
    step = max(1, _BLOCK_ELEMS // max(1, M * N))
    for s in range(0, K, step):
        sums = A[:, s : s + step, np.newaxis] + B[np.newaxis, s : s + step, :]
        out = reduce(out, reduce.reduce(sums, axis=1))
    return out


def tropical_accumulate(C, A, B, semiring):
    """``C ⊕ (A ⊗ B)``: fold the tropical product into ``C``."""
    return _reducer(semiring)(C, tropical_product(A, B, semiring))
