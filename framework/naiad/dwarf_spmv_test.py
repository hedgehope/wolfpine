"""The spmv dwarf (kernels/naiad/dwarfs/spmv): BSR Y = A X, each variant on Naiad against the fp64 product."""

import importlib.util
import shutil

import numpy as np
import pytest

from framework.naiad.dwarfs import NAIAD_CC, load_spec, run_point

SPEC = load_spec("spmv")
SMALLEST = SPEC.SIZES[0]
FD_PAIRS = [("fp32", "fp32"), ("fp64", "fp64")]

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


def _pairs(variant):
    if variant == "scalar":
        return FD_PAIRS
    return [tuple(p) for p in SPEC.VARIANT_FORMATS.get(variant, SPEC.FORMATS)]


def test_sizes_and_block_counts():
    assert SPEC.dims((128, 0.5)) == (128, 0.5, 16, 32)
    assert SPEC.dims((256, 0.01, 1)) == (256, 0.01, 1, 3)
    assert SPEC.dims((64, 1e-9))[3] == 1  # never an empty matrix
    for bad in (100, (100, 0.5), (128, 0.0), (128, 1.5), (128, 0.5, 17)):
        with pytest.raises(ValueError, match="spmv"):
            SPEC.dims(bad)


def test_inputs_are_bsr_with_the_requested_blocks():
    rng = np.random.default_rng(3)
    inputs = SPEC.make_inputs((128, 0.25, 4), rng)
    nb = 8
    assert inputs["rowptr"][0] == 0
    assert inputs["rowptr"][-1] == 16
    assert np.all(np.diff(inputs["rowptr"]) >= 0)
    pos = [
        (bi, int(c))
        for bi in range(nb)
        for c in inputs["colidx"][inputs["rowptr"][bi] : inputs["rowptr"][bi + 1]]
    ]
    assert len(set(pos)) == 16
    assert pos == sorted(pos)
    A = SPEC.dense(inputs)
    assert np.count_nonzero(A) == 16 * 256
    assert np.allclose(SPEC.golden(inputs)["Y"], A @ inputs["X"])
    assert SPEC.useful_flops((128, 0.25, 4)) == 2 * 256 * 16 * 4


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize(
    "variant,formats",
    [(v, p) for v in sorted(SPEC.VARIANTS) for p in _pairs(v)],
)
def test_variant_matches_golden(variant, formats, tmp_path):
    r = run_point(SPEC, variant, SMALLEST, *formats, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.region_cycles
    assert r.region_cycles > 0
    assert r.passed, r.error
    nnzb = SPEC.dims(SMALLEST)[3]
    assert r.counters.get("matrix.macs", 0) == (4096 * nnzb if variant == "ame" else 0)


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize("variant", sorted(SPEC.VARIANTS))
@pytest.mark.parametrize("nv", [1, 5])
def test_fewer_vectors(variant, nv, tmp_path):
    """nv < 16: the AME b tile's spare columns are junk that must never reach Y; RVV nv = 1 is a dot form."""
    r = run_point(SPEC, variant, (64, 0.4, nv), "fp32", "fp32", build_root=tmp_path)
    assert r.passed, (r.error, r.console)


@needs_clang
@needs_ml_dtypes
def test_empty_block_rows_are_zero(tmp_path):
    """A very sparse matrix leaves block rows without blocks: their Y rows must be written as 0."""
    size = (128, 0.03)  # 2 blocks in 8 block rows
    r = run_point(SPEC, "ame", size, "fp32", "fp32", build_root=tmp_path)
    assert r.passed, (r.error, r.console)
