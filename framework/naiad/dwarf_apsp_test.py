"""The apsp dwarf (kernels/naiad/dwarfs/apsp): blocked Floyd-Warshall on the tropical mma, exact."""

import importlib.util
import shutil

import numpy as np
import pytest

from framework.naiad.dwarfs import NAIAD_CC, load_spec, run_point

SPEC = load_spec("apsp")
SMALLEST = min(SPEC.SIZES)
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


def test_size_must_be_whole_blocks():
    with pytest.raises(ValueError, match="multiple of 16"):
        SPEC.cflags(40, "fp32", "fp32")
    with pytest.raises(ValueError, match="supports"):
        SPEC.cflags(32, "bf16", "fp32")


def test_inputs_are_connected_and_nontrivial():
    inputs = SPEC.make_inputs(64, np.random.default_rng(0))
    D = inputs["D"]
    assert np.all(np.diag(D) == 0)
    assert np.isinf(D).any()
    g = SPEC.golden(inputs)["D"]
    assert np.all(np.isfinite(g))  # the ring connects every pair
    assert (g < D).mean() > 0.5  # most distances come from paths, not edges
    assert g.max() < 256  # exact in bf16 too


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize(
    "variant,formats",
    [(v, p) for v in sorted(SPEC.VARIANTS) for p in _pairs(v)],
)
def test_variant_is_exact(variant, formats, tmp_path):
    r = run_point(SPEC, variant, SMALLEST, *formats, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.region_cycles
    assert r.region_cycles > 0
    assert r.passed, r.error
    assert r.error["D"]["max_abs"] == 0.0
    nb = SMALLEST // 16
    # (nb^2 + 3) tropical mma per diagonal block: closure (4), panels, the rest.
    assert r.counters.get("matrix.macs", 0) == (
        4096 * nb * (nb * nb + 3) if variant == "ame" else 0
    )


@needs_clang
@needs_ml_dtypes
def test_ame_exact_at_mid_size(tmp_path):
    r = run_point(SPEC, "ame", 64, "fp32", "fp32", build_root=tmp_path, seed=5)
    assert r.passed, (r.error, r.console)
    assert r.error["D"]["max_abs"] == 0.0
