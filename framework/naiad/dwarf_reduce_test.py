"""The reduce dwarf (kernels/naiad/dwarfs/reduce): each variant on Naiad against the fp64 sum."""

import importlib.util
import shutil

import pytest

from framework.naiad.dwarfs import NAIAD_CC, load_spec, run_point

SPEC = load_spec("reduce")
SMALLEST = min(SPEC.SIZES)

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


def test_size_must_be_whole_tile_groups():
    with pytest.raises(ValueError, match="multiple of 1024"):
        SPEC.cflags(1000, "fp32", "fp32")


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize("variant", sorted(SPEC.VARIANTS))
@pytest.mark.parametrize("formats", [("fp32", "fp32"), ("bf16", "fp32")])
def test_variant_matches_golden(variant, formats, tmp_path):
    r = run_point(SPEC, variant, SMALLEST, *formats, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.region_cycles
    assert r.region_cycles > 0
    assert r.passed, r.error
    # One mma per 256 inputs, plus the finishing mma.
    assert r.counters.get("matrix.macs", 0) == (
        4096 * (SMALLEST // 256 + 1) if variant == "ame" else 0
    )


@needs_clang
@needs_ml_dtypes
def test_scalar_finish_agrees(tmp_path):
    r = run_point(
        SPEC,
        "ame",
        SMALLEST,
        "fp32",
        "fp32",
        build_root=tmp_path,
        extra_cflags=SPEC.switch_cflags({"finish": "scalar"}),
    )
    assert r.passed, (r.error, r.console)
    assert r.counters["matrix.macs"] == 4096 * (SMALLEST // 256)
