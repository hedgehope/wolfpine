"""The jacobi dwarf (kernels/naiad/dwarfs/jacobi): host-side format checks, then each variant on Naiad."""

import importlib.util
import shutil

import numpy as np
import pytest

from framework.naiad.dwarfs import NAIAD_CC, load_spec, run_point

SPEC = load_spec("jacobi")
SMALLEST = min(SPEC.SIZES)
TILES = ((SMALLEST - 2) // 16) ** 2

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
# naiad-dwarfs.yaml lists bf16, so building its matrix unit needs ml_dtypes.
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


@pytest.mark.parametrize("fmt", ["fp16", "bf16", "fp32"])
def test_encode_rounds_once_like_the_matrix_unit(fmt):
    pytest.importorskip("ml_dtypes")
    from framework.matrix.formats import get_format

    rng = np.random.default_rng(3)
    x = rng.standard_normal(20000) * 10.0 ** rng.integers(-4, 4, 20000)
    got = SPEC.dwf.as_f64(SPEC.dwf.encode(x, fmt), fmt)
    np.testing.assert_array_equal(got, np.asarray(get_format(fmt).round(x), np.float64))


def test_golden_keeps_the_boundary_and_averages_the_interior():
    u = np.random.default_rng(0).random((SMALLEST, SMALLEST))
    g = SPEC.golden({"grid": u}, iters=1)["out"]
    np.testing.assert_array_equal(g[0], u[0])
    np.testing.assert_array_equal(g[:, -1], u[:, -1])
    assert g[5, 7] == pytest.approx((u[4, 7] + u[6, 7] + u[5, 6] + u[5, 8]) / 4)


def test_sizes_must_tile_the_interior():
    with pytest.raises(ValueError, match="16 t \\+ 2"):
        SPEC.cflags(32, "fp32", "fp32")
    with pytest.raises(ValueError, match="unknown jacobi switch"):
        SPEC.cflags(34, "fp32", "fp32", {"nonsense": 1})


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
    macs = r.counters.get("matrix.macs", 0)
    if variant in ("ame", "ame_mc"):  # ame_mc on one hart is ame
        # Default form: 5 mma per output tile per sweep, plus bf16's rounding mma
        # on every sweep but the last.
        mmas = SPEC.ITERS * TILES * 5 + (
            0 if formats[0] == "fp32" else (SPEC.ITERS - 1) * TILES
        )
        assert macs == mmas * 4096
    else:
        assert macs == 0


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize(
    "switches, mma_per_tile",
    [({"form": "window"}, 4), ({"tshift": False}, 5), ({"epilogue": True}, 5)],
)
def test_feature_switches_agree(switches, mma_per_tile, tmp_path):
    r = run_point(
        SPEC,
        "ame",
        SMALLEST,
        "fp32",
        "fp32",
        build_root=tmp_path,
        extra_cflags=SPEC.switch_cflags(switches),
    )
    assert r.passed, (r.error, r.console)
    assert r.counters["matrix.macs"] == SPEC.ITERS * TILES * mma_per_tile * 4096
