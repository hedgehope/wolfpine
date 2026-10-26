"""The jacobi dwarf (kernels/naiad/dwarfs/jacobi): host-side format checks, then each variant on Naiad."""

import importlib.util
import shutil

import numpy as np
import pytest

from framework.naiad.dwarfs import (
    NAIAD_CC,
    DwarfError,
    build_point,
    load_spec,
    run_point,
)

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


# -- radius r > 1: size (n, r), the 4 r + 1 point star stencil --------------------------------

RADIUS_SIZES = {r: 16 + 2 * r for r in (2, 4)}  # one 16 x 16 output tile per sweep


def test_radius_one_is_the_plain_size():
    assert SPEC.cflags((66, 1), "fp32", "fp32") == SPEC.cflags(66, "fp32", "fp32")
    assert not any("RADIUS" in f for f in SPEC.cflags(66, "fp32", "fp32"))
    assert "-DJACOBI_RADIUS=2" in SPEC.cflags((68, 2), "fp32", "fp32")
    assert (
        SPEC.useful_flops((66, 1)) == SPEC.useful_flops(66) == 4 * 64 * 64 * SPEC.ITERS
    )
    assert SPEC.useful_flops((72, 4)) == 16 * 64 * 64 * SPEC.ITERS


def test_radius_sizes_are_checked():
    with pytest.raises(ValueError, match="16 t \\+ 4"):
        SPEC.cflags((66, 2), "fp32", "fp32")
    with pytest.raises(ValueError, match="radius must be one of"):
        SPEC.cflags((70, 3), "fp32", "fp32")
    with pytest.raises(ValueError, match="radius 1 only"):
        SPEC.cflags((68, 2), "fp32", "fp32", {"form": "window", "tshift2": True})


@pytest.mark.parametrize("r", [2, 4])
def test_golden_averages_the_radius_r_star(r):
    n = RADIUS_SIZES[r] + 16
    u = np.random.default_rng(1).random((n, n))
    g = SPEC.golden({"grid": u, "radius": r}, iters=1)["out"]
    np.testing.assert_array_equal(g[:r], u[:r])
    np.testing.assert_array_equal(g[:, -r:], u[:, -r:])
    i, j = r + 3, n - r - 1
    star = sum(
        u[i - k, j] + u[i + k, j] + u[i, j - k] + u[i, j + k] for k in range(1, r + 1)
    )
    assert g[i, j] == pytest.approx(star / (4 * r))


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize("r", [2, 4])
@pytest.mark.parametrize("variant", ["ame", "rvv", "scalar"])
@pytest.mark.parametrize(
    "formats", [("fp32", "fp32"), ("bf16", "fp32"), ("fp64", "fp64")]
)
def test_radius_variant_matches_golden(r, variant, formats, tmp_path):
    if variant == "scalar" and formats[0] == "bf16":
        pytest.skip("the scalar baseline is F/D only")
    r_point = run_point(
        SPEC, variant, (RADIUS_SIZES[r], r), *formats, build_root=tmp_path
    )
    assert r_point.exit_code == 0, r_point.console
    assert r_point.passed, r_point.error
    macs = r_point.counters.get("matrix.macs", 0)
    if variant == "ame":
        # Banded: 5 mma per output tile per sweep at every radius (one tile here),
        # plus bf16's rounding mma on every sweep but the last.
        rounding = 0 if formats[0] != "bf16" else SPEC.ITERS - 1
        assert macs == (SPEC.ITERS * 5 + rounding) * 4096
    else:
        assert macs == 0


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize("r", [2, 4])
def test_radius_rvv_adds_in_the_scalar_order(r, tmp_path):
    size = (RADIUS_SIZES[r] + 16, r)
    got = [
        run_point(SPEC, v, size, "fp32", "fp32", build_root=tmp_path)
        for v in ("rvv", "scalar")
    ]
    assert all(p.passed for p in got)
    assert got[0].error == got[1].error


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize("r", [2, 4])
@pytest.mark.parametrize(
    "switches, mma_per_tile",
    [({"form": "window"}, 4), ({"epilogue": True}, 5), ({"tshift": False}, 5)],
)
def test_radius_feature_switches_agree(r, switches, mma_per_tile, tmp_path):
    size = (RADIUS_SIZES[r] + 16, r)  # 2 x 2 tiles: the rotation and both window halos
    res = run_point(
        SPEC,
        "ame",
        size,
        "fp32",
        "fp32",
        build_root=tmp_path,
        extra_cflags=SPEC.switch_cflags(switches),
    )
    assert res.passed, (res.error, res.console)
    assert res.counters["matrix.macs"] == SPEC.ITERS * 4 * mma_per_tile * 4096


@needs_clang
@pytest.mark.parametrize("variant", SPEC.RADIUS1_VARIANTS)
def test_multicore_variants_refuse_radius_r(variant, tmp_path):
    with pytest.raises(DwarfError, match="radius 1 only"):
        build_point(
            SPEC, variant, (RADIUS_SIZES[2], 2), "fp32", "fp32", build_root=tmp_path
        )
