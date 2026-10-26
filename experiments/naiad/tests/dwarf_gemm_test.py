import importlib.util
import shutil

import pytest

from experiments.naiad.dwarfs import DEFAULT_DESIGN, NAIAD_CC, load_spec, run_point
from framework.naiad.design import load_design

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
HAVE_ML_DTYPES = importlib.util.find_spec("ml_dtypes") is not None

SPEC = load_spec("gemm")


def _formats():
    for pair in SPEC.FORMATS:
        marks = []
        if "bf16" in pair and not HAVE_ML_DTYPES:
            marks.append(pytest.mark.skip(reason="bf16 needs ml_dtypes"))
        yield pytest.param(*pair, marks=marks, id="/".join(pair))


def _check(result):
    assert result.exit_code == 0, result.console
    assert result.passed, result.error
    assert result.region_cycles
    assert result.region_cycles < result.cycles
    return result


@needs_clang
@pytest.mark.parametrize("in_fmt,acc_fmt", list(_formats()))
def test_ame_matches_golden(tmp_path, in_fmt, acc_fmt):
    size = min(SPEC.SIZES)
    r = _check(run_point(SPEC, "ame", size, in_fmt, acc_fmt, build_root=tmp_path))
    # Every MAC goes through the unit: (n/16)^3 mma of 16^3 MACs for a block-aligned size.
    assert r.counters["matrix.macs"] == size**3


@needs_clang
@pytest.mark.parametrize(
    "in_fmt,acc_fmt", [("fp32", "fp32"), ("fp64", "fp64"), ("bf16", "fp32")]
)
def test_scalar_matches_golden(tmp_path, in_fmt, acc_fmt):
    if in_fmt == "bf16" and not HAVE_ML_DTYPES:
        pytest.skip("bf16 needs ml_dtypes")
    r = _check(
        run_point(SPEC, "scalar", min(SPEC.SIZES), in_fmt, acc_fmt, build_root=tmp_path)
    )
    assert r.counters.get("matrix.macs", 0) == 0


@needs_clang
@pytest.mark.parametrize("variant", ["ame", "scalar"])
def test_ragged_shape(tmp_path, variant):
    # Not a multiple of the 32 x 32 block or the tile K: the driver's padded edges.
    r = _check(
        run_point(SPEC, variant, (40, 24, 20), "fp32", "fp32", build_root=tmp_path)
    )
    if variant == "ame":
        assert r.counters["matrix.macs"] > 40 * 24 * 20


@needs_clang
def test_fp64_reaches_most_of_its_peak(tmp_path):
    """The sanity anchor: fp64 is the one format whose 2x2 blocking is not tile-load bound.

    On naiad-dwarfs a 16^3 fp64 mma takes 32 cycles and a 2 KiB fp64 tile load 32, so four
    loads per four mma balance; at 64^3 the block transitions still cost a quarter.
    """
    timing = load_design(DEFAULT_DESIGN).timing.matrix
    r = _check(run_point(SPEC, "ame", 64, "fp64", "fp64", build_root=tmp_path))
    peak = timing.macs_per_cycle / timing.format_passes["fp64"]
    assert r.counters["matrix.macs"] / (r.region_cycles * peak) > 0.7


@needs_clang
@pytest.mark.parametrize("in_fmt,acc_fmt", list(_formats()))
def test_rvv_matches_golden(tmp_path, in_fmt, acc_fmt):
    r = _check(
        run_point(SPEC, "rvv", min(SPEC.SIZES), in_fmt, acc_fmt, build_root=tmp_path)
    )
    assert r.counters.get("matrix.macs", 0) == 0


@needs_clang
def test_rvv_ragged_shape(tmp_path):
    # 37 rows (balanced 5 + 5 + ... blocks with a remainder), 45 columns (a partial panel), odd K.
    _check(run_point(SPEC, "rvv", (37, 45, 19), "fp32", "fp32", build_root=tmp_path))


@needs_clang
def test_only_vector_variants_get_v_in_march(tmp_path):
    """On naiad-dwarfs (which has RVV) the scalar baseline stays scalar: no auto-vectorisation."""
    size = min(SPEC.SIZES)
    scalar = _check(
        run_point(
            SPEC, "scalar", size, "fp32", "fp32", build_root=tmp_path, count_all=True
        )
    )
    rvv = _check(
        run_point(
            SPEC, "rvv", size, "fp32", "fp32", build_root=tmp_path, count_all=True
        )
    )
    assert scalar.counters["instret.vector"] == 0
    assert rvv.counters["instret.vector"] > 0
