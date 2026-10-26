import importlib.util
import shutil

import numpy as np
import pytest

from experiments.naiad.dwarfs import NAIAD_CC, load_spec, run_point

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
HAVE_ML_DTYPES = importlib.util.find_spec("ml_dtypes") is not None

SPEC = load_spec("tensorprod")


def _pairs():
    for pair in SPEC.FORMATS:
        marks = []
        if "bf16" in pair and not HAVE_ML_DTYPES:
            marks.append(pytest.mark.skip(reason="bf16 needs ml_dtypes"))
        yield pytest.param(*pair, marks=marks, id="/".join(pair))


@pytest.mark.parametrize("size", [8, 10, 12, 16, (5, 3), (7, 1), (1, 2)])
def test_layouts_round_trip_through_decode(size):
    """Place the golden in the padded layouts the kernels write; decode must give it back."""
    p, e = SPEC.dims(size)
    lay = SPEC.layout(size)
    gold = SPEC.golden(SPEC.make_inputs(size, np.random.default_rng(0)))
    outs = {
        name: np.full(shape, np.nan)
        for name, (_, shape) in SPEC.outputs(size, "fp64").items()
    }
    outs["tp_dims"] = np.array([p, e])
    q, T = lay["q"], lay["T"]
    for ei in range(e):
        for k in range(p):
            outs["out_t"][ei * p + k, : p * p] = gold["ut"][ei, k].reshape(-1)
            for j in range(p):
                row = (ei * p + k) * p + j
                outs["out_s"][row, :p] = gold["us"][ei, k, j]
                outs["out_r"][row // q, (row % q) * p : (row % q + 1) * p] = gold["ur"][
                    ei, k, j
                ]
    got = SPEC.decode(outs, "fp64")
    for name in gold:
        np.testing.assert_array_equal(got[name], gold[name])
    assert lay["rows_r"] % T == 0
    assert lay["LDT"] % T == 0


@needs_clang
@pytest.mark.parametrize("in_fmt,acc_fmt", list(_pairs()))
def test_ame_matches_golden(tmp_path, in_fmt, acc_fmt):
    size = min(SPEC.SIZES)
    r = run_point(SPEC, "ame", size, in_fmt, acc_fmt, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    assert r.counters["matrix.mma"] == SPEC.mma_count(size)


@needs_clang
@pytest.mark.parametrize("size", [10, (5, 3), (16, 2)])
def test_ame_one_slab_per_tile_and_ragged(tmp_path, size):
    # q = 1 (p > 8), p^2 not a multiple of 16, E not a multiple of q: every padding path.
    r = run_point(SPEC, "ame", size, "fp32", "fp32", build_root=tmp_path)
    assert r.passed, (r.error, r.console)
    assert r.counters["matrix.mma"] == SPEC.mma_count(size)


@needs_clang
@pytest.mark.parametrize(
    "in_fmt,acc_fmt", [("fp32", "fp32"), ("fp64", "fp64"), ("fp16", "fp32")]
)
def test_scalar_matches_golden(tmp_path, in_fmt, acc_fmt):
    r = run_point(SPEC, "scalar", (8, 2), in_fmt, acc_fmt, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    assert r.region_cycles
    assert r.counters.get("matrix.macs", 0) == 0


@needs_clang
@pytest.mark.parametrize("in_fmt,acc_fmt", list(_pairs()))
def test_rvv_matches_golden(tmp_path, in_fmt, acc_fmt):
    r = run_point(SPEC, "rvv", (8, 2), in_fmt, acc_fmt, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    assert r.region_cycles
    assert r.counters.get("matrix.macs", 0) == 0


@needs_clang
@pytest.mark.parametrize("size", [(10, 1), (5, 3), (16, 1)])
def test_rvv_odd_p_and_layouts(tmp_path, size):
    # p = 5 (Q * P != T: the per-group ur path), 10 (Q = 1) and 16 (full tiles).
    r = run_point(SPEC, "rvv", size, "fp32", "fp32", build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
