"""compare() and evaluate_formats(), including the zero conventions."""

import numpy as np
import pytest

from framework.matrix.accuracy import compare, evaluate_formats
from framework.matrix.mac import tile_mac


def test_compare_basic_metrics():
    g = np.array([1.0, 2.0, 4.0, -8.0])
    r = np.array([1.0, 2.5, 4.0, -8.0], dtype=np.float32)
    m = compare(r, g)
    assert m["max_abs"] == 0.5
    assert m["max_rel"] == 0.25
    assert m["normwise"] == pytest.approx(0.5 / np.sqrt(85))
    assert m["ulp_max"] == 0.5 / 2.0**-22  # result is fp32, ulp at 2 is 2**-22


def test_compare_zero_golden_conventions():
    # An exact zero is no error; an error at a zero golden is excluded from max_rel only
    m = compare(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert m == {"max_abs": 0.0, "max_rel": 0.0, "normwise": 0.0, "ulp_max": 0.0}
    m = compare(np.array([1e-30, 1.0]), np.array([0.0, 1.0]), "fp32")
    assert m["max_rel"] == 0.0
    assert m["max_abs"] == 1e-30
    assert m["ulp_max"] == pytest.approx(1e-30 / 2.0**-149)
    # An all-zero golden: 0 if matched, inf if not
    assert compare(np.zeros(3), np.zeros(3))["normwise"] == 0.0
    m = compare(np.array([0.0, 1e-9]), np.zeros(2))
    assert m["max_rel"] == np.inf
    assert m["normwise"] == np.inf


def test_compare_infinities_and_nans():
    m = compare(np.array([np.inf, 1.0]), np.array([np.inf, 1.0]))
    assert m["max_abs"] == 0.0
    assert np.isnan(compare(np.array([np.nan, 1.0]), np.array([0.0, 1.0]))["max_rel"])
    with pytest.raises(ValueError, match="shape"):
        compare(np.zeros(2), np.zeros(3))


def test_evaluate_formats_against_fp64_golden():
    pytest.importorskip("ml_dtypes")
    rng = np.random.default_rng(10)
    A, B = rng.uniform(-1, 1, (8, 256)), rng.uniform(-1, 1, (256, 8))
    run = {
        acc: evaluate_formats(
            tile_mac, None, A, B, formats={"in_fmt": "bf16", "acc_fmt": acc}
        )
        for acc in ("fp32", "bf16")
    }
    np.testing.assert_allclose(run["fp32"]["golden"], A @ B, rtol=1e-12)
    # bf16 inputs dominate the fp32-accumulated error; bf16 accumulation adds to it
    assert 1e-4 < run["fp32"]["normwise"] < run["bf16"]["normwise"] < 2**-6
    # ulp_max is inferred from the result's dtype: bf16 ulps are 2**16 x coarser than fp32's
    assert run["bf16"]["ulp_max"] < run["fp32"]["ulp_max"]
