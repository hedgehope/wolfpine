"""The fft dwarf (kernels/naiad/dwarfs/fft): host-side model checks, then each variant on Naiad."""

import importlib.util
import shutil

import numpy as np
import pytest

from framework.naiad.dwarfs import NAIAD_CC, load_spec, run_point

SPEC = load_spec("fft")
SMALLEST = (min(SPEC.SIZES), 2)

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
# naiad-dwarfs.yaml lists bf16, so building its matrix unit needs ml_dtypes.
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


@pytest.mark.parametrize("n", [256, 512, 1024, 4096])
def test_stockham_model_is_the_fft(n):
    x = np.random.default_rng(n).standard_normal((2, n)) * (1 + 1j)
    np.testing.assert_allclose(
        SPEC.model(x, SPEC.plan(n)), np.fft.fft(x), atol=1e-11 * n
    )


def test_dft_constants_are_exact_on_the_axes():
    f = SPEC.dft_matrix(16)
    assert f[0, 4] == 1
    assert f[1, 4] == -1j
    assert f[1, 8] == -1
    assert SPEC.stage_twiddles(256).shape == (256,)


@pytest.mark.parametrize("fmt", ["fp16", "bf16"])
def test_narrow_storage_rounds_once(fmt):
    pytest.importorskip("ml_dtypes")
    from framework.matrix.formats import get_format

    x = np.random.default_rng(1).standard_normal(20000) * 10.0 ** np.random.default_rng(
        2
    ).integers(-6, 4, 20000)
    got = SPEC.from_storage(SPEC.to_storage(x, fmt), fmt)
    np.testing.assert_array_equal(got, np.asarray(get_format(fmt).round(x), np.float64))


@needs_clang
@needs_ml_dtypes
# Variants needing optional matrix features run in dwarf_fused_test.py.
@pytest.mark.parametrize(
    "variant", sorted(set(SPEC.VARIANTS) - set(SPEC.VARIANT_FEATURES))
)
@pytest.mark.parametrize("formats", [("fp32", "fp32"), ("bf16", "fp32")])
def test_variant_matches_golden(variant, formats, tmp_path):
    r = run_point(SPEC, variant, SMALLEST, *formats, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.region_cycles
    assert r.region_cycles > 0
    assert r.passed, r.error
    if variant.startswith("ame"):
        assert r.counters["matrix.macs"] > 0
