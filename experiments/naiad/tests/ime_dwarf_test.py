"""The gemm dwarf's ``ime`` variant end to end (moved from ``framework/naiad/ime_test.py``)."""

import importlib.util
import shutil

import pytest

from experiments.naiad.dwarfs import NAIAD_CC, load_spec, run_point
from framework.naiad import DESIGNS_DIR

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)

IME = DESIGNS_DIR / "naiad-ime.yaml"
IME_TIMED = DESIGNS_DIR / "naiad-ime-timed.yaml"
needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


@needs_clang
@pytest.mark.parametrize(
    "pair", ["fp32:fp32", "fp64:fp64", "bf16:fp32", "fp16:fp32", "bf16:bf16"]
)
def test_gemm_dwarf_ime_variant(tmp_path, pair):
    spec = load_spec("gemm")
    assert spec.VARIANTS["ime"] == "ime"
    in_fmt, acc_fmt = pair.split(":")
    r = run_point(
        spec, "ime", 32, in_fmt, acc_fmt, design=IME_TIMED, build_root=tmp_path
    )
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    assert r.counters["matrix.macs"] == 32**3
    assert 0 < r.region_cycles < r.cycles


@needs_clang
def test_gemm_dwarf_ime_ragged_edges(tmp_path):
    spec = load_spec("gemm")
    for size in ((40, 24, 20), (17, 9, 7)):
        for in_fmt, acc_fmt in (("fp32", "fp32"), ("bf16", "fp32")):
            r = run_point(
                spec, "ime", size, in_fmt, acc_fmt, design=IME, build_root=tmp_path
            )
            assert r.passed, (size, in_fmt, r.error, r.console)
            m, n, k = size
            assert r.counters["matrix.macs"] >= m * n * k
