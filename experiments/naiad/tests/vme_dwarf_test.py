"""The gemm dwarf's ``vme`` variant end to end (moved from ``framework/naiad/vme_test.py``)."""

import importlib.util
import shutil

import pytest

from experiments.naiad.dwarfs import NAIAD_CC, load_spec, run_point
from framework.naiad import DESIGNS_DIR

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)

VME = DESIGNS_DIR / "naiad-vme.yaml"
VME_TIMED = DESIGNS_DIR / "naiad-vme-timed.yaml"
needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


# -- the gemm dwarf's vme variant ----------------------------------------------------------------------


def _formats():
    return load_spec("gemm").VARIANT_FORMATS["vme"]


@needs_clang
@pytest.mark.parametrize("pair", _formats(), ids="/".join)
def test_gemm_dwarf_vme_variant(tmp_path, pair):
    spec = load_spec("gemm")
    r = run_point(spec, "vme", 64, *pair, design=VME_TIMED, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    assert r.region_cycles
    assert r.counters["matrix.macs"] >= 64**3
    assert r.counters["matrix.mma"] > 0


@needs_clang
@pytest.mark.parametrize("size", [(40, 24, 20), (17, 9, 7)])
@pytest.mark.parametrize(
    "pair",
    [("fp32", "fp32"), ("bf16", "fp32"), ("fp64", "fp64"), ("bf16", "bf16")],
    ids="/".join,
)
def test_gemm_dwarf_vme_ragged_edges(tmp_path, size, pair):
    r = run_point(
        load_spec("gemm"), "vme", size, *pair, design=VME, build_root=tmp_path
    )
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
