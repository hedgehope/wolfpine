"""reduce's ime / vme variants: one emitted-GEMM-driver call against a ones operand, on IME / VME designs."""

import importlib.util
import shutil

import pytest

from experiments.naiad.dwarfs import NAIAD_CC, load_spec, run_point
from framework.naiad import DESIGNS_DIR

SPEC = load_spec("reduce")
SMALLEST = min(SPEC.SIZES)
DESIGNS = DESIGNS_DIR
#: variant -> (design, MACs per input element: IME's NB, VME's MB at that design's geometry)
CASES = {
    "ime": (DESIGNS / "naiad-ime-timed.yaml", 8),
    "vme": (DESIGNS / "naiad-vme-timed.yaml", 64),
}

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


def test_variants_listed_with_every_pair():
    for variant in CASES:
        assert SPEC.VARIANTS[variant] == "gemm"
        assert tuple(SPEC.VARIANT_FORMATS[variant]) == tuple(SPEC.FORMATS)


def test_cflags_name_the_formats_for_the_driver():
    flags = SPEC.cflags(SMALLEST, "bf16", "fp32")
    for want in ("-DREDUCE_IN=bf16", "-DREDUCE_ACC_UP=FP32", "-DREDUCE_IN_BF16=1"):
        assert want in flags


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize("variant", sorted(CASES))
@pytest.mark.parametrize(
    "formats", [("fp32", "fp32"), ("bf16", "fp32"), ("fp64", "fp64")]
)
def test_variant_matches_golden(variant, formats, tmp_path):
    design, macs_per_element = CASES[variant]
    r = run_point(SPEC, variant, SMALLEST, *formats, design=design, build_root=tmp_path)
    assert r.exit_code == 0, r.console
    assert r.region_cycles
    assert r.region_cycles > 0
    assert r.passed, r.error
    # The driver's padded block: every input element meets NB (IME) / MB (VME) ones.
    # fp64 halves IME's NB (C at SEW 64) and VME's MB (TEW-64 tiles).
    want = macs_per_element // (2 if formats[0] == "fp64" else 1)
    assert r.counters.get("matrix.macs", 0) == SMALLEST * want
