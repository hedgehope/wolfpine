"""The gemm dwarf's ``tensix`` variant end to end on ``naiad-tensix.yaml`` (moved from ``framework/ip/tensix/tensix_test.py``)."""

import importlib.util
import shutil

import pytest

from experiments.naiad.dwarfs import load_spec, run_point
from framework.ip.tensix import IMAGES_DIR
from framework.ip.tensix import image as tensix_image
from framework.naiad import DESIGNS_DIR

TRACE_FP32 = IMAGES_DIR / "bh_gemm_bf16_fp32.trace"
TRACE_BF16 = IMAGES_DIR / "bh_gemm_bf16_bf16.trace"
DESIGN = DESIGNS_DIR / "naiad-tensix.yaml"
DWARFS_DESIGN = DESIGNS_DIR / "naiad-dwarfs.yaml"
HAVE_ML_DTYPES = importlib.util.find_spec("ml_dtypes") is not None

# -- end to end: the gemm dwarf's tensix variant ------------------------------

needs_clang = pytest.mark.skipif(
    shutil.which("/opt/llvm/20.1.7/bin/clang") is None
    or shutil.which("make") is None
    or not HAVE_ML_DTYPES,
    reason="needs make, clang and ml_dtypes",
)


@needs_clang
@pytest.mark.parametrize("acc_fmt", ["fp32", "bf16"])
def test_gemm_tensix_variant(tmp_path, acc_fmt):
    spec = load_spec("gemm")
    r = run_point(
        spec, "tensix", 64, "bf16", acc_fmt, design=DESIGN, build_root=tmp_path
    )
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
    assert 0 < r.region_cycles < r.cycles
    c = r.counters
    esize = 4 if acc_fmt == "fp32" else 2
    image = tensix_image.from_trace(TRACE_FP32 if acc_fmt == "fp32" else TRACE_BF16)
    # The image (boot + launch) and both operands in, C out: every byte through the DMA engine.
    operands = 2 * 64 * 64 * 2
    image_bytes = sum(len(d) for _, d in tensix_image._segments(image.boot))
    image_bytes += sum(len(d) for _, d in tensix_image._segments(image.launch))
    assert c["dma.scratchpad_to_tensix_bytes"] == image_bytes + operands
    assert c["dma.tensix_to_scratchpad_bytes"] == 64 * 64 * esize
    assert c["dma.errors"] == 0
    assert c["tensix.launches"] == 1
    assert 0 < c["tensix.busy_cycles"] < r.region_cycles


@needs_clang
def test_ame_on_the_tensix_preset_is_unchanged(tmp_path):
    """naiad-tensix is naiad-dwarfs plus two blocks: an AME point times identically on both."""
    spec = load_spec("gemm")
    a = run_point(
        spec, "ame", 64, "bf16", "fp32", design=DWARFS_DESIGN, build_root=tmp_path
    )
    b = run_point(spec, "ame", 64, "bf16", "fp32", design=DESIGN, build_root=tmp_path)
    assert a.passed
    assert b.passed
    assert (a.cycles, a.region_cycles) == (b.cycles, b.region_cycles)
