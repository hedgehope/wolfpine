"""The gemm dwarf's ``vme`` variant, driven by the emitted header, end to end.

Moved from ``framework/naiad/emit/gemm_vme_test.py``.
"""

import importlib.util
import shutil

import pytest

from experiments.naiad.dwarfs import load_spec, run_point
from framework.naiad import DESIGNS_DIR

VME_TIMED = DESIGNS_DIR / "naiad-vme-timed.yaml"
NAIAD_CC = "/opt/llvm/20.1.7/bin/clang"
needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
needs_ml_dtypes = pytest.mark.skipif(
    importlib.util.find_spec("ml_dtypes") is None, reason="needs ml_dtypes"
)


# -- end to end ------------------------------------------------------------------------------------


@needs_clang
@needs_ml_dtypes
@pytest.mark.parametrize(
    "size, pair",
    [
        (
            (32, 32, 32),
            ("fp64", "fp64"),
        ),  # N < the 2 x 4 block: a narrow variant, B in place
        (
            (64, 96, 64),
            ("fp32", "fp32"),
        ),  # 96 = 64 + 32: the full block, then the narrow one
        (
            (16, 64, 300),
            ("fp32", "fp32"),
        ),  # K past one chunk: C reloaded (the _acc path)
        (
            (33, 64, 40),
            ("bf16", "fp32"),
        ),  # K not a multiple of the body: the padded path
    ],
    ids=lambda v: "x".join(map(str, v)) if isinstance(v[0], int) else "/".join(v),
)
def test_emitted_driver_end_to_end(tmp_path, size, pair):
    r = run_point(
        load_spec("gemm"), "vme", size, *pair, design=VME_TIMED, build_root=tmp_path
    )
    assert r.exit_code == 0, r.console
    assert r.passed, r.error
