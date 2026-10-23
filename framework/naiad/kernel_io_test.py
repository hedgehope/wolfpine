import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from framework.naiad.kernel_io import run_kernel

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "kernels" / "naiad"
DESIGN = REPO / "framework" / "naiad" / "designs" / "naiad-base-rv64.yaml"
NAIAD_CC = "/opt/llvm/20.1.7/bin/clang"


@pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)
def test_inputs_reach_the_kernel_and_outputs_come_back(tmp_path):
    subprocess.run(
        [
            "make",
            "-C",
            str(KERNELS),
            f"BUILD={tmp_path}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
            "PROG=iotest",
            f"DESIGN={DESIGN}",
        ],
        check=True,
        capture_output=True,
    )
    x = np.arange(10, dtype=np.int64) - 3
    run = run_kernel(
        DESIGN,
        tmp_path / "iotest.elf",
        inputs={"n": 10, "x": x},
        outputs={"y": (np.int64, (10,))},
    )
    assert run.ok
    np.testing.assert_array_equal(run.outputs["y"], 2 * x + 10)
    assert run.counters["instret"] > 0
