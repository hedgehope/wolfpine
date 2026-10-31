"""Build and run every driver/naiad example, through its Makefile, and assert it passes.

Each example checks itself and exits non-zero on a mismatch; ``make run-<ex>``
builds it against its design and runs it with ``wolfpine run naiad`` (ex5 also
runs its host driver, ex5.py). Skipped without make and clang (``NAIAD_CC``).

    pytest driver/naiad/examples_test.py
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")
EXAMPLES = ["ex1", "ex2", "ex3", "ex4", "ex5", "ex6", "ex7"]

needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


def make(target, build_dir):
    return subprocess.run(
        [
            "make",
            "-s",
            "-C",
            str(HERE),
            target,
            f"BUILD={build_dir}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
        ],
        capture_output=True,
        text=True,
    )


@needs_clang
@pytest.mark.parametrize("example", EXAMPLES)
def test_example_passes(example, tmp_path):
    run = make(f"run-{example}", tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.startswith(f"{example}: ") or f"\n{example}: " in run.stdout
    assert "MISMATCH" not in run.stdout
    assert run.stdout.endswith("PASS\n")
    assert re.search(r"naiad: exit code 0 after \d+ cycles", run.stderr), run.stderr


@needs_clang
def test_ex5_host_driver_checks_c_against_numpy(tmp_path):
    assert make("ex5", tmp_path).returncode == 0
    spec = importlib.util.spec_from_file_location("ex5_host", HERE / "ex5" / "ex5.py")
    ex5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex5)
    result, host_ok = ex5.run(tmp_path / "ex5" / "ex5.elf", shape=(17, 50, 33), seed=7)
    assert result.ok
    assert host_ok
    assert "ex5: 17 x 50 x 33 int8 GEMM" in result.console
    assert result.console.endswith("PASS\n")
