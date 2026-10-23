"""F / D on Naiad: the design wiring, and kernels/naiad/fpcheck end to end."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.run import main
from framework.pe.rv.isa.fd_isa import RV64_D_ISA, RV64_F_ISA

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS = REPO_ROOT / "kernels" / "naiad"
FP_DESIGN = DESIGNS_DIR / "naiad-fp-rv64.yaml"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")


def _cores(isa):
    blocks = [{"name": "dram", "type": "dram", "base": 0x80000000, "window": 0x100000}]
    return {"name": "t", "cores": {"xlen": 64, "isa": isa}, "blocks": blocks}


@pytest.mark.parametrize(
    "isa", [["i", "d"], ["i", "zfbfmin"], ["i", "m", "d", "zicsr"]]
)
def test_d_and_zfbfmin_require_f(isa):
    with pytest.raises(DesignError, match="require 'f'"):
        parse_design(_cores(isa))


def test_fp_design_builds_a_core_with_f_registers():
    design = load_design(FP_DESIGN)
    assert design.cores.march == "rv64imafd_zicsr_zicntr"
    assert design.cores.mabi == "lp64d"
    core = build(design).cores[0]
    assert RV64_F_ISA in core.isas
    assert RV64_D_ISA in core.isas
    assert core.register_file["f31"].read() == bytes(8)
    # misa reports F and D alongside I, M and A.
    misa = core.csrs.misa
    assert all(misa >> (ord(c) - ord("A")) & 1 for c in "IMAFD")


needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


@needs_clang
def test_fpcheck_builds_runs_and_passes(tmp_path, capsys):
    subprocess.run(
        [
            "make",
            "-C",
            str(KERNELS),
            f"BUILD={tmp_path}",
            f"PYTHON={sys.executable}",
            f"NAIAD_CC={NAIAD_CC}",
            f"DESIGN={FP_DESIGN}",
            "PROG=fpcheck",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert main([str(FP_DESIGN), str(tmp_path / "fpcheck.elf")]) == 0
    out, err = capsys.readouterr()
    assert "MISMATCH" not in out
    assert "double, fmadd: bfd8c93642e523a4 (expect bfd8c93642e523a4) ok" in out
    assert "fcvt.w.d 1e10 saturates: 000000007fffffff" in out
    assert out.endswith("PASS\n")
    assert "exit code 0" in err
