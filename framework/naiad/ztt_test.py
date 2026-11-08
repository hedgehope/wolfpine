"""``matrix: {mode: ztt}`` on a Naiad SoC: the shipped design, the builder, traps, the C kernel.

The trap tests hand-assemble their programs (``trap_test``'s helpers), so they need no
compiler; the kernel test builds ``kernels/naiad/ztt_gemm`` with clang when it is there.
"""

import os
import shutil
import subprocess
import sys

import pytest
import yaml

from framework.matrix import ztt
from framework.matrix.ztt import descriptor, encode
from framework.matrix.ztt_isa import RV64_ZTT_ISA, ZttHart
from framework.naiad import DESIGNS_DIR
from framework.naiad.builder import build
from framework.naiad.design import DesignError, load_design, parse_design
from framework.naiad.run import main
from framework.pe.rv.trap_test import A0, RES, S1, T0, T1, Prog, csrs, handler, run

ZTT_DESIGN = DESIGNS_DIR / "naiad-ztt06.yaml"
KERNELS = DESIGNS_DIR.parents[2] / "kernels" / "naiad"
NAIAD_CC = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")
MS = 1 << ztt.DEFAULT_MS_SHIFT


def _raw(**matrix):
    with open(ZTT_DESIGN) as f:
        raw = yaml.safe_load(f)
    raw["matrix"].update(matrix)
    return raw


def test_the_shipped_design_builds_with_the_unit_its_csrs_and_counters():
    soc = build(load_design(ZTT_DESIGN), count_all=True)
    core = soc.cores[0]
    assert isinstance(core.matrix, ZttHart)
    assert RV64_ZTT_ISA in core.isas
    assert core.matrix.spec.n == 16
    assert core.matrix.spec.acc_regs == 8
    c = soc.counters()
    for key in ("mma", "macs", "mem_to_m_bytes", "unsupported"):
        assert f"hart0.matrix.{key}" in c
    with pytest.raises(Exception, match="MS Off"):
        core.csrs.read(0xCC0)
    soc.reset()
    assert core.matrix.backend.owner is None


def test_a_shared_backend_is_one_object_for_every_hart():
    raw = _raw(backend="shared")
    raw["cores"]["count"] = 2
    soc = build(parse_design(raw))
    a, b = (c.matrix for c in soc.cores)
    assert a.backend is b.backend
    assert (a.hart_id, b.hart_id) == (0, 1)
    raw["matrix"]["backend"] = "dedicated"
    soc = build(parse_design(raw))
    assert soc.cores[0].matrix.backend is not soc.cores[1].matrix.backend


def test_a_ztt_unit_is_untimed():
    raw = _raw()
    raw["timing"] = {"matrix": {}}
    with pytest.raises(DesignError, match="no cycle model"):
        parse_design(raw)


def _enabled(p):
    """mtvec, MS = Initial, ame.acquire into A0."""
    p.setup(with_clint=False)
    p.li(T0, MS)
    p.emit(csrs(0x300, T0), encode("ame.acquire", rd=A0, rs1=0))
    return p


def test_an_illegal_ztt_instruction_traps_with_the_word_in_mtval(tmp_path):
    p = _enabled(Prog(64))
    bad = encode("mls.rm", md=16, rs1=T1)  # m16 on a 16-register unit
    bad_pc = p.pc()
    p.emit(bad)
    p.emit(
        p.sd(S1, A0, 40)
    )  # the handler skipped it: the acquire result lands in RES + 40
    p.exit()
    soc, result, res = run(
        tmp_path, p, handler(p, "skip"), design=load_design(ZTT_DESIGN)
    )
    assert result.exit_code == 0
    assert (res(0), res(1), res(2), res(4)) == (2, bad, bad_pc, 1)
    assert res(5) == 1  # GRANTED
    # mstatus in the handler: the grant set MS to Initial, and nothing has made it Dirty (no SD)
    assert (res(3) >> ztt.DEFAULT_MS_SHIFT) & 3 == ztt.MS_INITIAL
    assert not res(3) >> 63


def test_a_misaligned_square_load_traps_with_its_address(tmp_path):
    p = _enabled(Prog(64))
    p.li(T0, descriptor("fp32"))
    p.emit(encode("msettyp", md=4, rs1=T0))
    p.li(T1, RES + 0x102)
    load_pc = p.pc()
    p.emit(encode("mls.rm", md=4, rs1=T1))
    p.exit()
    _, result, res = run(
        tmp_path, p, handler(p, "skip"), design=load_design(ZTT_DESIGN)
    )
    assert result.exit_code == 0
    assert (res(0), res(1), res(2)) == (4, RES + 0x102, load_pc)


def test_ms_off_traps_even_ame_acquire(tmp_path):
    p = Prog(64).setup(with_clint=False)
    word = encode("ame.acquire", rd=A0, rs1=0)
    p.emit(word)
    p.exit()
    _, result, res = run(
        tmp_path, p, handler(p, "skip"), design=load_design(ZTT_DESIGN)
    )
    assert (res(0), res(1)) == (2, word)


needs_clang = pytest.mark.skipif(
    shutil.which(NAIAD_CC) is None or shutil.which("make") is None,
    reason=f"needs make and {NAIAD_CC}",
)


@needs_clang
def test_the_ztt_gemm_kernel_builds_runs_and_passes(tmp_path, capsys):
    args = [
        "make",
        "-C",
        str(KERNELS),
        f"BUILD={tmp_path}",
        f"PYTHON={sys.executable}",
        f"NAIAD_CC={NAIAD_CC}",
    ]
    subprocess.run(
        [*args, "PROG=ztt_gemm", f"DESIGN={ZTT_DESIGN}"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert main([str(ZTT_DESIGN), str(tmp_path / "ztt_gemm.elf")]) == 0
    out, err = capsys.readouterr()
    assert "MISMATCH" not in out
    assert out.endswith("ZTT_GEMM OK\n")
    assert "hart0.matrix.mma = 48" in err
    assert "hart0.matrix.unsupported = 1" in err
