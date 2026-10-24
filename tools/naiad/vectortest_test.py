"""Tests for tools/naiad/vectortest.py and the vector_env environment.

    ~/projects/riscv/venv/bin/python -m pytest tools/naiad/vectortest_test.py

The failure analysis is tested on a synthetic stage-2 source. The Spike tests
need the suite built by ``tools/naiad/build_vector_tests.sh`` (they skip
otherwise): they run real tests through the runner, and break one expected value
to check that the exit code names the failing check.
"""

import re
import subprocess

import pytest

from tools.naiad import vectortest as vt

STAGE2 = """\
# Generating tests for VL: 3, LMUL: m2, SEW: e32, Mask: false

# -------------- TEST BEGIN --------------
li t0, 3
vsetvli t1, t0, e32,m2,tu,mu
vadd.vv v4, v24, v26
# -------------- TEST END   --------------
la a0, resultdata

# Store v4 register group into result data area.
vsetvli t1, x0, e32,m2,tu,mu
vse32.v v4, (a0)

  TEST_CASE(6, t0, 0x10, ld t0, 0(a0); addi a0, a0, 8)
  TEST_CASE(7, t0, 0x20, ld t0, 0(a0); addi a0, a0, 8)
  TEST_CASE(8, t0, 0x30, ld t0, 0(a0); addi a0, a0, 8)
  TEST_CASE(9, t0, 0x40, ld t0, 0(a0); addi a0, a0, 8)
  TEST_CASE(10, t0, 0x50, ld t0, 0(a0); addi a0, a0, 8)
  TEST_CASE(11, t0, 0x1, csrr t0, vxsat)
"""


def test_instruction_of():
    assert vt.instruction_of("vadd_vv-3") == "vadd.vv"
    assert vt.instruction_of("vfwcvt_f_f_v-0") == "vfwcvt.f.f.v"


def test_explain_failure_names_the_register_word_and_code(tmp_path):
    src = tmp_path / "t.S"
    src.write_text(STAGE2)
    info = vt.explain_failure(src, 10)
    assert info["expected"] == 0x50
    assert info["word"] == 4
    assert info["stored"] == 4
    assert info["config"] == "VL: 3, LMUL: m2, SEW: e32, Mask: false"
    assert info["code"][-1] == "vadd.vv v4, v24, v26"
    text = vt.describe(info, actual=0x51)
    # VLEN 256 holds four 64-bit words per register: word 4 is v5's first.
    assert "v5 dword 0" in text
    assert "expected 0x50, got 0x51" in text
    assert "vxsat" in vt.describe(vt.explain_failure(src, 11))
    assert vt.explain_failure(src, 99) is None


def test_summarise_keeps_the_first_failing_part():
    results = [
        {"test": "vadd_vv-1", "insn": "vadd.vv", "status": "FAIL", "detail": "b"},
        {"test": "vadd_vv-0", "insn": "vadd.vv", "status": "PASS", "detail": ""},
        {"test": "vadd_vv-2", "insn": "vadd.vv", "status": "UNIMPL", "detail": "c"},
        {"test": "vsub_vv-0", "insn": "vsub.vv", "status": "PASS", "detail": ""},
    ]
    rows = vt.summarise(results)
    assert rows["vadd.vv"]["status"] == "FAIL"
    assert rows["vadd.vv"]["first"]["test"] == "vadd_vv-1"
    assert (rows["vadd.vv"]["passed"], rows["vadd.vv"]["parts"]) == (1, 3)
    assert rows["vsub.vv"]["status"] == "PASS"


def _suite():
    bin_dir = vt.DEFAULT_TESTS / "bin"
    if not (bin_dir / "vadd_vv-0").exists():
        pytest.skip("riscv-vector-tests not built (tools/naiad/build_vector_tests.sh)")
    from framework.naiad.difftest import find_spike

    spike = find_spike()
    if spike is None:
        pytest.skip("needs spike")
    return spike


def test_real_tests_pass_on_spike_through_the_runner():
    spike = _suite()
    for test in ("vadd_vv-0", "vfadd_vv-0", "vle32_v-0", "vfwmaccbf16_vv-0"):
        r = vt.run_one((test, str(vt.DEFAULT_TESTS), "spike", None, spike, 0, 120))
        assert r["status"] == "PASS", r


def test_a_broken_expectation_is_named_by_its_check_number(tmp_path):
    spike = _suite()
    import os

    src = (vt.DEFAULT_TESTS / "stage2" / "vadd_vv-0.S").read_text()
    cases = re.findall(r"TEST_CASE\((\d+), t0, (0x[0-9a-f]+)", src)
    number, value = cases[len(cases) // 2]
    broken = src.replace(
        f"TEST_CASE({number}, t0, {value},",
        f"TEST_CASE({number}, t0, {int(value, 16) ^ 1:#x},",
        1,
    )
    (tmp_path / "stage2").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "stage2" / "vadd_vv-0.S").write_text(broken)
    env = vt.REPO_ROOT / "tools" / "naiad" / "vector_env"
    macros = vt.DEFAULT_TESTS.parents[1] / "macros" / "general"
    cc = os.environ.get("NAIAD_CC", "/opt/llvm/20.1.7/bin/clang")
    march = vt.SPIKE_ISA.removesuffix("_zicntr_zvl256b")
    subprocess.run(
        [cc, "--target=riscv64-unknown-elf", f"-march={march}", "-mabi=lp64d", "-static", "-mcmodel=medany",
         "-nostdlib", "-nostartfiles", f"-I{env}", f"-I{macros}", f"-T{env / 'link.ld'}",
         str(tmp_path / "stage2" / "vadd_vv-0.S"), "-o", str(tmp_path / "bin" / "vadd_vv-0")],
        check=True,
    )  # fmt: skip
    r = vt.run_one(("vadd_vv-0", str(tmp_path), "spike", None, spike, 0, 120))
    assert r["status"] == "FAIL"
    assert r["detail"].startswith(f"check {number}: ")
    assert f"expected {int(value, 16) ^ 1:#x}" in r["detail"]
